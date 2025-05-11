import torch
import argparse
from safetensors.torch import load_file, save_file
import os
import json  # Added for SVDQuant
import importlib  # Added for SVDQuant

# Imports from our new modules
from quantization import stochastic_round_tensor_to_fp8

# Make sure SVDQuantLinear and other necessary utils are importable
from quantization.svdquant_layer import SVDQuantLinear
from quantization.calibration_utils import (
    find_optimal_alpha_svdquant,
)  # Already used by SVDQuantLinear
from quantization.svd_utils import (
    decompose_weights_svd,
)  # Already used by SVDQuantLinear
from quantization.smoothing_utils import (
    calculate_smoothing_factors,
    apply_smoothing_to_weights,
    apply_smoothing_to_activations,
)  # Used by SVDQuantLinear or helpers
from quantization.gptq_utils import (
    gptq_quantize_layer_residual_refined,
)  # Used by SVDQuantLinear


from scaling import get_fp8_constants_for_owlscale
from plotting_utils import MATPLOTLIB_AVAILABLE, generate_comparison_plots

"""
Quantizes a .safetensors model to FP8 using various stochastic rounding techniques,
or to SVDQuant (4-bit with low-rank decomposition) format.

FP8 Path:
This script provides multiple methods for converting floating-point tensors 
(typically FP32, FP16, or BF16) within a .safetensors file to FP8 (E4M3FN or E5M2 formats).
Stochastic rounding is employed instead of deterministic rounding (like Round-to-Nearest-Even)
to introduce a controlled amount of noise, which can sometimes help preserve model performance
by ensuring that, on average, the rounding errors do not systematically bias values up or down.

SVDQuant Path:
Replaces nn.Linear layers with SVDQuantLinear layers, which use a low-rank decomposition
(L1, L2) for the main weights and quantize the residual (R) to a low bitwidth (e.g., 4-bit)
using GPTQ. This path requires a model architecture and calibration data.
"""

# Global dtypes for Owlscale - these are set in main() if --owlscale is used.
OWLSCALE_COMPUTE_DTYPE = None
OWLSCALE_SCALE_DTYPE = None
OWLSCALE_FP8_MIN = None
OWLSCALE_FP8_MAX = None
OWLSCALE_FP8_MIN_POS = None


# --- Helper functions for SVDQuant ---
def get_model_architecture_from_str(
    loader_str: str, arch_params_json_str: str, device: torch.device
):
    """
    Loads a model architecture using a string like 'module.submodule:function_name'.
    This function needs to be adapted by the user based on their project structure.

    Args:
        loader_str (str): String specifying the module and function to load the model.
        arch_params_json_str (str): JSON string of keyword arguments for the loader function.
        device (torch.device): Device to initially load the model on.

    Returns:
        torch.nn.Module: The loaded model architecture.
    """
    print(
        f"Attempting to load model architecture from: {loader_str} with params: {arch_params_json_str}"
    )
    try:
        module_path, func_name = loader_str.rsplit(":", 1)
        module = importlib.import_module(module_path)
        loader_func = getattr(module, func_name)
        arch_params = json.loads(arch_params_json_str)
        # Model should be loaded to CPU first, then moved to device after state_dict load
        model = loader_func(**arch_params)
        print(f"Model architecture loaded successfully from {loader_str}.")
        return model
    except Exception as e:
        print(f"Error loading model architecture from '{loader_str}': {e}")
        print(
            "Please ensure --model_arch_loader points to a valid Python module and function (e.g., 'myproject.models:get_my_model')"
        )
        print(
            "and that the function can be called with arguments provided in --model_arch_params."
        )
        raise


def get_target_linear_layers_info(model: torch.nn.Module, target_keys: list[str]):
    """
    Identifies nn.Linear layers in the model that match the target_keys (name or suffix).
    Returns a list of tuples: (name, module_instance, parent_module, weight_clone, bias_clone).
    Weight and bias are cloned to CPU to avoid modification if model is on GPU.
    """
    infos = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(name == tk or name.endswith(tk) for tk in target_keys):
                # Get parent module and the name of the child (linear layer) within the parent
                parent_module = model
                name_parts = name.split(".")
                child_name_in_parent = name_parts[-1]
                if len(name_parts) > 1:
                    for part in name_parts[:-1]:
                        parent_module = getattr(parent_module, part)

                weight_clone = module.weight.detach().clone().cpu()
                bias_clone = (
                    module.bias.detach().clone().cpu()
                    if module.bias is not None
                    else None
                )
                infos.append(
                    (
                        name,
                        module,
                        parent_module,
                        child_name_in_parent,
                        weight_clone,
                        bias_clone,
                    )
                )
    if not infos:
        print(f"Warning: No nn.Linear layers found matching target keys: {target_keys}")
    return infos


captured_inputs_for_layers = {}


def capture_input_hook(layer_name):
    def hook(module, input, output):
        # input is a tuple. We are interested in the first element for nn.Linear.
        if layer_name not in captured_inputs_for_layers:
            captured_inputs_for_layers[layer_name] = []
        captured_inputs_for_layers[layer_name].append(input[0].detach().cpu())

    return hook


def run_calibration_pass_for_svd(
    model: torch.nn.Module,
    calibration_input_batches: list[torch.Tensor],
    target_layer_names: list[str],
    device: torch.device,
    calibration_batch_size: int,  # Used for creating DataLoader if needed
):
    """
    Runs calibration data through the model and captures input activations for specified layers.
    Args:
        model (torch.nn.Module): The model (with original weights).
        calibration_input_batches (list[torch.Tensor]): A list of input tensor batches for the model.
                                                     Each tensor is a batch.
        target_layer_names (list[str]): Names of nn.Linear layers to capture inputs for.
        device (torch.device): Device to run calibration on.
        calibration_batch_size (int): Batch size to use if creating a DataLoader.
                                     Currently assumes calibration_input_batches are already batched.

    Returns:
        dict: {layer_name: concatenated_input_activations_tensor (on CPU)}
    """
    global captured_inputs_for_layers
    captured_inputs_for_layers = {}  # Reset for each call

    hooks = []
    for layer_name in target_layer_names:
        try:
            module_to_hook = model.get_submodule(layer_name)
            if isinstance(module_to_hook, torch.nn.Linear):
                hook_handle = module_to_hook.register_forward_hook(
                    capture_input_hook(layer_name)
                )
                hooks.append(hook_handle)
            else:
                print(
                    f"Warning: Module {layer_name} is not nn.Linear, cannot attach hook."
                )
        except AttributeError:
            print(f"Warning: Could not find submodule {layer_name} to attach hook.")

    if not hooks:
        print(
            "Error: No valid hooks attached for calibration. Aborting SVDQuant calibration pass."
        )
        return {}

    model.eval()  # Ensure model is in eval mode
    model.to(device)

    print(
        f"  Running calibration forward pass with {len(calibration_input_batches)} batches..."
    )
    with torch.no_grad():
        for i, batch_input in enumerate(calibration_input_batches):
            if isinstance(batch_input, torch.Tensor):  # If it's a single tensor batch
                # Model might expect a tuple or dict, adapt if necessary
                model(batch_input.to(device))
            elif isinstance(
                batch_input, (list, tuple)
            ):  # If batch_input is a list/tuple of tensors
                model(*(t.to(device) for t in batch_input))
            elif isinstance(batch_input, dict):  # If batch_input is a dict of tensors
                model(**{k: v.to(device) for k, v in batch_input.items()})
            else:
                print(
                    f"Warning: Calibration batch {i} has an unsupported type: {type(batch_input)}. Skipping."
                )
                continue
            if i % 10 == 0 and i > 0:
                print(f"    Processed {i} calibration batches...")

    for hook_handle in hooks:
        hook_handle.remove()

    # Collate captured activations
    collated_activations = {}
    for layer_name, act_list in captured_inputs_for_layers.items():
        if act_list:
            collated_activations[layer_name] = torch.cat(act_list, dim=0)
            print(
                f"  Captured {collated_activations[layer_name].shape[0]} activation samples for {layer_name}, shape: {collated_activations[layer_name].shape}"
            )
        else:
            print(f"Warning: No activations captured for {layer_name}.")

    captured_inputs_for_layers = {}  # Clear global cache
    model.to("cpu")  # Move model back to CPU after calibration
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return collated_activations


# --- End of Helper functions for SVDQuant ---


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a .safetensors model to FP8 with stochastic rounding, or to SVDQuant format."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input .safetensors model file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the quantized .safetensors model file.",
    )
    # FP8 specific arguments
    parser.add_argument(
        "--fp8_type",
        type=str,
        default="e4m3",
        choices=["e4m3", "e5m2"],
        help="FP8 type to use for FP8 quantization path: e4m3 (torch.float8_e4m3fn) or e5m2 (torch.float8_e5m2).",
    )
    parser.add_argument(
        "--keys_to_quantize_suffix",
        type=str,
        nargs="+",
        default=[".weight", ".bias"],
        help="Suffixes of keys to identify tensors for FP8 quantization (e.g., '.weight' '.bias').",
    )
    parser.add_argument(
        "--complex_rounding",
        action="store_true",
        help="Use a more complex neighbor finding method for FP8 stochastic rounding.",
    )
    parser.add_argument(
        "--shifturb",
        action="store_true",
        help="Use shift-and-perturb (additive noise) FP8 stochastic rounding method.",
    )
    parser.add_argument(
        "--owlshift",
        action="store_true",
        help="Use owlshift (manual stochastic mantissa rounding) FP8 method.",
    )
    parser.add_argument(
        "--owlscale",
        action="store_true",
        help="Apply per-tensor max-abs scaling before FP8 stochastic rounding (from reference script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stochastic rounding methods that use it (e.g., owlshift).",
    )

    # Common arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computations (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable generation of comparison plots (requires matplotlib). Saved to --plot_dir. (FP8 path only)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./quant_plots/",
        help="Directory to save generated plots. (FP8 path only)",
    )
    parser.add_argument(
        "--plot_max_tensors",
        type=int,
        default=5,
        help="Maximum number of tensors for which to generate plots. (FP8 path only)",
    )
    parser.add_argument(
        "--plot_sample_size",
        type=int,
        default=5000,
        help="Number of points to sample for scatter plots of large tensors. (FP8 path only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug print statements to trace execution flow and flag states.",
    )

    # SVDQuant specific arguments
    parser.add_argument(
        "--svd_quantize",
        action="store_true",
        help="Enable SVDQuant: replace nn.Linear with SVDQuantLinear and prepare them.",
    )
    parser.add_argument(
        "--model_arch_loader",
        type=str,
        default=None,  # e.g., "my_project.models:get_my_unet_model"
        help="Specify a function to load the model architecture, e.g., 'module:function_name'. Required if --svd_quantize is used.",
    )
    parser.add_argument(
        "--model_arch_params",
        type=str,
        default="{}",
        help="JSON string of keyword arguments for the model architecture loader function.",
    )
    parser.add_argument(
        "--svd_target_linear_keys",
        type=str,
        nargs="+",
        default=[
            "Linear"
        ],  # Example, user needs to be specific, e.g. ".weight" or specific layer names
        help="Full names or suffixes of nn.Linear layers to target for SVDQuant. Use module names, not state_dict key suffixes.",
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=32,
        help="Rank for SVD decomposition in SVDQuant.",
    )
    parser.add_argument(
        "--svd_num_bits",
        type=int,
        default=4,
        help="Number of bits for residual quantization in SVDQuant (typically 4).",
    )
    parser.add_argument(
        "--svd_group_size",
        type=int,
        default=64,
        help="Group size for residual quantization in SVDQuant.",
    )
    parser.add_argument(
        "--svd_alpha",
        type=float,
        default=0.5,
        help="Default smoothing alpha for SVDQuant (if not performing search).",
    )
    parser.add_argument(
        "--svd_low_rank_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for L1, L2 low-rank factors in SVDQuant.",
    )
    parser.add_argument(
        "--svd_scale_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for quantization scales in SVDQuant.",
    )
    parser.add_argument(
        "--svd_perform_alpha_search",
        action="store_true",
        help="Perform alpha search for SVDQuant smoothing.",
    )
    parser.add_argument(
        "--svd_alpha_search_values",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="List of alpha values to search for SVDQuant smoothing.",
    )
    parser.add_argument(
        "--svd_gptq_percdamp",
        type=float,
        default=0.01,
        help="Percentage dampening for Hessian inverse in SVDQuant GPTQ.",
    )
    parser.add_argument(
        "--svd_gptq_act_order",
        action="store_true",
        help="Enable heuristic column reordering (act_order) in SVDQuant GPTQ.",
    )
    parser.add_argument(
        "--svd_gptq_compensation_strength",
        type=float,
        default=0.1,
        help="Error compensation strength in SVDQuant GPTQ for residual.",
    )
    parser.add_argument(
        "--svd_gptq_verbose",
        action="store_true",
        help="Enable verbose output from the SVDQuant GPTQ process.",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help="Path to a .pt file containing calibration data (e.g., a list of input tensor batches). Required if --svd_quantize is used.",
    )
    parser.add_argument(
        "--calibration_batch_size",  # This argument is currently informational for run_calibration_pass_for_svd
        type=int,  # as it assumes pre-batched data. Could be used if TensorDataset is used.
        default=8,
        help="Batch size for processing calibration data if creating a DataLoader internally (currently assumes pre-batched data in list).",
    )

    args = parser.parse_args()
    main_device = torch.device(args.device)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    if args.svd_quantize:
        # --- SVDQuant Path ---
        if not args.model_arch_loader:
            print("Error: --model_arch_loader is required when --svd_quantize is used.")
            return
        if not args.calibration_file:
            print("Error: --calibration_file is required when --svd_quantize is used.")
            return
        if not os.path.exists(args.calibration_file):
            print(f"Error: Calibration file not found: {args.calibration_file}")
            return

        # 1. Load Model Architecture (on CPU initially)
        print("SVDQuant Path: Loading model architecture...")
        model_arch_params_dict = json.loads(args.model_arch_params)
        # Model loaded to CPU by default in helper
        model = get_model_architecture_from_str(
            args.model_arch_loader, args.model_arch_params, torch.device("cpu")
        )

        # 2. Load Original State Dict into the model structure
        print(
            f"SVDQuant Path: Loading original state_dict from: {args.input_file} into model structure."
        )
        original_state_dict = load_file(args.input_file, device="cpu")
        try:
            model.load_state_dict(original_state_dict, strict=True)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print(
                "This might be due to architectural mismatches or if the SVDQuant process was already partially run."
            )
            print(
                "Ensure the model architecture matches the state_dict from --input_file."
            )
            return
        model.eval()
        print("SVDQuant Path: Original state_dict loaded into model.")

        # 3. Prepare Calibration Data
        print(f"SVDQuant Path: Loading calibration data from: {args.calibration_file}")
        # Expecting a list of pre-batched tensors, or a single tensor to be used as one batch.
        # Adapt this loading logic if your calibration_file has a different structure.
        raw_calibration_data = torch.load(args.calibration_file, map_location="cpu")
        if isinstance(
            raw_calibration_data, torch.Tensor
        ):  # If it's a single tensor, make it a list of one batch
            calibration_input_batches = [raw_calibration_data]
        elif isinstance(raw_calibration_data, list) and all(
            isinstance(t, torch.Tensor) for t in raw_calibration_data
        ):
            calibration_input_batches = raw_calibration_data
        else:
            print(
                f"Error: Calibration file {args.calibration_file} must contain a torch.Tensor or a list of torch.Tensors."
            )
            return
        print(
            f"SVDQuant Path: Loaded {len(calibration_input_batches)} calibration batch(es). Total samples (approx first batch): {calibration_input_batches[0].shape[0] if calibration_input_batches else 0}"
        )

        # 4. Identify Target Linear Layers
        print("SVDQuant Path: Identifying target nn.Linear layers...")
        target_linear_layers_info = get_target_linear_layers_info(
            model, args.svd_target_linear_keys
        )
        if not target_linear_layers_info:
            print("No target linear layers found for SVDQuant. Exiting.")
            return
        print(
            f"SVDQuant Path: Found {len(target_linear_layers_info)} target linear layers."
        )

        # 5. Run Calibration Pass to get Input Activations for these specific layers (model on main_device)
        print("SVDQuant Path: Running calibration pass to capture input activations...")
        captured_activations = run_calibration_pass_for_svd(
            model,
            calibration_input_batches,
            [info[0] for info in target_linear_layers_info],
            main_device,
            args.calibration_batch_size,
        )

        # 6. Replace Target Layers with SVDQuantLinear and Prepare them
        print(
            "SVDQuant Path: Replacing nn.Linear layers with SVDQuantLinear and preparing them..."
        )
        svd_low_rank_torch_dtype = getattr(torch, args.svd_low_rank_dtype)
        svd_scale_torch_dtype = getattr(torch, args.svd_scale_dtype)

        temp_model_for_replacement = model.to("cpu")  # Perform layer replacement on CPU

        for (
            layer_name,
            _,
            parent_module,
            child_name_in_parent,
            original_weight_cpu,
            original_bias_cpu,
        ) in target_linear_layers_info:
            print(f"  Processing SVDQuant for layer: {layer_name}")

            # Get original module to extract features, then replace
            original_linear_module = getattr(parent_module, child_name_in_parent)

            svd_layer = SVDQuantLinear(
                original_in_features=original_linear_module.in_features,
                original_out_features=original_linear_module.out_features,
                rank=args.svd_rank,
                num_bits=args.svd_num_bits,
                group_size=args.svd_group_size,
                alpha=args.svd_alpha,
                bias=original_bias_cpu is not None,
                low_rank_dtype=svd_low_rank_torch_dtype,
                scale_dtype=svd_scale_torch_dtype,
                device=main_device,  # Intended device for parameters after prepare
                dtype=original_weight_cpu.dtype,
            )
            setattr(parent_module, child_name_in_parent, svd_layer)

            layer_calib_activations = captured_activations.get(layer_name)
            if layer_calib_activations is None:
                print(
                    f"  Warning: No calibration activations found for {layer_name}. Skipping prepare."
                )
                continue

            # Ensure SVDLayer is on the target device before calling prepare, its params will be created there.
            svd_layer.to(main_device)

            print(
                f"    Preparing SVDQuantLinear for {layer_name} on device {main_device}..."
            )
            svd_layer.prepare(
                original_weight_tensor=original_weight_cpu.to(main_device),
                calibration_activations=layer_calib_activations.to(main_device),
                original_bias_tensor=(
                    original_bias_cpu.to(main_device)
                    if original_bias_cpu is not None
                    else None
                ),
                perform_alpha_search=args.svd_perform_alpha_search,
                alpha_search_values=args.svd_alpha_search_values,
                verbose_alpha_search=args.debug,
                gptq_percdamp=args.svd_gptq_percdamp,
                gptq_act_order=args.svd_gptq_act_order,
                gptq_compensation_strength=args.svd_gptq_compensation_strength,
                gptq_verbose=args.svd_gptq_verbose or args.debug,
            )
            print(f"  Layer {layer_name} prepared for SVDQuant.")
            # Move layer back to CPU after prepare if model is on CPU.
            # The main model instance will be moved to CPU before saving state_dict.
            if main_device.type != "cpu":
                svd_layer.to("cpu")  # Parameters are now on CPU
                torch.cuda.empty_cache()

        # 7. Save the Quantized State Dict
        final_model_for_state_dict = temp_model_for_replacement.to(
            "cpu"
        )  # Ensure entire model is on CPU for state_dict
        print("SVDQuant Path: Saving SVDQuantized model state_dict...")
        quantized_state_dict = final_model_for_state_dict.state_dict()

        save_file(quantized_state_dict, args.output_file)
        print(f"SVDQuantized model saved to: {args.output_file}")
        return  # End of SVDQuant path

    # --- Existing FP8 Quantization Path ---
    else:
        if args.fp8_type == "e4m3":
            fp8_dtype = torch.float8_e4m3fn
        elif args.fp8_type == "e5m2":
            fp8_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported fp8_type: {args.fp8_type}")

        print(f"FP8 Path: Loading model from: {args.input_file}")
        try:
            state_dict = load_file(args.input_file, device="cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        quantized_state_dict = {}
        total_tensors = len(state_dict)

        print(
            f"FP8 Path: Target FP8 type: {fp8_dtype}, Device for quantization: {main_device}"
        )
        print(
            f"FP8 Path: Will attempt to quantize tensors with keys ending in: {args.keys_to_quantize_suffix}"
        )
        if args.complex_rounding:
            print(
                "FP8 Path: Using complex neighbor finding method for stochastic rounding."
            )
        if args.shifturb:
            print(
                "FP8 Path: Using shift-and-perturb (additive noise) stochastic rounding."
            )
        if args.owlshift:
            print(
                f"FP8 Path: Using owlshift (manual stochastic mantissa rounding) with seed: {args.seed}."
            )
        if args.owlscale:
            print(
                f"FP8 Path: Using owlscale (per-tensor max-abs scaling pre-quantization)."
            )
            global OWLSCALE_COMPUTE_DTYPE, OWLSCALE_SCALE_DTYPE
            global OWLSCALE_FP8_MIN, OWLSCALE_FP8_MAX, OWLSCALE_FP8_MIN_POS
            OWLSCALE_COMPUTE_DTYPE = torch.float64
            OWLSCALE_SCALE_DTYPE = torch.float64
            OWLSCALE_FP8_MIN, OWLSCALE_FP8_MAX, OWLSCALE_FP8_MIN_POS = (
                get_fp8_constants_for_owlscale(fp8_dtype)
            )
            print(
                f"  Owlscale FP8 Target: {fp8_dtype}, Range: [{OWLSCALE_FP8_MIN}, {OWLSCALE_FP8_MAX}], Min Pos: {OWLSCALE_FP8_MIN_POS:.2e}"
            )

        if args.plot and not MATPLOTLIB_AVAILABLE:
            print(
                "Warning: --plot was specified, but matplotlib is not installed. Plotting will be disabled."
            )
            args.plot = False
        if args.plot:
            os.makedirs(args.plot_dir, exist_ok=True)
            print(f"Plots will be saved to: {args.plot_dir}")

        main_seed = args.seed
        plots_generated_count = 0

        for i, (key, tensor) in enumerate(state_dict.items()):
            original_tensor_for_plot_and_type = None
            if args.plot and plots_generated_count < args.plot_max_tensors:
                original_tensor_for_plot_and_type = tensor.detach().clone().cpu()

            scale_factor_for_comfyui_to_save = None

            if args.debug:
                print(
                    f"DEBUG main loop for tensor {key}: complex_flag={args.complex_rounding}, shifturb_flag={args.shifturb}, owlshift_flag={args.owlshift}, owlscale_flag={args.owlscale}"
                )
                if tensor.numel() > 0:
                    print(
                        f"  Tensor {key} before any processing: min={tensor.min().item():.4g}, max={tensor.max().item():.4g}, mean={tensor.mean().item():.4g}, isfinite={torch.isfinite(tensor).all().item()}, dtype={tensor.dtype}"
                    )
                else:
                    print(f"  Tensor {key} is empty.")

            print(
                f"Processing tensor {i+1}/{total_tensors}: {key} (dtype: {tensor.dtype}, shape: {tensor.shape})",
                end="",
            )

            original_tensor_device = tensor.device  # Should be CPU from load_file
            tensor_to_process = tensor.to(main_device)

            if (
                args.owlscale
                and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
                and tensor_to_process.is_floating_point()
                and (
                    not key.endswith(".bias")
                )  # Exclude bias from ComfyUI-style owlscale path
            ):
                print(" (Applying ComfyUI-style scaled FP8 quantization)...", end="")

                # 1. Determine scale_factor_for_comfyui
                original_hp_tensor = tensor_to_process.to(OWLSCALE_COMPUTE_DTYPE)
                abs_max = torch.max(torch.abs(original_hp_tensor))

                scale_factor_for_comfyui_val = abs_max.clamp(min=1e-12)

                scale_factor_for_comfyui_to_save = scale_factor_for_comfyui_val.to(
                    OWLSCALE_SCALE_DTYPE
                ).to(original_tensor_device)

                # 2. Prepare tensor for quantization: original_tensor / scale
                input_for_quantization_hp = (
                    tensor_to_process.to(scale_factor_for_comfyui_val.dtype)
                    / scale_factor_for_comfyui_val
                )
                input_for_quantization = input_for_quantization_hp.to(
                    tensor_to_process.dtype
                )

                if args.debug:
                    if input_for_quantization.numel() > 0:
                        print(
                            f"  Tensor {key} input_for_quantization (orig / scale): min={input_for_quantization.min().item():.4g}, max={input_for_quantization.max().item():.4g}, mean={input_for_quantization.mean().item():.4g}, isfinite={torch.isfinite(input_for_quantization).all().item()}, dtype={input_for_quantization.dtype}"
                        )
                    else:
                        print(
                            f"  Tensor {key} input_for_quantization (orig / scale) is empty."
                        )

                # 3. Quantize the scaled tensor
                quantized_tensor = stochastic_round_tensor_to_fp8(
                    input_for_quantization,
                    fp8_dtype,
                    args.complex_rounding,
                    args.shifturb,
                    args.owlshift,
                    seed=main_seed + i,
                    debug_mode=args.debug,
                )

                # 4. Save quantized tensor and the scale factor
                quantized_state_dict[key] = quantized_tensor.to(original_tensor_device)

                _scale_key_to_use_ = ""  # Renamed from scale_key in user code to avoid conflict if used later
                scale_key_parts = key.split(".")
                if scale_key_parts[-1] == "weight":
                    scale_key_parts[-1] = "scale_weight"
                    _scale_key_to_use_ = ".".join(scale_key_parts)
                else:
                    _scale_key_to_use_ = key + ".scale_absmax"

                if _scale_key_to_use_ != key:
                    quantized_state_dict[_scale_key_to_use_] = (
                        scale_factor_for_comfyui_to_save
                    )
                else:
                    # This print should be conditional on args.debug or be a warning
                    print(
                        f" Warning: Could not determine unique scale key for {key} (generated scale key was the same: {_scale_key_to_use_}). Scale not saved separately."
                    )

                quantized_count += 1
                # The following print statements replace the general " Done." that was previously outside this if/elif/else block
                print(
                    f" Done. New dtype: {quantized_tensor.dtype}", end=""
                )  # Keep end="" to allow scale factor print on same conceptual line
                if scale_factor_for_comfyui_to_save is not None:
                    print(
                        f" Scale factor for {key} (as {_scale_key_to_use_}): {scale_factor_for_comfyui_to_save.item():.5g}",
                        end="",
                    )
                # The final newline will be handled by the outer loop's print(" Done.") structure if that's kept,
                # or this line needs its own implicit newline if the outer " Done." is removed.
                # For now, ending this block's specific prints.

            elif (
                any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
                and tensor_to_process.is_floating_point()
            ):
                print(" (Applying general FP8 quantization)...", end="")
                quantized_tensor_fp8 = stochastic_round_tensor_to_fp8(
                    tensor=tensor_to_process,
                    fp8_dtype=fp8_dtype,
                    use_complex_method=args.complex_rounding,
                    use_shift_perturb_method=args.shifturb,
                    use_owlshift_method=args.owlshift,
                    seed=(
                        main_seed + i if args.owlshift else main_seed
                    ),  # Vary seed per tensor for owlshift
                    debug_mode=args.debug,
                )
                quantized_state_dict[key] = quantized_tensor_fp8.to(
                    original_tensor_device
                )  # Store on CPU
            else:
                print(" (Skipping quantization, copying as is)...", end="")
                quantized_state_dict[key] = tensor.to(
                    original_tensor_device
                )  # Copy to CPU

            print(" Done.")

            if (
                args.plot
                and original_tensor_for_plot_and_type is not None
                and plots_generated_count < args.plot_max_tensors
                and key in quantized_state_dict
            ):
                if tensor_to_process.is_floating_point() and any(
                    key.endswith(suffix) for suffix in args.keys_to_quantize_suffix
                ):
                    print(f"  Generating plot for {key}...")
                    # Prepare arguments for generate_comparison_plots
                    original_tensor_cpu_for_plot = (
                        original_tensor_for_plot_and_type  # Already on CPU
                    )
                    quantized_tensor_cpu_for_plot = quantized_state_dict[key].cpu()
                    dequantized_tensor_cpu_for_plot = quantized_tensor_cpu_for_plot.to(
                        original_tensor_cpu_for_plot.dtype
                    )

                    # Construct a safe filename for the plot
                    safe_tensor_key_for_filename = key.replace("/", "_").replace(
                        ".", "_"
                    )
                    plot_file_path = os.path.join(
                        args.plot_dir, f"{safe_tensor_key_for_filename}_comparison.png"
                    )

                    original_dtype_name = str(original_tensor_cpu_for_plot.dtype)
                    fp8_dtype_name = str(
                        fp8_dtype
                    )  # fp8_dtype is like torch.float8_e4m3fn

                    generate_comparison_plots(
                        original_tensor_cpu=original_tensor_cpu_for_plot,
                        quantized_fp8_tensor_cpu=quantized_tensor_cpu_for_plot,
                        dequantized_tensor_cpu=dequantized_tensor_cpu_for_plot,
                        tensor_key=key,
                        plot_filename=plot_file_path,
                        original_dtype_str=original_dtype_name,
                        fp8_dtype_str=fp8_dtype_name,
                        sample_size=args.plot_sample_size,
                    )
                    plots_generated_count += 1

            if main_device.type == "cuda":
                torch.cuda.empty_cache()

        print("Saving FP8 quantized model...")
        save_file(quantized_state_dict, args.output_file)
        print(f"FP8 Quantized model saved to: {args.output_file}")


if __name__ == "__main__":
    main()
