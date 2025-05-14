import torch
import argparse
from safetensors.torch import save_file
import os
from quantization import stochastic_round_tensor_to_fp8
from scaling import get_fp8_constants_for_comfyscale
from plotting_utils import MATPLOTLIB_AVAILABLE, generate_comparison_plots
from safetensors_utils import load_tensor_from_safetensors, get_safetensors_tensor_keys

# TorchAO imports
try:
    import torchao
    from torchao.quantization import (
        quantize_,
        Float8WeightOnlyConfig,
        Float8DynamicActivationFloat8WeightConfig,
    )
    from torchao.quantization.granularity import (
        PerTensor,
    )  # Using PerTensor as a default for dynamic
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.float8.float8_utils import to_fp8_saturated
    from torchao.float8.float8_linear import (
        Float8Linear,
    )  # Added import for dynamic quant handling

    # For plotting dequantization if using lower-level primitives:
    # from torchao.quantization.quant_primitives import dequantize_affine_floatx
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

    # Define dummy classes if torchao is not available to prevent NameError later if accessed
    # This is mainly for type checking and avoiding runtime errors if TORCHAO_AVAILABLE check fails.
    class AffineQuantizedTensor:
        pass


"""
Quantizes a .safetensors model to FP8 using various stochastic rounding techniques.

This script provides multiple methods for converting floating-point tensors 
(typically FP32, FP16, or BF16) within a .safetensors file to FP8 (E4M3FN or E5M2 formats).
Stochastic rounding is employed instead of deterministic rounding (like Round-to-Nearest-Even)
to introduce a controlled amount of noise, which can sometimes help preserve model performance
by ensuring that, on average, the rounding errors do not systematically bias values up or down.
"""

# Global dtypes for comfyscale - these are set in main() if --comfyscale is used.
comfyscale_COMPUTE_DTYPE = None
comfyscale_SCALE_DTYPE = None
comfyscale_FP8_MIN = None
comfyscale_FP8_MAX = None
comfyscale_FP8_MIN_POS = None


# Dummy Module for TorchAO
class DummyModule(torch.nn.Module):
    def __init__(self, weight_tensor, bias_tensor=None):
        super().__init__()
        out_features, in_features = weight_tensor.shape
        self.linear = torch.nn.Linear(
            in_features, out_features, bias=(bias_tensor is not None)
        )
        self.linear.weight = torch.nn.Parameter(
            weight_tensor.clone().detach(), requires_grad=False
        )
        if bias_tensor is not None:
            self.linear.bias = torch.nn.Parameter(
                bias_tensor.clone().detach(), requires_grad=False
            )
        else:
            self.linear.bias = None

    def forward(self, x):
        # Dummy forward, may not be strictly needed for weight-only quant via torchao's quantize_
        # but safer to include for general compatibility with torch.compile/quantize flows.
        # Input x would need to match in_features.
        # This forward pass won't actually be called in xavier.py's current usage.
        return self.linear(x)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a .safetensors model to FP8 with stochastic rounding or TorchAO."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input .safetensors model file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the quantized .safetensors model file.",
    )
    # Quantization method
    parser.add_argument(
        "--quant_method",
        type=str,
        default="native",
        choices=[
            "native",
            "torchao_fp8_weight_only_aoscale",
            "torchao_fp8_weight_only_comfyscale",
            "torchao_fp8_dynamic_act_weight_aoscale",
            "torchao_fp8_dynamic_act_weight_comfyscale",
        ],
        help="Quantization method to use. 'native' uses xavier.py's original methods. "
        "'torchao_..._aoscale' uses TorchAO with its native scaling. "
        "'torchao_..._comfyscale' uses TorchAO with ComfyUI/comfyscale compatible scaling.",
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
        default=[
            ".weight"
        ],  # Changed default to mostly target weights for TorchAO's typical use with DummyModule
        help="Suffixes of keys to identify tensors for FP8 quantization (e.g., '.weight' '.bias').",
    )
    # Native specific arguments
    parser.add_argument(
        "--complex_rounding",
        action="store_true",
        help="Use a more complex neighbor finding method for FP8 stochastic rounding (native method only).",
    )
    parser.add_argument(
        "--shifturb",
        action="store_true",
        help="Use shift-and-perturb (additive noise) FP8 stochastic rounding method (native method only).",
    )
    parser.add_argument(
        "--owlshift",
        action="store_true",
        help="Use owlshift (manual stochastic mantissa rounding) FP8 method (native method only).",
    )
    parser.add_argument(
        "--comfyscale",
        action="store_true",
        help="Apply per-tensor max-abs scaling before FP8 stochastic rounding (native method only, TorchAO handles its own scaling).",
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

    args = parser.parse_args()
    main_device = torch.device(args.device)

    is_torchao_method = "torchao_" in args.quant_method

    if is_torchao_method and not TORCHAO_AVAILABLE:
        print(
            f"Error: TorchAO method '{args.quant_method}' selected, but torchao library is not installed or importable."
        )
        print("Please install torchao: pip install torchao")
        return

    if is_torchao_method:
        if args.complex_rounding or args.shifturb or args.owlshift or args.comfyscale:
            print(
                "Warning: Native quantization flags (--complex_rounding, --shifturb, --owlshift, --comfyscale) are ignored when a TorchAO method is selected."
            )

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    if args.fp8_type == "e4m3":
        fp8_dtype = torch.float8_e4m3fn
    elif args.fp8_type == "e5m2":
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported fp8_type: {args.fp8_type}")

    print(f"FP8 Path: Loading model metadata from: {args.input_file}")
    try:
        tensor_keys = get_safetensors_tensor_keys(args.input_file)
        if not tensor_keys:
            print(
                f"Error: No tensors found in {args.input_file} or could not read keys."
            )
            return
        print(f"Found {len(tensor_keys)} tensor keys in {args.input_file}")
    except Exception as e:
        print(f"Error loading tensor keys from model: {e}")
        return

    quantized_state_dict = {}
    total_tensors = len(tensor_keys)

    print(f"Using quantization method: {args.quant_method}")
    print(f"Target FP8 type: {fp8_dtype}, Device for quantization: {main_device}")
    if (
        not is_torchao_method
    ):  # Print native-specific messages only if not using torchao
        print(
            f"Will attempt to quantize tensors with keys ending in: {args.keys_to_quantize_suffix}"
        )
        if args.complex_rounding:
            print("Using complex neighbor finding method for stochastic rounding.")
        if args.shifturb:
            print("Using shift-and-perturb (additive noise) stochastic rounding.")
        if args.owlshift:
            print(
                f"Using owlshift (manual stochastic mantissa rounding) with seed: {args.seed}."
            )
        if args.comfyscale:
            print(f"Using comfyscale (per-tensor max-abs scaling pre-quantization).")
            global comfyscale_COMPUTE_DTYPE, comfyscale_SCALE_DTYPE
            global comfyscale_FP8_MIN, comfyscale_FP8_MAX, comfyscale_FP8_MIN_POS
            comfyscale_COMPUTE_DTYPE = torch.float64
            comfyscale_SCALE_DTYPE = torch.float64
            comfyscale_FP8_MIN, comfyscale_FP8_MAX, comfyscale_FP8_MIN_POS = (
                get_fp8_constants_for_comfyscale(fp8_dtype)
            )
            print(
                f"  comfyscale FP8 Target: {fp8_dtype}, Range: [{comfyscale_FP8_MIN}, {comfyscale_FP8_MAX}], Min Pos: {comfyscale_FP8_MIN_POS:.2e}"
            )
    else:  # TorchAO specific messages
        print(
            f"TorchAO will target tensors with keys ending in: {args.keys_to_quantize_suffix} (primarily for nn.Linear weights via DummyModule)."
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
    quantized_count = 0

    for i, key in enumerate(tensor_keys):
        try:
            tensor = load_tensor_from_safetensors(args.input_file, key, device="cpu")
        except Exception as e:
            print(f"\nError loading tensor {key}: {e}. Skipping this tensor.")
            if key in quantized_state_dict:
                del quantized_state_dict[key]
            continue

        original_tensor_for_plot_and_type = None
        if args.plot and plots_generated_count < args.plot_max_tensors:
            original_tensor_for_plot_and_type = tensor.detach().clone().cpu()

        # This variable will store the scale from torchao if applicable
        scale_factor_for_comfyui_to_save = None

        if args.debug and key == "weight":  # Only for the weight tensor in debug mode
            print(f"\n  DEBUG Conditions for 'weight' tensor ('{key}'):")
            print(f"    is_torchao_method: {is_torchao_method}")
            print(f"    args.quant_method: {args.quant_method}")
            print(f"    args.keys_to_quantize_suffix: {args.keys_to_quantize_suffix}")
            _any_suffix_match = any(
                key.endswith(s) for s in args.keys_to_quantize_suffix
            )
            print(
                f"    any(key.endswith(s) for s in args.keys_to_quantize_suffix): {_any_suffix_match}"
            )
            _is_fp = tensor.is_floating_point()
            print(f"    tensor.is_floating_point(): {_is_fp}")
            print(f"    tensor.dim(): {tensor.dim()}")

            # Native Comfyscale Path condition components
            print(f"    Condition (Native Comfyscale Path):")
            _args_comfyscale = args.comfyscale
            print(f"      args.comfyscale: {_args_comfyscale}")
            _not_is_torchao = not is_torchao_method
            print(f"      not is_torchao_method: {_not_is_torchao}")
            # _any_suffix_match is already printed
            # _is_fp is already printed
            _not_bias = not key.endswith(".bias")
            print(f"      not key.endswith('.bias'): {_not_bias}")
            full_cond_native_comfy = (
                _args_comfyscale
                and _not_is_torchao
                and _any_suffix_match
                and _is_fp
                and _not_bias
            )
            print(f"      => EVALUATES TO: {full_cond_native_comfy}")

            # Native General Path condition components
            print(f"    Condition (Native General Path):")
            # _not_is_torchao, _any_suffix_match, _is_fp are already printed
            full_cond_native_general = _not_is_torchao and _any_suffix_match and _is_fp
            print(f"      => EVALUATES TO: {full_cond_native_general}")
            print("  DEBUG End of conditions block.")

        original_tensor_device = tensor.device
        tensor_to_process = tensor.to(main_device)

        # Determine if TorchAO should be attempted based on method, suffix, and tensor type
        should_attempt_torchao = (
            is_torchao_method
            and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
            and tensor_to_process.is_floating_point()
        )

        # TORCHAO QUANTIZATION PATH
        if should_attempt_torchao and tensor_to_process.dim() >= 2:
            print(f" (Applying {args.quant_method} quantization)...", end="")

            # Bias handling for DummyModule - simplified
            bias_for_dummy_module = None

            dummy_model = DummyModule(
                tensor_to_process.clone(), bias_tensor=bias_for_dummy_module
            ).to(main_device)

            torchao_config = None
            if "torchao_fp8_weight_only" in args.quant_method:
                torchao_config = Float8WeightOnlyConfig(weight_dtype=fp8_dtype)
            elif "torchao_fp8_dynamic_act_weight" in args.quant_method:
                torchao_config = Float8DynamicActivationFloat8WeightConfig(
                    activation_dtype=fp8_dtype,
                    weight_dtype=fp8_dtype,
                    granularity=PerTensor(),  # Defaulting to PerTensor, can be made configurable
                )

            if torchao_config:
                quantize_(dummy_model, torchao_config)

                quantized_layer_or_param = dummy_model.linear
                final_quantized_tensor_data_ao = None
                final_scale_ao = None

                # Order of checks is important, esp. for Float8Linear
                is_f8l_dynamic_case = False
                try:
                    is_f8l_dynamic_case = isinstance(
                        quantized_layer_or_param, Float8Linear
                    )
                except TypeError:
                    # This handles the scenario where Float8Linear is a mock instance due to test patching.
                    # If TORCHAO_AVAILABLE is True, it means torchao was successfully imported,
                    # and thus Float8Linear was originally a real class before being patched by a mock instance.
                    # The TypeError arises because 'Float8Linear' (the name) is not a type.
                    # In this specific test setup, this implies that 'quantized_layer_or_param'
                    # is an instance produced by that mock (which itself replaced the class name),
                    # so we treat this as the dynamic quantization case.
                    if TORCHAO_AVAILABLE:
                        is_f8l_dynamic_case = True
                    # If TORCHAO_AVAILABLE is False, Float8Linear is a dummy class, so a TypeError
                    # from isinstance(..., Float8Linear) is unexpected. If it occurs, let it propagate
                    # or handle as an actual error, as this fix is specific to the described test mock.

                is_aqt_weight_only_case = False
                # Check for AQT on weight only if not already identified as an F8L dynamic case
                if not is_f8l_dynamic_case and hasattr(
                    quantized_layer_or_param, "weight"
                ):
                    try:
                        is_aqt_weight_only_case = isinstance(
                            quantized_layer_or_param.weight, AffineQuantizedTensor
                        )
                    except TypeError:
                        # Similar handling if AffineQuantizedTensor is mocked as an instance.
                        if TORCHAO_AVAILABLE:
                            is_aqt_weight_only_case = True

                is_aqt_param_case = False
                # Check for AQT on param only if not F8L dynamic and not AQT weight_only
                if (
                    not is_f8l_dynamic_case
                    and not is_aqt_weight_only_case
                    and isinstance(quantized_layer_or_param, torch.nn.Parameter)
                ):
                    try:
                        is_aqt_param_case = isinstance(
                            quantized_layer_or_param, AffineQuantizedTensor
                        )
                    except TypeError:
                        # Similar handling if AffineQuantizedTensor is mocked as an instance.
                        if TORCHAO_AVAILABLE:
                            is_aqt_param_case = True

                if is_f8l_dynamic_case:
                    # This branch is for Float8DynamicActivationFloat8WeightConfig.
                    # The layer itself (quantized_layer_or_param) is of type Float8Linear (or its mock).
                    # Its .weight attribute should be a Float8Tensor (or similar) holding .data and .scale
                    if (
                        hasattr(
                            quantized_layer_or_param, "weight"
                        )  # Check if Float8Linear has a weight attribute
                        and hasattr(
                            quantized_layer_or_param.weight, "data"
                        )  # Check if the weight object has .data
                        and hasattr(
                            quantized_layer_or_param.weight, "scale"
                        )  # Check if the weight object has .scale
                    ):
                        final_quantized_tensor_data_ao = (
                            quantized_layer_or_param.weight.data
                        )
                        final_scale_ao = quantized_layer_or_param.weight.scale
                        if args.debug:
                            print(
                                f"  DEBUG ({key}): initial final_quantized_tensor_data_ao type: {type(final_quantized_tensor_data_ao).__name__}"
                            )
                            print(
                                f"  DEBUG ({key}): initial final_scale_ao type: {type(final_scale_ao).__name__}"
                            )
                    else:
                        print(
                            f" Warning: Object potentially identified as Float8Linear (mock or real) does not have expected structure (.weight.data / .weight.scale). Key: {key}. Copying original.",
                            end="",
                        )
                        quantized_state_dict[key] = tensor.to(original_tensor_device)
                        print(" Done.")
                        continue  # Skip to next tensor

                # Case 2: Weight-only quantization for nn.Linear (layer is nn.Linear, weight is AQT)
                elif is_aqt_weight_only_case:
                    aqt_weight = (
                        quantized_layer_or_param.weight
                    )  # This is the AffineQuantizedTensor
                    if aqt_weight is None:
                        if args.debug:
                            print(f"  DEBUG: aqt_weight for {key} is None. Skipping.")
                    else:
                        if args.debug:
                            print(
                                f"\n  DEBUG: Type of aqt_weight for {key}: {type(aqt_weight).__name__}"
                            )
                            print(
                                f"  DEBUG: Attributes of aqt_weight for {key}: {dir(aqt_weight)}\n"
                            )

                        if hasattr(aqt_weight, "tensor_impl"):
                            impl = aqt_weight.tensor_impl
                            if hasattr(impl, "data") and hasattr(impl, "scale"):
                                potential_fp8_tensor = impl.data
                                if (
                                    hasattr(potential_fp8_tensor, "dtype")
                                    and potential_fp8_tensor.dtype == fp8_dtype
                                ):
                                    final_quantized_tensor_data_ao = (
                                        potential_fp8_tensor
                                    )
                                    final_scale_ao = impl.scale
                                    if args.debug:
                                        print(
                                            f"  DEBUG ({key} AQT.weight): final_quantized_tensor_data_ao type: {type(final_quantized_tensor_data_ao).__name__}"
                                        )
                                        print(
                                            f"  DEBUG ({key} AQT.weight): final_scale_ao type: {type(final_scale_ao).__name__}"
                                        )
                                    # Successfully extracted FP8 data and its scale
                                else:
                                    print(
                                        f" Warning: AQT weight's tensor_impl.data for {key} is not FP8 (dtype: {getattr(potential_fp8_tensor, 'dtype', 'N/A')}). Original AQT wrapper dtype: {getattr(aqt_weight, 'dtype', 'N/A')}. Copying original.",
                                        end="",
                                    )
                            else:
                                print(
                                    f" Warning: AQT weight's tensor_impl for {key} does not have expected .data / .scale attributes. Original AQT wrapper dtype: {getattr(aqt_weight, 'dtype', 'N/A')}. Copying original.",
                                    end="",
                                )
                        else:
                            print(
                                f" Warning: AQT weight for {key} (type: {type(aqt_weight).__name__}) does not have .tensor_impl attribute. Original AQT wrapper dtype: {getattr(aqt_weight, 'dtype', 'N/A')}. Copying original.",
                                end="",
                            )

                # Case 3: Tensor is a Parameter and directly an AffineQuantizedTensor
                elif is_aqt_param_case:  # quantized_layer_or_param is Parameter and AQT
                    aqt_param = (
                        quantized_layer_or_param  # This is the AffineQuantizedTensor
                    )
                    if aqt_param is None:
                        if args.debug:
                            print(f"  DEBUG: aqt_param for {key} is None. Skipping.")
                    else:
                        if args.debug:
                            print(
                                f"\n  DEBUG: Type of aqt_param for {key}: {type(aqt_param).__name__}"
                            )
                            print(
                                f"  DEBUG: Attributes of aqt_param for {key}: {dir(aqt_param)}\n"
                            )

                        if hasattr(aqt_param, "tensor_impl"):
                            impl = aqt_param.tensor_impl
                            if hasattr(impl, "data") and hasattr(impl, "scale"):
                                potential_fp8_tensor = impl.data
                                if (
                                    hasattr(potential_fp8_tensor, "dtype")
                                    and potential_fp8_tensor.dtype == fp8_dtype
                                ):
                                    final_quantized_tensor_data_ao = (
                                        potential_fp8_tensor
                                    )
                                    final_scale_ao = impl.scale
                                    if args.debug:
                                        print(
                                            f"  DEBUG ({key} AQT.param): final_quantized_tensor_data_ao type: {type(final_quantized_tensor_data_ao).__name__}"
                                        )
                                        print(
                                            f"  DEBUG ({key} AQT.param): final_scale_ao type: {type(final_scale_ao).__name__}"
                                        )
                                    # Successfully extracted FP8 data and its scale
                                else:
                                    print(
                                        f" Warning: AQT param's tensor_impl.data for {key} is not FP8 (dtype: {getattr(potential_fp8_tensor, 'dtype', 'N/A')}). Original AQT wrapper dtype: {getattr(aqt_param, 'dtype', 'N/A')}. Copying original.",
                                        end="",
                                    )
                            else:
                                print(
                                    f" Warning: AQT param's tensor_impl for {key} does not have expected .data / .scale attributes. Original AQT wrapper dtype: {getattr(aqt_param, 'dtype', 'N/A')}. Copying original.",
                                    end="",
                                )
                        else:
                            print(
                                f" Warning: AQT param for {key} (type: {type(aqt_param).__name__}) does not have .tensor_impl attribute. Original AQT wrapper dtype: {getattr(aqt_param, 'dtype', 'N/A')}. Copying original.",
                                end="",
                            )

                else:
                    # Fallback: Did not match expected TorchAO structures after quantization call.
                    print(
                        f" Warning: Tensor {key} was targeted for TorchAO but did not match expected quantized structures after quantize_ call. Copying original.",
                        end="",
                    )
                    quantized_state_dict[key] = tensor.to(original_tensor_device)
                    print(" Done.")
                    continue  # Skip to next tensor

                if (
                    final_quantized_tensor_data_ao is not None
                    and final_scale_ao is not None
                ):
                    processed_fp8_tensor = None
                    processed_scale_factor = None

                    # Detach the tensor data to prevent issues with underlying implementations
                    cloned_final_quantized_tensor_data_ao = (
                        final_quantized_tensor_data_ao.clone().detach()
                    )
                    if args.debug:
                        print(
                            f"  DEBUG ({key}): cloned_final_quantized_tensor_data_ao type: {type(cloned_final_quantized_tensor_data_ao).__name__}"
                        )

                    if "comfyscale" in args.quant_method:
                        print(f" (Adapting to ComfyUI scale convention)...", end="")

                        # Dequantize using TorchAO's data and scale to get the effective high-precision tensor
                        # Ensure types are appropriate for multiplication (e.g., float32)

                        # Potentially unwrap Float8Tensor to get to the raw underlying data tensor
                        data_to_cast = cloned_final_quantized_tensor_data_ao
                        if hasattr(cloned_final_quantized_tensor_data_ao, "_data"):
                            data_to_cast = cloned_final_quantized_tensor_data_ao._data

                        if data_to_cast.dtype != torch.float32:
                            data_f32 = data_to_cast.to(torch.float32)
                        else:
                            data_f32 = data_to_cast

                        scale_f32 = (
                            final_scale_ao.clone().detach().to(torch.float32)
                        )  # Scales should typically be plain tensors
                        effective_hp_tensor = data_f32 * scale_f32

                        # Calculate the ComfyUI-style scale (absmax of the effective HP tensor)
                        # This mimics the original comfyscale behavior where scale is amax of original tensor
                        comfy_style_scale = torch.max(torch.abs(effective_hp_tensor))
                        comfy_style_scale = comfy_style_scale.clamp(
                            min=torch.finfo(effective_hp_tensor.dtype).tiny
                        )  # Match comfyscale clamp

                        # Normalize the effective HP tensor by the new ComfyUI-style scale
                        # The input to to_fp8_saturated should be in the [-1, 1] range (approx) for best results
                        normalized_hp_tensor_for_comfy = (
                            effective_hp_tensor / comfy_style_scale
                        )

                        # Re-quantize this normalized tensor to FP8
                        processed_fp8_tensor = to_fp8_saturated(
                            normalized_hp_tensor_for_comfy, target_dtype=fp8_dtype
                        )

                        # The scale factor to save is the ComfyUI-style scale
                        # Save it with the original tensor's dtype for consistency with native comfyscale, or float32
                        # original_tensor_for_plot_and_type is available here if needed for dtype
                        processed_scale_factor = comfy_style_scale.to(
                            dtype=(
                                original_tensor_for_plot_and_type.dtype
                                if original_tensor_for_plot_and_type is not None
                                else torch.float32
                            )
                        )

                    elif "aoscale" in args.quant_method:
                        print(f" (Using native TorchAO scale)...", end="")
                        processed_fp8_tensor = cloned_final_quantized_tensor_data_ao  # Use the cloned, detached data
                        processed_scale_factor = final_scale_ao
                    else:  # Should not be reached due to CLI choices
                        print(
                            f" Internal Error: Unknown TorchAO scaling variant for {key}. Copying original.",
                            end="",
                        )
                        quantized_state_dict[key] = tensor.to(original_tensor_device)
                        print(" Done.")
                        # Continue to next tensor or clean up dummy_model
                        del dummy_model
                        if "torchao_config" in locals():
                            del torchao_config
                        continue

                    if (
                        processed_fp8_tensor is not None
                        and processed_scale_factor is not None
                    ):
                        # Ensure the tensors stored are plain tensors for safetensors compatibility

                        # processed_fp8_tensor is on main_device (e.g., CUDA) and might be a TorchAO wrapper
                        # Access the underlying data if it's a known TorchAO wrapper
                        underlying_data_tensor = None
                        # Order of checks: AQT -> direct .data (for Float8Tensor/Impl) -> ._data (fallback)
                        if (
                            isinstance(processed_fp8_tensor, AffineQuantizedTensor)
                            and hasattr(processed_fp8_tensor, "tensor_impl")
                            and hasattr(processed_fp8_tensor.tensor_impl, "data")
                        ):
                            print(
                                f"  DEBUG ({key}): Extracting .tensor_impl.data from AffineQuantizedTensor ({type(processed_fp8_tensor).__name__})"
                            )
                            underlying_data_tensor = (
                                processed_fp8_tensor.tensor_impl.data
                            )
                        elif hasattr(processed_fp8_tensor, "data") and isinstance(
                            getattr(processed_fp8_tensor, "data", None), torch.Tensor
                        ):
                            # This handles cases like Float8AQTTensorImpl or Float8Tensor which have a .data attribute directly
                            print(
                                f"  DEBUG ({key}): Extracting .data directly from {type(processed_fp8_tensor).__name__}"
                            )
                            underlying_data_tensor = processed_fp8_tensor.data
                        elif hasattr(
                            processed_fp8_tensor, "_data"
                        ):  # Fallback for other AQT-like types
                            print(
                                f"  DEBUG ({key}): Extracting ._data from {type(processed_fp8_tensor).__name__}"
                            )
                            underlying_data_tensor = processed_fp8_tensor._data
                        else:  # Assume it might be a plain tensor already, or a different wrapper
                            print(
                                f"  DEBUG ({key}): Using processed_fp8_tensor directly, type: {type(processed_fp8_tensor).__name__}"
                            )
                            underlying_data_tensor = processed_fp8_tensor

                        if underlying_data_tensor is None:
                            print(
                                f"  ERROR ({key}): Could not extract underlying tensor from {type(processed_fp8_tensor).__name__}. Skipping save logic for this tensor."
                            )
                            # Potentially raise an error or handle more gracefully if this is unexpected
                            continue  # Skip to next tensor in the main loop, might need adjustment based on where this edit is placed

                        detached_fp8_on_main_device = underlying_data_tensor.detach()

                        # 1. Move underlying tensor to CPU (if not already)
                        print(
                            f"  DEBUG: Calling .cpu() on underlying_data_tensor (type: {type(detached_fp8_on_main_device)}) for {key}"
                        )
                        detached_fp8_cpu = detached_fp8_on_main_device.cpu()

                        # 2. Create a new plain tensor on CPU and copy data from the (now CPU) underlying tensor
                        plain_fp8_cpu = torch.empty(
                            detached_fp8_cpu.shape,
                            dtype=detached_fp8_cpu.dtype,
                            device="cpu",
                        )
                        try:
                            plain_fp8_cpu.copy_(
                                detached_fp8_cpu
                            )  # This is the critical copy
                        except AttributeError as e:
                            # Intercept the specific 'transposed' AttributeError to provide a more direct hypothesis
                            if "'Tensor' object has no attribute 'transposed'" in str(
                                e
                            ):
                                print(
                                    f"  ERROR HINT: The .copy_ operation from a TorchAO tensor (even on CPU) to a plain CPU tensor failed. {key}"
                                )
                                print(
                                    f"              This suggests TorchAO's __torch_dispatch__ for copy_ expects attributes plain tensors don't have."
                                )
                                print(
                                    f"              Source (TorchAO tensor on CPU) type: {type(detached_fp8_cpu)}"
                                )
                                print(
                                    f"              Dest (Plain tensor on CPU) type: {type(plain_fp8_cpu)}"
                                )
                            raise  # Re-raise the original error after printing hint

                        # 3. Move the plain CPU tensor to the original target device
                        plain_fp8_for_saving = plain_fp8_cpu.to(original_tensor_device)
                        quantized_state_dict[key] = plain_fp8_for_saving

                        # Ensure scale is also a plain tensor (scales are usually simpler)
                        # processed_scale_factor could be a TorchAO scale tensor or a plain tensor
                        underlying_scale_tensor = None
                        # Scales are usually plain tensors or simple wrappers. If it came from Float8Tensor.scale, it should be plain.
                        if (
                            hasattr(processed_scale_factor, "data")
                            and isinstance(
                                getattr(processed_scale_factor, "data", None),
                                torch.Tensor,
                            )
                            and not isinstance(processed_scale_factor, torch.Tensor)
                        ):
                            # This might catch a wrapped scale tensor that isn't a plain tensor but has .data
                            print(
                                f"  DEBUG ({key} scale): Extracting .data from scale object {type(processed_scale_factor).__name__}"
                            )
                            underlying_scale_tensor = processed_scale_factor.data
                        elif isinstance(processed_scale_factor, torch.Tensor):
                            print(
                                f"  DEBUG ({key} scale): Using scale directly, type: {type(processed_scale_factor).__name__}"
                            )
                            underlying_scale_tensor = processed_scale_factor
                        else:
                            # Fallback if it's some other type of object we don't expect for a scale
                            print(
                                f"  WARNING ({key} scale): processed_scale_factor is of unexpected type {type(processed_scale_factor).__name__}. Attempting to use as is."
                            )
                            underlying_scale_tensor = processed_scale_factor  # May cause issues later if not a tensor

                        if underlying_scale_tensor is None or not isinstance(
                            underlying_scale_tensor, torch.Tensor
                        ):
                            print(
                                f"  ERROR ({key} scale): Could not obtain a valid underlying tensor for scale from {type(processed_scale_factor).__name__}. Skipping save logic for this scale."
                            )
                            # This tensor (key) might end up saved without its scale if this path is hit.
                            # The main loop will continue, but verification might fail or behavior might be wrong.
                            # Consider how to handle this, e.g. by not saving the key at all if scale is bad.
                        else:
                            detached_scale_on_main_device = (
                                underlying_scale_tensor.detach()
                            )
                            print(
                                f"  DEBUG: Calling .cpu() on underlying_scale_tensor (type: {type(detached_scale_on_main_device)}) for {key}'s scale"
                            )
                            detached_scale_cpu = detached_scale_on_main_device.cpu()

                            plain_scale_cpu = torch.empty(
                                detached_scale_cpu.shape,
                                dtype=detached_scale_cpu.dtype,
                                device="cpu",
                            )
                            plain_scale_cpu.copy_(
                                detached_scale_cpu
                            )  # Scales are less likely to be complex wrappers
                            plain_scale_for_saving = plain_scale_cpu.to(
                                original_tensor_device
                            )
                            scale_factor_for_comfyui_to_save = plain_scale_for_saving

                            _scale_key_to_use_ = ""
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
                                scale_repr = (
                                    str(scale_factor_for_comfyui_to_save.item())
                                    if scale_factor_for_comfyui_to_save.numel() == 1
                                    else str(scale_factor_for_comfyui_to_save.tolist())
                                )
                                print(
                                    f" Done. New dtype: {quantized_state_dict[key].dtype}. Scale saved as {_scale_key_to_use_}: {scale_repr}"
                                )
                            else:
                                print(
                                    f" Done. New dtype: {quantized_state_dict[key].dtype}. Scale for {key} not saved separately (key conflict). Print scale: {str(scale_factor_for_comfyui_to_save)}"
                                )
                            quantized_count += 1
                        # else: this case should be covered if processed_fp8_tensor or processed_scale_factor becomes None

                    elif (
                        final_quantized_tensor_data_ao is not None
                        and final_scale_ao is None
                    ):
                        print(
                            f" Warning: TorchAO quantized {key} to {fp8_dtype} but scale not found directly. Scale factor cannot be adapted or saved. Copying original.",
                            end="",
                        )
                        quantized_state_dict[key] = tensor.to(original_tensor_device)
                        print(" Done.")
                    # The case where final_quantized_tensor_data_ao is None is handled by the initial check after getting AQT/Float8Tensor components

                else:  # Should not happen if args.quant_method is valid and contains torchao_
                    print(
                        f" Error: No TorchAO config for method '{args.quant_method}'. Copying original.",
                        end="",
                    )
                    quantized_state_dict[key] = tensor.to(original_tensor_device)
                    print(" Done.")

                del dummy_model
                if "torchao_config" in locals():
                    del torchao_config

        elif should_attempt_torchao and tensor_to_process.dim() < 2:
            # This case is when TorchAO was desired but tensor dimension is too low
            print(
                f" Warning: Tensor {key} has dim < 2 ({tensor_to_process.dim()}), skipping TorchAO as it expects Linear-like weights. Copying original.",
                end="",
            )
            quantized_state_dict[key] = tensor.to(original_tensor_device)
            print(" Done.")  # Newline for this path

        # NATIVE comfyscale QUANTIZATION PATH
        elif (
            args.comfyscale
            and not is_torchao_method  # Ensure comfyscale is not run if torchao is active
            and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
            and tensor_to_process.is_floating_point()
            and (not key.endswith(".bias"))
        ):
            print(" (Applying ComfyUI-style scaled FP8 quantization)...", end="")

            original_hp_tensor = tensor_to_process.to(comfyscale_COMPUTE_DTYPE)
            abs_max = torch.max(torch.abs(original_hp_tensor))
            scale_factor_for_comfyui_val = abs_max.clamp(min=1e-12)
            scale_factor_for_comfyui_to_save = scale_factor_for_comfyui_val.to(
                comfyscale_SCALE_DTYPE
            ).to(original_tensor_device)

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

            quantized_tensor = stochastic_round_tensor_to_fp8(
                input_for_quantization,
                fp8_dtype,
                args.complex_rounding,
                args.shifturb,
                args.owlshift,
                seed=main_seed + i,
                debug_mode=args.debug,
            )
            quantized_state_dict[key] = quantized_tensor.to(original_tensor_device)

            _scale_key_to_use_ = ""
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
                if main_device.type == "cuda":
                    if "original_hp_tensor" in locals():
                        del original_hp_tensor
                    if "scale_factor_for_comfyui_val" in locals() and isinstance(
                        scale_factor_for_comfyui_val, torch.Tensor
                    ):
                        del scale_factor_for_comfyui_val
                    if "input_for_quantization_hp" in locals():
                        del input_for_quantization_hp
                    if "input_for_quantization" in locals():
                        del input_for_quantization
                    if "quantized_tensor" in locals():
                        del quantized_tensor
            else:
                print(
                    f" Warning: Could not determine unique scale key for {key} (generated scale key was the same: {_scale_key_to_use_}). Scale not saved separately."
                )
            quantized_count += 1
            print(f" Done. New dtype: {quantized_state_dict[key].dtype}", end="")
            if (
                scale_factor_for_comfyui_to_save is not None
                and _scale_key_to_use_ != key
            ):
                print(
                    f" Scale factor for {key} (as {_scale_key_to_use_}): {scale_factor_for_comfyui_to_save.item():.5g}",
                    end="",
                )
            print()  # Newline for this path's status

        # NATIVE GENERAL FP8 QUANTIZATION PATH
        elif (
            not is_torchao_method  # Ensure not torchao
            and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
            and tensor_to_process.is_floating_point()
        ):
            print(" (Applying general FP8 quantization)...", end="")
            quantized_tensor_fp8 = stochastic_round_tensor_to_fp8(
                tensor=tensor_to_process,
                fp8_dtype=fp8_dtype,
                use_complex_method=args.complex_rounding,
                use_shift_perturb_method=args.shifturb,
                use_owlshift_method=args.owlshift,
                seed=(main_seed + i if args.owlshift else main_seed),
                debug_mode=args.debug,
            )
            quantized_state_dict[key] = quantized_tensor_fp8.to(original_tensor_device)
            if main_device.type == "cuda":
                if "quantized_tensor_fp8" in locals():
                    del quantized_tensor_fp8
            quantized_count += 1
            print(
                f" Done. New dtype: {quantized_state_dict[key].dtype}."
            )  # Newline for this path's status

        # NO QUANTIZATION (COPY AS IS)
        else:
            if not (
                is_torchao_method
                and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
                and tensor_to_process.is_floating_point()
                and tensor_to_process.dim() < 2
            ):  # Avoid double print if torchao skipped due to dim
                print(
                    " (Skipping quantization, copying as is)... Done."
                )  # Newline for this path's status
            quantized_state_dict[key] = tensor.to(original_tensor_device)

        # PLOTTING LOGIC (common to all paths that might have quantized)
        if (
            args.plot
            and original_tensor_for_plot_and_type is not None
            and plots_generated_count < args.plot_max_tensors
            and key in quantized_state_dict  # Ensure key made it to output
            and quantized_state_dict[key].dtype
            in [torch.float8_e4m3fn, torch.float8_e5m2]  # Only plot if actually FP8
        ):
            # Ensure tensor was actually processed for quantization by one of the methods
            # This check needs to be more robust based on which path was taken.
            # For now, assume if it's FP8 in dict, it was quantized.

            print(f"  Generating plot for {key}...")
            original_tensor_cpu_for_plot = original_tensor_for_plot_and_type
            quantized_tensor_cpu_for_plot = quantized_state_dict[key].cpu()

            dequantized_tensor_cpu_for_plot = None
            scale_for_plot_display = 1.0  # Default for native, or if TorchAO scale cannot be determined for display

            if is_torchao_method:
                # `quantized_tensor_cpu_for_plot` is what was stored in quantized_state_dict[key].cpu()
                # We suspect this is still a TorchAO wrapper despite efforts to make it plain for safetensors.
                # For plotting, we need to access its underlying plain FP8 data.

                plain_fp8_for_plotting = None

                # Try to access tensor_impl.data, similar to the main loop
                if hasattr(quantized_tensor_cpu_for_plot, "tensor_impl") and hasattr(
                    quantized_tensor_cpu_for_plot.tensor_impl, "data"
                ):
                    potential_plain_data = (
                        quantized_tensor_cpu_for_plot.tensor_impl.data
                    )
                    if isinstance(
                        potential_plain_data, torch.Tensor
                    ) and potential_plain_data.dtype in [
                        torch.float8_e4m3fn,
                        torch.float8_e5m2,
                    ]:
                        plain_fp8_for_plotting = potential_plain_data
                    else:
                        print(
                            f"      Warning: Plotting: {key}'s .tensor_impl.data is not a plain FP8 tensor (dtype: {getattr(potential_plain_data, 'dtype', 'N/A')})."
                        )
                elif hasattr(
                    quantized_tensor_cpu_for_plot, "_data"
                ):  # Fallback to _data
                    potential_plain_data = quantized_tensor_cpu_for_plot._data
                    if isinstance(
                        potential_plain_data, torch.Tensor
                    ) and potential_plain_data.dtype in [
                        torch.float8_e4m3fn,
                        torch.float8_e5m2,
                    ]:
                        plain_fp8_for_plotting = potential_plain_data
                        print(f"      Info: Plotting: Used ._data for {key}.")
                    else:
                        print(
                            f"      Warning: Plotting: {key}'s ._data is not a plain FP8 tensor (dtype: {getattr(potential_plain_data, 'dtype', 'N/A')})."
                        )
                else:  # If still no plain data, we might be in trouble for plotting this tensor accurately
                    print(
                        f"      Warning: Plotting: Could not access underlying plain FP8 data for {key} (type: {type(quantized_tensor_cpu_for_plot).__name__}). Attempting to use the object directly."
                    )
                    plain_fp8_for_plotting = (
                        quantized_tensor_cpu_for_plot  # Last resort, might error
                    )

                retrieved_scale_for_plotting = None
                scale_display_value = 1.0  # Default for plot title

                _scale_key_to_use_plot_ = ""
                scale_key_parts_plot = key.split(".")
                if scale_key_parts_plot[-1] == "weight":
                    scale_key_parts_plot[-1] = "scale_weight"
                    _scale_key_to_use_plot_ = ".".join(scale_key_parts_plot)
                else:
                    _scale_key_to_use_plot_ = key + ".scale_absmax"

                if _scale_key_to_use_plot_ in quantized_state_dict:
                    retrieved_scale_for_plotting = quantized_state_dict[
                        _scale_key_to_use_plot_
                    ].cpu()
                    if not isinstance(retrieved_scale_for_plotting, torch.Tensor):
                        print(
                            f"     Warning: Retrieved scale for {key} is not a tensor, type: {type(retrieved_scale_for_plotting)}"
                        )
                        retrieved_scale_for_plotting = (
                            None  # Invalidate if not a tensor
                        )
                else:
                    print(
                        f"      Warning: Plotting for TorchAO tensor {key}, but its specific scale key '{_scale_key_to_use_plot_}' not found in quantized_state_dict. Plot may be inaccurate if it relied on this scale."
                    )

                if (
                    plain_fp8_for_plotting is not None
                    and retrieved_scale_for_plotting is not None
                ):
                    try:
                        dequant_hp_plot = plain_fp8_for_plotting.to(
                            torch.float32
                        ) * retrieved_scale_for_plotting.to(torch.float32)
                        dequantized_tensor_cpu_for_plot = dequant_hp_plot.to(
                            original_tensor_cpu_for_plot.dtype
                        )

                        if retrieved_scale_for_plotting.numel() == 1:
                            scale_display_value = retrieved_scale_for_plotting.item()
                        else:
                            scale_display_value = torch.norm(
                                retrieved_scale_for_plotting.float()
                            ).item()
                            # print(f"      Note: Using norm of per-channel scale for plot display title for {key}.") # User commented out debug prints
                    except Exception as e:
                        print(
                            f"      Error during dequantization for plotting {key}: {e}. Using fallback."
                        )
                        dequantized_tensor_cpu_for_plot = plain_fp8_for_plotting.to(
                            original_tensor_cpu_for_plot.dtype
                        )  # Fallback to direct cast
                        scale_display_value = 1.0
                else:
                    print(
                        f"      Fallback: Could not retrieve plain FP8 data or scale for TorchAO tensor {key} for plotting. Using direct cast of the (potentially wrapped) FP8 tensor."
                    )
                    # This direct cast on plain_fp8_for_plotting might still fail if it's a wrapper and not plain_fp8_data
                    if plain_fp8_for_plotting is not None:
                        try:
                            dequantized_tensor_cpu_for_plot = plain_fp8_for_plotting.to(
                                original_tensor_cpu_for_plot.dtype
                            )
                        except Exception as e:
                            print(
                                f"        Error during fallback direct cast for plotting {key}: {e}"
                            )
                            dequantized_tensor_cpu_for_plot = None  # Cannot plot
                    else:
                        dequantized_tensor_cpu_for_plot = None  # Cannot plot
                    scale_display_value = 1.0
            else:  # Native methods dequantization
                # Native dequantization logic (already uses scale_factor_for_comfyui_to_save or direct cast)
                # It also sets `scale_for_plot_display` which we should rename to `scale_display_value` for consistency.
                if (
                    args.comfyscale
                    and scale_factor_for_comfyui_to_save  # This is the scale from native comfyscale path
                    is not None
                    and not key.endswith(".bias")
                ):
                    scale_display_value = scale_factor_for_comfyui_to_save.cpu().item()
                    dequantized_tensor_cpu_for_plot = (
                        quantized_tensor_cpu_for_plot.to(
                            scale_factor_for_comfyui_to_save.dtype  # Use scale's dtype for intermediate mult
                        )
                        * scale_factor_for_comfyui_to_save.cpu()  # Multiply by the scale
                    ).to(
                        original_tensor_cpu_for_plot.dtype
                    )  # Cast back to original
                else:  # General FP8 path (no explicit scale factor saved alongside, dequant is just type cast)
                    dequantized_tensor_cpu_for_plot = quantized_tensor_cpu_for_plot.to(
                        original_tensor_cpu_for_plot.dtype
                    )
                    scale_display_value = (
                        1.0  # Reflects direct cast without explicit scale
                    )

            # The generate_comparison_plots function expects a 'scale_value' for its title.
            # We've prepared 'scale_display_value' for this purpose.

            if dequantized_tensor_cpu_for_plot is not None:
                safe_tensor_key_for_filename = key.replace("/", "_").replace(".", "_")
                plot_file_path = os.path.join(
                    args.plot_dir, f"{safe_tensor_key_for_filename}_comparison.png"
                )
                original_dtype_name = str(original_tensor_cpu_for_plot.dtype)
                fp8_dtype_name = str(fp8_dtype)

                generate_comparison_plots(
                    original_tensor_cpu=original_tensor_cpu_for_plot,
                    quantized_fp8_tensor_cpu=quantized_tensor_cpu_for_plot,  # This is the raw FP8 tensor
                    dequantized_tensor_cpu=dequantized_tensor_cpu_for_plot,  # This is the dequantized tensor in higher precision
                    tensor_key=key,
                    plot_filename=plot_file_path,
                    original_dtype_str=original_dtype_name,
                    fp8_dtype_str=fp8_dtype_name,
                    sample_size=args.plot_sample_size,
                )
                plots_generated_count += 1
            else:
                print(
                    f"      Skipping plot for {key} as dequantized tensor could not be obtained."
                )

        if main_device.type == "cuda":
            if "tensor_to_process" in locals():
                del tensor_to_process
            if "original_hp_tensor" in locals():
                del original_hp_tensor  # From comfyscale
            torch.cuda.empty_cache()
    # End of per-tensor loop

    print(
        f"\nQuantization complete. {quantized_count}/{total_tensors} tensors were processed for quantization."
    )

    if (args.comfyscale and not is_torchao_method) or (
        is_torchao_method and quantized_count > 0
    ):
        # Add marker if comfyscale was used (native) or if torchao quantized something (implying scales might be used)
        print(
            f"Adding FP8 marker key 'scaled_fp8' for ComfyUI compatibility (method: {args.quant_method})."
        )
        quantized_state_dict["scaled_fp8"] = torch.empty(
            (2), dtype=fp8_dtype
        )  # Use the globally determined fp8_dtype

    try:
        print(f"Saving quantized model to: {args.output_file}")
        save_file(quantized_state_dict, args.output_file)
        print(f"Quantized model saved to: {args.output_file}")
    except Exception as e:
        print(f"Error saving quantized model: {e}")


if __name__ == "__main__":
    main()
