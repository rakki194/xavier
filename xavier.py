import torch
import argparse
from safetensors.torch import load_file, save_file
import os

# Imports from our new modules
from quantization import stochastic_round_tensor_to_fp8
from scaling import get_fp8_constants_for_owlscale
from plotting_utils import MATPLOTLIB_AVAILABLE, generate_comparison_plots

"""
Quantizes a .safetensors model to FP8 using various stochastic rounding techniques.

This script provides multiple methods for converting floating-point tensors 
(typically FP32, FP16, or BF16) within a .safetensors file to FP8 (E4M3FN or E5M2 formats).
Stochastic rounding is employed instead of deterministic rounding (like Round-to-Nearest-Even)
to introduce a controlled amountof noise, which can sometimes help preserve model performance
by ensuring that, on average, the rounding errors do not systematically bias values up or down.
"""

# Global dtypes for Owlscale - these are set in main() if --owlscale is used.
# It's often better to pass such configurations as parameters, but for now,
# we'll mirror the existing global approach.
OWLSCALE_COMPUTE_DTYPE = None
OWLSCALE_SCALE_DTYPE = None
OWLSCALE_FP8_MIN = None
OWLSCALE_FP8_MAX = None
OWLSCALE_FP8_MIN_POS = None


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a .safetensors model to FP8 with stochastic rounding."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input .safetensors model file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the quantized .safetensors model file.",
    )
    parser.add_argument(
        "--fp8_type",
        type=str,
        default="e4m3",
        choices=["e4m3", "e5m2"],
        help="FP8 type to use: e4m3 (torch.float8_e4m3fn) or e5m2 (torch.float8_e5m2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computations (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--keys_to_quantize_suffix",
        type=str,
        nargs="+",
        default=[".weight", ".bias"],
        help="Suffixes of keys to identify tensors for quantization (e.g., '.weight' '.bias').",
    )
    parser.add_argument(
        "--complex_rounding",
        action="store_true",
        help="Use a more complex neighbor finding method for stochastic rounding.",
    )
    parser.add_argument(
        "--shifturb",
        action="store_true",
        help="Use shift-and-perturb (additive noise) stochastic rounding method.",
    )
    parser.add_argument(
        "--owlshift",
        action="store_true",
        help="Use owlshift (manual stochastic mantissa rounding) method.",
    )
    parser.add_argument(
        "--owlscale",
        action="store_true",
        help="Apply per-tensor max-abs scaling before stochastic rounding (from reference script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stochastic rounding methods that use it (e.g., owlshift).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable generation of comparison plots (requires matplotlib). Saved to --plot_dir.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./quant_plots/",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--plot_max_tensors",
        type=int,
        default=5,
        help="Maximum number of tensors for which to generate plots.",
    )
    parser.add_argument(
        "--plot_sample_size",
        type=int,
        default=5000,
        help="Number of points to sample for scatter plots of large tensors.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug print statements to trace execution flow and flag states.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    if args.fp8_type == "e4m3":
        fp8_dtype = torch.float8_e4m3fn
    elif args.fp8_type == "e5m2":
        fp8_dtype = torch.float8_e5m2
    else:
        # Should be caught by choices, but as a safeguard
        raise ValueError(f"Unsupported fp8_type: {args.fp8_type}")

    print(f"Loading model from: {args.input_file}")
    # Load metadata to check for device incompatibilities if model was saved on a different device type
    try:
        state_dict = load_file(args.input_file, device="cpu")  # Load to CPU first
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    quantized_state_dict = {}
    total_tensors = len(state_dict)
    quantized_count = 0

    print(f"Target FP8 type: {fp8_dtype}, Device for quantization: {args.device}")
    print(
        f"Will attempt to quantize tensors with keys ending in: {args.keys_to_quantize_suffix}"
    )
    print(
        "Stochastic rounding for FP8 is complex and the method used here has known approximations."
    )
    if args.complex_rounding:
        print("Using complex neighbor finding method for stochastic rounding.")
    if args.shifturb:
        print("Using shift-and-perturb (additive noise) stochastic rounding.")
    if args.owlshift:
        print(
            f"Using owlshift (manual stochastic mantissa rounding) with seed: {args.seed}."
        )
    if args.owlscale:
        print(f"Using owlscale (per-tensor max-abs scaling pre-quantization).")
        # Make OWLSCALE_... variables global so they can be set here and used if --owlscale
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

    print("Evaluate the quantized model carefully.")

    if args.plot and not MATPLOTLIB_AVAILABLE:
        print(
            "Warning: --plot was specified, but matplotlib is not installed. Plotting will be disabled."
        )
        args.plot = False  # Disable plotting if library not found

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {args.plot_dir}")

    # Global seed for owlshift if used, to ensure deterministic output for the whole run if desired
    # The generator inside owlshift will be re-seeded for each tensor for consistency with reference, but this main seed can make runs reproducible.
    main_seed = args.seed
    plots_generated_count = 0

    for i, (key, tensor) in enumerate(state_dict.items()):
        original_tensor_for_plot_and_type = (
            None  # Store original tensor for plotting and its dtype
        )
        if args.plot and plots_generated_count < args.plot_max_tensors:
            original_tensor_for_plot_and_type = tensor.detach().clone().cpu()

        # This variable will store the scale factor if ComfyUI-style scaling is used
        scale_factor_for_comfyui_to_save = None

        if args.debug:
            print(
                f"DEBUG main loop for tensor {key}: complex_flag={args.complex_rounding}, shifturb_flag={args.shifturb}, owlshift_flag={args.owlshift}, owlscale_flag={args.owlscale}"
            )
            if tensor.numel() > 0:
                print(
                    f"  Tensor {key} before any processing (on device {args.device}): min={tensor.min().item():.4g}, max={tensor.max().item():.4g}, mean={tensor.mean().item():.4g}, isfinite={torch.isfinite(tensor).all().item()}, dtype={tensor.dtype}"
                )
            else:
                print(
                    f"  Tensor {key} before any processing (on device {args.device}) is empty."
                )

        print(
            f"Processing tensor {i+1}/{total_tensors}: {key} (dtype: {tensor.dtype}, shape: {tensor.shape})",
            end="",
        )

        original_tensor_device = tensor.device
        tensor_to_process = tensor.to(
            args.device
        )  # Move to target device for processing

        if (
            args.owlscale  # This block now implements ComfyUI-compatible scaled FP8
            and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
            and tensor_to_process.is_floating_point()
            and (
                not key.endswith(".bias")
            )  # Exclude bias from ComfyUI-style owlscale path
        ):
            print(" (Applying ComfyUI-style scaled FP8 quantization)...", end="")

            # 1. Determine scale_factor_for_comfyui (this will be saved as layer.scale_info[0] or similar)
            original_hp_tensor = tensor_to_process.to(
                OWLSCALE_COMPUTE_DTYPE
            )  # Use a high-precision dtype for abs_max
            abs_max = torch.max(torch.abs(original_hp_tensor))

            # Use abs_max as the scale, ensuring it's not too small.
            # This normalizes the tensor to roughly [-1, 1] before quantization.
            scale_factor_for_comfyui_val = abs_max.clamp(min=1e-12)

            # This is the actual scale value that will be saved (e.g., as model.0.scale for ComfyUI)
            scale_factor_for_comfyui_to_save = scale_factor_for_comfyui_val.to(
                OWLSCALE_SCALE_DTYPE
            ).to(original_tensor_device)

            # 2. Prepare tensor for quantization: original_tensor / scale
            # Perform division in high precision, then cast back to original tensor's dtype for stochastic rounder
            # Ensure tensor_to_process is compatible with scale_factor_for_comfyui_val's dtype for division
            input_for_quantization_hp = (
                tensor_to_process.to(scale_factor_for_comfyui_val.dtype)
                / scale_factor_for_comfyui_val
            )
            # The stochastic rounders might have their own dtype expectations (e.g., owlshift uses .half() internally)
            # but they operate on the input tensor's original dtype after this scaling.
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

            # Determine the key for the scale factor
            scale_key_parts = key.split(".")
            if scale_key_parts[-1] == "weight":
                scale_key_parts[-1] = (
                    "scale_weight"  # Or what ComfyUI expects, e.g., "scale_info.0"
                )
                scale_key = ".".join(scale_key_parts)
            else:  # Fallback for non-weight keys if they were ever to be scaled this way
                scale_key = key + ".scale_absmax"  # More descriptive fallback

            if scale_key != key:  # Ensure it's a different key
                quantized_state_dict[scale_key] = scale_factor_for_comfyui_to_save
            else:
                print(
                    f"Warning: Could not determine unique scale key for {key} (generated scale key was the same: {scale_key}). Scale not saved separately for ComfyUI-style scaled FP8."
                )

            quantized_count += 1
            print(f" Done. New dtype: {quantized_tensor.dtype}")
            if scale_factor_for_comfyui_to_save is not None:
                print(
                    f"  ComfyUI-style scale factor for {key} (saved as {scale_key}): {scale_factor_for_comfyui_to_save.item():.5g}"
                )

        elif (  # Original (non-owlscale) quantization path, also excluding bias from direct quantization if no scaling applied
            any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
            and (not key.endswith(".bias"))
            and tensor_to_process.is_floating_point()
        ):
            print(" -> Quantizing (direct, no ComfyUI-style scaling)...", end="")
            quantized_tensor = stochastic_round_tensor_to_fp8(
                tensor_to_process,  # Original tensor values passed directly
                fp8_dtype,
                args.complex_rounding,
                args.shifturb,
                args.owlshift,
                seed=main_seed + i,
                debug_mode=args.debug,  # Pass debug flag
            )
            quantized_state_dict[key] = quantized_tensor.to(original_tensor_device)
            quantized_count += 1
            print(f" Done. New dtype: {quantized_tensor.dtype}")

        else:  # Not a target key, or a bias (if not doing ComfyUI scaling for it), or not a float
            print(
                " -> Skipping quantization (not a target key/suffix, or a bias, or not a float)."
            )
            quantized_state_dict[key] = tensor_to_process.to(
                original_tensor_device
            )  # Copy as is, ensuring it's on original device

        # --- Plotting Logic (Adjusted for ComfyUI-style scaling if used) ---
        if (
            args.plot
            and original_tensor_for_plot_and_type is not None
            and plots_generated_count < args.plot_max_tensors
            # Ensure we plot only if quantization actually happened on this tensor or if it's a float we tried to process
            and (
                key in quantized_state_dict
                and quantized_state_dict[key].dtype == fp8_dtype
            )
        ):
            print(f"  Generating plot for {key}...")
            quantized_tensor_for_plot = quantized_state_dict[
                key
            ].cpu()  # FP8 tensor on CPU
            dequantized_for_plot = None

            if (
                args.owlscale
                and scale_factor_for_comfyui_to_save is not None
                and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
                and tensor_to_process.is_floating_point()
                and not key.endswith(".bias")
            ):
                # Dequantize using the ComfyUI-style scale factor
                scale_for_dequant = scale_factor_for_comfyui_to_save.cpu().to(
                    original_tensor_for_plot_and_type.dtype
                )
                dequantized_for_plot = (
                    quantized_tensor_for_plot.to(
                        original_tensor_for_plot_and_type.dtype
                    )
                    * scale_for_dequant
                )
            elif (
                not args.owlscale
                and any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
                and tensor_to_process.is_floating_point()
                and not key.endswith(".bias")
            ):
                # For non-scaled, just cast quantized FP8 back to original dtype for comparison
                dequantized_for_plot = quantized_tensor_for_plot.to(
                    original_tensor_for_plot_and_type.dtype
                )

            if dequantized_for_plot is not None:
                plot_filename = os.path.join(
                    args.plot_dir, f"{key.replace('.', '_')}_{fp8_dtype}.png"
                )
                generate_comparison_plots(
                    original_tensor_for_plot_and_type,
                    quantized_tensor_for_plot,  # Send CPU copy of FP8 tensor
                    dequantized_for_plot,
                    key,
                    plot_filename,
                    str(original_tensor_for_plot_and_type.dtype),
                    str(fp8_dtype),
                    args.plot_sample_size,
                )
                plots_generated_count += 1
            else:
                if args.debug:
                    print(
                        f"  DEBUG: Skipping plot for {key} as dequantized tensor could not be prepared (check conditions)."
                    )
        # --- End Plotting Logic ---

    print(
        f"Quantization complete. {quantized_count}/{total_tensors} tensors were processed for quantization."
    )

    if (
        args.owlscale
    ):  # Add marker only if ComfyUI-style scaling was attempted for some weights
        print("Adding FP8 marker key 'scaled_fp8' for ComfyUI compatibility.")
        # We use the fp8_dtype determined from args.fp8_type for the marker
        quantized_state_dict["scaled_fp8"] = torch.empty((2), dtype=fp8_dtype)

    try:
        print(f"Saving quantized model to: {args.output_file}")
        save_file(quantized_state_dict, args.output_file)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving quantized model: {e}")


if __name__ == "__main__":
    main()
