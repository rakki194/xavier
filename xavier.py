import torch
import argparse
from safetensors.torch import load_file, save_file
import os
from quantization import stochastic_round_tensor_to_fp8
from scaling import get_fp8_constants_for_owlscale
from plotting_utils import MATPLOTLIB_AVAILABLE, generate_comparison_plots

"""
Quantizes a .safetensors model to FP8 using various stochastic rounding techniques.

This script provides multiple methods for converting floating-point tensors 
(typically FP32, FP16, or BF16) within a .safetensors file to FP8 (E4M3FN or E5M2 formats).
Stochastic rounding is employed instead of deterministic rounding (like Round-to-Nearest-Even)
to introduce a controlled amount of noise, which can sometimes help preserve model performance
by ensuring that, on average, the rounding errors do not systematically bias values up or down.
"""

# Global dtypes for Owlscale - these are set in main() if --owlscale is used.
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

    args = parser.parse_args()
    main_device = torch.device(args.device)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

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
        print("FP8 Path: Using shift-and-perturb (additive noise) stochastic rounding.")
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
    quantized_count = 0

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
            quantized_state_dict[key] = tensor.to(original_tensor_device)  # Copy to CPU

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
                safe_tensor_key_for_filename = key.replace("/", "_").replace(".", "_")
                plot_file_path = os.path.join(
                    args.plot_dir, f"{safe_tensor_key_for_filename}_comparison.png"
                )

                original_dtype_name = str(original_tensor_cpu_for_plot.dtype)
                fp8_dtype_name = str(fp8_dtype)  # fp8_dtype is like torch.float8_e4m3fn

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
        print(f"FP8 Quantized model saved to: {args.output_file}")
    except Exception as e:
        print(f"Error saving quantized model: {e}")


if __name__ == "__main__":
    main()
