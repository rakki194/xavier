import torch
import argparse
from safetensors.torch import load_file, save_file
import os

# Attempt to import matplotlib for plotting
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import numpy as np  # Usually comes with matplotlib or is a common dependency

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass  # Plotting will be disabled if matplotlib is not found

"""
Quantizes a .safetensors model to FP8 using various stochastic rounding techniques.

This script provides multiple methods for converting floating-point tensors 
(typically FP32, FP16, or BF16) within a .safetensors file to FP8 (E4M3FN or E5M2 formats).
Stochastic rounding is employed instead of deterministic rounding (like Round-to-Nearest-Even)
to introduce a controlled amountof noise, which can sometimes help preserve model performance
by ensuring that, on average, the rounding errors do not systematically bias values up or down.

Key Features & Techniques:

1.  **Target FP8 Formats**:
    *   `--fp8_type e4m3`: Uses `torch.float8_e4m3fn` (4 exponent bits, 3 mantissa bits, NaN-preserving).
    *   `--fp8_type e5m2`: Uses `torch.float8_e5m2` (5 exponent bits, 2 mantissa bits).

2.  **Core Stochastic Rounding Dispatcher (`stochastic_round_tensor_to_fp8`)**:
    This function is the main entry point for quantizing a single tensor. It selects the
    rounding method based on command-line flags.

3.  **Default Stochastic Rounding (No specific flag, or if others are false)**:
    *   Method: Finds two FP8-representable candidates that are likely to bracket the input value.
        One candidate is the result of PyTorch's default Round-to-Nearest-Even (RNE) cast to FP8.
        The other is its FP8 neighbor in the direction of the original input value, found using an 
        approximation (`get_fp8_neighbor`).
    *   Probability: The probability of rounding to the 'higher' candidate is proportional to the
        input value's position between the two candidates.
        `P(round_to_high) = (input - low_candidate) / (high_candidate - low_candidate)`.
    *   Implementation: `stochastic_round_tensor_to_fp8` (else branch).

4.  **Complex Neighbor Stochastic Rounding (`--complex_rounding`)**:
    *   Method: Uses a more sophisticated approach (`get_fp8_bracketing_candidates_complex`) to find
        the two FP8-representable values that strictly bracket the input tensor values.
        It handles cases where the input is exactly representable, or between two FP8 values, more directly.
        This method also relies on `torch.nextafter` on the FP8 grid (approximated by casting back and forth
        from the original precision).
    *   Probability: Same probabilistic choice as the default method, but with potentially more accurate candidates.
    *   Activation: `--complex_rounding` flag.

5.  **Shift-and-Perturb Stochastic Rounding (`--shifturb`)**:
    *   Method: Adds uniform random noise to the input tensor *before* quantizing it with standard
        Round-to-Nearest-Even (RNE).
        The noise is scaled by `delta_approx`, which is the difference between bracketing FP8 candidates
        (obtained using `get_fp8_bracketing_candidates_complex`).
        `noise = U[-delta_approx/2, +delta_approx/2]`
        `quantized = RNE(input + noise)`
    *   Implementation: `stochastic_round_shift_perturb`.
    *   Activation: `--shifturb` flag.

6.  **Owlshift - Manual Stochastic Mantissa Rounding (`--owlshift`)**:
    *   Method: Directly implements stochastic rounding by manipulating the mantissa bits of the
        floating-point numbers (after conversion to an intermediate `.half()` representation).
        It calculates the exponent and mantissa, adds uniform random noise `U[0,1)` to the scaled mantissa,
        then floors it. The number is then reconstructed from the stochastically rounded mantissa.
        This method is ported from an external reference script and includes logic for handling normal
        and subnormal numbers, as well as tensor slicing for large tensors to manage memory/computation.
    *   Implementation: `stochastic_round_owlshift_method` (which uses `_owlshift_manual_stochastic_round_to_float8`
        and `_owlshift_calc_mantissa`).
    *   Activation: `--owlshift` flag. Uses the `--seed` argument for its random number generator.

7.  **Owlscale - Per-Tensor Max-Absolute Scaling (`--owlscale`)**:
    *   Method: A pre-processing step that can be combined with any of the above rounding methods.
        For each tensor, it calculates a scale factor to map the tensor's maximum absolute value
        into the representable range of the target FP8 format (specifically, `FP8_MAX - FP8_MIN_POS`).
        The tensor is then scaled by this factor and clamped to `[FP8_MIN, FP8_MAX]` before being passed
        to the chosen stochastic rounding function.
    *   Output: If `--owlscale` is used, additional tensors representing the dequantization scales
        (1.0 / scale_factor) are saved in the output file (e.g., `layer.weight` gets a corresponding
        `layer.scale_weight`).
    *   Implementation: `apply_owlscale_preprocess` (called in `main` if flag is active).
        Uses `OWLSCALE_COMPUTE_DTYPE` (torch.float64) for internal scale calculations.
    *   Activation: `--owlscale` flag.

Combining Flags:
*   `--owlscale` can be combined with `--complex_rounding`, `--shifturb`, `--owlshift`, or the default.
    The scaling is applied first, then the chosen rounding method operates on the scaled tensor.
*   `--complex_rounding`, `--shifturb`, and `--owlshift` are mutually exclusive in terms of the core
    rounding logic applied by `stochastic_round_tensor_to_fp8`. If multiple are specified, the order of
    preference in the dispatcher is: owlshift, then shifturb, then complex_rounding, then default.

Usage:
    python xavier.py <input.safetensors> <output.safetensors> \
                     [--fp8_type <e4m3|e5m2>] [--device <cpu|cuda>] \
                     [--keys_to_quantize_suffix .weight .bias] \
                     [--complex_rounding] [--shifturb] [--owlshift] [--owlscale] [--seed <int>]
"""


def get_fp8_neighbor(
    value_in_orig_prec: torch.Tensor,
    direction_tensor: torch.Tensor,
    fp8_dtype_target: torch.dtype,
) -> torch.Tensor:
    """
    Attempts to find an FP8 neighbor of value_in_orig_prec in the given direction.
    value_in_orig_prec is assumed to be an FP8-representable value, but in original precision (e.g., bf16).
    direction_tensor indicates the direction (+1 for next, -1 for prev relative to value_in_orig_prec).
    """
    # Take a small step in the original precision along the direction.
    # torch.finfo(value_in_orig_prec.dtype).tiny is the smallest positive representable number
    # in the original precision.
    stepped_value_orig_prec = torch.nextafter(
        value_in_orig_prec,
        value_in_orig_prec
        + direction_tensor * torch.finfo(value_in_orig_prec.dtype).tiny,
    )

    # Project this stepped value onto the FP8 grid and cast back to original precision.
    # This is an approximation of finding the true FP8 neighbor.
    neighbor_fp8_orig_prec = stepped_value_orig_prec.to(fp8_dtype_target).to(
        value_in_orig_prec.dtype
    )

    return neighbor_fp8_orig_prec


def get_fp8_bracketing_candidates_complex(
    tensor: torch.Tensor, fp8_dtype_target: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds two FP8-representable candidates that bracket the input tensor,
    using a more direct neighbor finding on the FP8 grid.
    Returns them in the original precision of the input tensor.
    """
    original_dtype = tensor.dtype

    # Value of tensor if directly cast to fp8 (uses RNE by default)
    x_cast_fp8_native = tensor.to(fp8_dtype_target)
    x_cast_orig_prec = x_cast_fp8_native.to(original_dtype)

    # Initialize candidate tensors
    low_candidate_orig_prec = torch.zeros_like(tensor)
    high_candidate_orig_prec = torch.zeros_like(tensor)

    # Mask for elements where tensor is greater than its RNE FP8 representation
    mask_tensor_gt_cast = tensor > x_cast_orig_prec
    # Mask for elements where tensor is less than its RNE FP8 representation
    mask_tensor_lt_cast = tensor < x_cast_orig_prec
    # Mask for elements where tensor is equal to its RNE FP8 representation
    mask_tensor_eq_cast = tensor == x_cast_orig_prec

    # Case 1: tensor > x_cast_orig_prec
    # x_cast_orig_prec is the lower bound. Find next FP8 value up.
    if torch.any(mask_tensor_gt_cast):
        current_low_fp8_native_subset = x_cast_fp8_native[mask_tensor_gt_cast]
        # Operate nextafter in original_dtype after initial snap to fp8 grid
        current_low_orig_prec_subset = current_low_fp8_native_subset.to(original_dtype)

        next_high_orig_prec_subset = torch.nextafter(
            current_low_orig_prec_subset,
            torch.full_like(
                current_low_orig_prec_subset, torch.finfo(original_dtype).max
            ),
        )
        # Ensure the result is snapped to the FP8 grid again
        next_high_fp8_snapped_orig_prec_subset = next_high_orig_prec_subset.to(
            fp8_dtype_target
        ).to(original_dtype)

        low_candidate_orig_prec[mask_tensor_gt_cast] = current_low_orig_prec_subset
        high_candidate_orig_prec[mask_tensor_gt_cast] = (
            next_high_fp8_snapped_orig_prec_subset
        )

    # Case 2: tensor < x_cast_orig_prec
    # x_cast_orig_prec is the upper bound. Find next FP8 value down.
    if torch.any(mask_tensor_lt_cast):
        current_high_fp8_native_subset = x_cast_fp8_native[mask_tensor_lt_cast]
        # Operate nextafter in original_dtype
        current_high_orig_prec_subset = current_high_fp8_native_subset.to(
            original_dtype
        )

        next_low_orig_prec_subset = torch.nextafter(
            current_high_orig_prec_subset,
            torch.full_like(
                current_high_orig_prec_subset, torch.finfo(original_dtype).min
            ),
        )
        # Ensure the result is snapped to the FP8 grid again
        next_low_fp8_snapped_orig_prec_subset = next_low_orig_prec_subset.to(
            fp8_dtype_target
        ).to(original_dtype)

        high_candidate_orig_prec[mask_tensor_lt_cast] = current_high_orig_prec_subset
        low_candidate_orig_prec[mask_tensor_lt_cast] = (
            next_low_fp8_snapped_orig_prec_subset
        )

    # Case 3: tensor == x_cast_orig_prec (tensor is exactly representable in fp8_dtype)
    # x_cast_orig_prec is one bound. Find its next FP8 neighbor for the other bound.
    # By convention, let x_cast_orig_prec be low_candidate and find the next higher FP8 value.
    if torch.any(mask_tensor_eq_cast):
        exact_match_fp8_native_subset = x_cast_fp8_native[mask_tensor_eq_cast]
        # Operate nextafter in original_dtype
        exact_match_orig_prec_subset = exact_match_fp8_native_subset.to(original_dtype)

        next_high_orig_prec_subset = torch.nextafter(
            exact_match_orig_prec_subset,
            torch.full_like(
                exact_match_orig_prec_subset, torch.finfo(original_dtype).max
            ),
        )
        # Ensure the result is snapped to the FP8 grid again
        next_high_fp8_snapped_orig_prec_subset = next_high_orig_prec_subset.to(
            fp8_dtype_target
        ).to(original_dtype)

        # If tensor is max representable FP8, nextafter(inf) might be itself or inf.
        # Ensure low_candidate is the tensor value.
        low_candidate_orig_prec[mask_tensor_eq_cast] = exact_match_orig_prec_subset
        high_candidate_orig_prec[mask_tensor_eq_cast] = (
            next_high_fp8_snapped_orig_prec_subset
        )

        # Handle case where exact_match is max_val and nextafter(max_val, inf) is max_val
        # In this scenario, high_candidate should still be itself, but if we want to force prob=0,
        # we need low and high to define an interval. Or, if low==high, prob becomes 0 due to non_degenerate_mask.
        # The current setup where prob_high = (tensor - low) / (high - low)
        # if tensor == low, and low == high (e.g. at max fp8 val), then (0 / 0) -> nan.
        # non_degenerate_mask handles this by keeping prob_high = 0, so it rounds to low_candidate (which is correct).

    return low_candidate_orig_prec, high_candidate_orig_prec


def stochastic_round_shift_perturb(
    tensor: torch.Tensor, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """
    Applies stochastic rounding using the shift-and-perturb (additive noise)
    method before standard RNE quantization.
    """
    if (
        not tensor.is_floating_point()
    ):  # Should have been caught earlier, but good practice
        try:
            return tensor.to(fp8_dtype)
        except Exception:
            return tensor
    if tensor.numel() == 0:
        return tensor.to(fp8_dtype)

    original_dtype = tensor.dtype

    # Get bracketing FP8 candidates (in original precision) using the complex method
    # as it provides a good local estimate for the quantization step.
    low_candidate_orig_prec, high_candidate_orig_prec = (
        get_fp8_bracketing_candidates_complex(tensor, fp8_dtype)
    )

    delta_approx = high_candidate_orig_prec - low_candidate_orig_prec

    # Generate uniform noise in [-delta_approx/2, delta_approx/2]
    # For elements where delta_approx is 0 (low == high, e.g., at representable FP8 max/min or if tensor is exactly FP8)
    # no noise should effectively be added, or noise will be 0.
    noise = torch.rand_like(tensor) * delta_approx - (delta_approx / 2.0)

    # Add noise to the original tensor
    perturbed_tensor = tensor + noise

    # Quantize the perturbed tensor using standard Round-to-Nearest-Even
    quantized_tensor = perturbed_tensor.to(fp8_dtype)

    return quantized_tensor


def get_fp8_constants_for_owlscale(
    fp8_dtype: torch.dtype,
) -> tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_min_pos = 2**-9  # Smallest subnormal for E4M3FN
    elif fp8_dtype == torch.float8_e5m2:
        fp8_min_pos = 2**-16  # Smallest subnormal for E5M2
    else:
        fp8_min_pos = finfo.tiny  # Fallback
    return float(finfo.min), float(finfo.max), float(fp8_min_pos)


def apply_owlscale_preprocess(
    tensor: torch.Tensor,
    target_fp8_dtype: torch.dtype,
    fp8_min_val: float,
    fp8_max_val: float,
    fp8_min_pos_val: float,
    compute_dtype: torch.dtype,
    debug_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies per-tensor scaling and clamping as per the reference script.
    Returns the processed tensor (ready for stochastic rounding) and the dequantization scale factor.
    """
    if debug_mode:
        print(
            f"DEBUG apply_owlscale_preprocess for tensor: shape={tensor.shape}, initial_dtype={tensor.dtype}, device={tensor.device}"
        )
        if tensor.numel() > 0:
            print(
                f"  Initial tensor stats: min={tensor.min().item():.4g}, max={tensor.max().item():.4g}, mean={tensor.mean().item():.4g}, isfinite={torch.isfinite(tensor).all().item()}"
            )
        else:
            print("  Initial tensor is empty.")

    original_tensor_dtype_for_output_scale = (
        tensor.dtype
    )  # Dtype for the output scale factor

    # Convert to high precision for calculations
    calc_tensor = tensor.to(compute_dtype)
    if debug_mode:
        print(f"  Tensor cast to compute_dtype: {calc_tensor.dtype}")

    if calc_tensor.numel() == 0:
        # Return tensor as is (will be converted to empty FP8 later) and a dummy scale
        if debug_mode:
            print("  Tensor is empty, returning original and dummy scale 1.0")
        return tensor, torch.tensor(
            1.0, dtype=original_tensor_dtype_for_output_scale, device=tensor.device
        )

    abs_max = torch.max(torch.abs(calc_tensor))

    if debug_mode:
        print(f"  Calculated abs_max: {abs_max.item():.4g} (on compute_dtype)")

    dequant_scale_val = torch.tensor(
        1.0, dtype=compute_dtype, device=calc_tensor.device
    )
    processed_tensor = calc_tensor  # Default to no scaling if not needed

    if abs_max < 1e-12:  # Near-zero max value
        # No scaling needed, dequant_scale remains 1.0
        if debug_mode:
            print(
                f"  Near-zero abs_max ({abs_max.item():.4g}). No scaling applied. Processed tensor dtype: {processed_tensor.dtype}"
            )
        pass
    else:
        # Clamp abs_max to smallest positive representable to avoid division by zero or extreme scales
        clamped_abs_max = abs_max.clamp(min=fp8_min_pos_val)
        if debug_mode:
            print(
                f"  Clamped abs_max: {clamped_abs_max.item():.4g} (min_pos_val used for clamping: {fp8_min_pos_val:.4g})"
            )

        # Scale factor to map values into [fp8_min_pos_val, fp8_max_val] range (approximately)
        # The reference script aims to map abs_max to (FP8_MAX - FP8_MIN_POS)
        # This seems to preserve a bit of headroom from the absolute max of FP8.
        quant_scale_factor = (fp8_max_val - fp8_min_pos_val) / clamped_abs_max
        if debug_mode:
            print(
                f"  Calculated quant_scale_factor: {quant_scale_factor.item():.4g} (using fp8_max_val={fp8_max_val:.4g})"
            )

        processed_tensor = calc_tensor.mul(quant_scale_factor)
        dequant_scale_val = quant_scale_factor.reciprocal()
        if debug_mode:
            if processed_tensor.numel() > 0:
                print(
                    f"  Processed_tensor (after mul by quant_scale_factor, before clamp) stats: min={processed_tensor.min().item():.4g}, max={processed_tensor.max().item():.4g}, mean={processed_tensor.mean().item():.4g}, isfinite={torch.isfinite(processed_tensor).all().item()}, dtype={processed_tensor.dtype}"
                )
            else:
                print(
                    "  Processed_tensor (after mul by quant_scale_factor, before clamp) is empty."
                )
            print(f"  Calculated dequant_scale_val: {dequant_scale_val.item():.4g}")

    # Clamp the scaled tensor to the representable FP8 range
    clamped_tensor = torch.clamp(processed_tensor, fp8_min_val, fp8_max_val)
    if debug_mode:
        if clamped_tensor.numel() > 0:
            print(
                f"  Clamped_tensor stats: min={clamped_tensor.min().item():.4g}, max={clamped_tensor.max().item():.4g}, mean={clamped_tensor.mean().item():.4g}, isfinite={torch.isfinite(clamped_tensor).all().item()}, dtype={clamped_tensor.dtype}"
            )
            print(
                f"  Clamping range was: min_val={fp8_min_val:.4g}, max_val={fp8_max_val:.4g}"
            )
        else:
            print("  Clamped_tensor is empty.")

    output_tensor = clamped_tensor
    output_dequant_scale = dequant_scale_val

    if debug_mode:
        print(
            f"  Returning processed tensor with dtype: {output_tensor.dtype}, and dequant_scale with dtype: {output_dequant_scale.dtype} (both should be compute_dtype: {compute_dtype})"
        )

    return output_tensor, output_dequant_scale


def stochastic_round_tensor_to_fp8(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    use_complex_method: bool,
    use_shift_perturb_method: bool,
    use_owlshift_method: bool,
    seed: int = 0,
    debug_mode: bool = False,
) -> torch.Tensor:
    """
    Applies stochastic rounding to convert a tensor to the specified FP8 dtype.
    Note: This is a challenging operation due to the nature of FP8 and relies on
    approximations for finding neighboring FP8 values.
    """
    if debug_mode:
        print(
            f"DEBUG stochastic_round_tensor_to_fp8: use_complex={use_complex_method}, use_shifturb={use_shift_perturb_method}, use_owlshift={use_owlshift_method}"
        )

    if not tensor.is_floating_point():
        # If not a float, attempt direct cast (might error if incompatible) or return
        try:
            return tensor.to(fp8_dtype)
        except Exception:
            print(
                f"Warning: Could not convert non-float tensor of dtype {tensor.dtype} to {fp8_dtype}. Returning original."
            )
            return tensor

    if tensor.numel() == 0:
        return tensor.to(fp8_dtype)

    # Candidate 1: Value obtained by PyTorch's default Round-to-Nearest-Even (RNE)
    # This value is representable in fp8_dtype, but stored in tensor.dtype.
    # x_rne_orig_prec = tensor.to(fp8_dtype).to(tensor.dtype) # Moved inside simple method

    # Determine the two FP8 values (in original precision) that are candidates for rounding.
    # One is x_rne_orig_prec. The other is its FP8 neighbor in the direction of the original 'tensor'.

    if use_owlshift_method:
        if debug_mode:
            print("DEBUG: Using Owlshift Method")
        return stochastic_round_owlshift_method(tensor, fp8_dtype, seed=seed)
    elif use_shift_perturb_method:
        if debug_mode:
            print("DEBUG: Using Shift-Perturb Method")
        return stochastic_round_shift_perturb(tensor, fp8_dtype)
    elif use_complex_method:
        if debug_mode:
            print("DEBUG: Using Complex Bracketing Method for candidate search")
        low_candidate, high_candidate = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype
        )
    else:
        if debug_mode:
            print(
                "DEBUG: Using Default (Simple) Bracketing Method for candidate search"
            )
        x_rne_orig_prec = tensor.to(fp8_dtype).to(tensor.dtype)
        # Determine direction towards the original 'tensor' from the RNE value 'x_rne_orig_prec'.
        direction_to_tensor = torch.sign(tensor - x_rne_orig_prec)

        # If 'tensor' is exactly 'x_rne_orig_prec', direction is 0.
        # For finding a distinct neighbor, we need a non-zero direction for get_fp8_neighbor.
        # If direction is 0, any neighbor would do, as probability will be 0 or 1. Default to +1.
        direction_for_neighbor_search = torch.where(
            direction_to_tensor == 0,
            torch.ones_like(direction_to_tensor),
            direction_to_tensor,
        )

        x_neighbor_orig_prec = get_fp8_neighbor(
            x_rne_orig_prec, direction_for_neighbor_search, fp8_dtype
        )

        # We now have two candidate FP8-representable values (in original precision):
        # x_rne_orig_prec and x_neighbor_orig_prec.
        # These should ideally bracket 'tensor', or one is equal to 'tensor'.

        low_candidate = torch.min(x_rne_orig_prec, x_neighbor_orig_prec)
        high_candidate = torch.max(x_rne_orig_prec, x_neighbor_orig_prec)

    # Denominator for probability calculation.
    denominator = high_candidate - low_candidate

    prob_high = torch.zeros_like(tensor)  # Default to 0 probability for high candidate

    # Mask for non-degenerate cases (where low_candidate < high_candidate)
    # Add a small epsilon to denominator check for floating point stability
    non_degenerate_mask = denominator > torch.finfo(denominator.dtype).eps

    # Calculate probability P(round to high_candidate)
    # P = (tensor - low_candidate) / (high_candidate - low_candidate)
    if torch.any(non_degenerate_mask):
        prob_high[non_degenerate_mask] = ((tensor - low_candidate) / denominator)[
            non_degenerate_mask
        ]

    # Clamp probabilities to [0, 1] to handle:
    # 1. Floating point inaccuracies.
    # 2. Cases where 'tensor' might be outside [low_candidate, high_candidate] due to
    #    the discrete nature of FP8 and approximation in neighbor finding.
    prob_high = torch.clamp(prob_high, 0.0, 1.0)

    # Uniform random numbers for stochastic choice
    random_draw = torch.rand_like(tensor)

    # Make the stochastic choice
    chosen_value_orig_prec = torch.where(
        random_draw < prob_high, high_candidate, low_candidate
    )

    return chosen_value_orig_prec.to(fp8_dtype)


def _owlshift_calc_mantissa(
    abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None
):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS))),
    )
    # Add uniform random noise U[0,1) for stochastic rounding effect via floor
    mantissa_scaled += torch.rand(
        mantissa_scaled.size(),
        dtype=mantissa_scaled.dtype,
        layout=mantissa_scaled.layout,
        device=mantissa_scaled.device,
        generator=generator,
    )
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


def _owlshift_manual_stochastic_round_to_float8(
    x_chunk: torch.Tensor, target_fp8_dtype: torch.dtype, generator=None
):
    if target_fp8_dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif target_fp8_dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError(
            f"Unsupported target FP8 dtype for owlshift: {target_fp8_dtype}"
        )

    x_compute = x_chunk.half()  # Using .half() (float16) for intermediate calcs
    original_device = x_chunk.device

    sign = torch.sign(x_compute)
    abs_x = x_compute.abs()
    sign = torch.where(
        abs_x == 0,
        torch.tensor(0.0, dtype=x_compute.dtype, device=original_device),
        sign,
    )

    abs_x_safe_for_log = torch.where(
        abs_x == 0,
        torch.tensor(1e-38, dtype=abs_x.dtype, device=original_device),
        abs_x,
    )
    exponent = torch.floor(torch.log2(abs_x_safe_for_log)) + EXPONENT_BIAS

    max_exponent_val = (2**EXPONENT_BITS) - 1
    exponent = torch.clamp(exponent, 0, max_exponent_val)

    normal_mask = (exponent > 0) & (exponent < max_exponent_val)
    mantissa_values = torch.zeros_like(abs_x)

    if torch.any(normal_mask):
        mantissa_values[normal_mask] = _owlshift_calc_mantissa(
            abs_x[normal_mask],
            exponent[normal_mask],
            torch.ones_like(abs_x[normal_mask], dtype=torch.bool),
            MANTISSA_BITS,
            EXPONENT_BIAS,
            generator=generator,
        )

    subnormal_mask = (exponent == 0) & (abs_x > 0)
    if torch.any(subnormal_mask):
        mantissa_val_subnormal = abs_x[subnormal_mask] / (
            2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)
        )
        mantissa_val_subnormal += torch.rand(
            mantissa_val_subnormal.size(),
            dtype=mantissa_val_subnormal.dtype,
            layout=mantissa_val_subnormal.layout,
            device=original_device,
            generator=generator,
        )
        mantissa_values[subnormal_mask] = mantissa_val_subnormal.floor() / (
            2**MANTISSA_BITS
        )

    reconstructed_abs_x = torch.zeros_like(abs_x)
    reconstructed_abs_x[normal_mask] = (
        2.0 ** (exponent[normal_mask] - EXPONENT_BIAS)
    ) * (1.0 + mantissa_values[normal_mask])
    reconstructed_abs_x[subnormal_mask] = (
        2.0 ** (-EXPONENT_BIAS + 1)
    ) * mantissa_values[subnormal_mask]

    reconstructed_value = sign * reconstructed_abs_x
    finfo_fp8 = torch.finfo(target_fp8_dtype)
    reconstructed_value = torch.clamp(
        reconstructed_value, min=finfo_fp8.min, max=finfo_fp8.max
    )

    return reconstructed_value.to(target_fp8_dtype)


def stochastic_round_owlshift_method(
    tensor: torch.Tensor, target_fp8_dtype: torch.dtype, seed: int = 0
) -> torch.Tensor:
    """
    Applies stochastic rounding using the 'owlshift' method (manual mantissa rounding).
    This version incorporates the slicing logic from the reference script.
    """
    if not tensor.is_floating_point():
        try:
            return tensor.to(target_fp8_dtype)
        except Exception:
            print(
                f"Warning (owlshift): Could not convert non-float tensor of dtype {tensor.dtype} to {target_fp8_dtype}. Returning original."
            )
            return tensor
    if tensor.numel() == 0:
        return tensor.to(target_fp8_dtype)

    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    output = torch.empty_like(tensor, dtype=target_fp8_dtype)
    MAX_SLICE_ELEMENTS = 2048 * 2048

    if tensor.numel() <= MAX_SLICE_ELEMENTS or tensor.ndim == 0:
        output.copy_(
            _owlshift_manual_stochastic_round_to_float8(
                tensor, target_fp8_dtype, generator=generator
            )
        )
    else:
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        output_flat = torch.empty_like(tensor_flat, dtype=target_fp8_dtype)
        num_slices = max(1, int(tensor_flat.numel() / MAX_SLICE_ELEMENTS))
        elements_per_slice = (tensor_flat.numel() + num_slices - 1) // num_slices

        for i in range(0, tensor_flat.numel(), elements_per_slice):
            chunk = tensor_flat[i : i + elements_per_slice]
            processed_chunk = _owlshift_manual_stochastic_round_to_float8(
                chunk, target_fp8_dtype, generator=generator
            )
            output_flat[i : i + elements_per_slice].copy_(processed_chunk)
        output = output_flat.reshape(original_shape)
    return output


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
        # Define owlscale compute/scale dtypes here, from reference script's config
        # These are not class members, so define them locally if owlscale is used.
        # Or better, make them global if they are fixed for owlscale.
        # For now, let's assume they are fixed as in the reference for owlscale.
        global OWLSCALE_COMPUTE_DTYPE, OWLSCALE_SCALE_DTYPE
        OWLSCALE_COMPUTE_DTYPE = torch.float64
        OWLSCALE_SCALE_DTYPE = torch.float64

        # Get constants for the target FP8 type for owlscale
        global OWLSCALE_FP8_MIN, OWLSCALE_FP8_MAX, OWLSCALE_FP8_MIN_POS
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

        original_tensor_for_plot_and_type = (
            None  # Store original tensor for plotting and its dtype
        )
        if args.plot and plots_generated_count < args.plot_max_tensors:
            original_tensor_for_plot_and_type = tensor.detach().clone().cpu()

        dequant_scale_factor_for_owlscale = None  # Initialize here

        if (
            any(key.endswith(suffix) for suffix in args.keys_to_quantize_suffix)
            and tensor_to_process.is_floating_point()
        ):
            tensor_actually_scaled = (
                False  # Flag to track if owlscale was applied to this tensor
            )
            if args.owlscale and key.endswith(
                ".weight"
            ):  # Apply owlscale only to .weight tensors
                print(" (Applying owlscale pre-processing to weight tensor)...", end="")
                tensor_to_process, dequant_scale_factor_for_owlscale = (
                    apply_owlscale_preprocess(
                        tensor_to_process,
                        fp8_dtype,
                        OWLSCALE_FP8_MIN,
                        OWLSCALE_FP8_MAX,
                        OWLSCALE_FP8_MIN_POS,
                        OWLSCALE_COMPUTE_DTYPE,
                        debug_mode=args.debug,  # Pass debug flag
                    )
                )
                tensor_actually_scaled = True  # Mark that scaling was done
                # Store the dequant scale factor, specific to .weight keys
                scale_key = key.replace(".weight", ".scale_weight")
                quantized_state_dict[scale_key] = dequant_scale_factor_for_owlscale.to(
                    OWLSCALE_SCALE_DTYPE
                ).to(original_tensor_device)
                if args.debug:
                    if tensor_to_process.numel() > 0:
                        print(
                            f"  Tensor {key} AFTER owlscale_preprocess (input to stochastic_round): min={tensor_to_process.min().item():.4g}, max={tensor_to_process.max().item():.4g}, mean={tensor_to_process.mean().item():.4g}, isfinite={torch.isfinite(tensor_to_process).all().item()}, dtype={tensor_to_process.dtype}"
                        )
                    else:
                        print(
                            f"  Tensor {key} AFTER owlscale_preprocess (input to stochastic_round) is empty."
                        )
            elif args.owlscale and not key.endswith(".weight"):
                if args.debug:
                    print(f"  DEBUG: Skipping owlscale for non-weight tensor: {key}")

            print(" -> Quantizing...", end="")
            quantized_tensor = stochastic_round_tensor_to_fp8(
                tensor_to_process,
                fp8_dtype,
                args.complex_rounding,
                args.shifturb,
                args.owlshift,
                seed=main_seed + i,  # Pass seed, vary per tensor
                debug_mode=args.debug,
            )
            quantized_state_dict[key] = quantized_tensor.to(
                original_tensor_device
            )  # Move back to original or CPU for saving
            quantized_count += 1
            print(f" Done. New dtype: {quantized_tensor.dtype}")
            if args.owlscale and dequant_scale_factor_for_owlscale is not None:
                print(
                    f"  Owlscale dequant factor for {key}: {dequant_scale_factor_for_owlscale.item():.5g}"
                )

            # --- Plotting Logic ---
            if (
                args.plot
                and original_tensor_for_plot_and_type is not None
                and plots_generated_count < args.plot_max_tensors
            ):
                print(f"  Generating plot for {key}...")
                dequantized_for_plot = None
                if (
                    tensor_actually_scaled
                    and dequant_scale_factor_for_owlscale is not None
                ):  # Use scale factor if tensor was actually scaled
                    # Dequantize using the scale factor
                    # Ensure scale factor is on the same device and correct dtype for multiplication
                    scale_for_dequant = dequant_scale_factor_for_owlscale.to(
                        tensor_to_process.device
                    ).to(tensor_to_process.dtype)
                    dequantized_for_plot = (
                        quantized_tensor.to(tensor_to_process.dtype) * scale_for_dequant
                    ).cpu()
                else:
                    # For non-scaled, or if scale factor is not available (e.g. bias with owlscale flag on)
                    # cast quantized back to original dtype for comparison
                    dequantized_for_plot = quantized_tensor.to(
                        original_tensor_for_plot_and_type.dtype
                    ).cpu()

                plot_filename = os.path.join(
                    args.plot_dir, f"{key.replace('.', '_')}_{fp8_dtype}.png"
                )
                generate_comparison_plots(
                    original_tensor_for_plot_and_type,
                    quantized_tensor.cpu(),  # Send CPU copy of FP8 tensor
                    dequantized_for_plot,
                    key,
                    plot_filename,
                    str(original_tensor_for_plot_and_type.dtype),
                    str(fp8_dtype),
                    args.plot_sample_size,
                )
                plots_generated_count += 1
            # --- End Plotting Logic ---

        else:
            print(" -> Skipping quantization (not a target key/suffix or not a float).")
            quantized_state_dict[key] = tensor_to_process.to(
                original_tensor_device
            )  # Copy as is, ensuring it's on original device

    print(
        f"Quantization complete. {quantized_count}/{total_tensors} tensors were processed for quantization."
    )

    if args.owlscale:  # Add marker only if owlscale was used
        print("Adding FP8 marker key 'scaled_fp8' for compatibility.")
        # We use the fp8_dtype determined from args.fp8_type for the marker
        quantized_state_dict["scaled_fp8"] = torch.empty((2), dtype=fp8_dtype)

    try:
        print(f"Saving quantized model to: {args.output_file}")
        save_file(quantized_state_dict, args.output_file)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving quantized model: {e}")


def generate_comparison_plots(
    original_tensor_cpu: torch.Tensor,
    quantized_fp8_tensor_cpu: torch.Tensor,  # This is still in FP8 format, but on CPU
    dequantized_tensor_cpu: torch.Tensor,  # This is in original_tensor_cpu's dtype
    tensor_key: str,
    plot_filename: str,
    original_dtype_str: str,
    fp8_dtype_str: str,
    sample_size: int,
):
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        original_np = original_tensor_cpu.float().numpy().flatten()
        # For histogram, cast FP8 to float to see its quantized levels
        quantized_for_hist_np = quantized_fp8_tensor_cpu.float().numpy().flatten()
        dequantized_np = dequantized_tensor_cpu.float().numpy().flatten()

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(
            f"Quantization Comparison: {tensor_key}\nOriginal ({original_dtype_str}) vs. Quantized ({fp8_dtype_str})",
            fontsize=16,
        )

        # Subplot 1: Histograms
        axs[0].hist(
            original_np,
            bins="auto",
            alpha=0.6,
            label=f"Original Values\n(min: {original_np.min():.3g}, max: {original_np.max():.3g})",
            color="blue",
            density=True,
        )
        axs[0].hist(
            quantized_for_hist_np,
            bins="auto",
            alpha=0.7,
            label=f"FP8 Values (cast to float)\n(min: {quantized_for_hist_np.min():.3g}, max: {quantized_for_hist_np.max():.3g})",
            color="red",
            density=True,
        )
        if dequantized_tensor_cpu is not None:
            axs[0].hist(
                dequantized_np,
                bins="auto",
                alpha=0.5,
                label=f"Dequantized Values\n(min: {dequantized_np.min():.3g}, max: {dequantized_np.max():.3g})",
                color="green",
                density=True,
            )
        axs[0].set_title("Value Distributions (Density)")
        axs[0].set_xlabel("Value")
        axs[0].set_ylabel("Density")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        # Use a symmetric log scale if data spans positive and negative with wide range, else linear
        if original_np.min() < 0 < original_np.max() and (
            original_np.max() / original_np.min() < -0.01
            or np.log10(np.abs(original_np.max())) - np.log10(np.abs(original_np.min()))
            > 2
        ):
            axs[0].set_yscale("symlog", linthresh=1e-5)
        else:
            axs[0].set_yscale("linear")  # or 'log' if appropriate for all positive

        # Subplot 2: Scatter Plot (Original vs. Dequantized/Quantized)
        # Sample data for scatter plot if tensors are too large
        num_elements = original_np.shape[0]
        indices = np.arange(num_elements)
        if num_elements > sample_size:
            indices = np.random.choice(num_elements, size=sample_size, replace=False)

        scatter_x = original_np[indices]
        scatter_y = dequantized_np[indices]

        axs[1].scatter(
            scatter_x,
            scatter_y,
            alpha=0.3,
            s=5,
            label="Original vs. Dequantized/FP8-as-float",
            color="purple",
        )
        # Add a y=x line for reference
        lims = [
            min(scatter_x.min(), scatter_y.min()),
            max(scatter_x.max(), scatter_y.max()),
        ]
        axs[1].plot(lims, lims, "k--", alpha=0.75, zorder=0, label="y=x (Ideal)")
        axs[1].set_title(f"Scatter Plot (Sampled {len(scatter_x)} points)")
        axs[1].set_xlabel(f"Original Value ({original_dtype_str})")
        axs[1].set_ylabel(f"Dequantized/FP8-as-float Value")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        axs[1].axhline(0, color="black", linewidth=0.5)
        axs[1].axvline(0, color="black", linewidth=0.5)

        plt.savefig(plot_filename)
        plt.close(fig)  # Close the figure to free memory
        print(f"    Plot saved to {plot_filename}")

    except Exception as e:
        print(f"Error generating plot for {tensor_key}: {e}")
        if "plt" in locals() and plt.gcf().get_axes():  # Check if a figure is open
            plt.close("all")  # Close all figures in case of an error during plotting


if __name__ == "__main__":
    main()

