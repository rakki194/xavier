import torch


def apply_owlscale_preprocess(
    tensor: torch.Tensor,
    target_fp8_dtype: torch.dtype,  # Not used in current body, but kept for signature consistency
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
