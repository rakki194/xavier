import torch


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
    # Handle zero specifically
    if torch.all(value_in_orig_prec == 0.0):
        fp8_tiny = torch.finfo(fp8_dtype_target).tiny
        # Create a tensor with the same shape and device as value_in_orig_prec
        # Fill with fp8_tiny or -fp8_tiny based on direction
        result = torch.full_like(value_in_orig_prec, fp8_tiny)
        result = result * direction_tensor  # Apply direction
        return result.to(value_in_orig_prec.dtype)  # Ensure correct output dtype

    # Original logic for non-zero values
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
