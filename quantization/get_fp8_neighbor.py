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
