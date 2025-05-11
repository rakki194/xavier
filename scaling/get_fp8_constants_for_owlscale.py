import torch


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
