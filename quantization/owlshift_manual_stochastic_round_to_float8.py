import torch
from .owlshift_calc_mantissa import owlshift_calc_mantissa


def owlshift_manual_stochastic_round_to_float8(
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
        mantissa_values[normal_mask] = owlshift_calc_mantissa(
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
