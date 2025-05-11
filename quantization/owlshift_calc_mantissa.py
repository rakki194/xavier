import torch


def owlshift_calc_mantissa(
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
