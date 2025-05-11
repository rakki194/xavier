import torch
from .get_fp8_bracketing_candidates_complex import get_fp8_bracketing_candidates_complex


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
