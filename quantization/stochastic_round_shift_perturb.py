import torch
from .get_fp8_bracketing_candidates_complex import get_fp8_bracketing_candidates_complex


def stochastic_round_shift_perturb(
    tensor: torch.Tensor, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """
    Applies stochastic rounding using the shift-and-perturb (additive noise)
    method before standard RNE quantization.
    """
    # If the tensor is complex, proceed with its real part.
    # This makes the conversion explicit and silences the PyTorch warning about discarding imaginary parts.
    # The associated test `test_non_floating_point_tensor` already expects this behavior.
    if tensor.is_complex():
        tensor_to_process = tensor.real
    else:
        tensor_to_process = tensor

    if not tensor_to_process.is_floating_point():
        try:
            return tensor_to_process.to(fp8_dtype)
        except Exception:
            return (
                tensor_to_process  # Return the (potentially modified) tensor_to_process
            )

    if tensor_to_process.numel() == 0:
        return tensor_to_process.to(fp8_dtype)

    # original_dtype = tensor_to_process.dtype # original_dtype is not actually used later

    # Get bracketing FP8 candidates (in original precision) using the complex method
    # as it provides a good local estimate for the quantization step.
    low_candidate_orig_prec, high_candidate_orig_prec = (
        get_fp8_bracketing_candidates_complex(
            tensor_to_process, fp8_dtype
        )  # Use tensor_to_process
    )

    delta_approx = high_candidate_orig_prec - low_candidate_orig_prec

    # Generate uniform noise in [-delta_approx/2, delta_approx/2]
    noise = torch.rand_like(tensor_to_process) * delta_approx - (
        delta_approx / 2.0
    )  # Use tensor_to_process for rand_like

    # Add noise to the original tensor (real part if original was complex)
    perturbed_tensor = tensor_to_process + noise

    # Quantize the perturbed tensor using standard Round-to-Nearest-Even
    quantized_tensor = perturbed_tensor.to(fp8_dtype)

    return quantized_tensor
