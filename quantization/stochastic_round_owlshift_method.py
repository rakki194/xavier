import torch
from .owlshift_manual_stochastic_round_to_float8 import (
    owlshift_manual_stochastic_round_to_float8,
)

MAX_SLICE_ELEMENTS = 2048 * 2048


def stochastic_round_owlshift_method(
    tensor: torch.Tensor, target_fp8_dtype: torch.dtype, seed: int = 0
) -> torch.Tensor:
    """
    Applies stochastic rounding using the 'owlshift' method (manual mantissa rounding).
    This version incorporates the slicing logic from the reference script.
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
            return tensor_to_process.to(target_fp8_dtype)
        except Exception:
            print(
                f"Warning (owlshift): Could not convert non-float tensor of dtype {tensor_to_process.dtype} to {target_fp8_dtype}. Returning original."
            )
            return tensor_to_process
    if tensor_to_process.numel() == 0:
        return tensor_to_process.to(target_fp8_dtype)

    generator = torch.Generator(device=tensor_to_process.device)
    generator.manual_seed(seed)
    output = torch.empty_like(tensor_to_process, dtype=target_fp8_dtype)

    if tensor_to_process.numel() <= MAX_SLICE_ELEMENTS or tensor_to_process.ndim == 0:
        output.copy_(
            owlshift_manual_stochastic_round_to_float8(
                tensor_to_process, target_fp8_dtype, generator=generator
            )
        )
    else:
        original_shape = tensor_to_process.shape
        tensor_flat = tensor_to_process.flatten()
        output_flat = torch.empty_like(tensor_flat, dtype=target_fp8_dtype)
        num_slices = max(1, int(tensor_flat.numel() / MAX_SLICE_ELEMENTS))
        elements_per_slice = (tensor_flat.numel() + num_slices - 1) // num_slices

        for i in range(0, tensor_flat.numel(), elements_per_slice):
            chunk = tensor_flat[i : i + elements_per_slice]
            processed_chunk = owlshift_manual_stochastic_round_to_float8(
                chunk, target_fp8_dtype, generator=generator
            )
            output_flat[i : i + elements_per_slice].copy_(processed_chunk)
        output = output_flat.reshape(original_shape)
    return output
