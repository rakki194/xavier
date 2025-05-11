import torch


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

    return low_candidate_orig_prec, high_candidate_orig_prec
