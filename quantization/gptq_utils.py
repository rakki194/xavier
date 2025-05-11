import torch
from .group_quant_utils import (
    quantize_symmetric_per_group,
    dequantize_symmetric_per_group,
)
import math  # For math.isclose for float comparisons


def get_act_order(calibration_activations_hat, num_cols_to_permute):
    """Helper to get an activation order for columns (simplistic version)."""
    if calibration_activations_hat.numel() == 0 or num_cols_to_permute == 0:
        return torch.arange(
            num_cols_to_permute, device=calibration_activations_hat.device
        )
    try:
        # Ensure X has at least 2 samples for variance calculation if not empty
        if (
            calibration_activations_hat.shape[0] < 2
            and calibration_activations_hat.shape[0] > 0
        ):
            # If only one sample, variance is not well-defined, use mean abs value as proxy
            col_metric = torch.mean(torch.abs(calibration_activations_hat), dim=0)
        elif (
            calibration_activations_hat.shape[0] == 0
        ):  # Should be caught by numel check, but defense
            return torch.arange(
                num_cols_to_permute, device=calibration_activations_hat.device
            )
        else:
            col_metric = torch.var(calibration_activations_hat, dim=0)

        perm = torch.argsort(col_metric, descending=True)

        # Ensure we only take valid indices for W columns if W has fewer columns than X features
        # This typically applies if X corresponds to W.T rows, and W has `out_features` columns.
        # Here, X is (samples, in_features), W is (in_features, out_features).
        # We are permuting *columns* of W. So num_cols_to_permute is out_features.
        # The metric should be based on something that makes sense for output columns.
        # Let's assume for now `get_act_order` is for permuting W_ij by importance of input features X_i.
        # If we want to permute columns of W (R_hp), based on features of X, this is tricky.
        # Typically act_order in GPTQ permutes columns of W based on their overall importance / variance.
        # The original GPTQ permutes columns of W (size `d_model x d_out`) using importance of output activations.
        # Here, W is `in_features x out_features`. `X` is `samples x in_features`.
        # A simple proxy for column importance in W could be L2 norm of columns.
        # For now, the passed `calibration_activations_hat` is used, but its columns correspond to *input features*.
        # A true `act_order` for columns of R_hp would require different logic if R_hp is (in_feat, out_feat).
        # Let's assume for this placeholder `get_act_order` refers to permuting based on `X`'s feature variance for now.
        # This means we'd be permuting *rows* of W if this function was directly used for row permutation.
        # For column permutation of W, we need a different metric or a different X.
        # Let's make get_act_order simple: permute based on L2 norm of columns of W itself as a proxy
        # if X cannot directly inform W column order.
        # For this function, let's assume it's about permuting features of X (rows of W).
        # If it's to permute *columns* of W, it needs to be adapted based on W's statistics.

        # For this version, let's assume `num_cols_to_permute` is `out_features`
        # and we are trying to find an order for columns of W (R_hp).
        # A simple heuristic for column order if X cannot be used: norm of columns of W itself.
        # This is not true 'activation order' but a common heuristic.
        # This `get_act_order` is now a bit mismatched with its typical GPTQ meaning if X is input activation.
        # Let's make it return a dummy order for now, or a simple norm-based order on a passed W.
        # For GPTQ act_order, it usually means columns of W are reordered based on variance of XW.
        # Sticking to the simple variance of X for now and acknowledging the mismatch for R_hp columns.
        if len(perm) > num_cols_to_permute:
            perm = perm[:num_cols_to_permute]
        elif len(perm) < num_cols_to_permute:
            # If X has fewer features than W columns, pad with remaining indices
            remaining_indices = torch.tensor(
                [i for i in range(num_cols_to_permute) if i not in perm],
                device=perm.device,
                dtype=perm.dtype,
            )
            perm = torch.cat((perm, remaining_indices))
        return perm

    except Exception as e:
        print(f"Warning: get_act_order failed: {e}. Returning default arange.")
        return torch.arange(
            num_cols_to_permute, device=calibration_activations_hat.device
        )


def get_heuristic_column_order(W_matrix):
    """
    Gets a heuristic column order for W based on L2 norm of columns.
    Args:
        W_matrix (torch.Tensor): Shape (in_features, out_features)
    Returns:
        perm_indices (torch.Tensor): Permutation indices for columns.
    """
    if W_matrix.numel() == 0 or W_matrix.shape[1] == 0:
        return torch.arange(W_matrix.shape[1], device=W_matrix.device)
    try:
        col_norms = torch.linalg.norm(W_matrix, dim=0)  # L2 norm for each column
        perm_indices = torch.argsort(col_norms, descending=True)
        return perm_indices
    except Exception as e:
        print(
            f"Warning: get_heuristic_column_order failed: {e}. Returning default arange."
        )
        return torch.arange(W_matrix.shape[1], device=W_matrix.device)


def gptq_quantize_layer_residual_refined(
    residual_tensor_hp,  # R_hp: (in_features, out_features)
    calibration_activations_hat,  # hat_X_cal: (num_samples, in_features)
    num_bits,
    group_size,
    scale_dtype=torch.float16,
    percdamp=0.01,  # Dampening for Hessian inverse
    act_order=False,  # Reorder columns of R based on a heuristic (e.g. column norms)
    compensation_strength=0.1,  # Factor for scaling the Hessian-based error compensation
    verbose=False,
):
    """
    Refined GPTQ-like quantization for the residual tensor R.
    Processes R column by column, updating *subsequent* columns to compensate for quantization error
    using a Hessian-derived update.
    """
    device = residual_tensor_hp.device
    in_features, out_features = residual_tensor_hp.shape
    num_calib_samples = calibration_activations_hat.shape[0]

    if verbose:
        print(
            f"Starting Refined GPTQ (v2) for R: {in_features}x{out_features}, {num_bits}-bit, group_size={group_size}"
        )

    W = residual_tensor_hp.clone().float()  # Work on a float32 copy
    X = calibration_activations_hat.float()

    if X.shape[1] != in_features:
        raise ValueError(f"X columns ({X.shape[1]}) must match W rows ({in_features})")

    # 1. Hessian Computation H = X^T @ X (relates to input features / rows of W)
    H_inv = None
    if (
        num_calib_samples < 1
    ):  # Need at least 1 sample for meaningful H, though more is better
        print(
            "Warning: Insufficient calibration samples for Hessian. Using dampened Identity for H_inv."
        )
        H = torch.eye(in_features, device=device, dtype=X.dtype)
        diag_mean_simulated = 1.0
    else:
        max_hessian_samples = min(num_calib_samples, 512)
        X_subset_for_H = X[:max_hessian_samples, :]
        H = 2 * X_subset_for_H.T @ X_subset_for_H  # Shape: (in_features, in_features)
        diag_H = torch.diag(H)
        diag_mean_simulated = diag_H.mean().item()

        # Dampen Hessian before inversion
        dead_neurons_mask = diag_H < 1e-9
        H[dead_neurons_mask, dead_neurons_mask] = 1e-6
        damp = percdamp * diag_mean_simulated
        if not math.isclose(damp, 0.0):
            H.add_(torch.eye(in_features, device=device, dtype=H.dtype), alpha=damp)

    try:
        H_inv = torch.inverse(H)
        if verbose:
            print("  Hessian inverse computed successfully.")
    except torch.linalg.LinAlgError as e:
        print(
            f"Warning: Hessian inversion failed: {e}. Will skip Hessian-based error compensation."
        )
        H_inv = None  # Fallback: no Hessian-based compensation
    except (
        RuntimeError
    ) as e:  # Catch other potential runtime errors from inverse like CUDA errors
        print(
            f"Warning: Runtime error during Hessian inversion: {e}. Will skip Hessian-based error compensation."
        )
        H_inv = None

    # Initialize output tensors
    num_groups_in_col = (in_features + group_size - 1) // group_size
    quantized_R_int8 = torch.empty_like(W, dtype=torch.int8, device=device)
    scales_R = torch.empty(
        (num_groups_in_col, out_features), dtype=scale_dtype, device=device
    )

    perm_indices = None
    inv_perm_indices = None
    if act_order and out_features > 1:
        if verbose:
            print("  Calculating column order (heuristic: L2 norm of columns)...")
        perm_indices = get_heuristic_column_order(W)
        W = W[:, perm_indices]
        inv_perm_indices = torch.argsort(perm_indices)

    loss_total = 0.0
    # 2. Iterate over columns of W (which is R_hp, possibly permuted)
    for c_idx in range(out_features):
        if verbose and (
            c_idx % max(1, out_features // 10) == 0 or c_idx == out_features - 1
        ):
            print(f"    Processing column {c_idx+1}/{out_features}...")

        w_col_current = W[:, c_idx].clone()

        # 3. Quantize current column
        q_col, s_col = quantize_symmetric_per_group(
            w_col_current, num_bits, group_size, scale_dtype=scale_dtype
        )
        quantized_R_int8[:, c_idx] = q_col
        scales_R[:, c_idx] = s_col

        # 4. Calculate Quantization Error for this column
        w_col_dequant = dequantize_symmetric_per_group(
            q_col, s_col, group_size, output_dtype=W.dtype
        )
        error_col = w_col_current - w_col_dequant  # Shape: (in_features,)
        loss_total += torch.sum(error_col**2)

        # 5. Error Compensation: Update *subsequent* unquantized columns in W
        if H_inv is not None and c_idx < out_features - 1:
            # scaled_error_update_vector = H_inv @ error_col (more robust: solve H @ x = error_col)
            # For this refined version, let's use H_inv directly if available.
            scaled_error_update_vector = H_inv @ error_col  # Shape: (in_features,)

            # Distribute this update to all subsequent columns
            # W[:, c_idx+1:] += compensation_strength * scaled_error_update_vector.unsqueeze(1) -> broadcasts to all remaining
            # This is one way. Another is to make the compensation proportional to the next column values.
            # Let's try the direct additive update for simplicity of this structure.
            W[
                :, c_idx + 1 :
            ] += compensation_strength * scaled_error_update_vector.unsqueeze(
                1
            ).expand_as(
                W[:, c_idx + 1 :]
            )

    if verbose:
        print(
            f"  Refined GPTQ: Total squared error during quantization: {loss_total.item():.4f}"
        )

    # If columns were permuted, un-permute the final Q_R_int8 and scales_R
    if act_order and inv_perm_indices is not None and out_features > 1:
        if verbose:
            print("  Un-permuting columns for Q_R_int8 and scales_R...")
        quantized_R_int8 = quantized_R_int8[:, inv_perm_indices]
        scales_R = scales_R[:, inv_perm_indices]

    if verbose:
        print("Refined GPTQ (v2) for R complete.")
    return quantized_R_int8, scales_R


if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Make dimensions slightly larger to better observe effects and test stability
    _in_feat_test, _out_feat_test = 128, 256
    _num_bits_main, _group_size_main = 4, 64
    _comp_strength = 0.05  # Compensation strength

    print(f"Device: {_device}")

    test_R_hp = (
        torch.randn(_in_feat_test, _out_feat_test, device=_device, dtype=torch.float32)
        * 0.1
    )
    test_hat_X_cal = torch.randn(
        256, _in_feat_test, device=_device, dtype=torch.float32
    )  # More calib samples

    print(
        f"\nTesting Refined GPTQ (v2) for R ({_in_feat_test}x{_out_feat_test}), compensation_strength={_comp_strength}..."
    )
    q_R_ref, scales_R_ref = gptq_quantize_layer_residual_refined(
        test_R_hp.clone(),
        test_hat_X_cal,
        num_bits=_num_bits_main,
        group_size=_group_size_main,
        scale_dtype=torch.float16,
        percdamp=0.01,
        act_order=False,
        compensation_strength=_comp_strength,
        verbose=True,
    )

    print(f"\nRefined GPTQ (v2) Output:")
    print(f"  Quantized R shape: {q_R_ref.shape}, dtype: {q_R_ref.dtype}")
    print(f"  Scales R shape: {scales_R_ref.shape}, dtype: {scales_R_ref.dtype}")

    dequant_R_ref = dequantize_symmetric_per_group(
        q_R_ref, scales_R_ref, _group_size_main, output_dtype=torch.float32
    )
    mse_ref = torch.mean((test_R_hp - dequant_R_ref) ** 2)
    print(
        f"  MSE of dequantized R (Refined GPTQ v2) vs original R: {mse_ref.item():.6e}"
    )

    # RTN for comparison
    q_R_rtn, scales_R_rtn = quantize_symmetric_per_group(
        test_R_hp, _num_bits_main, _group_size_main, torch.float16
    )
    dequant_R_rtn = dequantize_symmetric_per_group(
        q_R_rtn, scales_R_rtn, _group_size_main, torch.float32
    )
    mse_rtn = torch.mean((test_R_hp - dequant_R_rtn) ** 2)
    print(
        f"  MSE of dequantized R (RTN) vs original R:              {mse_rtn.item():.6e}"
    )
    if mse_ref < mse_rtn:
        print("  (Refined GPTQ v2 shows improvement over RTN as expected)")
    else:
        print(
            "  (Warning: Refined GPTQ v2 did NOT show improvement over RTN. Check parameters/logic.)"
        )

    if _out_feat_test > 10:
        print(f"\nTesting Refined GPTQ (v2) with act_order=True...")
        q_R_ao, scales_R_ao = gptq_quantize_layer_residual_refined(
            test_R_hp.clone(),
            test_hat_X_cal,
            num_bits=_num_bits_main,
            group_size=_group_size_main,
            scale_dtype=torch.float16,
            percdamp=0.01,
            act_order=True,
            compensation_strength=_comp_strength,
            verbose=True,
        )
        dequant_R_ao = dequantize_symmetric_per_group(
            q_R_ao, scales_R_ao, _group_size_main, torch.float32
        )
        mse_ao = torch.mean((test_R_hp - dequant_R_ao) ** 2)
        print(f"  MSE (act_order, Refined v2): {mse_ao.item():.6e}")
