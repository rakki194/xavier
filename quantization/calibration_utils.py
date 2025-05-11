import torch
import math  # For infinity

from .smoothing_utils import (
    calculate_smoothing_factors,
    apply_smoothing_to_weights,
    apply_smoothing_to_activations,
)
from .svd_utils import decompose_weights_svd
from .group_quant_utils import (
    quantize_symmetric_per_group,
    dequantize_symmetric_per_group,
)


def find_optimal_alpha_svdquant(
    original_weight_tensor,  # Shape: (out_features, in_features)
    calibration_activations,  # Shape: (num_calib_samples, in_features)
    rank,
    num_bits,
    group_size,
    alpha_values=None,
    low_rank_dtype=torch.float16,
    scale_dtype=torch.float16,
    verbose=False,
):
    """
    Finds the optimal smoothing factor alpha for a given layer using SVDQuant principles.

    Args:
        original_weight_tensor (torch.Tensor): Original weights (out_features, in_features).
        calibration_activations (torch.Tensor): Calibration input activations.
        rank (int): Rank for SVD.
        num_bits (int): Bits for residual and activation quantization.
        group_size (int): Group size for quantization.
        alpha_values (list, optional): List of alpha values to test. Defaults to a range.
        low_rank_dtype (torch.dtype): Dtype for L1, L2.
        scale_dtype (torch.dtype): Dtype for quantization scales.
        verbose (bool): If True, prints MSE for each alpha.

    Returns:
        dict: A dictionary containing:
            - 'best_alpha': The alpha value that minimized MSE.
            - 'best_lambda_val': Corresponding smoothing factors.
            - 'best_L1': Corresponding L1 matrix.
            - 'best_L2': Corresponding L2 matrix.
            - 'best_Q_R': Corresponding quantized residual R.
            - 'best_scales_R': Corresponding scales for Q_R.
            - 'min_mse': The minimum MSE achieved.
    """
    device = original_weight_tensor.device
    if calibration_activations.device != device:
        calibration_activations = calibration_activations.to(device)

    # Transpose original_weight_tensor for processing: (in_features, out_features)
    W_proc = (
        original_weight_tensor.T.contiguous().float()
    )  # Work with float32 for precision

    if alpha_values is None:
        alpha_values = [i / 10.0 for i in range(0, 11)]  # e.g., [0.0, 0.1, ..., 1.0]

    best_result = {
        "best_alpha": None,
        "best_lambda_val": None,
        "best_L1": None,
        "best_L2": None,
        "best_Q_R": None,
        "best_scales_R": None,
        "min_mse": math.inf,
    }

    for alpha in alpha_values:
        try:
            # 1. Calculate Smoothing Factors
            lambda_val = calculate_smoothing_factors(
                calibration_activations, W_proc, alpha
            )

            # 2. Apply Smoothing to get hat_W and hat_X_cal
            hat_W = apply_smoothing_to_weights(W_proc, lambda_val)
            hat_X_cal = apply_smoothing_to_activations(
                calibration_activations.float(), lambda_val
            )

            # 3. Decompose hat_W into L1, L2, R (all float32 initially)
            L1_hp, L2_hp, R_hp = decompose_weights_svd(hat_W, rank)
            L1 = L1_hp.to(low_rank_dtype)
            L2 = L2_hp.to(low_rank_dtype)

            # 4. Quantize Residual R (using round-to-nearest for now)
            # TODO: This is where GPTQ for R would be integrated later.
            Q_R, scales_R = quantize_symmetric_per_group(
                R_hp, num_bits, group_size, scale_dtype=scale_dtype
            )
            R_dequant = dequantize_symmetric_per_group(
                Q_R,
                scales_R,
                group_size,
                output_dtype=low_rank_dtype,  # Dequant to low_rank_dtype for matmul
            )

            # 5. Calculate SVDQuant output and Target Output
            # Target output is hat_X_cal @ hat_W (output after smoothing, before SVD/quantization of residual)
            output_target = hat_X_cal.to(low_rank_dtype) @ hat_W.to(low_rank_dtype)

            # SVDQuant output path
            # output_svdq_low_rank = hat_X_cal.to(low_rank_dtype) @ L1 @ L2
            # For R_dequant, ensure it's compatible with matmul with hat_X_cal
            # output_svdq_residual = hat_X_cal.to(low_rank_dtype) @ R_dequant # This is incorrect path for residual
            # The SVDQuant output uses hat_X_cal @ L1 @ L2 + Dequant(Q(hat_X_cal)) @ Dequant(Q(R))
            # For alpha search, paper says: "minimize layer output MSE after SVD on calibration dataset"
            # This can be interpreted as comparing XW vs X_smooth (L1L2 + Q(R_smooth))
            # Or, as paper (Sec 4.2) E(hat_X, R) = || hat_X R - Q(hat_X)Q(R) ||_F
            # Let's follow: minimize MSE of (hat_X_cal @ (L1@L2 + R_dequant)) vs (hat_X_cal @ hat_W)
            # This measures how well the SVD + quantized residual reconstructs the smoothed weights' output.

            reconstructed_hat_W_effective = (L1 @ L2) + R_dequant
            output_svdq_reconstruction = (
                hat_X_cal.to(low_rank_dtype) @ reconstructed_hat_W_effective
            )

            mse = torch.mean((output_svdq_reconstruction - output_target) ** 2)

            if verbose:
                print(f"Alpha: {alpha:.2f}, MSE: {mse.item():.6e}")

            if mse.item() < best_result["min_mse"]:
                best_result["min_mse"] = mse.item()
                best_result["best_alpha"] = alpha
                best_result["best_lambda_val"] = lambda_val
                best_result["best_L1"] = (
                    L1_hp  # Store high precision before casting for SVDQuantLinear
                )
                best_result["best_L2"] = L2_hp  # Store high precision
                best_result["best_Q_R"] = Q_R
                best_result["best_scales_R"] = scales_R
        except Exception as e:
            if verbose:
                print(f"Error during alpha search for alpha={alpha:.2f}: {e}")
            continue  # Skip this alpha if an error occurs (e.g., SVD fails, numerical issues)

    if best_result["best_alpha"] is None:
        raise RuntimeError(
            "Alpha search failed to find a valid alpha. Check input data or alpha range."
        )

    if verbose:
        print(
            f"Best Alpha: {best_result['best_alpha']:.2f}, Min MSE: {best_result['min_mse']:.6e}"
        )

    return best_result


if __name__ == "__main__":
    # Example Usage for find_optimal_alpha_svdquant
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _in_feat, _out_feat = 64, 128
    _rank_test, _num_bits_test, _group_size_test = 16, 4, 32

    # Dummy original weights (out_features, in_features) and calibration data
    test_orig_W = (
        torch.randn(_out_feat, _in_feat, device=_device, dtype=torch.float32) * 0.1
    )
    test_cal_act = torch.randn(100, _in_feat, device=_device, dtype=torch.float32)

    print(f"Starting alpha search for a {_out_feat}x{_in_feat} layer...")
    search_results = find_optimal_alpha_svdquant(
        test_orig_W,
        test_cal_act,
        rank=_rank_test,
        num_bits=_num_bits_test,
        group_size=_group_size_test,
        alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9],  # Smaller set for quick test
        low_rank_dtype=torch.float16,
        scale_dtype=torch.float16,
        verbose=True,
    )

    print("\nAlpha Search Complete.")
    if search_results and search_results["best_alpha"] is not None:
        print(f"  Best Alpha: {search_results['best_alpha']}")
        print(f"  Min MSE: {search_results['min_mse']}")
        print(f"  Best Lambda Val shape: {search_results['best_lambda_val'].shape}")
        print(f"  Best L1 shape: {search_results['best_L1'].shape}")
        print(f"  Best L2 shape: {search_results['best_L2'].shape}")
        print(f"  Best Q_R shape: {search_results['best_Q_R'].shape}")
        print(f"  Best Scales_R shape: {search_results['best_scales_R'].shape}")
    else:
        print("Alpha search did not yield a best result.")
