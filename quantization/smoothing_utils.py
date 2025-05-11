import torch


def calculate_smoothing_factors(calibration_activations, weights, alpha):
    """
    Calculates the per-channel smoothing factors (lambda).
    Formula from SVDQuant paper (Appendix D), based on SmoothQuant:
    lambda_i = max(|X_cal[:,i]|)^alpha / max(|W[i,:]|)^(1-alpha)
    where X_cal is (batch_size, in_features) and W is (in_features, out_features).

    Args:
        calibration_activations (torch.Tensor): A batch of representative activations
                                                (e.g., from a calibration dataset).
                                                Shape: (batch_size, in_features).
        weights (torch.Tensor): The original weight matrix of the layer.
                                Shape: (in_features, out_features).
        alpha (float): The migration strength hyperparameter (typically between 0 and 1).

    Returns:
        torch.Tensor: The per-channel smoothing factors (lambda_val). Shape: (in_features,).
    """
    if not isinstance(calibration_activations, torch.Tensor) or not isinstance(
        weights, torch.Tensor
    ):
        raise TypeError(
            "Inputs calibration_activations and weights must be PyTorch Tensors."
        )
    if not calibration_activations.ndim == 2 or not weights.ndim == 2:
        raise ValueError(
            "Inputs calibration_activations and weights must be 2D matrices."
        )
    if calibration_activations.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Number of features in activations ({calibration_activations.shape[1]}) "
            f"must match input features in weights ({weights.shape[0]})."
        )
    if not (0 <= alpha <= 1):
        # Paper doesn't strictly limit alpha, but typical range is [0,1]
        print(f"Warning: alpha ({alpha}) is outside the typical [0, 1] range.")

    device = weights.device
    if calibration_activations.device != device:
        calibration_activations = calibration_activations.to(device)

    # max_abs_act: max(|X_cal[:,i]|) across batch for each feature i
    max_abs_act = torch.max(
        torch.abs(calibration_activations), dim=0
    ).values  # Shape: (in_features,)

    # max_abs_weights: max(|W[i,:]|) across output dimension for each input feature i (i.e. for each row of W)
    max_abs_weights = torch.max(
        torch.abs(weights), dim=1
    ).values  # Shape: (in_features,)

    epsilon = torch.finfo(weights.dtype).eps  # Use dtype-appropriate epsilon
    max_abs_act = torch.clamp(max_abs_act, min=epsilon)
    max_abs_weights = torch.clamp(max_abs_weights, min=epsilon)

    lambda_val = (max_abs_act.pow(alpha)) / (max_abs_weights.pow(1 - alpha) + epsilon)

    # It might be necessary to clamp lambda_val to prevent extreme values leading to NaN/Inf
    # lambda_val = torch.clamp(lambda_val, min=1e-5, max=1e5) # Example clamp

    return lambda_val


def apply_smoothing_to_weights(weights, lambda_val):
    """
    Applies smoothing to the weight matrix: hat_W = diag(lambda_val) * W.
    This means each row i of W is scaled by lambda_val[i].
    This is a one-time transformation of the weights.

    Args:
        weights (torch.Tensor): The original weight matrix. Shape: (in_features, out_features).
        lambda_val (torch.Tensor): The pre-calculated per-channel smoothing factors. Shape: (in_features,).

    Returns:
        torch.Tensor: The transformed weight matrix (hat_W). Shape: (in_features, out_features).
    """
    if not isinstance(weights, torch.Tensor) or not isinstance(
        lambda_val, torch.Tensor
    ):
        raise TypeError("Inputs weights and lambda_val must be PyTorch Tensors.")
    if not weights.ndim == 2 or not lambda_val.ndim == 1:
        raise ValueError(
            "Input weights must be a 2D matrix and lambda_val a 1D vector."
        )
    if weights.shape[0] != lambda_val.shape[0]:
        raise ValueError(
            f"Input features in weights ({weights.shape[0]}) "
            f"must match size of lambda_val ({lambda_val.shape[0]})."
        )

    device = weights.device
    if lambda_val.device != device:
        lambda_val = lambda_val.to(device)

    # hat_W = diag(lambda_val) * W is equivalent to scaling row i of W by lambda_val[i]
    # lambda_val.unsqueeze(1) results in shape (in_features, 1)
    # Broadcasting: (in_features, 1) * (in_features, out_features) -> (in_features, out_features)
    hat_W = lambda_val.unsqueeze(1) * weights
    return hat_W


def apply_smoothing_to_activations(activations, lambda_val):
    """
    Applies smoothing to the activation tensor: hat_X = X * diag(lambda_val)^(-1).
    This means each column j of X is scaled by 1/lambda_val[j].
    This is applied to activations during inference.

    Args:
        activations (torch.Tensor): The input activation tensor. Shape: (batch_size, in_features).
        lambda_val (torch.Tensor): The pre-calculated per-channel smoothing factors. Shape: (in_features,).

    Returns:
        torch.Tensor: The smoothed activation tensor (hat_X). Shape: (batch_size, in_features).
    """
    if not isinstance(activations, torch.Tensor) or not isinstance(
        lambda_val, torch.Tensor
    ):
        raise TypeError("Inputs activations and lambda_val must be PyTorch Tensors.")
    if not activations.ndim == 2 or not lambda_val.ndim == 1:
        raise ValueError(
            "Input activations must be a 2D matrix and lambda_val a 1D vector."
        )
    if activations.shape[1] != lambda_val.shape[0]:
        raise ValueError(
            f"Number of features in activations ({activations.shape[1]}) "
            f"must match size of lambda_val ({lambda_val.shape[0]})."
        )

    device = activations.device
    if lambda_val.device != device:
        lambda_val = lambda_val.to(device)

    epsilon = torch.finfo(lambda_val.dtype).eps  # Use dtype-appropriate epsilon
    lambda_inv = 1.0 / (lambda_val + epsilon)

    # hat_X = X * diag(lambda_val)^(-1) is equivalent to scaling column j of X by 1/lambda_val[j]
    # lambda_inv.unsqueeze(0) results in shape (1, in_features)
    # Broadcasting: (batch_size, in_features) * (1, in_features) -> (batch_size, in_features)
    hat_X = activations * lambda_inv.unsqueeze(0)
    return hat_X


if __name__ == "__main__":
    # Example Usage
    batch_s, in_feat, out_feat = 32, 64, 128
    alpha_test = 0.5

    # Dummy data
    cal_act = torch.randn(
        batch_s * 5, in_feat, dtype=torch.float32
    )  # More samples for calibration
    orig_W = torch.randn(in_feat, out_feat, dtype=torch.float32)
    infer_act = torch.randn(batch_s, in_feat, dtype=torch.float32)

    print(f"Original W shape: {orig_W.shape}")
    print(f"Calibration activations shape: {cal_act.shape}")
    print(f"Inference activations shape: {infer_act.shape}")

    # 1. Calculate smoothing factors
    lambda_factors = calculate_smoothing_factors(cal_act, orig_W, alpha_test)
    print(
        f"Calculated lambda_factors shape: {lambda_factors.shape}"
    )  # Expected: (in_feat,)

    # 2. Apply smoothing to weights (offline)
    hat_W_test = apply_smoothing_to_weights(orig_W, lambda_factors)
    print(f"Smoothed hat_W shape: {hat_W_test.shape}")  # Expected: (in_feat, out_feat)

    # 3. Apply smoothing to activations (inference time)
    hat_X_test = apply_smoothing_to_activations(infer_act, lambda_factors)
    print(f"Smoothed hat_X shape: {hat_X_test.shape}")  # Expected: (batch_s, in_feat)

    # Check the core identity: X @ W should be approx hat_X @ hat_W
    # X @ W = (hat_X @ diag(lambda)) @ (diag(lambda)^(-1) @ hat_W) -> ideally if transformations are perfect inverses
    # More directly: X @ W = (X L^-1) @ (L W) where L = diag(lambda_factors)
    # So, (X_cal @ W_orig) should be 'close' to (hat_X_cal @ hat_W_orig)
    # Let's test on inference activations
    original_output = infer_act @ orig_W
    smoothed_output = hat_X_test @ hat_W_test

    mse = torch.mean((original_output - smoothed_output) ** 2)
    print(
        f"MSE between (infer_act @ orig_W) and (hat_X_test @ hat_W_test): {mse.item()}"
    )
    # This MSE should be very close to zero, indicating the transformation is almost perfectly reversible.

    # Test edge cases for calculate_smoothing_factors
    zero_weights = torch.zeros_like(orig_W)
    lambda_zero_W = calculate_smoothing_factors(cal_act, zero_weights, alpha_test)
    print(f"Lambda with zero weights: {lambda_zero_W[:5]}...")  # Should not be NaN/Inf

    zero_cal_act = torch.zeros_like(cal_act)
    lambda_zero_act = calculate_smoothing_factors(zero_cal_act, orig_W, alpha_test)
    print(
        f"Lambda with zero activations: {lambda_zero_act[:5]}..."
    )  # Should not be NaN/Inf
