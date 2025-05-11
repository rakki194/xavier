import torch


def decompose_weights_svd(hat_W, r):
    """
    Decomposes the transformed weight matrix hat_W using SVD into
    low-rank factors L1, L2 and a residual R.

    Args:
        hat_W (torch.Tensor): The m x n transformed weight matrix.
                              Should be on the correct device.
        r (int): The desired rank for the low-rank approximation.

    Returns:
        L1 (torch.Tensor): The m x r low-rank factor. Intended for FP16 storage/use.
        L2 (torch.Tensor): The r x n low-rank factor. Intended for FP16 storage/use.
        R (torch.Tensor): The m x n residual matrix (hat_W - L1 @ L2).
    """
    if not isinstance(hat_W, torch.Tensor):
        raise TypeError("Input hat_W must be a PyTorch Tensor.")
    if not hat_W.ndim == 2:
        raise ValueError("Input hat_W must be a 2D matrix.")
    if not isinstance(r, int) or r <= 0:
        raise ValueError("Rank r must be a positive integer.")
    if r > min(hat_W.shape):
        raise ValueError(
            f"Rank r ({r}) cannot be greater than the smallest dimension of hat_W ({min(hat_W.shape)})."
        )

    # Ensure hat_W is in float32 for SVD computation for numerical stability
    hat_W_float32 = hat_W.float()

    # 1. Perform SVD
    # torch.linalg.svd returns U, S (singular values as a 1D tensor), V (not Vh or Vt)
    # V is the matrix whose columns are the right singular vectors.
    # So, V.mH (adjoint) or V.T (if real) gives V_transpose.
    try:
        U, S_diag, V = torch.linalg.svd(hat_W_float32, full_matrices=False)
    except Exception as e:
        # Handle potential SVD convergence issues, though rare with full_matrices=False
        # and float32 inputs for well-behaved matrices.
        raise RuntimeError(f"SVD computation failed: {e}")

    # Vh is V conjugate transpose (or just transpose if hat_W is real)
    Vh = V.mH  # Use .mH for generality, it's .T if matrix is real.

    # 2. Rank r is an input, already validated.

    # 3. Form L1 and L2
    # L1 = U[:, :r] @ torch.diag(S_diag[:r])
    # A more direct way for L1: scale first r columns of U by first r singular values
    L1_high_precision = U[:, :r] * S_diag[:r].unsqueeze(
        0
    )  # S_diag[:r] is (r), unsqueeze to (1,r) for broadcasting
    L2_high_precision = Vh[:r, :]

    # According to the paper, L1 and L2 are stored/used in 16-bit precision.
    # We return them in high precision here; conversion to FP16 can be done by the caller
    # if needed for storage or the 16-bit branch computation.
    # Example: L1_fp16 = L1_high_precision.to(dtype=torch.float16)
    #          L2_fp16 = L2_high_precision.to(dtype=torch.float16)

    # 4. Calculate Residual R
    # This should be done using the high precision L1, L2 and original hat_W precision
    # to maintain accuracy for the residual that will be quantized.
    # If L1 and L2 were immediately cast to fp16, R = hat_W_float32 - (L1_fp16.float() @ L2_fp16.float())
    # But it's better to compute R from higher precision components first.
    R = hat_W_float32 - (L1_high_precision @ L2_high_precision)

    # The caller can decide the dtype for L1, L2 (e.g., convert to torch.float16)
    # R will be further processed (quantized to 4-bit).
    return L1_high_precision, L2_high_precision, R


if __name__ == "__main__":
    # Example Usage (for testing the function)
    m, n, r_val = 50, 60, 10
    example_hat_W = torch.randn(m, n, dtype=torch.float32)

    print(f"Original hat_W shape: {example_hat_W.shape}")

    L1, L2, R = decompose_weights_svd(example_hat_W, r_val)

    print(f"L1 shape: {L1.shape}, dtype: {L1.dtype}")  # Expected: (m, r_val)
    print(f"L2 shape: {L2.shape}, dtype: {L2.dtype}")  # Expected: (r_val, n)
    print(f"R shape: {R.shape}, dtype: {R.dtype}")  # Expected: (m, n)

    # Check reconstruction error (should be small if r is close to rank of original)
    reconstructed_W = (L1 @ L2) + R
    error = torch.norm(example_hat_W - reconstructed_W)
    print(f"Norm of (hat_W - (L1@L2 + R)): {error.item()}")  # Should be close to 0

    # Test with a known low-rank matrix
    rank_deficient_W = torch.randn(m, r_val) @ torch.randn(r_val, n)
    L1_rd, L2_rd, R_rd = decompose_weights_svd(rank_deficient_W.float(), r_val)
    error_rd = torch.norm(
        rank_deficient_W.float() - (L1_rd @ L2_rd)
    )  # R_rd should be very small
    print(f"Norm of (rank_deficient_W - (L1_rd@L2_rd)): {error_rd.item()}")
    print(f"Norm of R_rd: {torch.norm(R_rd).item()}")

    # Test invalid rank
    try:
        decompose_weights_svd(example_hat_W, m + 1)
    except ValueError as e:
        print(f"Caught expected error for invalid rank: {e}")

    # Test non-2D tensor
    try:
        decompose_weights_svd(torch.randn(m, n, 1), r_val)
    except ValueError as e:
        print(f"Caught expected error for non-2D tensor: {e}")
