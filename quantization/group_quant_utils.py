import torch


def quantize_symmetric_per_group(
    input_tensor, num_bits, group_size, scale_dtype=torch.float16
):
    """
    Performs per-group symmetric quantization on an input tensor.
    The tensor is expected to be 1D or 2D, with grouping applied along the last dimension.

    Args:
        input_tensor (torch.Tensor): Floating-point tensor to quantize. Assumed to be 1D or 2D.
                                     If 2D, shape (dim0, dim1), grouping is along dim1.
        num_bits (int): Number of bits for quantization (e.g., 4 for INT4).
        group_size (int): Size of groups for quantization along the last dimension.
                          The last dimension must be divisible by group_size.
        scale_dtype (torch.dtype): Data type for storing scales (e.g., torch.float16).

    Returns:
        quantized_values (torch.Tensor): Tensor of quantized integer values. Stored in torch.int8,
                                         but values are clamped to [-(2**(num_bits-1)), 2**(num_bits-1)-1].
        scales (torch.Tensor): Tensor of scales for each group, in scale_dtype.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input tensor must be a PyTorch Tensor.")
    if not (input_tensor.ndim == 1 or input_tensor.ndim == 2):
        raise ValueError("Input tensor must be 1D or 2D.")
    if not (1 < num_bits <= 8):
        raise ValueError("num_bits must be between 2 and 8 for int8 storage.")

    original_input_shape = input_tensor.shape
    last_dim_size = original_input_shape[-1]
    device = input_tensor.device

    # Pad the tensor if the last dimension is not divisible by group_size
    padding_needed = (group_size - (last_dim_size % group_size)) % group_size
    if padding_needed > 0:
        if input_tensor.ndim == 1:
            pad_dims = (0, padding_needed)  # Pad only at the end for 1D
        else:  # 2D
            # For 2D tensor (dim0, dim1), pad_dims should be (left, right, top, bottom)
            # We only want to pad the last dimension (dim1) on the right.
            pad_dims = (0, padding_needed, 0, 0)
        input_tensor_processed = torch.nn.functional.pad(
            input_tensor, pad_dims, mode="constant", value=0
        )
    else:
        input_tensor_processed = input_tensor

    current_shape = (
        input_tensor_processed.shape
    )  # Shape of the (potentially padded) tensor

    if input_tensor_processed.ndim == 1:
        # Reshape to (-1, group_size) for group-wise operations
        reshaped_tensor = input_tensor_processed.reshape(-1, group_size)
    else:  # 2D
        # Reshape to (original_dim0, num_groups_in_last_dim, group_size)
        # then flatten leading two to (-1, group_size)
        # This ensures scales correspond to original_dim0 * num_groups
        num_groups_in_last_dim = current_shape[-1] // group_size
        reshaped_tensor = input_tensor_processed.reshape(
            current_shape[0] * num_groups_in_last_dim, group_size
        )

    # Calculate scale for each group
    max_abs_vals = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True).values

    # For symmetric quantization, q_max corresponds to 2^(num_bits-1) - 1 for signed types
    q_max = (2 ** (num_bits - 1)) - 1

    # Perform scale calculation in float32 for potentially better precision,
    # especially if max_abs_vals are low-precision floats (bf16, fp16).
    max_abs_vals_fp32 = max_abs_vals.to(torch.float32)
    # Use float32 epsilon for the calculation to avoid issues if max_abs_vals_fp32 is near zero.
    epsilon_fp32 = torch.finfo(torch.float32).eps
    scales_fp32 = max_abs_vals_fp32 / (q_max + epsilon_fp32)

    # Convert calculated scales to the desired scale_dtype
    scales = scales_fp32.to(dtype=scale_dtype)

    # Quantize
    # Convert reshaped_tensor to float32 for stable calculations
    reshaped_tensor_fp32 = reshaped_tensor.to(torch.float32)

    # Calculate inverse of scales in float32
    inv_scales_fp32 = 1.0 / scales.to(torch.float32)

    # Perform scaling in float32
    scaled_values_fp32 = reshaped_tensor_fp32 * inv_scales_fp32

    # Round in float32, then convert to int8
    quantized_values = torch.round(scaled_values_fp32).to(torch.int8)

    # Clamp to the representable range for num_bits
    q_min = -(2 ** (num_bits - 1))
    quantized_values = torch.clamp(quantized_values, min=q_min, max=q_max)

    # Reshape scales and quantized_values
    # Scales should correspond to the number of groups in the processed tensor
    # Quantized values should have the shape of the processed tensor

    if input_tensor_processed.ndim == 1:
        scales = scales.squeeze(-1)  # Shape (num_total_groups,)
        # quantized_values is already (num_total_groups, group_size), reshape to processed shape
        quantized_values = quantized_values.reshape(current_shape)
    else:  # 2D
        num_groups_last_dim = current_shape[-1] // group_size
        # scales_shape = (current_shape[0], num_groups_last_dim)
        # The reshape for scales earlier already made it (current_shape[0] * num_groups_last_dim, 1)
        # So it should be reshaped to (current_shape[0], num_groups_last_dim)
        scales = scales.reshape(current_shape[0], num_groups_last_dim)
        # quantized_values is (current_shape[0] * num_groups_last_dim, group_size)
        quantized_values = quantized_values.reshape(current_shape)

    return quantized_values, scales


def dequantize_symmetric_per_group(
    quantized_values, scales, group_size, output_dtype=torch.float32
):
    """
    Performs per-group symmetric dequantization.
    The tensor is expected to be 1D or 2D, with grouping applied along the last dimension.

    Args:
        quantized_values (torch.Tensor): Tensor of quantized integer values (e.g., torch.int8).
                                         If 2D, shape (dim0, dim1).
        scales (torch.Tensor): Tensor of scales for each group.
                               If quantized_values is 2D (dim0, dim1), scales is (dim0, dim1 // group_size).
        group_size (int): Size of groups used during quantization.
        output_dtype (torch.dtype): Desired data type for the dequantized tensor (e.g., torch.float32).

    Returns:
        torch.Tensor: Dequantized floating-point tensor.
    """
    if not isinstance(quantized_values, torch.Tensor) or not isinstance(
        scales, torch.Tensor
    ):
        raise TypeError("Inputs quantized_values and scales must be PyTorch Tensors.")
    if not (quantized_values.ndim == 1 or quantized_values.ndim == 2):
        raise ValueError("Input quantized_values must be 1D or 2D.")
    if quantized_values.dtype != torch.int8:
        # This function expects int8 that holds values in a smaller bit range.
        # Actual num_bits is implicitly defined by how scales were calculated.
        print(
            f"Warning: quantized_values dtype is {quantized_values.dtype}, expected torch.int8."
        )

    original_shape = quantized_values.shape
    last_dim_size = original_shape[-1]
    device = quantized_values.device

    if scales.device != device:
        scales = scales.to(device)

    scales = scales.to(
        output_dtype
    )  # Ensure scales are in the target float type for multiplication

    if quantized_values.ndim == 1:
        reshaped_quantized = quantized_values.reshape(-1, group_size)
        # Scales expected shape: (num_groups_last_dim,)
        # Need to unsqueeze scales for broadcasting with reshaped_quantized
        if scales.ndim == 1:
            scales_broadcastable = scales.unsqueeze(-1)
        else:
            raise ValueError("Scales for 1D quantized input should be 1D.")
    else:  # 2D input
        # quantized_values shape (dim0, dim1)
        # scales shape (dim0, dim1 // group_size)
        # Reshape quantized to (dim0, dim1 // group_size, group_size)
        num_groups_last_dim = last_dim_size // group_size
        reshaped_quantized = quantized_values.reshape(
            original_shape[0], num_groups_last_dim, group_size
        )

        # Scales need to be unsqueezed to (dim0, dim1 // group_size, 1) for broadcasting
        if scales.shape == (original_shape[0], num_groups_last_dim):
            scales_broadcastable = scales.unsqueeze(-1)
        else:
            raise ValueError(
                f"Scales shape {scales.shape} incompatible with quantized_values {original_shape} and group_size {group_size}"
            )

    dequantized_tensor_grouped = (
        reshaped_quantized.to(output_dtype) * scales_broadcastable
    )
    dequantized_tensor = dequantized_tensor_grouped.reshape(original_shape)

    return dequantized_tensor


if __name__ == "__main__":
    # Example Usage
    num_b = 4  # 4-bit quantization
    grp_size = 64  # As per paper for INT4
    scale_dt = torch.float16

    # Test 1D tensor
    print("\n--- Testing 1D Tensor ---")
    tensor_1d = torch.randn(256, dtype=torch.float32) * 10  # (num_elements)
    print(f"Original 1D Tensor (first 10): {tensor_1d[:10]}")
    quant_1d, scales_1d = quantize_symmetric_per_group(
        tensor_1d, num_b, grp_size, scale_dt
    )
    print(f"Quantized 1D (first 10 of 1st group): {quant_1d[:10]}")
    print(
        f"Scales 1D shape: {scales_1d.shape}, dtype: {scales_1d.dtype}"
    )  # Expected (256/64,)
    dequant_1d = dequantize_symmetric_per_group(
        quant_1d, scales_1d, grp_size, output_dtype=torch.float32
    )
    print(f"Dequantized 1D (first 10): {dequant_1d[:10]}")
    mse_1d = torch.mean((tensor_1d - dequant_1d) ** 2)
    print(f"MSE 1D: {mse_1d.item()}")

    # Test 2D tensor (e.g., weights: out_features x in_features)
    print("\n--- Testing 2D Tensor ---")
    out_f, in_f = 32, 128
    tensor_2d = torch.randn(out_f, in_f, dtype=torch.float32) * 5  # (dim0, dim1)
    print(f"Original 2D Tensor shape: {tensor_2d.shape}")

    quant_2d, scales_2d = quantize_symmetric_per_group(
        tensor_2d, num_b, grp_size, scale_dt
    )
    print(f"Quantized 2D shape: {quant_2d.shape}")
    print(
        f"Scales 2D shape: {scales_2d.shape}, dtype: {scales_2d.dtype}"
    )  # Expected (out_f, in_f/grp_size)

    dequant_2d = dequantize_symmetric_per_group(
        quant_2d, scales_2d, grp_size, output_dtype=torch.float32
    )
    print(f"Dequantized 2D shape: {dequant_2d.shape}")
    mse_2d = torch.mean((tensor_2d - dequant_2d) ** 2)
    print(f"MSE 2D: {mse_2d.item()}")

    # Example for activations (batch_size, features)
    print("\n--- Testing 2D Tensor (Activations) ---")
    batch_s, features = 16, 256
    activations = torch.rand(
        batch_s, features, dtype=torch.float32
    )  # Typically smaller range like [0,1] or [-k, k]
    quant_act, scales_act = quantize_symmetric_per_group(
        activations, num_b, grp_size, scale_dt
    )
    print(f"Quantized Activations shape: {quant_act.shape}")
    print(f"Scales Activations shape: {scales_act.shape}, dtype: {scales_act.dtype}")
    dequant_act = dequantize_symmetric_per_group(quant_act, scales_act, grp_size)
    mse_act = torch.mean((activations - dequant_act) ** 2)
    print(f"MSE Activations: {mse_act.item()}")

    # Test with num_bits = 8 (should have lower MSE)
    print("\n--- Testing 2D Tensor with num_bits = 8 ---")
    quant_2d_8bit, scales_2d_8bit = quantize_symmetric_per_group(
        tensor_2d, 8, grp_size, scale_dt
    )
    dequant_2d_8bit = dequantize_symmetric_per_group(
        quant_2d_8bit, scales_2d_8bit, grp_size
    )
    mse_2d_8bit = torch.mean((tensor_2d - dequant_2d_8bit) ** 2)
    print(f"MSE 2D (8-bit): {mse_2d_8bit.item()}")
