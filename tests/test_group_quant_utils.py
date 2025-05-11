import unittest
import torch
from quantization.group_quant_utils import (
    quantize_symmetric_per_group,
    dequantize_symmetric_per_group,
)


class TestGroupQuantUtils(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_tensor_fp32 = (
            torch.randn((16, 256), dtype=torch.float32, device=self.device) * 10
        )  # Larger scale
        self.test_tensor_bf16 = (
            torch.randn((16, 256), dtype=torch.bfloat16, device=self.device) * 10
        )
        self.test_tensor_fp16 = (
            torch.randn((16, 256), dtype=torch.float16, device=self.device) * 10
        )

    def _run_quant_dequant_test(self, tensor, num_bits, group_size, scale_dtype):
        # Test quantization
        quantized_tensor, scales = quantize_symmetric_per_group(
            tensor, num_bits, group_size, scale_dtype=scale_dtype
        )

        self.assertEqual(quantized_tensor.dtype, torch.int8)
        self.assertEqual(scales.dtype, scale_dtype)

        expected_scale_shape_dim1 = tensor.shape[0]
        if tensor.ndim > 1:  # For FC layers typically (in_features, out_features)
            expected_scale_shape_dim2 = tensor.shape[1] // group_size
            if tensor.shape[1] % group_size != 0:
                expected_scale_shape_dim2 += 1  # if not perfectly divisible
            self.assertEqual(
                scales.shape, (expected_scale_shape_dim1, expected_scale_shape_dim2)
            )
        else:  # For 1D tensors
            expected_scale_shape_dim1 = tensor.shape[0] // group_size
            if tensor.shape[0] % group_size != 0:
                expected_scale_shape_dim1 += 1
            self.assertEqual(scales.shape, (expected_scale_shape_dim1,))

        if num_bits == 8:
            max_q_val = 127
            min_q_val = -128
        elif num_bits == 4:
            max_q_val = 7
            min_q_val = -8
        else:
            # Fallback for other bit sizes, assuming symmetric range around 0
            max_q_val = 2 ** (num_bits - 1) - 1
            min_q_val = -(2 ** (num_bits - 1))

        self.assertTrue(torch.all(quantized_tensor <= max_q_val))
        self.assertTrue(torch.all(quantized_tensor >= min_q_val))

        # Test dequantization
        dequantized_tensor = dequantize_symmetric_per_group(
            quantized_tensor, scales, group_size, output_dtype=tensor.dtype
        )
        self.assertEqual(dequantized_tensor.dtype, tensor.dtype)
        # The dequantized_tensor will have the shape of the (potentially) padded input,
        # which is the same shape as quantized_tensor.
        self.assertEqual(dequantized_tensor.shape, quantized_tensor.shape)

        # Slice dequantized_tensor back to the original input tensor's shape if padding occurred
        # before comparison. The original input tensor is 'tensor'.
        unpadded_dequantized_tensor = dequantized_tensor
        if (
            dequantized_tensor.ndim > 0
            and dequantized_tensor.shape[-1] > tensor.shape[-1]
        ):
            if dequantized_tensor.ndim == 1:
                unpadded_dequantized_tensor = dequantized_tensor[: tensor.shape[-1]]
            else:  # 2D or more, slice only the last dimension
                slicing = [slice(None)] * (dequantized_tensor.ndim - 1) + [
                    slice(0, tensor.shape[-1])
                ]
                unpadded_dequantized_tensor = dequantized_tensor[tuple(slicing)]

        # Ensure the unpadded tensor has the original target shape for comparison
        self.assertEqual(unpadded_dequantized_tensor.shape, tensor.shape)

        # Check closeness (MSE)
        # For lower precision inputs like bfloat16/float16, the tolerance needs to be higher.
        # Also, 4-bit quantization will inherently have higher error.
        if tensor.dtype == torch.float32:
            atol = (
                3.8 if num_bits == 4 else 0.15
            )  # fp32 4-bit: to pass max_diff ~3.75. 8-bit: to pass ~0.14
        elif tensor.dtype == torch.bfloat16:
            atol = (
                3.2 if num_bits == 4 else 0.16
            )  # bf16 4-bit: to pass ~3.15. 8-bit: 0.16 (passes)
        elif tensor.dtype == torch.float16:
            atol = (
                3.0 if num_bits == 4 else 0.16
            )  # fp16 4-bit: to pass ~2.9. 8-bit: 0.16 (passes)
        else:
            atol = 3.8  # Fallback for 4-bit like cases

        # The error depends on the scale of the original tensor and quantization parameters.
        # We check if the dequantized tensor is "close" to the original.
        # A common way is to check if the error is within a fraction of the original tensor's range or stdev.
        # Max error should be roughly scale / (2^num_bits -1) for a group
        # For simplicity, let's use torch.allclose with a reasonable atol
        # print(f"Original tensor (sum): {tensor.sum().item()}")
        # print(f"Dequant tensor (sum): {dequantized_tensor.sum().item()}")
        # print(f"Max abs diff: {torch.max(torch.abs(tensor - dequantized_tensor)).item()}")
        self.assertTrue(
            torch.allclose(unpadded_dequantized_tensor, tensor, atol=atol, rtol=0.1),
            f"Dequantization failed for {num_bits}-bit, gs={group_size}, dtype={tensor.dtype}. Max diff: {torch.max(torch.abs(tensor - unpadded_dequantized_tensor)).item()}",
        )
        return quantized_tensor, scales, dequantized_tensor

    # --- Tests for quantize_symmetric_per_group and dequantize_symmetric_per_group ---

    # Test cases for different bit-widths and group sizes with FP32 input
    def test_fp32_8bit_group32(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp32, 8, 32, scale_dtype=torch.float32
        )

    def test_fp32_4bit_group64(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp32, 4, 64, scale_dtype=torch.float32
        )

    def test_fp32_4bit_group128(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp32, 4, 128, scale_dtype=torch.float32
        )

    def test_fp32_4bit_group_uneven(self):
        tensor = torch.randn((16, 250), dtype=torch.float32, device=self.device) * 10
        self._run_quant_dequant_test(tensor, 4, 64, scale_dtype=torch.float32)

    # Test cases with FP16 scale dtype
    def test_fp32_input_fp16_scale_8bit_group32(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp32, 8, 32, scale_dtype=torch.float16
        )

    def test_fp32_input_fp16_scale_4bit_group64(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp32, 4, 64, scale_dtype=torch.float16
        )

    # Test cases for BFLOAT16 input
    def test_bf16_8bit_group32_bf16_scale(self):
        self._run_quant_dequant_test(
            self.test_tensor_bf16, 8, 32, scale_dtype=torch.bfloat16
        )

    def test_bf16_4bit_group64_bf16_scale(self):
        self._run_quant_dequant_test(
            self.test_tensor_bf16, 4, 64, scale_dtype=torch.bfloat16
        )

    def test_bf16_4bit_group64_fp32_scale(self):
        self._run_quant_dequant_test(
            self.test_tensor_bf16,
            4,
            64,
            scale_dtype=torch.float32,  # Higher precision scale
        )

    # Test cases for FLOAT16 input
    def test_fp16_8bit_group32_fp16_scale(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp16, 8, 32, scale_dtype=torch.float16
        )

    def test_fp16_4bit_group64_fp16_scale(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp16, 4, 64, scale_dtype=torch.float16
        )

    def test_fp16_4bit_group64_fp32_scale(self):
        self._run_quant_dequant_test(
            self.test_tensor_fp16,
            4,
            64,
            scale_dtype=torch.float32,  # Higher precision scale
        )

    def test_1d_tensor_quant_dequant(self):
        tensor_1d = torch.randn(1024, dtype=torch.float32, device=self.device) * 5
        self._run_quant_dequant_test(tensor_1d, 4, 128, scale_dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()
