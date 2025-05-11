import torch
import pytest

from scaling.apply_owlscale_preprocess import apply_owlscale_preprocess

# Define some FP8 constants for testing (can be simplified versions)
# These would typically come from get_fp8_constants_for_owlscale or be specific to a test case
FP8_E4M3_MIN_VAL = -448.0
FP8_E4M3_MAX_VAL = 448.0
FP8_E4M3_MIN_POS_VAL = 2**-9  # Smallest subnormal for E4M3FN, as an example

FP8_E5M2_MIN_VAL = -57344.0
FP8_E5M2_MAX_VAL = 57344.0
FP8_E5M2_MIN_POS_VAL = 2**-16  # Smallest subnormal for E5M2, as an example

# Test tensor dtypes (original)
ORIG_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Compute dtypes to test with (as used inside the function)
COMPUTE_DTYPES = [torch.float32, torch.float64]
# Target FP8 dtypes (for context, though not directly used by preprocess func for calcs)
FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


@pytest.mark.parametrize("orig_dtype", ORIG_DTYPES)
@pytest.mark.parametrize("compute_dtype_to_use", COMPUTE_DTYPES)
class TestApplyOwlscalePreprocess:

    def test_empty_tensor(self, orig_dtype, compute_dtype_to_use):
        tensor = torch.empty((0,), dtype=orig_dtype)
        processed_tensor, dequant_scale = apply_owlscale_preprocess(
            tensor,
            target_fp8_dtype=torch.float8_e4m3fn,  # Dummy, not used for calcs
            fp8_min_val=FP8_E4M3_MIN_VAL,
            fp8_max_val=FP8_E4M3_MAX_VAL,
            fp8_min_pos_val=FP8_E4M3_MIN_POS_VAL,
            compute_dtype=compute_dtype_to_use,
        )
        assert processed_tensor.dtype == orig_dtype  # Empty tensor returned as is
        assert processed_tensor.numel() == 0
        assert dequant_scale.dtype == orig_dtype  # Scale is in original dtype
        assert dequant_scale.item() == 1.0

    def test_near_zero_abs_max(self, orig_dtype, compute_dtype_to_use):
        tensor = torch.tensor([1e-15, -1e-14, 0.0], dtype=orig_dtype)
        processed_tensor, dequant_scale = apply_owlscale_preprocess(
            tensor,
            target_fp8_dtype=torch.float8_e4m3fn,
            fp8_min_val=FP8_E4M3_MIN_VAL,
            fp8_max_val=FP8_E4M3_MAX_VAL,
            fp8_min_pos_val=FP8_E4M3_MIN_POS_VAL,
            compute_dtype=compute_dtype_to_use,
        )
        # No scaling expected, tensor returned in compute_dtype, scale is 1.0 in compute_dtype
        expected_processed_tensor = tensor.to(compute_dtype_to_use)
        torch.testing.assert_close(processed_tensor, expected_processed_tensor)
        assert processed_tensor.dtype == compute_dtype_to_use
        assert dequant_scale.dtype == compute_dtype_to_use
        assert dequant_scale.item() == 1.0

    @pytest.mark.parametrize(
        "fp8_params",
        [
            (FP8_E4M3_MIN_VAL, FP8_E4M3_MAX_VAL, FP8_E4M3_MIN_POS_VAL, "E4M3"),
            (FP8_E5M2_MIN_VAL, FP8_E5M2_MAX_VAL, FP8_E5M2_MIN_POS_VAL, "E5M2"),
        ],
    )
    def test_normal_scaling_and_clamping(
        self, orig_dtype, compute_dtype_to_use, fp8_params
    ):
        fp8_min, fp8_max, fp8_min_pos, fp8_name = fp8_params

        # Tensor that will require scaling and potentially clamping
        # Values chosen to exceed typical FP8 ranges before scaling, and some within
        tensor = torch.tensor([-600.0, -10.0, 0.0, 20.0, 500.0], dtype=orig_dtype)
        calc_tensor = tensor.to(compute_dtype_to_use)

        processed_tensor, dequant_scale = apply_owlscale_preprocess(
            tensor,
            target_fp8_dtype=torch.float8_e4m3fn,  # Dummy
            fp8_min_val=fp8_min,
            fp8_max_val=fp8_max,
            fp8_min_pos_val=fp8_min_pos,
            compute_dtype=compute_dtype_to_use,
        )

        # --- Expected Calculation ---
        expected_abs_max = torch.max(torch.abs(calc_tensor))
        expected_dequant_scale_val = torch.tensor(
            1.0, dtype=compute_dtype_to_use, device=calc_tensor.device
        )
        expected_processed_tensor_no_clamp = calc_tensor.clone()

        if expected_abs_max >= 1e-12:
            clamped_exp_abs_max = expected_abs_max.clamp(min=fp8_min_pos)
            quant_scale_factor = (fp8_max - fp8_min_pos) / clamped_exp_abs_max
            expected_processed_tensor_no_clamp = calc_tensor.mul(quant_scale_factor)
            expected_dequant_scale_val = quant_scale_factor.reciprocal()

        expected_clamped_tensor = torch.clamp(
            expected_processed_tensor_no_clamp, fp8_min, fp8_max
        )
        # --- End Expected Calculation ---

        assert processed_tensor.dtype == compute_dtype_to_use
        torch.testing.assert_close(
            processed_tensor,
            expected_clamped_tensor,
            msg=f"Processed tensor for {fp8_name}",
        )

        assert dequant_scale.dtype == compute_dtype_to_use
        torch.testing.assert_close(
            dequant_scale,
            expected_dequant_scale_val,
            msg=f"Dequant scale for {fp8_name}",
        )

    @pytest.mark.parametrize(
        "fp8_params",
        [
            (FP8_E4M3_MIN_VAL, FP8_E4M3_MAX_VAL, FP8_E4M3_MIN_POS_VAL, "E4M3"),
        ],
    )
    def test_abs_max_smaller_than_min_pos(
        self, orig_dtype, compute_dtype_to_use, fp8_params
    ):
        fp8_min, fp8_max, fp8_min_pos, fp8_name = fp8_params

        # Tensor whose abs_max is very small, smaller than fp8_min_pos
        tensor = torch.tensor(
            [fp8_min_pos / 10.0, -fp8_min_pos / 20.0], dtype=orig_dtype
        )
        calc_tensor = tensor.to(compute_dtype_to_use)

        processed_tensor, dequant_scale = apply_owlscale_preprocess(
            tensor,
            target_fp8_dtype=torch.float8_e4m3fn,  # Dummy
            fp8_min_val=fp8_min,
            fp8_max_val=fp8_max,
            fp8_min_pos_val=fp8_min_pos,
            compute_dtype=compute_dtype_to_use,
        )

        # --- Expected Calculation (abs_max will be clamped to fp8_min_pos for quant_scale_factor calc) ---
        expected_abs_max = torch.max(torch.abs(calc_tensor))
        # clamped_abs_max for scale calculation will be fp8_min_pos itself
        clamped_exp_abs_max_for_scale = torch.tensor(
            fp8_min_pos, dtype=compute_dtype_to_use, device=calc_tensor.device
        )
        quant_scale_factor = (fp8_max - fp8_min_pos) / clamped_exp_abs_max_for_scale
        expected_processed_tensor_no_clamp = calc_tensor.mul(quant_scale_factor)
        expected_dequant_scale_val = quant_scale_factor.reciprocal()
        expected_clamped_tensor = torch.clamp(
            expected_processed_tensor_no_clamp, fp8_min, fp8_max
        )
        # --- End Expected Calculation ---

        assert processed_tensor.dtype == compute_dtype_to_use
        torch.testing.assert_close(
            processed_tensor,
            expected_clamped_tensor,
            msg=f"Processed tensor small abs_max for {fp8_name}",
        )

        assert dequant_scale.dtype == compute_dtype_to_use
        torch.testing.assert_close(
            dequant_scale,
            expected_dequant_scale_val,
            msg=f"Dequant scale small abs_max for {fp8_name}",
        )

    def test_debug_mode(self, orig_dtype, compute_dtype_to_use, capsys):
        tensor = torch.tensor([1.0, 2.0], dtype=orig_dtype)
        apply_owlscale_preprocess(
            tensor,
            target_fp8_dtype=torch.float8_e4m3fn,
            fp8_min_val=FP8_E4M3_MIN_VAL,
            fp8_max_val=FP8_E4M3_MAX_VAL,
            fp8_min_pos_val=FP8_E4M3_MIN_POS_VAL,
            compute_dtype=compute_dtype_to_use,
            debug_mode=True,
        )
        captured = capsys.readouterr()
        assert "DEBUG apply_owlscale_preprocess" in captured.out
        assert "Initial tensor stats" in captured.out
        assert "Calculated abs_max" in captured.out
        assert "Clamped_tensor stats" in captured.out
