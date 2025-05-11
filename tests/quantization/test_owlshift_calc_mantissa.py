import torch
import pytest

from quantization.owlshift_calc_mantissa import owlshift_calc_mantissa

# For FP8 params helper
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
SUPPORTED_FP8_DTYPES = [FP8_E4M3, FP8_E5M2]


# Helper to get FP8 params (mirrored from test_owlshift_manual_stochastic_round_to_float8.py)
def get_fp8_params(fp8_dtype):
    if fp8_dtype == FP8_E4M3:
        return 4, 3, 7  # EXPONENT_BITS (unused here), MANTISSA_BITS, EXPONENT_BIAS
    elif fp8_dtype == FP8_E5M2:
        return 5, 2, 15  # EXPONENT_BITS (unused here), MANTISSA_BITS, EXPONENT_BIAS
    else:
        raise ValueError("Unsupported dtype for params")


# Input tensor dtype for abs_x, exponent (typically float16 from caller)
COMPUTE_DTYPE = torch.float16


@pytest.mark.parametrize("fp8_dtype_target", SUPPORTED_FP8_DTYPES)
class TestOwlshiftCalcMantissa:

    def test_path_a_normal_mask_true(self, fp8_dtype_target):
        _, MANTISSA_BITS, EXPONENT_BIAS = get_fp8_params(fp8_dtype_target)
        seed = 123

        # Example: abs_x = 1.5, actual exponent = 0. So biased_exponent = EXPONENT_BIAS + 0.
        # calc_A = (1.5 / (2**(EXPONENT_BIAS - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS)
        #        = (1.5 / 1.0 - 1.0) * (2**MANTISSA_BITS)
        #        = 0.5 * (2**MANTISSA_BITS)
        # If MANTISSA_BITS = 3, this is 0.5 * 8 = 4.0
        # If MANTISSA_BITS = 2, this is 0.5 * 4 = 2.0
        abs_x_input = torch.tensor(
            [1.5, 2.75], dtype=COMPUTE_DTYPE
        )  # 2.75 = 1.375 * 2^1
        # For 1.5, actual_exp = 0. For 2.75, actual_exp = 1.
        biased_exponent_input = torch.tensor(
            [EXPONENT_BIAS + 0, EXPONENT_BIAS + 1], dtype=COMPUTE_DTYPE
        )
        normal_mask_input = torch.tensor([True, True], dtype=torch.bool)

        generator = torch.Generator(device=abs_x_input.device)
        generator.manual_seed(seed)

        # --- Expected Calculation ---
        # Path A logic:
        mantissa_scaled_no_noise_expected = (
            abs_x_input / (2.0 ** (biased_exponent_input - EXPONENT_BIAS)) - 1.0
        ) * (2**MANTISSA_BITS)

        # Expected random noise (re-seed generator for this part of calc)
        expected_rand_gen = torch.Generator(device=abs_x_input.device).manual_seed(seed)
        random_noise_expected = torch.rand(
            mantissa_scaled_no_noise_expected.size(),
            dtype=COMPUTE_DTYPE,
            device=abs_x_input.device,
            generator=expected_rand_gen,
        )
        mantissa_scaled_with_noise_expected = (
            mantissa_scaled_no_noise_expected + random_noise_expected
        )
        expected_result_fraction = mantissa_scaled_with_noise_expected.floor() / (
            2**MANTISSA_BITS
        )
        # --- End Expected Calculation ---

        result = owlshift_calc_mantissa(
            abs_x_input,
            biased_exponent_input,
            normal_mask_input,
            MANTISSA_BITS,
            EXPONENT_BIAS,
            generator=generator,
        )

        assert result.dtype == COMPUTE_DTYPE
        torch.testing.assert_close(result, expected_result_fraction)
        # Check that the output is a fraction <= (2^MANTISSA_BITS - 1) / (2^MANTISSA_BITS)
        assert torch.all(result <= (2**MANTISSA_BITS - 1) / (2**MANTISSA_BITS))
        assert torch.all(result >= 0.0)

    def test_path_b_normal_mask_false(self, fp8_dtype_target):
        _, MANTISSA_BITS, EXPONENT_BIAS = get_fp8_params(fp8_dtype_target)
        seed = 456

        # For this path, `exponent` input to function is not used in calc_B.
        # calc_B = (abs_x / (2.0**(-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
        # This is the subnormal-like scaling.
        # Example: abs_x could be a small subnormal value.
        smallest_normal_val_fp8_scale = 2.0 ** (-EXPONENT_BIAS + 1)
        abs_x_input = torch.tensor(
            [smallest_normal_val_fp8_scale / 2.0, smallest_normal_val_fp8_scale / 4.0],
            dtype=COMPUTE_DTYPE,
        )
        # Dummy exponent, not used by this path if normal_mask is False
        biased_exponent_input = torch.tensor([0, 0], dtype=COMPUTE_DTYPE)
        normal_mask_input = torch.tensor([False, False], dtype=torch.bool)

        generator = torch.Generator(device=abs_x_input.device)
        generator.manual_seed(seed)

        # --- Expected Calculation ---
        # Path B logic:
        subnormal_scaling_factor = 2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)
        mantissa_scaled_no_noise_expected = abs_x_input / subnormal_scaling_factor

        expected_rand_gen = torch.Generator(device=abs_x_input.device).manual_seed(seed)
        random_noise_expected = torch.rand(
            mantissa_scaled_no_noise_expected.size(),
            dtype=COMPUTE_DTYPE,
            device=abs_x_input.device,
            generator=expected_rand_gen,
        )
        mantissa_scaled_with_noise_expected = (
            mantissa_scaled_no_noise_expected + random_noise_expected
        )
        expected_result_fraction = mantissa_scaled_with_noise_expected.floor() / (
            2**MANTISSA_BITS
        )
        # --- End Expected Calculation ---

        result = owlshift_calc_mantissa(
            abs_x_input,
            biased_exponent_input,
            normal_mask_input,
            MANTISSA_BITS,
            EXPONENT_BIAS,
            generator=generator,
        )

        assert result.dtype == COMPUTE_DTYPE
        torch.testing.assert_close(result, expected_result_fraction)
        # Range check as above
        assert torch.all(result <= (2**MANTISSA_BITS - 1) / (2**MANTISSA_BITS))
        assert torch.all(result >= 0.0)

    def test_mixed_mask_paths(self, fp8_dtype_target):
        _, MANTISSA_BITS, EXPONENT_BIAS = get_fp8_params(fp8_dtype_target)
        seed = 789

        # Element 1: Normal (Path A)
        abs_x1 = torch.tensor([1.5], dtype=COMPUTE_DTYPE)
        exp1 = torch.tensor([EXPONENT_BIAS + 0], dtype=COMPUTE_DTYPE)
        mask1 = torch.tensor([True], dtype=torch.bool)

        # Element 2: Subnormal-like (Path B)
        smallest_normal_fp8_scale = 2.0 ** (-EXPONENT_BIAS + 1)
        abs_x2 = torch.tensor([smallest_normal_fp8_scale / 2.0], dtype=COMPUTE_DTYPE)
        exp2 = torch.tensor([0], dtype=COMPUTE_DTYPE)  # Dummy for path B
        mask2 = torch.tensor([False], dtype=torch.bool)

        abs_x_input = torch.cat([abs_x1, abs_x2])
        biased_exponent_input = torch.cat([exp1, exp2])
        normal_mask_input = torch.cat([mask1, mask2])

        generator = torch.Generator(device=abs_x_input.device).manual_seed(seed)
        expected_rand_gen = torch.Generator(device=abs_x_input.device).manual_seed(seed)

        # --- Expected Calculation ---
        # Path A part (for element 1)
        m_scaled_A_no_noise = (abs_x1 / (2.0 ** (exp1 - EXPONENT_BIAS)) - 1.0) * (
            2**MANTISSA_BITS
        )
        noise_A = torch.rand(
            m_scaled_A_no_noise.size(),
            dtype=COMPUTE_DTYPE,
            device=abs_x_input.device,
            generator=expected_rand_gen,
        )
        res_A_frac = (m_scaled_A_no_noise + noise_A).floor() / (2**MANTISSA_BITS)

        # Path B part (for element 2)
        sub_scaling_B = 2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)
        m_scaled_B_no_noise = abs_x2 / sub_scaling_B
        noise_B = torch.rand(
            m_scaled_B_no_noise.size(),
            dtype=COMPUTE_DTYPE,
            device=abs_x_input.device,
            generator=expected_rand_gen,
        )
        res_B_frac = (m_scaled_B_no_noise + noise_B).floor() / (2**MANTISSA_BITS)

        expected_result_fraction = torch.cat([res_A_frac, res_B_frac])
        # --- End Expected Calculation ---

        result = owlshift_calc_mantissa(
            abs_x_input,
            biased_exponent_input,
            normal_mask_input,
            MANTISSA_BITS,
            EXPONENT_BIAS,
            generator=generator,
        )
        assert result.dtype == COMPUTE_DTYPE
        torch.testing.assert_close(result, expected_result_fraction)
