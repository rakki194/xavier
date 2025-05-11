import torch
import pytest
from unittest.mock import patch, MagicMock
import warnings

from quantization.stochastic_round_shift_perturb import stochastic_round_shift_perturb

# Available FP8 dtypes for testing
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
FP8_DTYPES_TARGET = [FP8_E4M3, FP8_E5M2]

# Input tensor dtypes
ORIG_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# Helper to get an FP8 value in original precision for setting up mocks
def fp8_val(val, fp8_dtype_target, orig_dtype):
    return torch.tensor([val], dtype=orig_dtype).to(fp8_dtype_target).to(orig_dtype)


@pytest.mark.parametrize("orig_dtype", ORIG_DTYPES)
@pytest.mark.parametrize("fp8_dtype_target", FP8_DTYPES_TARGET)
class TestStochasticRoundShiftPerturb:

    def test_empty_tensor(self, orig_dtype, fp8_dtype_target):
        tensor = torch.empty((0,), dtype=orig_dtype)
        result = stochastic_round_shift_perturb(tensor, fp8_dtype_target)
        assert result.dtype == fp8_dtype_target
        assert result.numel() == 0

    def test_non_floating_point_tensor(self, orig_dtype, fp8_dtype_target, capsys):
        # Using int tensor as example, other non-float might fail conversion
        int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = stochastic_round_shift_perturb(int_tensor, fp8_dtype_target)
        if result.dtype == fp8_dtype_target:  # Conversion was possible
            expected = int_tensor.to(torch.float).to(
                fp8_dtype_target
            )  # approx expected
            torch.testing.assert_close(
                result.to(torch.float32), expected.to(torch.float32), atol=1, rtol=1
            )  # loose check
        else:  # Conversion failed, original returned
            assert result.dtype == int_tensor.dtype
            torch.testing.assert_close(result, int_tensor)

        complex_tensor = torch.tensor([1 + 1j], dtype=torch.complex64)
        result_complex = stochastic_round_shift_perturb(
            complex_tensor, fp8_dtype_target
        )
        assert result_complex.dtype == fp8_dtype_target
        expected_complex_real_fp8 = complex_tensor.real.to(fp8_dtype_target)
        torch.testing.assert_close(
            result_complex.to(torch.float32),
            expected_complex_real_fp8.to(torch.float32),
        )

    @patch(
        "quantization.stochastic_round_shift_perturb.get_fp8_bracketing_candidates_complex"
    )
    def test_delta_approx_positive(
        self, mock_get_candidates, orig_dtype, fp8_dtype_target
    ):
        """Test when delta_approx is positive, noise is generated and applied."""
        tensor_val = 0.9375  # Example value, e.g. halfway between 0.875 and 1.0
        tensor = torch.tensor([tensor_val], dtype=orig_dtype)
        seed = 42

        # Mock candidates to control delta_approx
        # Let low_cand = 0.875, high_cand = 1.0 (in orig_dtype, but FP8 representable)
        low_cand_mock = fp8_val(0.875, fp8_dtype_target, orig_dtype)
        high_cand_mock = fp8_val(1.0, fp8_dtype_target, orig_dtype)
        mock_get_candidates.return_value = (low_cand_mock, high_cand_mock)

        delta_approx_expected = high_cand_mock - low_cand_mock

        torch.manual_seed(seed)
        # Manually calculate expected noise based on the seed for a single element tensor
        # torch.rand_like(tensor) will produce a single random number
        rand_val = torch.rand(1, dtype=orig_dtype)
        noise_expected = rand_val * delta_approx_expected - (
            delta_approx_expected / 2.0
        )

        perturbed_tensor_expected = tensor + noise_expected
        expected_output = perturbed_tensor_expected.to(fp8_dtype_target)

        # Reset seed before calling the function to ensure it uses the same random number
        torch.manual_seed(seed)
        result = stochastic_round_shift_perturb(tensor, fp8_dtype_target)

        mock_get_candidates.assert_called_once_with(tensor, fp8_dtype_target)
        assert result.dtype == fp8_dtype_target
        torch.testing.assert_close(
            result.to(torch.float32), expected_output.to(torch.float32)
        )

    @patch(
        "quantization.stochastic_round_shift_perturb.get_fp8_bracketing_candidates_complex"
    )
    def test_delta_approx_zero(self, mock_get_candidates, orig_dtype, fp8_dtype_target):
        """Test when delta_approx is zero (e.g., tensor is exact FP8), noise is zero."""
        tensor_val = 1.0  # An exact FP8 value
        tensor = fp8_val(tensor_val, fp8_dtype_target, orig_dtype)
        seed = 42  # Seed shouldn't matter if noise is zero

        # Mock candidates such that low_cand == high_cand
        exact_fp8_cand_mock = fp8_val(tensor_val, fp8_dtype_target, orig_dtype)
        mock_get_candidates.return_value = (exact_fp8_cand_mock, exact_fp8_cand_mock)

        delta_approx_expected = exact_fp8_cand_mock - exact_fp8_cand_mock  # Should be 0
        assert delta_approx_expected.item() == 0.0

        noise_expected = torch.tensor([0.0], dtype=orig_dtype)  # Expect zero noise
        perturbed_tensor_expected = tensor + noise_expected
        expected_output = perturbed_tensor_expected.to(fp8_dtype_target)
        # Which is same as tensor.to(fp8_dtype_target)

        torch.manual_seed(seed)  # For consistency, though rand_like * 0 is 0
        result = stochastic_round_shift_perturb(tensor, fp8_dtype_target)

        mock_get_candidates.assert_called_once_with(tensor, fp8_dtype_target)
        assert result.dtype == fp8_dtype_target
        torch.testing.assert_close(
            result.to(torch.float32), expected_output.to(torch.float32)
        )
        # Also check if it's identical to direct conversion, as noise should be zero
        torch.testing.assert_close(
            result.to(torch.float32), tensor.to(fp8_dtype_target).to(torch.float32)
        )

    @patch(
        "quantization.stochastic_round_shift_perturb.get_fp8_bracketing_candidates_complex"
    )
    def test_multiple_elements(self, mock_get_candidates, orig_dtype, fp8_dtype_target):
        """Test with multiple elements, some with delta > 0, some with delta = 0."""
        # Element 1: delta > 0 (e.g. 0.9375 for E4M3, low=0.875, high=1.0)
        # Element 2: delta = 0 (e.g. 1.0 for E4M3, low=1.0, high=1.0)
        tensor_vals_py = [0.9375, 1.0]
        tensor = torch.tensor(tensor_vals_py, dtype=orig_dtype)
        seed = 0

        # Mock candidates
        low_cand_e1 = fp8_val(0.875, fp8_dtype_target, orig_dtype)
        high_cand_e1 = fp8_val(1.0, fp8_dtype_target, orig_dtype)
        exact_cand_e2 = fp8_val(1.0, fp8_dtype_target, orig_dtype)

        mock_low_cands = torch.cat([low_cand_e1, exact_cand_e2])
        mock_high_cands = torch.cat([high_cand_e1, exact_cand_e2])
        mock_get_candidates.return_value = (mock_low_cands, mock_high_cands)

        delta_approx_expected = mock_high_cands - mock_low_cands

        torch.manual_seed(seed)
        rand_vals = torch.rand_like(tensor)
        noise_expected = rand_vals * delta_approx_expected - (
            delta_approx_expected / 2.0
        )
        perturbed_tensor_expected = tensor + noise_expected
        expected_output = perturbed_tensor_expected.to(fp8_dtype_target)

        torch.manual_seed(seed)
        result = stochastic_round_shift_perturb(tensor, fp8_dtype_target)

        mock_get_candidates.assert_called_once_with(tensor, fp8_dtype_target)
        assert result.dtype == fp8_dtype_target
        torch.testing.assert_close(
            result.to(torch.float32), expected_output.to(torch.float32)
        )
