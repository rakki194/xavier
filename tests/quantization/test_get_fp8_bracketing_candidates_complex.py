import torch
import pytest

from quantization.get_fp8_bracketing_candidates_complex import (
    get_fp8_bracketing_candidates_complex,
)

# Available FP8 dtypes for testing
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
FP8_DTYPES_TARGET = [FP8_E4M3, FP8_E5M2]

# Input tensor dtypes
ORIG_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# Helper to get an FP8 value in original precision
def fp8_val(val, fp8_dtype_target, orig_dtype):
    return torch.tensor([val], dtype=orig_dtype).to(fp8_dtype_target).to(orig_dtype)


# Helper to get the *actual* next FP8 value from a given FP8 value (fp8_value_orig_prec must be FP8 representable in orig_dtype)
def get_next_fp8_val(fp8_value_orig_prec, fp8_dtype_target, direction=+1.0):
    orig_dtype = fp8_value_orig_prec.dtype
    # Determine the target for nextafter based on direction
    if direction > 0:
        target_for_nextafter = torch.tensor(float("inf"), dtype=orig_dtype)
    else:
        target_for_nextafter = torch.tensor(float("-inf"), dtype=orig_dtype)

    # Step in the original precision
    stepped_val = torch.nextafter(fp8_value_orig_prec, target_for_nextafter)
    # Snap to FP8 grid and back to original precision
    return stepped_val.to(fp8_dtype_target).to(orig_dtype)


@pytest.mark.parametrize("orig_dtype", ORIG_DTYPES)
@pytest.mark.parametrize("fp8_dtype_target", FP8_DTYPES_TARGET)
class TestGetFP8BracketingCandidatesComplex:

    def test_output_properties(self, orig_dtype, fp8_dtype_target):
        tensor = torch.tensor([0.9375], dtype=orig_dtype)
        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )

        assert low_cand.dtype == orig_dtype
        assert high_cand.dtype == orig_dtype
        assert low_cand.shape == tensor.shape
        assert high_cand.shape == tensor.shape
        # Check if candidates are FP8 representable (by seeing if converting them changes their value)
        torch.testing.assert_close(
            low_cand.to(torch.float32),
            low_cand.to(fp8_dtype_target).to(orig_dtype).to(torch.float32),
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32),
            high_cand.to(fp8_dtype_target).to(orig_dtype).to(torch.float32),
        )
        assert torch.all(
            low_cand.to(torch.float32) <= high_cand.to(torch.float32)
        )  # low should always be <= high

    def test_case1_tensor_gt_cast(self, orig_dtype, fp8_dtype_target):
        """Test case where tensor > x_cast_orig_prec."""
        # Example: E4M3, val=0.875. If tensor is 0.9, x_cast_orig_prec is 0.875.
        # low_candidate should be 0.875. high_candidate should be next_fp8(0.875) = 1.0.
        val_0_875_fp8 = fp8_val(0.875, fp8_dtype_target, orig_dtype)
        tensor = torch.tensor([0.9], dtype=orig_dtype)
        # Ensure test condition: tensor > tensor.to(fp8).to(orig)
        # This depends on RNE. For 0.9 (E4M3), RNE is 0.875. So 0.9 > 0.875 is true.
        # If tensor was 0.95, RNE for E4M3 is 1.0. So 0.95 < 1.0 (Case 2).
        # We need to pick tensor such that it is indeed > x_cast
        x_cast_check = tensor.to(fp8_dtype_target).to(orig_dtype)
        if not torch.all(tensor.to(torch.float32) > x_cast_check.to(torch.float32)):
            pytest.skip(
                f"Test condition tensor > x_cast not met for {tensor.item()} with {fp8_dtype_target} (x_cast={x_cast_check.item()})"
            )

        expected_low_cand = x_cast_check
        expected_high_cand = get_next_fp8_val(
            expected_low_cand, fp8_dtype_target, direction=+1.0
        )

        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )

        torch.testing.assert_close(
            low_cand.to(torch.float32), expected_low_cand.to(torch.float32)
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32), expected_high_cand.to(torch.float32)
        )

    def test_case2_tensor_lt_cast(self, orig_dtype, fp8_dtype_target):
        """Test case where tensor < x_cast_orig_prec."""
        # Example: E4M3, val=1.0. If tensor is 0.95, x_cast_orig_prec is 1.0.
        # high_candidate should be 1.0. low_candidate should be prev_fp8(1.0) = 0.875.
        tensor = torch.tensor([0.95], dtype=orig_dtype)
        x_cast_check = tensor.to(fp8_dtype_target).to(orig_dtype)
        if not torch.all(tensor.to(torch.float32) < x_cast_check.to(torch.float32)):
            pytest.skip(
                f"Test condition tensor < x_cast not met for {tensor.item()} with {fp8_dtype_target} (x_cast={x_cast_check.item()})"
            )

        expected_high_cand = x_cast_check
        expected_low_cand = get_next_fp8_val(
            expected_high_cand, fp8_dtype_target, direction=-1.0
        )

        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )

        torch.testing.assert_close(
            low_cand.to(torch.float32), expected_low_cand.to(torch.float32)
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32), expected_high_cand.to(torch.float32)
        )

    def test_case3_tensor_eq_cast(self, orig_dtype, fp8_dtype_target):
        """Test case where tensor == x_cast_orig_prec (tensor is an FP8 value)."""
        # Example: E4M3, tensor = 1.0. x_cast_orig_prec is 1.0.
        # low_candidate should be 1.0. high_candidate should be next_fp8(1.0) = 1.125.
        tensor = fp8_val(
            1.0, fp8_dtype_target, orig_dtype
        )  # Ensure it's an exact FP8 value
        x_cast_check = tensor.to(fp8_dtype_target).to(orig_dtype)
        assert torch.all(
            tensor.to(torch.float32) == x_cast_check.to(torch.float32)
        ), "Test setup: tensor should be exact FP8"

        expected_low_cand = tensor
        expected_high_cand = get_next_fp8_val(tensor, fp8_dtype_target, direction=+1.0)

        # Special case: if tensor is max_fp8, then next_fp8 will also be max_fp8.
        max_fp8_val = fp8_val(
            torch.finfo(fp8_dtype_target).max, fp8_dtype_target, orig_dtype
        )
        if torch.equal(tensor, max_fp8_val):
            expected_high_cand = max_fp8_val

        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )

        torch.testing.assert_close(
            low_cand.to(torch.float32), expected_low_cand.to(torch.float32)
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32), expected_high_cand.to(torch.float32)
        )

    def test_max_value(self, orig_dtype, fp8_dtype_target):
        """Test with tensor being the max FP8 value."""
        max_val_py = torch.finfo(fp8_dtype_target).max
        tensor = fp8_val(max_val_py, fp8_dtype_target, orig_dtype)

        expected_low_cand = tensor
        expected_high_cand = tensor  # Next FP8 of max is max

        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )
        torch.testing.assert_close(
            low_cand.to(torch.float32),
            expected_low_cand.to(torch.float32),
            msg="Max val: Low cand",
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32),
            expected_high_cand.to(torch.float32),
            msg="Max val: High cand",
        )

    def test_min_value(self, orig_dtype, fp8_dtype_target):
        """Test with tensor being the min FP8 value."""
        # Min value for FP8 is negative. Case 3 applies if it's an exact FP8 value.
        # low_cand = min_val, high_cand = next_fp8_up(min_val)
        min_val_py = torch.finfo(fp8_dtype_target).min
        tensor = fp8_val(min_val_py, fp8_dtype_target, orig_dtype)

        expected_low_cand = tensor
        expected_high_cand = get_next_fp8_val(tensor, fp8_dtype_target, direction=+1.0)

        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )
        torch.testing.assert_close(
            low_cand.to(torch.float32),
            expected_low_cand.to(torch.float32),
            msg="Min val: Low cand",
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32),
            expected_high_cand.to(torch.float32),
            msg="Min val: High cand",
        )

    def test_mixed_values(self, orig_dtype, fp8_dtype_target):
        """Test with a tensor containing values that fall into all three cases."""
        # Values for testing (ensure they trigger different cases for common fp8_dtype_target)
        # For E4M3FN (approximate): x_cast(0.9) = 0.875 (Case 1: T > X_cast)
        #                        x_cast(0.95) = 1.0   (Case 2: T < X_cast)
        #                        x_cast(1.0) = 1.0   (Case 3: T == X_cast)
        # Need to ensure these values correctly trigger cases for E5M2 as well, or make values dynamic.
        # Let's pick values more generally.
        # val1: such that tensor > x_cast (e.g. RNE_val + a bit, but not enough to reach next RNE)
        # val2: such that tensor < x_cast (e.g. RNE_val - a bit)
        # val3: an exact FP8 value

        exact_fp8 = fp8_val(1.0, fp8_dtype_target, orig_dtype)
        val_for_case1 = exact_fp8 + fp8_val(
            torch.finfo(fp8_dtype_target).eps / 2.0, fp8_dtype_target, orig_dtype
        )  # slightly above an fp8 value
        # Ensure val_for_case1 itself is not an FP8 value and RNE(val_for_case1) is exact_fp8
        val_for_case1 = (
            exact_fp8 + get_next_fp8_val(exact_fp8, fp8_dtype_target, +1.0)
        ) / 2.0  # Midpoint, should RNE to one of them
        if val_for_case1.to(fp8_dtype_target).to(orig_dtype) >= val_for_case1:
            val_for_case1 = (
                exact_fp8 + get_next_fp8_val(exact_fp8, fp8_dtype_target, -1.0)
            ) / 2.0  # try other midpoint
            if val_for_case1.to(fp8_dtype_target).to(orig_dtype) >= val_for_case1:
                pytest.skip("Could not reliably create value for Case 1 for mixed test")

        prev_exact_fp8 = get_next_fp8_val(exact_fp8, fp8_dtype_target, -1.0)
        val_for_case2 = (exact_fp8 + prev_exact_fp8) / 2.0  # Midpoint
        if val_for_case2.to(fp8_dtype_target).to(orig_dtype) <= val_for_case2:
            pytest.skip("Could not reliably create value for Case 2 for mixed test")

        test_values_py = [val_for_case1.item(), val_for_case2.item(), exact_fp8.item()]
        # Filter out skips if they happened during value generation
        # This setup for mixed values is getting complex, might simplify or use fixed known values that work for both e4m3/e5m2 for these cases.
        # For now, let's use more stable, less dynamic values, even if they don't hit all cases for ALL fp8 types perfectly.
        # The individual case tests are more robust for that.

        if fp8_dtype_target == FP8_E4M3:
            test_values_py = [0.9, 0.95, 1.0]  # Case1, Case2, Case3 for E4M3
        elif fp8_dtype_target == FP8_E5M2:
            # E5M2 has more precision. 0.9->0.875 (C1), 0.95->0.9375 (C1), 0.98->1.0 (C2), 1.0->1.0 (C3)
            test_values_py = [0.9, 0.98, 1.0]  # Approx Case1, Case2, Case3 for E5M2
        else:
            test_values_py = [0.9, 0.95, 1.0]  # Default

        tensor = torch.tensor(test_values_py, dtype=orig_dtype)

        expected_low_list = []
        expected_high_list = []

        for val_py_single in test_values_py:
            ts = torch.tensor([val_py_single], dtype=orig_dtype)
            x_cast_s = ts.to(fp8_dtype_target).to(orig_dtype)

            elc_s, ehc_s = None, None
            if ts > x_cast_s:  # Case 1
                elc_s = x_cast_s
                ehc_s = get_next_fp8_val(x_cast_s, fp8_dtype_target, +1.0)
            elif ts < x_cast_s:  # Case 2
                ehc_s = x_cast_s
                elc_s = get_next_fp8_val(x_cast_s, fp8_dtype_target, -1.0)
            else:  # Case 3
                elc_s = x_cast_s
                ehc_s = get_next_fp8_val(x_cast_s, fp8_dtype_target, +1.0)
                max_fp8_s = fp8_val(
                    torch.finfo(fp8_dtype_target).max, fp8_dtype_target, orig_dtype
                )
                if torch.equal(x_cast_s, max_fp8_s):
                    ehc_s = max_fp8_s

            expected_low_list.append(elc_s)
            expected_high_list.append(ehc_s)

        expected_low_cand = torch.cat(expected_low_list)
        expected_high_cand = torch.cat(expected_high_list)

        low_cand, high_cand = get_fp8_bracketing_candidates_complex(
            tensor, fp8_dtype_target
        )

        torch.testing.assert_close(
            low_cand.to(torch.float32),
            expected_low_cand.to(torch.float32),
            msg="Mixed values: Low cand",
        )
        torch.testing.assert_close(
            high_cand.to(torch.float32),
            expected_high_cand.to(torch.float32),
            msg="Mixed values: High cand",
        )
