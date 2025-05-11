import torch
import pytest

from quantization.get_fp8_neighbor import get_fp8_neighbor

# Available FP8 dtypes for testing
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
FP8_DTYPES_TARGET = [FP8_E4M3, FP8_E5M2]

# Input tensor dtypes for value_in_orig_prec
ORIG_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# Helper to convert a list of Python floats to a tensor of a given dtype
def _t(values, dtype):
    return torch.tensor(values, dtype=dtype)


# Helper to get an FP8 value in original precision
def fp8_val(val, fp8_dtype_target, orig_dtype):
    return torch.tensor([val], dtype=orig_dtype).to(fp8_dtype_target).to(orig_dtype)


@pytest.mark.parametrize("orig_dtype", ORIG_DTYPES)
@pytest.mark.parametrize("fp8_dtype_target", FP8_DTYPES_TARGET)
class TestGetFP8Neighbor:

    def test_output_dtype(self, orig_dtype, fp8_dtype_target):
        """Test that the output tensor has the same dtype as value_in_orig_prec."""
        value = fp8_val(1.0, fp8_dtype_target, orig_dtype)
        direction = _t([1.0], dtype=orig_dtype)
        neighbor = get_fp8_neighbor(value, direction, fp8_dtype_target)
        assert neighbor.dtype == orig_dtype

    def test_zero_neighbors(self, orig_dtype, fp8_dtype_target):
        """Test neighbors of zero."""
        zero = fp8_val(0.0, fp8_dtype_target, orig_dtype)

        # Next neighbor of 0.0 (positive direction)
        direction_pos = _t([1.0], dtype=orig_dtype)
        neighbor_pos = get_fp8_neighbor(zero, direction_pos, fp8_dtype_target)
        expected_smallest_pos = torch.finfo(
            fp8_dtype_target
        ).tiny  # Smallest positive subnormal
        torch.testing.assert_close(
            neighbor_pos, fp8_val(expected_smallest_pos, fp8_dtype_target, orig_dtype)
        )
        assert neighbor_pos.item() > 0.0

        # Prev neighbor of 0.0 (negative direction)
        direction_neg = _t([-1.0], dtype=orig_dtype)
        neighbor_neg = get_fp8_neighbor(zero, direction_neg, fp8_dtype_target)
        expected_smallest_neg = -expected_smallest_pos
        torch.testing.assert_close(
            neighbor_neg, fp8_val(expected_smallest_neg, fp8_dtype_target, orig_dtype)
        )
        assert neighbor_neg.item() < 0.0

    # Known E4M3FN values and their neighbors
    # Value | Prev    | Next
    # 1.0   | 0.875   | 1.125
    # 0.5   | 0.4375  | 0.5625
    # Max   | Max-eps | Max
    # Min   | Min     | Min+eps
    # (Need to confirm these with torch for E4M3FN and E5M2 specifically)

    # Test structure for specific values:
    # (value, prev_expected, next_expected, fp8_type_comment)
    # We need to derive these expected values carefully using torch itself.

    # Example: For E4M3FN, value = 1.0.
    # Stepping from 1.0 up: nextafter(1.0, +inf) -> convert to e4m3fn -> back to orig_dtype. This is the expected next.
    # Stepping from 1.0 down: nextafter(1.0, -inf) -> convert to e4m3fn -> back to orig_dtype. This is the expected prev.

    @pytest.mark.parametrize("val_py", [0.5, 1.0, 1.5, 2.0, -0.5, -1.0])
    def test_specific_values_next(self, val_py, orig_dtype, fp8_dtype_target):
        value_orig_prec = fp8_val(
            val_py, fp8_dtype_target, orig_dtype
        )  # This is already an FP8 representable value
        direction = _t([1.0], dtype=orig_dtype)

        # Calculate expected next FP8 value more robustly
        # 1. Take value_orig_prec (which is an FP8 point in high prec)
        # 2. Find the true next representable float in high precision
        true_next_in_orig_prec = torch.nextafter(
            value_orig_prec, torch.tensor(float("inf"), dtype=orig_dtype)
        )
        # 3. Convert this true_next to FP8 and back to see what FP8 value it maps to.
        #    This is the expected FP8 neighbor if true_next_in_orig_prec itself isn't the original value_orig_prec already at a boundary.
        #    If value_orig_prec is already max_fp8, then true_next_in_orig_prec.to(fp8).to(orig) will still be max_fp8.
        expected_neighbor_val = true_next_in_orig_prec.to(fp8_dtype_target).to(
            orig_dtype
        )

        # If value_orig_prec is the max representable FP8 value, then its "next" FP8 neighbor is itself.
        max_fp8 = fp8_val(
            torch.finfo(fp8_dtype_target).max, fp8_dtype_target, orig_dtype
        )
        if torch.equal(value_orig_prec, max_fp8):
            expected_neighbor_val = max_fp8
        # A special case: if true_next_in_orig_prec converts to the *same* fp8 value as value_orig_prec,
        # it means value_orig_prec was on one side of a sparse FP8 boundary and the next FP8 value is further away.
        # The tested function steps by `tiny` then converts. We need to match that logic.
        # The function under test does: nextafter(value, value + direction * tiny).to(fp8).to(orig)
        # So, the expected_neighbor_val should be derived this way too.
        stepped_val_for_expected = torch.nextafter(
            value_orig_prec, value_orig_prec + direction * torch.finfo(orig_dtype).tiny
        )
        expected_neighbor_val = stepped_val_for_expected.to(fp8_dtype_target).to(
            orig_dtype
        )

        neighbor = get_fp8_neighbor(value_orig_prec, direction, fp8_dtype_target)
        torch.testing.assert_close(
            neighbor, expected_neighbor_val, msg=f"Next for {val_py}"
        )
        if not torch.equal(
            value_orig_prec, max_fp8
        ):  # If not max, neighbor should be greater or equal
            assert neighbor.item() >= value_orig_prec.item()
        else:  # If max, neighbor is max
            assert neighbor.item() == value_orig_prec.item()

    @pytest.mark.parametrize("val_py", [0.5, 1.0, 1.5, 2.0, -0.5, -1.0])
    def test_specific_values_prev(self, val_py, orig_dtype, fp8_dtype_target):
        value_orig_prec = fp8_val(val_py, fp8_dtype_target, orig_dtype)
        direction = _t([-1.0], dtype=orig_dtype)

        stepped_val_for_expected = torch.nextafter(
            value_orig_prec, value_orig_prec + direction * torch.finfo(orig_dtype).tiny
        )
        expected_neighbor_val = stepped_val_for_expected.to(fp8_dtype_target).to(
            orig_dtype
        )

        # If value_orig_prec is the min representable FP8 value, then its "prev" FP8 neighbor is itself.
        min_fp8 = fp8_val(
            torch.finfo(fp8_dtype_target).min, fp8_dtype_target, orig_dtype
        )
        if torch.equal(value_orig_prec, min_fp8):
            expected_neighbor_val = min_fp8

        neighbor = get_fp8_neighbor(value_orig_prec, direction, fp8_dtype_target)
        torch.testing.assert_close(
            neighbor, expected_neighbor_val, msg=f"Prev for {val_py}"
        )
        if not torch.equal(
            value_orig_prec, min_fp8
        ):  # If not min, neighbor should be less or equal
            assert neighbor.item() <= value_orig_prec.item()
        else:  # If min, neighbor is min
            assert neighbor.item() == value_orig_prec.item()

    def test_max_value_next(self, orig_dtype, fp8_dtype_target):
        """Test next neighbor of max FP8 value."""
        max_val = torch.finfo(fp8_dtype_target).max
        value_orig_prec = fp8_val(max_val, fp8_dtype_target, orig_dtype)
        direction = _t([1.0], dtype=orig_dtype)
        neighbor = get_fp8_neighbor(value_orig_prec, direction, fp8_dtype_target)
        torch.testing.assert_close(
            neighbor, value_orig_prec, msg="Next of max"
        )  # Should be max itself

    def test_min_value_prev(self, orig_dtype, fp8_dtype_target):
        """Test prev neighbor of min FP8 value."""
        min_val = torch.finfo(fp8_dtype_target).min
        value_orig_prec = fp8_val(min_val, fp8_dtype_target, orig_dtype)
        direction = _t([-1.0], dtype=orig_dtype)
        neighbor = get_fp8_neighbor(value_orig_prec, direction, fp8_dtype_target)
        torch.testing.assert_close(
            neighbor, value_orig_prec, msg="Prev of min"
        )  # Should be min itself
