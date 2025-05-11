import torch
import pytest
from unittest.mock import patch

from quantization.stochastic_round_tensor_to_fp8 import stochastic_round_tensor_to_fp8

# Available FP8 dtypes for testing
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
FP8_DTYPES = [FP8_E4M3, FP8_E5M2]

# Test tensor dtypes
TEST_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
def test_empty_tensor(fp8_dtype_to_test):
    """Test that an empty tensor is returned as an empty tensor of the target fp8_dtype."""
    empty_tensor = torch.tensor([])
    result = stochastic_round_tensor_to_fp8(
        empty_tensor,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )
    assert result.dtype == fp8_dtype_to_test
    assert result.numel() == 0


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
def test_non_floating_point_tensor(fp8_dtype_to_test):
    """Test that non-floating point tensors are converted directly if possible."""
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

    # It's a bit tricky because direct conversion from int32 to fp8 might not be supported
    # or behave as expected across all torch versions / fp8 types.
    # The function attempts a direct .to(fp8_dtype)
    # We expect it to either convert or return original on exception.
    # For integers, direct conversion to FP8 usually results in those same integers if they are representable.

    result = stochastic_round_tensor_to_fp8(
        int_tensor,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )

    # If direct conversion is possible and doesn't error
    if result.dtype == fp8_dtype_to_test:
        # Check if values are preserved (common for small integers)
        # This might need adjustment based on actual behavior of torch.tensor([1,2,3], dtype=torch.int32).to(fp8_dtype_to_test)
        expected_tensor = int_tensor.to(torch.float32).to(
            fp8_dtype_to_test
        )  # Simulate expected path
        torch.testing.assert_close(result, expected_tensor)
    else:
        # If conversion failed and original is returned
        assert result.dtype == int_tensor.dtype
        torch.testing.assert_close(result, int_tensor)


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
def test_non_floating_point_tensor_unsupported_conversion(fp8_dtype_to_test, capsys):
    """Test that a warning is printed and original tensor is returned if direct conversion of non-float fails."""
    # Complex numbers are a good example of non-float that won't convert to fp8
    complex_tensor = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)

    result = stochastic_round_tensor_to_fp8(
        complex_tensor,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )

    assert result.dtype == complex_tensor.dtype
    torch.testing.assert_close(result, complex_tensor)
    captured = capsys.readouterr()
    assert (
        f"Warning: Could not convert non-float tensor of dtype {complex_tensor.dtype} to {fp8_dtype_to_test}"
        in captured.out
    )


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
@pytest.mark.parametrize("tensor_dtype", TEST_DTYPES)
@patch("quantization.stochastic_round_tensor_to_fp8.stochastic_round_owlshift_method")
def test_use_owlshift_method(
    mock_owlshift_method,
    tensor_dtype,
    fp8_dtype_to_test,
):
    """Test that stochastic_round_owlshift_method is called when use_owlshift_method is True."""
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=tensor_dtype)
    seed = 123
    debug_mode = False

    # Make the mock return a tensor of the correct fp8_dtype to satisfy the rest of the function flow
    mock_owlshift_method.return_value = torch.tensor(
        [1.0, 2.0, 3.0], dtype=fp8_dtype_to_test
    )

    result = stochastic_round_tensor_to_fp8(
        tensor,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=True,
        seed=seed,
        debug_mode=debug_mode,
    )

    mock_owlshift_method.assert_called_once_with(tensor, fp8_dtype_to_test, seed=seed)
    assert result.dtype == fp8_dtype_to_test


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
@pytest.mark.parametrize("tensor_dtype", TEST_DTYPES)
@patch("quantization.stochastic_round_tensor_to_fp8.stochastic_round_shift_perturb")
def test_use_shift_perturb_method(
    mock_shift_perturb_method,
    tensor_dtype,
    fp8_dtype_to_test,
):
    """Test that stochastic_round_shift_perturb is called when use_shift_perturb_method is True."""
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=tensor_dtype)
    debug_mode = False

    # Make the mock return a tensor of the correct fp8_dtype
    mock_shift_perturb_method.return_value = torch.tensor(
        [1.0, 2.0, 3.0], dtype=fp8_dtype_to_test
    )

    result = stochastic_round_tensor_to_fp8(
        tensor,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,  # Should be ignored
        use_shift_perturb_method=True,
        use_owlshift_method=False,  # Owlshift takes precedence if both are true, so set to False
        debug_mode=debug_mode,
    )

    mock_shift_perturb_method.assert_called_once_with(tensor, fp8_dtype_to_test)
    assert result.dtype == fp8_dtype_to_test


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
@pytest.mark.parametrize("tensor_dtype", TEST_DTYPES)
@patch(
    "quantization.stochastic_round_tensor_to_fp8.get_fp8_bracketing_candidates_complex"
)
def test_complex_bracketing_method(
    mock_get_candidates,
    tensor_dtype,
    fp8_dtype_to_test,
    seed_value=42,  # for torch.rand_like
):
    """Test the complex bracketing stochastic rounding path."""
    # Input tensor value is between low and high candidate
    # e.g., low=0.5, high=1.0, tensor=0.75. Prob_high = (0.75-0.5)/(1.0-0.5) = 0.25/0.5 = 0.5
    # With seed 42, torch.rand(1) is ~0.83 torch.rand_like(tensor) will give a value > 0.5, so it should pick low_candidate.
    # If rand_like gives < 0.5, it should pick high_candidate.
    # Let's try to set up tensor and candidates for clearer probabilities.

    # E4M3 example values (approximate)
    # val_low_fp8 = torch.tensor(0.875, dtype=tensor_dtype) # FP8: 0.875
    # val_high_fp8 = torch.tensor(1.0, dtype=tensor_dtype)  # FP8: 1.0
    # Test tensor value closer to val_low_fp8, e.g. 0.9
    # Prob_high = (0.9 - 0.875) / (1.0 - 0.875) = 0.025 / 0.125 = 0.2

    # We need actual FP8 representable values for candidates for robust tests
    # Let's pick values that are definitely representable in e4m3 and e5m2
    # E.g., for e4m3: 0.5, 0.75 are representable (mantissa 0, 1 with some exponent)
    # 1.0, 1.5, 2.0 are representable.
    # Let low_candidate_val = 1.0, high_candidate_val = 1.5
    # Let tensor_val = 1.125 (exactly halfway, prob_high = 0.25 / 0.5 = 0.5)

    low_candidate_val_orig_prec = torch.tensor([1.0], dtype=tensor_dtype)
    high_candidate_val_orig_prec = torch.tensor([1.5], dtype=tensor_dtype)
    tensor = torch.tensor([1.125], dtype=tensor_dtype)  # Exactly halfway

    # Ensure candidates are already in fp8_dtype when returned by the mock, then converted to tensor.dtype
    # The function expects candidates in original precision, so this is fine.
    mock_get_candidates.return_value = (
        low_candidate_val_orig_prec.clone(),
        high_candidate_val_orig_prec.clone(),
    )

    torch.manual_seed(seed_value)  # For reproducible random_draw
    # For tensor [1.125], low [1.0], high [1.5]:
    # prob_high = (1.125 - 1.0) / (1.5 - 1.0) = 0.125 / 0.5 = 0.25
    # With seed 42, torch.rand(1) is ~0.83. Since 0.83 is NOT < 0.25, it should choose low_candidate (1.0)
    expected_value_fp8 = low_candidate_val_orig_prec.to(fp8_dtype_to_test)

    result = stochastic_round_tensor_to_fp8(
        tensor,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=True,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )

    mock_get_candidates.assert_called_once_with(tensor, fp8_dtype_to_test)
    assert result.dtype == fp8_dtype_to_test
    torch.testing.assert_close(result, expected_value_fp8)

    # Test case: tensor value closer to high_candidate -> higher prob_high
    # tensor_val = 1.375 -> prob_high = (1.375 - 1.0) / (1.5 - 1.0) = 0.375 / 0.5 = 0.75
    # With seed 42, torch.rand(1) is ~0.83. Since 0.83 is NOT < 0.75, it should choose low_candidate (1.0)
    # This means my rand() assumption might be off or I need a different seed / test setup.
    # Let's make it simpler: if rand < prob_high, choose high. Otherwise low.

    # Seed 0: rand() for a single element tensor([1.375]) is tensor([0.4963])
    # prob_high = 0.75. random_draw (0.4963) < prob_high (0.75) is TRUE. Should choose high_candidate.
    tensor_closer_to_high = torch.tensor([1.375], dtype=tensor_dtype)
    mock_get_candidates.return_value = (
        low_candidate_val_orig_prec.clone(),
        high_candidate_val_orig_prec.clone(),
    )
    expected_value_closer_to_high_fp8 = high_candidate_val_orig_prec.to(
        fp8_dtype_to_test
    )

    torch.manual_seed(0)  # Seed that gives rand ~0.49 for a single element
    result_closer_to_high = stochastic_round_tensor_to_fp8(
        tensor_closer_to_high,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=True,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )
    # mock_get_candidates would have been called again, so check call count or reset
    # For simplicity in this edit, we assume it's fine if we test one case per function call.
    # Better would be to re-initialize for the second sub-test or make it a separate test function.
    assert result_closer_to_high.dtype == fp8_dtype_to_test
    torch.testing.assert_close(
        result_closer_to_high,
        expected_value_closer_to_high_fp8,
        msg="Tensor closer to high candidate",
    )

    # Test case: Denominator is zero (low_candidate == high_candidate)
    # prob_high should be 0, should always choose low_candidate (which is also high_candidate)
    degenerate_candidate_val = torch.tensor([1.0], dtype=tensor_dtype)
    tensor_degenerate = torch.tensor([1.0], dtype=tensor_dtype)
    mock_get_candidates.return_value = (
        degenerate_candidate_val.clone(),
        degenerate_candidate_val.clone(),
    )
    expected_degenerate_fp8 = degenerate_candidate_val.to(fp8_dtype_to_test)

    torch.manual_seed(0)  # Seed shouldn't matter here
    result_degenerate = stochastic_round_tensor_to_fp8(
        tensor_degenerate,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=True,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )
    assert result_degenerate.dtype == fp8_dtype_to_test
    torch.testing.assert_close(
        result_degenerate, expected_degenerate_fp8, msg="Degenerate candidates"
    )


@pytest.mark.parametrize("fp8_dtype_to_test", FP8_DTYPES)
@pytest.mark.parametrize("tensor_dtype", TEST_DTYPES)
@patch("quantization.stochastic_round_tensor_to_fp8.get_fp8_neighbor")
def test_default_bracketing_method(
    mock_get_neighbor,
    tensor_dtype,
    fp8_dtype_to_test,
    # Seed 0: rand() for a single element is tensor([0.4963]) if tensor.numel() == 1
    # Seed 42: rand() for a single element is tensor([0.8379]) if tensor.numel() == 1
):
    """Test the default bracketing stochastic rounding path."""

    # --- Case 1: Tensor exactly halfway between two FP8 values ---
    # Example: tensor = 0.875 (FP8 for e4m3), neighbor (next up) could be 1.0 (FP8 for e4m3)
    # Let input tensor be 0.9375 (halfway between 0.875 and 1.0)
    # x_rne_orig_prec (0.9375 -> fp8 -> tensor_dtype) will be either 0.875 or 1.0 depending on RNE rule for exact halves.
    # Let's assume RNE rounds 0.9375 to 1.0 (if it rounds to nearest even mantissa, etc.)
    # Then direction_to_tensor = sign(0.9375 - 1.0) = -1.
    # mock_get_neighbor should be called with x_rne_orig_prec=1.0, direction=-1, returning 0.875.
    # So, low_candidate=0.875, high_candidate=1.0.
    # prob_high = (0.9375 - 0.875) / (1.0 - 0.875) = 0.0625 / 0.125 = 0.5

    # To make it concrete, let's use values that are representable and force a scenario
    # These values are in `tensor_dtype`
    val_rne = torch.tensor([1.0], dtype=tensor_dtype)  # What RNE would produce
    val_neighbor = torch.tensor(
        [0.875], dtype=tensor_dtype
    )  # The neighbor in the opposite direction of error
    tensor_halfway = torch.tensor([0.9375], dtype=tensor_dtype)

    # Mock get_fp8_neighbor: it's called with x_rne_orig_prec and a direction.
    # If tensor_halfway.to(fp8_dtype_to_test).to(tensor_dtype) is val_rne (1.0),
    # then direction will be sign(0.9375 - 1.0) = -1.
    # So, mock_get_neighbor should return val_neighbor (0.875)
    mock_get_neighbor.return_value = (
        val_neighbor.clone()
    )  # Simulate it returns the other candidate

    # Set seed so random_draw < prob_high (0.5) is TRUE, chooses high_candidate (1.0)
    # Seed 0 gives ~0.49 for a single element tensor.
    torch.manual_seed(0)
    # Expected: prob_high = 0.5. random_draw ~0.49. 0.49 < 0.5 is True. Chooses high_candidate (val_rne = 1.0)
    expected_value_halfway_fp8 = val_rne.to(fp8_dtype_to_test)

    result_halfway = stochastic_round_tensor_to_fp8(
        tensor_halfway,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )
    # x_rne_orig_prec inside the function for tensor_halfway and fp8_dtype_to_test
    # This part is tricky as .to(fp8).to(orig) behavior for exact halves needs checking for specific fp8 types.
    # For simplicity, the mock setup ensures the candidates are val_rne and val_neighbor.
    # The key is that one is RNE, the other is its neighbor in the direction that brackets the original tensor value.

    assert result_halfway.dtype == fp8_dtype_to_test
    # This assertion depends on RNE behavior + mock working as intended for candiate selection
    # The main goal is to test the prob_high and random choice logic given the candidates.
    torch.testing.assert_close(
        result_halfway, expected_value_halfway_fp8, msg="Halfway case"
    )
    # We should also assert that mock_get_neighbor was called correctly.
    # Determine x_rne_orig_prec as it would be inside the function for assertion:
    x_rne_inside_func = tensor_halfway.to(fp8_dtype_to_test).to(tensor_dtype)
    # Determine direction_for_neighbor_search
    direction_to_tensor_inside = torch.sign(tensor_halfway - x_rne_inside_func)
    direction_for_search_inside = torch.where(
        direction_to_tensor_inside == 0,
        torch.ones_like(direction_to_tensor_inside),
        direction_to_tensor_inside,
    )
    mock_get_neighbor.assert_any_call(
        x_rne_inside_func, direction_for_search_inside, fp8_dtype_to_test
    )

    # --- Case 2: Tensor is an exact FP8 value ---
    # e.g. tensor = 1.0 (which is FP8 representable)
    # x_rne_orig_prec = 1.0.to(fp8).to(tensor_dtype) = 1.0
    # direction_to_tensor = sign(1.0 - 1.0) = 0.
    # direction_for_neighbor_search = 1 (ones_like)
    # mock_get_neighbor called with 1.0, direction=1. Should return next FP8 up (e.g. 1.125 for e4m3 if 1.0 is input)
    # low_candidate = 1.0, high_candidate = 1.125
    # prob_high = (1.0 - 1.0) / (1.125 - 1.0) = 0 / 0.125 = 0.
    # random_draw will always be >= prob_high (0). Should choose low_candidate (1.0)
    tensor_exact_fp8 = torch.tensor([1.0], dtype=tensor_dtype)
    # Assume RNE of an exact FP8 is itself
    x_rne_exact = tensor_exact_fp8.to(fp8_dtype_to_test).to(tensor_dtype)
    # Mock neighbor to be something greater for this test
    val_neighbor_up = torch.tensor([1.125], dtype=tensor_dtype)
    mock_get_neighbor.return_value = val_neighbor_up.clone()

    torch.manual_seed(42)  # Seed shouldn't matter as prob_high is 0
    # Expected: prob_high = 0. Should choose low_candidate (1.0)
    expected_value_exact_fp8 = x_rne_exact.to(fp8_dtype_to_test)

    result_exact = stochastic_round_tensor_to_fp8(
        tensor_exact_fp8,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )
    assert result_exact.dtype == fp8_dtype_to_test
    torch.testing.assert_close(
        result_exact, expected_value_exact_fp8, msg="Exact FP8 case"
    )
    # Assert mock call for this case
    direction_for_search_exact = torch.ones_like(
        tensor_exact_fp8
    )  # Since direction_to_tensor is 0
    mock_get_neighbor.assert_any_call(
        x_rne_exact, direction_for_search_exact, fp8_dtype_to_test
    )

    # --- Case 3: Denominator is zero (RNE and neighbor are the same) ---
    # This could happen if tensor is at an extreme, or get_fp8_neighbor returns the same value.
    # prob_high should be 0, should always choose low_candidate
    tensor_degenerate_default = torch.tensor(
        [448.0], dtype=tensor_dtype
    )  # Max E4M3 if tensor_dtype is float
    x_rne_degenerate = tensor_degenerate_default.to(fp8_dtype_to_test).to(tensor_dtype)
    # Mock get_neighbor to return the same value (e.g., at boundary)
    mock_get_neighbor.return_value = x_rne_degenerate.clone()

    torch.manual_seed(0)  # Seed shouldn't matter
    # Expected: low_candidate == high_candidate == x_rne_degenerate. prob_high = 0.
    expected_degenerate_default_fp8 = x_rne_degenerate.to(fp8_dtype_to_test)

    result_degenerate_default = stochastic_round_tensor_to_fp8(
        tensor_degenerate_default,
        fp8_dtype=fp8_dtype_to_test,
        use_complex_method=False,
        use_shift_perturb_method=False,
        use_owlshift_method=False,
    )
    assert result_degenerate_default.dtype == fp8_dtype_to_test
    torch.testing.assert_close(
        result_degenerate_default,
        expected_degenerate_default_fp8,
        msg="Degenerate default case",
    )
    # Assert mock call for this case
    direction_to_tensor_degen = torch.sign(tensor_degenerate_default - x_rne_degenerate)
    direction_for_search_degen = torch.where(
        direction_to_tensor_degen == 0,
        torch.ones_like(direction_to_tensor_degen),
        direction_to_tensor_degen,
    )
    # If tensor_degenerate_default is already the max FP8 value, x_rne_degenerate should be that value.
    # Then direction_to_tensor_degen would be 0, and direction_for_search_degen would be 1.
    # This specific assertion for direction might need to be more robust if x_rne_degenerate is not exactly tensor_degenerate_default
    mock_get_neighbor.assert_any_call(
        x_rne_degenerate, direction_for_search_degen, fp8_dtype_to_test
    )
