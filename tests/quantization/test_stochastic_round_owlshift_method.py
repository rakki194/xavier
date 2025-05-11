import torch
import pytest
from unittest.mock import patch, MagicMock
import warnings  # Ensure warnings is imported

from quantization.stochastic_round_owlshift_method import (
    stochastic_round_owlshift_method,
    MAX_SLICE_ELEMENTS,  # Import for test calculations
)

# Available FP8 dtypes for testing
FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
FP8_DTYPES_TARGET = [FP8_E4M3, FP8_E5M2]

# Input tensor dtypes
ORIG_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("orig_dtype", ORIG_DTYPES)
@pytest.mark.parametrize("fp8_dtype_target", FP8_DTYPES_TARGET)
class TestStochasticRoundOwlshiftMethod:

    def test_empty_tensor(self, orig_dtype, fp8_dtype_target):
        tensor = torch.empty((0,), dtype=orig_dtype)
        result = stochastic_round_owlshift_method(tensor, fp8_dtype_target, seed=0)
        assert result.dtype == fp8_dtype_target
        assert result.numel() == 0

    def test_non_floating_point_tensor(self, orig_dtype, fp8_dtype_target, capsys):
        int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result_int = stochastic_round_owlshift_method(
            int_tensor, fp8_dtype_target, seed=0
        )
        # Behavior for non-float can be direct conversion or return original with warning
        if result_int.dtype == fp8_dtype_target:
            # Loosely check if values are somewhat preserved if conversion happened
            torch.testing.assert_close(
                result_int.to(torch.float32),
                int_tensor.to(torch.float).to(fp8_dtype_target).to(torch.float32),
                atol=1,
                rtol=1,
            )
        else:
            assert result_int.dtype == int_tensor.dtype
            torch.testing.assert_close(result_int, int_tensor)
            captured = capsys.readouterr()
            assert (
                f"Warning (owlshift): Could not convert non-float tensor of dtype {int_tensor.dtype}"
                in captured.out
            )

        complex_tensor = torch.tensor([1 + 1j], dtype=torch.complex64)
        result_complex = stochastic_round_owlshift_method(
            complex_tensor, fp8_dtype_target, seed=0
        )
        assert result_complex.dtype == fp8_dtype_target
        expected_complex_real_fp8 = complex_tensor.real.to(fp8_dtype_target)
        torch.testing.assert_close(
            result_complex.to(torch.float32),
            expected_complex_real_fp8.to(torch.float32),
        )

    @patch(
        "quantization.stochastic_round_owlshift_method.owlshift_manual_stochastic_round_to_float8"
    )
    def test_small_tensor_no_slicing(
        self, mock_manual_round, orig_dtype, fp8_dtype_target
    ):
        tensor = torch.randn(
            10, 10, dtype=orig_dtype
        )  # numel = 100 <= MAX_SLICE_ELEMENTS
        seed = 123

        # Mock the return value to be of the target_fp8_dtype
        mock_manual_round.return_value = torch.zeros_like(
            tensor, dtype=fp8_dtype_target
        )

        result = stochastic_round_owlshift_method(tensor, fp8_dtype_target, seed=seed)

        mock_manual_round.assert_called_once()
        args, kwargs = mock_manual_round.call_args
        torch.testing.assert_close(args[0], tensor)  # First arg is the tensor
        assert args[1] == fp8_dtype_target  # Second arg is the fp8_dtype
        assert isinstance(kwargs["generator"], torch.Generator)
        # Check if the generator was seeded correctly - difficult to check exact state,
        # but can check if a generator object was passed.
        # Further seed check: if seed is fixed, the *sequence* of numbers from generator would be fixed.
        # For now, checking instance is a good start for mocking.
        assert result.dtype == fp8_dtype_target
        assert result.shape == tensor.shape

    @patch(
        "quantization.stochastic_round_owlshift_method.owlshift_manual_stochastic_round_to_float8"
    )
    def test_zero_dim_tensor_no_slicing(
        self, mock_manual_round, orig_dtype, fp8_dtype_target
    ):
        tensor = torch.tensor(3.14, dtype=orig_dtype)  # 0-dim tensor
        seed = 42
        mock_manual_round.return_value = torch.zeros_like(
            tensor, dtype=fp8_dtype_target
        )

        result = stochastic_round_owlshift_method(tensor, fp8_dtype_target, seed=seed)
        mock_manual_round.assert_called_once()
        args, kwargs = mock_manual_round.call_args
        torch.testing.assert_close(args[0], tensor)
        assert args[1] == fp8_dtype_target
        assert isinstance(kwargs["generator"], torch.Generator)
        assert result.dtype == fp8_dtype_target
        assert result.shape == tensor.shape

    @patch(
        "quantization.stochastic_round_owlshift_method.owlshift_manual_stochastic_round_to_float8"
    )
    def test_large_tensor_with_slicing(
        self, mock_manual_round, orig_dtype, fp8_dtype_target
    ):
        num_elements_large = MAX_SLICE_ELEMENTS + 100  # Ensure it's larger
        # Create a 1D tensor for simplicity in checking slices
        tensor = torch.randn(num_elements_large, dtype=orig_dtype)
        seed = 456

        # To check assembly, have the mock return identifiable chunks
        def side_effect_slicing(*args, **kwargs):
            called_tensor_slice = args[0]
            # Return a tensor of ones in target_fp8_dtype, with same shape as slice
            return torch.ones_like(called_tensor_slice, dtype=fp8_dtype_target)

        mock_manual_round.side_effect = side_effect_slicing

        result = stochastic_round_owlshift_method(tensor, fp8_dtype_target, seed=seed)

        # Calculate expected call count based on source logic
        expected_call_count = 0
        if tensor.numel() == 0:
            expected_call_count = 0  # Covered by empty_tensor test, mock_manual_round not called by slicing path
        elif tensor.numel() <= MAX_SLICE_ELEMENTS or tensor.ndim == 0:
            expected_call_count = 1
        else:
            # Slicing logic from source
            _N = tensor.numel()
            _num_slices_var = max(1, int(_N / MAX_SLICE_ELEMENTS))
            _elements_per_slice_var = (_N + _num_slices_var - 1) // _num_slices_var
            if _elements_per_slice_var > 0:
                expected_call_count = (
                    _N + _elements_per_slice_var - 1
                ) // _elements_per_slice_var
            else:  # Should not happen with N > 0
                expected_call_count = 0

        if (
            tensor.numel() > 0
        ):  # Only assert call_count if mock is expected to be called
            assert mock_manual_round.call_count == expected_call_count
        else:
            mock_manual_round.assert_not_called()  # Explicitly for empty tensor in this path

        # Check that the same generator object was used for all calls
        if mock_manual_round.call_args_list:
            first_generator = mock_manual_round.call_args_list[0][1]["generator"]
            for call_args in mock_manual_round.call_args_list:
                assert call_args[1]["generator"] is first_generator

        assert result.dtype == fp8_dtype_target
        assert result.shape == tensor.shape
        # Because the side_effect returns ones, the result should be all ones if slicing happened.
        if (
            tensor.numel() > 0 and expected_call_count > 0
        ):  # Check result if processing happened
            torch.testing.assert_close(
                result.to(torch.float32),
                torch.ones_like(tensor, dtype=fp8_dtype_target).to(torch.float32),
            )

    @patch(
        "quantization.stochastic_round_owlshift_method.owlshift_manual_stochastic_round_to_float8"
    )
    def test_seed_propagation(self, mock_manual_round, orig_dtype, fp8_dtype_target):
        tensor = torch.randn(10, dtype=orig_dtype)
        seed1 = 111
        seed2 = 222

        mock_manual_round.return_value = torch.zeros_like(
            tensor, dtype=fp8_dtype_target
        )

        # Call with seed1
        stochastic_round_owlshift_method(tensor, fp8_dtype_target, seed=seed1)
        _, kwargs1 = mock_manual_round.call_args
        generator1 = kwargs1["generator"]
        # Note: initial_seed() is a property of torch.Generator
        assert generator1.initial_seed() == seed1

        # Call with seed2
        stochastic_round_owlshift_method(tensor, fp8_dtype_target, seed=seed2)
        _, kwargs2 = mock_manual_round.call_args  # This will be the latest call args
        generator2 = kwargs2["generator"]
        assert generator2.initial_seed() == seed2

        # Ensure generators are different if seeds are different
        assert (
            generator1 is not generator2
        )  # They are different objects from different calls to SUT
