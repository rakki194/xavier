import torch
import pytest
import argparse
import os
from unittest.mock import patch, MagicMock, mock_open
import io
from safetensors.torch import save_file
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock, call  # ensure call is imported

# Assuming xavier.py is in the root or PYTHONPATH is set up
import xavier
from xavier import main


# Helper to create dummy safetensors file content
def create_dummy_safetensors_content(data_dict):
    # Use a temporary file to get bytes
    with tempfile.NamedTemporaryFile(delete=False) as tmp_f:
        tmp_name = tmp_f.name
    try:
        save_file(data_dict, tmp_name)
        with open(tmp_name, "rb") as f:
            content_bytes = f.read()
    finally:
        os.remove(tmp_name)
    return content_bytes


@pytest.fixture
def mock_args(tmp_path):
    """Fixture to create mock command line arguments."""
    # Create dummy input and output file paths
    input_file = os.path.abspath(tmp_path / "input.safetensors")
    output_file = os.path.abspath(tmp_path / "output.safetensors")
    plot_dir = os.path.abspath(tmp_path / "quant_plots_test_fixture")

    # Create a dummy safetensors file
    dummy_data = {"tensor1": torch.randn(2, 2), "tensor2": torch.randn(3, 3)}
    # Safetensors save_file needs a string path
    with open(input_file, "wb") as f_input:
        f_input.write(create_dummy_safetensors_content(dummy_data))

    return argparse.Namespace(
        input_file=str(input_file),
        output_file=str(output_file),
        fp8_type="e4m3",
        device="cpu",
        keys_to_quantize_suffix=[".weight"],
        complex_rounding=False,
        shifturb=False,
        owlshift=False,
        comfycale=False,
        seed=0,
        plot=False,
        plot_dir=str(plot_dir),
        plot_max_tensors=5,
        plot_sample_size=5000,
        debug=False,
    )


@patch("xavier.get_safetensors_tensor_keys")
@patch("xavier.load_tensor_from_safetensors")
@patch("xavier.save_file")
@patch("xavier.stochastic_round_tensor_to_fp8")
@patch(
    "argparse.ArgumentParser.parse_args"
)  # Keep this as the primary patch for parse_args
class TestXavierMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir_manager = tempfile.TemporaryDirectory()
        cls.temp_dir_path = cls.temp_dir_manager.name
        # Create a dummy input file for tests that need it
        cls.dummy_input_path = os.path.abspath(
            os.path.join(cls.temp_dir_path, "input.safetensors")
        )
        cls.dummy_output_path = os.path.abspath(
            os.path.join(cls.temp_dir_path, "output.safetensors")
        )
        cls.dummy_plot_dir = os.path.abspath(
            os.path.join(cls.temp_dir_path, "quant_plots_test_class")
        )

        # Save a minimal safetensor file
        dummy_tensor_data = {"test.weight": torch.randn(2, 2)}
        xavier.save_file(dummy_tensor_data, cls.dummy_input_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir_manager.cleanup()

    def setUp(self):
        # Reset or reinitialize mocks if they are instance-specific
        # Create a fresh Namespace object for each test
        self.mock_args = argparse.Namespace(
            input_file=self.dummy_input_path,
            output_file=self.dummy_output_path,
            quant_method="native",  # Default, can be overridden by tests
            fp8_type="e4m3",
            keys_to_quantize_suffix=[".weight"],
            complex_rounding=False,
            shifturb=False,
            owlshift=False,
            comfyscale=False,
            seed=0,
            device="cpu",  # Default to CPU for tests unless CUDA is explicitly tested
            plot=False,
            plot_dir=self.dummy_plot_dir,  # Use class-defined temp plot_dir
            plot_max_tensors=5,
            plot_sample_size=5000,
            debug=False,
        )
        # For tests that directly patch and pass mock_parse_args, this setUp won't affect them
        # But for tests relying on class-level patch, this ensures self.mock_args is fresh.

    # Removed individual @patch("argparse.ArgumentParser.parse_args") from test methods below
    # The class level patch will provide mock_parse_args to each method that needs it.
    def test_input_file_not_found(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        capsys,  # Ensure capsys is present here
    ):
        self.mock_args.input_file = "non_existent_file.safetensors"
        mock_parse_args.return_value = self.mock_args

        # Mock os.path.exists specifically for this test's context if not globally
        with patch("os.path.exists", return_value=False) as mock_exists:
            xavier.main()
            mock_exists.assert_called_with("non_existent_file.safetensors")

        captured = capsys.readouterr()
        self.assertIn("Error: Input file not found", captured.out)

    def test_basic_quantization_run(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        mock_parse_args.return_value = self.mock_args  # Use args from setUp

        mock_get_keys.return_value = ["test.weight"]
        dummy_tensor = torch.randn(2, 2)
        mock_load_tensor.return_value = dummy_tensor
        mock_stochastic_round.return_value = dummy_tensor.to(torch.float8_e4m3fn)

        xavier.main()

        mock_get_keys.assert_called_once_with(self.mock_args.input_file)
        mock_load_tensor.assert_called_once_with(
            self.mock_args.input_file, "test.weight", device="cpu"
        )
        mock_stochastic_round.assert_called_once()
        mock_save_file.assert_called_once()
        # Check scale key is NOT saved for basic native run without comfyscale
        saved_dict = mock_save_file.call_args[0][0]
        self.assertNotIn("test.scale_weight", saved_dict)
        self.assertNotIn("scaled_fp8", saved_dict)

    def test_fp8_type_e5m2(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        self.mock_args.fp8_type = "e5m2"
        mock_parse_args.return_value = self.mock_args

        dummy_key = "t.weight"
        dummy_tensor_val = torch.randn(2, 2)
        mock_get_keys.return_value = [dummy_key]
        mock_load_tensor.return_value = dummy_tensor_val

        # Ensure mock_stochastic_round returns the correct dtype for assertion if any
        mock_stochastic_round.return_value = torch.randn(2, 2).to(torch.float8_e5m2)

        xavier.main()

        # Verify stochastic_round_tensor_to_fp8 was called with torch.float8_e5m2
        self.assertEqual(
            mock_stochastic_round.call_args.kwargs["fp8_dtype"], torch.float8_e5m2
        )
        saved_dict = mock_save_file.call_args[0][0]
        self.assertEqual(saved_dict[dummy_key].dtype, torch.float8_e5m2)

    def _run_boolean_flag_test(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_load_tensor,
        mock_get_keys,
        flag_name,
        expect_true_for_kwarg,
    ):
        current_test_args = argparse.Namespace(**vars(self.mock_args))
        current_test_args.complex_rounding = False
        current_test_args.shifturb = False
        current_test_args.owlshift = False
        setattr(current_test_args, flag_name, True)
        current_test_args.quant_method = "native"
        mock_parse_args.return_value = current_test_args

        dummy_key = "test.weight"
        dummy_tensor_val = torch.randn(2, 2, dtype=torch.float32)
        mock_get_keys.return_value = [dummy_key]
        mock_load_tensor.return_value = dummy_tensor_val
        mock_stochastic_round.return_value = dummy_tensor_val.to(torch.float8_e4m3fn)

        xavier.main()
        self.assertTrue(mock_stochastic_round.call_args.kwargs[expect_true_for_kwarg])

    def test_boolean_flag_complex_rounding(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        self._run_boolean_flag_test(
            mock_parse_args,
            mock_stochastic_round,
            mock_load_tensor,
            mock_get_keys,
            "complex_rounding",
            "use_complex_method",
        )

    def test_boolean_flag_shifturb(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        self._run_boolean_flag_test(
            mock_parse_args,
            mock_stochastic_round,
            mock_load_tensor,
            mock_get_keys,
            "shifturb",
            "use_shift_perturb_method",
        )

    def test_boolean_flag_owlshift(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        self._run_boolean_flag_test(
            mock_parse_args,
            mock_stochastic_round,
            mock_load_tensor,
            mock_get_keys,
            "owlshift",
            "use_owlshift_method",
        )

    def test_keys_to_quantize_suffix_variations(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        dummy_tensors = {
            "layer1.weight": torch.randn(2, 2),
            "layer1.bias": torch.randn(2),  # Will not be quantized by default native
            "layer2.data": torch.randn(3, 3),  # Not a default suffix
            "another.weight": torch.randn(4, 4),
        }
        dummy_keys = list(dummy_tensors.keys())
        mock_get_keys.return_value = dummy_keys

        def load_side_effect(file_path, key, device):
            return dummy_tensors[key]

        mock_load_tensor.side_effect = load_side_effect

        # Make stochastic_round return a tensor of the input's shape and target dtype
        mock_stochastic_round.side_effect = (
            lambda tensor, fp8_dtype, **kwargs: tensor.to(
                fp8_dtype
            )  # Adjusted to accept **kwargs
        )

        # Case 1: Multiple suffixes, matching .weight and .data
        current_args_case1 = argparse.Namespace(**vars(self.mock_args))
        current_args_case1.keys_to_quantize_suffix = [".weight", ".data"]
        mock_parse_args.return_value = current_args_case1

        xavier.main()

        quant_calls_case1 = mock_stochastic_round.call_count
        # Expected: layer1.weight, layer2.data, another.weight (3 calls)
        self.assertEqual(
            quant_calls_case1, 3, "Failed for suffixes ['.weight', '.data']"
        )
        mock_stochastic_round.reset_mock()  # Reset for next case

        # Case 2: Single suffix, matching .bias
        current_args_case2 = argparse.Namespace(**vars(self.mock_args))
        current_args_case2.keys_to_quantize_suffix = [".bias"]
        mock_parse_args.return_value = current_args_case2

        xavier.main()
        quant_calls_case2 = mock_stochastic_round.call_count
        # Expected: layer1.bias (1 call)
        self.assertEqual(quant_calls_case2, 1, "Failed for suffix ['.bias']")
        mock_stochastic_round.reset_mock()

        # Case 3: No matching suffix
        current_args_case3 = argparse.Namespace(**vars(self.mock_args))
        current_args_case3.keys_to_quantize_suffix = [".nonexistent"]
        mock_parse_args.return_value = current_args_case3
        xavier.main()
        quant_calls_case3 = mock_stochastic_round.call_count
        self.assertEqual(quant_calls_case3, 0, "Failed for suffix ['.nonexistent']")

    def test_comfyscale_saves_scale_and_marker(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        self.mock_args.comfyscale = True
        self.mock_args.keys_to_quantize_suffix = [
            ".weight"
        ]  # Ensure it targets a quantizable tensor
        mock_parse_args.return_value = self.mock_args

        test_key = "test.weight"
        original_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mock_get_keys.return_value = [test_key]
        mock_load_tensor.return_value = original_tensor

        # stochastic_round is called with normalized tensor
        # its return value is the quantized tensor
        mock_stochastic_round.return_value = (original_tensor / 4.0).to(
            torch.float8_e4m3fn
        )

        xavier.main()

        mock_save_file.assert_called_once()
        saved_dict = mock_save_file.call_args[0][0]

        self.assertIn(test_key, saved_dict)
        self.assertEqual(saved_dict[test_key].dtype, torch.float8_e4m3fn)

        scale_key = "test.scale_weight"
        self.assertIn(scale_key, saved_dict)
        # Expected scale is amax(abs(original_tensor)) which is 4.0
        # The scale saved by comfyscale is float64
        self.assertEqual(saved_dict[scale_key].item(), 4.0)
        self.assertEqual(saved_dict[scale_key].dtype, torch.float64)

        self.assertIn("scaled_fp8", saved_dict)
        self.assertEqual(saved_dict["scaled_fp8"].dtype, torch.float8_e4m3fn)

    def test_plot_flag_triggers_plotting_logic(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
    ):
        self.mock_args.plot = True
        self.mock_args.quant_method = "native"  # Ensure it's a path that can plot
        self.mock_args.comfyscale = (
            True  # comfyscale path involves dequant for plotting
        )
        mock_parse_args.return_value = self.mock_args

        test_key = "plot.weight"
        original_tensor = torch.randn(5, 5, dtype=torch.float32) * 10
        mock_get_keys.return_value = [test_key]
        mock_load_tensor.return_value = original_tensor

        # Mock stochastic_round to return FP8
        quantized_fp8_tensor = (
            original_tensor / torch.max(torch.abs(original_tensor))
        ).to(torch.float8_e4m3fn)
        mock_stochastic_round.return_value = quantized_fp8_tensor

        with patch("xavier.generate_comparison_plots") as mock_generate_plots, patch(
            "xavier.MATPLOTLIB_AVAILABLE", True
        ), patch("os.makedirs") as mock_makedirs:

            xavier.main()

            mock_makedirs.assert_called_with(self.mock_args.plot_dir, exist_ok=True)
            mock_generate_plots.assert_called_once()

            # Check some args of generate_comparison_plots
            call_args = mock_generate_plots.call_args[1]  # kwargs
            self.assertEqual(call_args["tensor_key"], test_key)
            self.assertTrue(
                torch.allclose(call_args["original_tensor_cpu"], original_tensor.cpu())
            )
            self.assertEqual(
                call_args["quantized_fp8_tensor_cpu"].dtype, torch.float8_e4m3fn
            )


# We might need more tests for TorchAO paths, but those are in test_xavier_torchao.py
# This file focuses on the original native paths and common argument parsing.

if __name__ == "__main__":
    unittest.main()
