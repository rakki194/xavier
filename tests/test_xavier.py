import torch
import pytest
import argparse
import os
from unittest.mock import patch, MagicMock, mock_open
import io
from safetensors.torch import save_file
import tempfile

# Assuming xavier.py is in the root or PYTHONPATH is set up
import xavier
from xavier import main, stochastic_round_tensor_to_fp8
from scaling.apply_owlscale_preprocess import apply_owlscale_preprocess


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
    input_file = tmp_path / "input.safetensors"
    output_file = tmp_path / "output.safetensors"
    # Create a dummy safetensors file
    dummy_data = {"tensor1": torch.randn(2, 2), "tensor2": torch.randn(3, 3)}
    input_file.write_bytes(create_dummy_safetensors_content(dummy_data))

    return argparse.Namespace(
        input_file=str(input_file),
        output_file=str(output_file),
        fp8_type="e4m3",
        device="cpu",
        keys_to_quantize_suffix=[".weight"],
        complex_rounding=False,
        shifturb=False,
        owlshift=False,
        owlscale=False,
        seed=0,
        plot=False,
        plot_dir="./quant_plots/",
        plot_max_tensors=5,
        plot_sample_size=5000,
        debug=False,
    )


class TestXavierMain:

    @patch("argparse.ArgumentParser.parse_args")
    def test_input_file_not_found(self, mock_parse_args, capsys, tmp_path):
        """Test that an error is printed if the input file does not exist."""
        # Make output_file also a temporary file for this test
        output_file = tmp_path / "output.safetensors"
        mock_parse_args.return_value = argparse.Namespace(
            input_file="non_existent_input.safetensors",
            output_file=str(output_file),
            # ... other necessary args ...
            fp8_type="e4m3",
            device="cpu",
            keys_to_quantize_suffix=[".weight"],
            complex_rounding=False,
            shifturb=False,
            owlshift=False,
            owlscale=False,
            seed=0,
            plot=False,
            plot_dir="./",
            plot_max_tensors=1,
            plot_sample_size=100,
            debug=False,
        )
        xavier.main()
        captured = capsys.readouterr()
        assert "Error: Input file not found" in captured.out

    @patch("xavier.get_safetensors_tensor_keys")
    @patch("xavier.load_tensor_from_safetensors")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_basic_quantization_run(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        mock_args,
        capsys,
    ):
        """Test a basic run through the quantization process, mocking external calls."""
        mock_parse_args.return_value = mock_args

        # Define the dummy tensors and their keys
        dummy_tensors = {
            "layer1.weight": torch.randn(10, 5, dtype=torch.float32),
            "layer1.bias": torch.randn(10, dtype=torch.float32),
            "other_tensor": torch.randint(0, 10, (5,), dtype=torch.int32),
        }
        dummy_keys = list(dummy_tensors.keys())

        # Mock get_safetensors_tensor_keys to return the list of keys
        mock_get_keys.return_value = dummy_keys

        # Mock load_tensor_from_safetensors to return the correct tensor based on the key
        def load_side_effect(file_path, key, device):
            return dummy_tensors[key]

        mock_load_tensor.side_effect = load_side_effect

        # Mock stochastic_round_tensor_to_fp8 to return a tensor of the target dtype
        def mock_round_func(
            tensor,
            fp8_dtype,
            use_complex_method,
            use_shift_perturb_method,
            use_owlshift_method,
            seed,
            debug_mode,
        ):
            return tensor.to(fp8_dtype)  # Simplified mock

        mock_stochastic_round.side_effect = mock_round_func

        xavier.main()

        # Assertions
        mock_get_keys.assert_called_once_with(mock_args.input_file)

        # Check that load_tensor_from_safetensors was called for each key
        assert mock_load_tensor.call_count == len(dummy_keys)
        for key in dummy_keys:
            mock_load_tensor.assert_any_call(mock_args.input_file, key, device="cpu")

        # Check that stochastic_round was called for the .weight tensor
        # It should be called once because only "layer1.weight" matches the suffix.
        assert mock_stochastic_round.call_count == 1
        call_args_list = mock_stochastic_round.call_args_list
        # call_args_list[0] is a tuple: (pos_args_tuple, kw_args_dict)
        # If stochastic_round_tensor_to_fp8 is called with keyword arguments,
        # pos_args_tuple will be empty. Access args via kw_args_dict.
        kw_args_dict = call_args_list[0][1]  # Get the keyword arguments dictionary

        # Access arguments by their names from the kwargs dictionary
        called_tensor_arg = kw_args_dict["tensor"]
        called_fp8_dtype_arg = kw_args_dict["fp8_dtype"]
        called_complex_method_arg = kw_args_dict["use_complex_method"]
        called_shift_perturb_arg = kw_args_dict["use_shift_perturb_method"]
        called_owlshift_arg = kw_args_dict["use_owlshift_method"]
        called_seed_arg = kw_args_dict["seed"]
        called_debug_mode_arg = kw_args_dict["debug_mode"]

        torch.testing.assert_close(
            called_tensor_arg, dummy_tensors["layer1.weight"].to(mock_args.device)
        )
        assert called_fp8_dtype_arg == torch.float8_e4m3fn
        assert called_complex_method_arg == mock_args.complex_rounding
        assert called_shift_perturb_arg == mock_args.shifturb
        assert called_owlshift_arg == mock_args.owlshift
        assert called_seed_arg == mock_args.seed
        assert called_debug_mode_arg == mock_args.debug

        # Check that save_file was called with the quantized state dict
        mock_save_file.assert_called_once()
        saved_state_dict_call = mock_save_file.call_args[0][0]
        assert "layer1.weight" in saved_state_dict_call
        assert saved_state_dict_call["layer1.weight"].dtype == torch.float8_e4m3fn
        assert "layer1.bias" in saved_state_dict_call  # Bias should be passed through
        torch.testing.assert_close(
            saved_state_dict_call["layer1.bias"],
            dummy_tensors["layer1.bias"].to(mock_args.device),
        )
        assert (
            "other_tensor" in saved_state_dict_call
        )  # Non-float tensor passed through
        torch.testing.assert_close(
            saved_state_dict_call["other_tensor"],
            dummy_tensors["other_tensor"].to(mock_args.device),
        )

        captured = capsys.readouterr()
        # Check for the specific line that confirms quantization and output file
        expected_quant_line_part = f"Saving quantized model to: {mock_args.output_file}"
        assert (
            expected_quant_line_part in captured.out
        ), f"Expected part '{expected_quant_line_part}' not found in stdout: {captured.out}"

    @patch("xavier.get_safetensors_tensor_keys")
    @patch("xavier.load_tensor_from_safetensors")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_fp8_type_e5m2(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        mock_args,
    ):
        mock_args.fp8_type = "e5m2"
        mock_parse_args.return_value = mock_args

        dummy_key = "t.weight"
        dummy_tensor_val = torch.randn(2, 2)
        mock_get_keys.return_value = [dummy_key]
        mock_load_tensor.return_value = dummy_tensor_val

        mock_stochastic_round.return_value = torch.randn(2, 2).to(torch.float8_e5m2)

        xavier.main()
        # Check that the correct fp8_dtype was passed to stochastic_round
        assert mock_stochastic_round.call_args.kwargs["fp8_dtype"] == torch.float8_e5m2
        # Check that the saved tensor is e5m2
        saved_state_dict = mock_save_file.call_args[0][0]
        assert saved_state_dict[dummy_key].dtype == torch.float8_e5m2

    @patch("xavier.get_safetensors_tensor_keys")
    @patch("xavier.load_tensor_from_safetensors")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_keys_to_quantize_suffix_variations(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        mock_args,
    ):
        dummy_tensors = {
            "layer1.weight": torch.randn(2, 2),
            "layer1.bias": torch.randn(2),
            "layer2.data": torch.randn(3, 3),
            "another.weight": torch.randn(4, 4),
        }
        dummy_keys = list(dummy_tensors.keys())
        mock_get_keys.return_value = dummy_keys

        def load_side_effect(file_path, key, device):
            return dummy_tensors[key]

        mock_load_tensor.side_effect = load_side_effect

        mock_stochastic_round.return_value = torch.randn(1, 1).to(
            torch.float8_e4m3fn
        )  # Dummy return

        # Case 1: Multiple suffixes, matching .weight and .data
        mock_args.keys_to_quantize_suffix = [".weight", ".data"]
        mock_parse_args.return_value = mock_args
        xavier.main()
        assert (
            mock_stochastic_round.call_count == 3
        )  # layer1.weight, layer2.data, another.weight
        mock_stochastic_round.reset_mock()

        # Case 2: Suffix that matches no tensors
        mock_args.keys_to_quantize_suffix = [".nonexistent"]
        mock_parse_args.return_value = mock_args
        xavier.main()
        assert mock_stochastic_round.call_count == 0
        mock_stochastic_round.reset_mock()

        # Case 3: Quantize only bias (if that were a use case, though typically not)
        mock_args.keys_to_quantize_suffix = [".bias"]
        mock_parse_args.return_value = mock_args
        xavier.main()
        assert mock_stochastic_round.call_count == 1
        mock_stochastic_round.reset_mock()

    @pytest.mark.parametrize(
        "flag_to_test", ["complex_rounding", "shifturb", "owlshift"]
    )
    @patch("xavier.get_safetensors_tensor_keys")
    @patch("xavier.load_tensor_from_safetensors")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_boolean_flags(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        mock_args,
        flag_to_test,
    ):
        # Set the specific flag to True, others to False
        mock_args.complex_rounding = flag_to_test == "complex_rounding"
        mock_args.shifturb = flag_to_test == "shifturb"
        mock_args.owlshift = flag_to_test == "owlshift"
        mock_parse_args.return_value = mock_args

        # Mock get_keys and load_tensor for this test
        # Assume a single tensor is enough for testing flags
        dummy_key = "test.weight"
        dummy_tensor_val = torch.randn(2, 2, dtype=torch.float32)
        mock_get_keys.return_value = [dummy_key]
        mock_load_tensor.return_value = dummy_tensor_val

        mock_stochastic_round.return_value = torch.randn(2, 2).to(torch.float8_e4m3fn)

        xavier.main()

        assert mock_stochastic_round.call_count == 1
        # call_args is (pos_args_tuple, kw_args_dict)
        # Assuming arguments are passed by keyword:
        kwargs_passed = mock_stochastic_round.call_args.kwargs

        # Argument names for stochastic_round_tensor_to_fp8:
        # tensor, fp8_dtype, use_complex_method, use_shift_perturb_method, use_owlshift_method, seed, debug_mode
        called_complex_flag = kwargs_passed["use_complex_method"]
        called_shifturb_flag = kwargs_passed["use_shift_perturb_method"]
        called_owlshift_flag = kwargs_passed["use_owlshift_method"]

        if flag_to_test == "complex_rounding":
            assert called_complex_flag is True
            assert called_shifturb_flag is False
            assert called_owlshift_flag is False
        elif flag_to_test == "shifturb":
            assert called_complex_flag is False
            assert called_shifturb_flag is True
            assert called_owlshift_flag is False
        elif flag_to_test == "owlshift":
            assert called_complex_flag is False
            assert called_shifturb_flag is False
            assert called_owlshift_flag is True

        # Optionally, verify other key arguments if necessary
        assert (
            kwargs_passed["fp8_dtype"] == torch.float8_e4m3fn
        )  # Default from mock_args
        assert kwargs_passed["seed"] == mock_args.seed
        assert kwargs_passed["debug_mode"] == mock_args.debug

    @patch("xavier.get_safetensors_tensor_keys")
    @patch("xavier.load_tensor_from_safetensors")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("xavier.get_fp8_constants_for_owlscale")
    @patch("argparse.ArgumentParser.parse_args")
    def test_owlscale_flag(
        self,
        mock_parse_args,
        mock_get_constants,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        mock_args,
        capsys,
    ):
        mock_args.owlscale = True
        mock_args.keys_to_quantize_suffix = [".weight"]
        mock_parse_args.return_value = mock_args

        # Mock get_keys and load_tensor
        dummy_tensors = {
            "layer.weight": torch.randn(5, 5, dtype=torch.float32)
            * 10,  # Multiplied to ensure abs_max is not too small
            "layer.bias": torch.randn(5, dtype=torch.float32),
        }
        dummy_keys = list(dummy_tensors.keys())
        mock_get_keys.return_value = dummy_keys

        def load_side_effect(file_path, key, device):
            return dummy_tensors[key]

        mock_load_tensor.side_effect = load_side_effect

        # Mock get_fp8_constants_for_owlscale
        fp8_dtype_target = torch.float8_e4m3fn
        # These values are used by xavier.py but not directly in this test's core logic for input scaling check
        mock_get_constants.return_value = (-448.0, 448.0, 2**-9)

        # Determine OWLSCALE_COMPUTE_DTYPE and OWLSCALE_SCALE_DTYPE as used in xavier.py
        # These are globally set in xavier.py when --owlscale is true.
        # For the test, we use them directly.
        OWLSCALE_COMPUTE_DTYPE = torch.float64
        OWLSCALE_SCALE_DTYPE = torch.float64

        # Calculate the expected input for quantization based on xavier.py's ComfyUI owlscale logic
        tensor_to_process_weight = dummy_tensors["layer.weight"].to(mock_args.device)
        original_hp_tensor_weight = tensor_to_process_weight.to(OWLSCALE_COMPUTE_DTYPE)
        abs_max_weight = torch.max(torch.abs(original_hp_tensor_weight))
        scale_factor_for_comfyui_val_weight = abs_max_weight.clamp(min=1e-12)

        expected_input_for_quantization_hp_weight = (
            tensor_to_process_weight.to(scale_factor_for_comfyui_val_weight.dtype)
            / scale_factor_for_comfyui_val_weight
        )
        expected_input_for_quantization_weight = (
            expected_input_for_quantization_hp_weight.to(tensor_to_process_weight.dtype)
        )

        # The scale factor that should be saved
        expected_scale_to_save_weight = scale_factor_for_comfyui_val_weight.to(
            OWLSCALE_SCALE_DTYPE
        )

        # Mock stochastic_round_tensor_to_fp8 to simply return the input tensor converted to target dtype
        # This allows us to check the input to this function accurately.
        def mock_round_side_effect(tensor, fp8_dtype, *args, **kwargs):
            return tensor.to(fp8_dtype)

        mock_stochastic_round.side_effect = mock_round_side_effect

        xavier.main()

        mock_get_constants.assert_called_once_with(fp8_dtype_target)

        assert mock_stochastic_round.call_count == 1
        # In xavier.py, stochastic_round_tensor_to_fp8 is called with positional args for the tensor and fp8_dtype
        # and keyword args for the rest.
        # However, the mock might capture them differently if not explicitly controlled.
        # Let's check kwargs first as they are more specific.
        # If the call was made as func(tensor_val, fp8_dtype_val, complex_rounding=False, ...)
        # then call_args will be ( (tensor_val, fp8_dtype_val), {complex_rounding:False, ...} )

        passed_tensor_to_round = mock_stochastic_round.call_args.kwargs.get(
            "tensor", None
        )
        if passed_tensor_to_round is None and mock_stochastic_round.call_args.args:
            passed_tensor_to_round = mock_stochastic_round.call_args.args[0]

        passed_fp8_dtype = mock_stochastic_round.call_args.kwargs.get("fp8_dtype", None)
        if passed_fp8_dtype is None and len(mock_stochastic_round.call_args.args) > 1:
            passed_fp8_dtype = mock_stochastic_round.call_args.args[1]

        torch.testing.assert_close(
            passed_tensor_to_round,
            expected_input_for_quantization_weight.to(mock_args.device),
            msg="Tensor passed to stochastic_round is not correctly pre-scaled for owlscale path.",
        )
        assert passed_fp8_dtype == fp8_dtype_target

        mock_save_file.assert_called_once()
        saved_state_dict = mock_save_file.call_args[0][0]

        assert "layer.weight" in saved_state_dict
        # The saved tensor should be the result of mock_stochastic_round (which is the pre-scaled tensor cast to fp8_dtype_target in our mock)
        # We primarily care about the dtype here, as the exact values depend on the mocked stochastic_round_tensor_to_fp8
        # torch.testing.assert_close(saved_state_dict["layer.weight"].to(expected_input_for_quantization_weight.dtype), expected_input_for_quantization_weight)
        assert saved_state_dict["layer.weight"].dtype == fp8_dtype_target

        expected_scale_key_weight = "layer.scale_weight"
        assert expected_scale_key_weight in saved_state_dict

        assert saved_state_dict[expected_scale_key_weight].dtype == OWLSCALE_SCALE_DTYPE
        torch.testing.assert_close(
            saved_state_dict[expected_scale_key_weight],
            expected_scale_to_save_weight,
            msg="Saved scale factor for owlscale is incorrect.",
        )

        assert "layer.bias" in saved_state_dict
        assert saved_state_dict["layer.bias"].dtype == dummy_tensors["layer.bias"].dtype
        assert "layer.bias_comfyui_scale" not in saved_state_dict

    @patch("os.makedirs")
    @patch("xavier.get_safetensors_tensor_keys")
    @patch("xavier.load_tensor_from_safetensors")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("xavier.generate_comparison_plots")
    @patch("argparse.ArgumentParser.parse_args")
    def test_plot_flag(
        self,
        mock_parse_args,
        mock_generate_plots,
        mock_stochastic_round,
        mock_save_file,
        mock_load_tensor,
        mock_get_keys,
        injected_mock_os_makedirs,
        mock_args,
    ):
        mock_args.plot = True
        mock_args.plot_dir = "./test_plot_dir"
        mock_args.keys_to_quantize_suffix = [".weight"]
        mock_args.plot_max_tensors = 1
        mock_parse_args.return_value = mock_args

        # Mock get_keys and load_tensor
        dummy_tensors = {
            "plot_tensor.weight": torch.randn(3, 3, dtype=torch.float32),
            "another.tensor": torch.randn(2, 2, dtype=torch.float32),
        }
        dummy_keys = list(dummy_tensors.keys())
        mock_get_keys.return_value = dummy_keys

        def load_side_effect(file_path, key, device):
            return dummy_tensors[key]

        mock_load_tensor.side_effect = load_side_effect

        # Mock stochastic_round to return something processable by the plot logic if it were real
        mock_stochastic_round.return_value = torch.randn(3, 3).to(torch.float8_e4m3fn)

        xavier.main()

        # Check that os.makedirs was called with the correct test plot directory
        injected_mock_os_makedirs.assert_called_once_with(
            mock_args.plot_dir, exist_ok=True
        )

        # Check that stochastic_round was called for the tensor we expect to plot
        assert mock_stochastic_round.call_count == 1
        # Ensure the tensor passed to stochastic_round was the one intended for plotting
        passed_round_tensor_arg = mock_stochastic_round.call_args.kwargs.get("tensor")
        assert (
            passed_round_tensor_arg is not None
        ), "Tensor argument not found in call to stochastic_round_tensor_to_fp8"
        torch.testing.assert_close(
            passed_round_tensor_arg,
            dummy_tensors["plot_tensor.weight"].to(mock_args.device),
        )

        # Check that generate_comparison_plots was called
        mock_generate_plots.assert_called_once()
        # You could add more specific assertions about the arguments passed to mock_generate_plots if needed
        # For example, check the tensor_key or parts of the plot_filename:
        call_kwargs = mock_generate_plots.call_args.kwargs
        assert call_kwargs["tensor_key"] == "plot_tensor.weight"
        assert mock_args.plot_dir in call_kwargs["plot_filename"]


# Further tests to consider:
# - Different fp8_type arguments (e5m2)
# - --keys_to_quantize_suffix variations (multiple suffixes, no match)
# - --complex_rounding, --shifturb, --owlshift flags set to True
# - --owlscale flag and its interaction with apply_owlscale_preprocess (needs mocking that too)
# - --plot flag and its interaction with plotting_utils (mock generate_comparison_plots)
# - Error handling for safetensors load/save failures (if possible to mock effectively)
# - Debug flag printing relevant messages
