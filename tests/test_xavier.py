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

    @patch("xavier.load_file")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_basic_quantization_run(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_file,
        mock_args,
        capsys,
    ):
        """Test a basic run through the quantization process, mocking external calls."""
        mock_parse_args.return_value = mock_args

        # Mock load_file to return a simple state dict
        dummy_state_dict = {
            "layer1.weight": torch.randn(10, 5, dtype=torch.float32),
            "layer1.bias": torch.randn(10, dtype=torch.float32),
            "other_tensor": torch.randint(
                0, 10, (5,), dtype=torch.int32
            ),  # Non-float tensor
        }
        mock_load_file.return_value = dummy_state_dict

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
        mock_load_file.assert_called_once_with(mock_args.input_file, device="cpu")

        # Check that stochastic_round was called for the .weight tensor
        # It should be called once because only "layer1.weight" matches the suffix.
        assert mock_stochastic_round.call_count == 1
        call_args_list = mock_stochastic_round.call_args_list
        # call_args_list[0] is a tuple: (pos_args_tuple, kw_args_dict)
        pos_args_tuple = call_args_list[0][0]
        kw_args_dict = call_args_list[0][1]

        called_tensor_arg = pos_args_tuple[0]
        called_fp8_dtype_arg = pos_args_tuple[1]
        called_complex_method_arg = pos_args_tuple[2]
        called_shift_perturb_arg = pos_args_tuple[3]
        called_owlshift_arg = pos_args_tuple[4]
        # Seed and debug_mode are in kw_args_dict

        torch.testing.assert_close(
            called_tensor_arg, dummy_state_dict["layer1.weight"].to(mock_args.device)
        )
        assert called_fp8_dtype_arg == torch.float8_e4m3fn
        assert called_complex_method_arg == mock_args.complex_rounding
        assert called_shift_perturb_arg == mock_args.shifturb
        assert called_owlshift_arg == mock_args.owlshift
        assert kw_args_dict["seed"] == mock_args.seed
        assert kw_args_dict["debug_mode"] == mock_args.debug

        # Check that save_file was called with the quantized state dict
        mock_save_file.assert_called_once()
        saved_state_dict_call = mock_save_file.call_args[0][0]
        assert "layer1.weight" in saved_state_dict_call
        assert saved_state_dict_call["layer1.weight"].dtype == torch.float8_e4m3fn
        assert "layer1.bias" in saved_state_dict_call  # Bias should be passed through
        torch.testing.assert_close(
            saved_state_dict_call["layer1.bias"],
            dummy_state_dict["layer1.bias"].to(mock_args.device),
        )
        assert (
            "other_tensor" in saved_state_dict_call
        )  # Non-float tensor passed through
        torch.testing.assert_close(
            saved_state_dict_call["other_tensor"],
            dummy_state_dict["other_tensor"].to(mock_args.device),
        )

        captured = capsys.readouterr()
        # Check for the specific line that confirms quantization and output file
        expected_quant_line_part = f"Saving quantized model to: {mock_args.output_file}"
        assert (
            expected_quant_line_part in captured.out
        ), f"Expected part '{expected_quant_line_part}' not found in stdout: {captured.out}"

    @patch("xavier.load_file")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_fp8_type_e5m2(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_file,
        mock_args,
    ):
        mock_args.fp8_type = "e5m2"
        mock_parse_args.return_value = mock_args
        mock_load_file.return_value = {"t.weight": torch.randn(2, 2)}
        mock_stochastic_round.return_value = torch.randn(2, 2).to(torch.float8_e5m2)

        xavier.main()
        # Check that the correct fp8_dtype was passed to stochastic_round
        assert mock_stochastic_round.call_args[0][1] == torch.float8_e5m2
        # Check that the saved tensor is e5m2
        saved_state_dict = mock_save_file.call_args[0][0]
        assert saved_state_dict["t.weight"].dtype == torch.float8_e5m2

    @patch("xavier.load_file")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_keys_to_quantize_suffix_variations(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_file,
        mock_args,
    ):
        dummy_state_dict = {
            "layer1.weight": torch.randn(2, 2),
            "layer1.bias": torch.randn(2),
            "layer2.data": torch.randn(3, 3),
            "another.weight": torch.randn(4, 4),
        }
        mock_load_file.return_value = dummy_state_dict
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
        assert mock_stochastic_round.call_count == 0

    @pytest.mark.parametrize(
        "flag_to_test", ["complex_rounding", "shifturb", "owlshift"]
    )
    @patch("xavier.load_file")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("argparse.ArgumentParser.parse_args")
    def test_boolean_flags(
        self,
        mock_parse_args,
        mock_stochastic_round,
        mock_save_file,
        mock_load_file,
        mock_args,
        flag_to_test,
    ):
        # Set the specific flag to True, others to False
        mock_args.complex_rounding = flag_to_test == "complex_rounding"
        mock_args.shifturb = flag_to_test == "shifturb"
        mock_args.owlshift = flag_to_test == "owlshift"
        mock_parse_args.return_value = mock_args

        mock_load_file.return_value = {"t.weight": torch.randn(2, 2)}
        mock_stochastic_round.return_value = torch.randn(2, 2).to(torch.float8_e4m3fn)

        xavier.main()

        assert mock_stochastic_round.call_count == 1
        # call_args is (pos_args_tuple, kw_args_dict)
        pos_args_passed = mock_stochastic_round.call_args[0]
        # kwargs_passed = mock_stochastic_round.call_args[1]

        # Positional args to stochastic_round_tensor_to_fp8 are:
        # (tensor, fp8_dtype, use_complex_method, use_shift_perturb_method, use_owlshift_method)
        called_complex_flag = pos_args_passed[2]
        called_shifturb_flag = pos_args_passed[3]
        called_owlshift_flag = pos_args_passed[4]

        if flag_to_test == "complex_rounding":
            assert called_complex_flag == True
            assert called_shifturb_flag == False
            assert called_owlshift_flag == False
        elif flag_to_test == "shifturb":
            assert called_complex_flag == False
            assert called_shifturb_flag == True
            assert called_owlshift_flag == False
        elif flag_to_test == "owlshift":
            assert called_complex_flag == False
            assert called_shifturb_flag == False
            assert called_owlshift_flag == True

    @patch("xavier.load_file")
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
        mock_load_file,
        mock_args,
        capsys,
    ):
        mock_args.owlscale = True
        mock_args.fp8_type = "e4m3"
        mock_args.device = "cpu"
        mock_parse_args.return_value = mock_args

        fp8_dtype_target = torch.float8_e4m3fn

        mock_get_constants.return_value = (-448.0, 448.0, 2**-9)

        dummy_weight_tensor = torch.randn(2, 2, dtype=torch.float32) * 10
        dummy_bias_tensor = torch.randn(2, dtype=torch.float32)
        mock_load_file.return_value = {
            "layer.weight": dummy_weight_tensor.clone(),
            "layer.bias": dummy_bias_tensor.clone(),
        }

        weight_hp = dummy_weight_tensor.to(torch.float64)
        abs_max_weight = torch.max(torch.abs(weight_hp))
        scale_val_weight = abs_max_weight.clamp(min=1e-12)
        expected_scaled_weight_for_stochastic_round = (weight_hp / scale_val_weight).to(
            dummy_weight_tensor.dtype
        )

        def mock_round_func(
            tensor,
            fp8_dtype,
            use_complex_method,
            use_shift_perturb_method,
            use_owlshift_method,
            seed,
            debug_mode,
        ):
            return tensor.to(fp8_dtype)

        mock_stochastic_round.side_effect = mock_round_func

        xavier.main()

        mock_get_constants.assert_called_once_with(fp8_dtype_target)

        assert mock_stochastic_round.call_count == 1
        called_args = mock_stochastic_round.call_args[0]
        # called_kwargs = mock_stochastic_round.call_args[1] # Not used, can be removed for cleanliness

        # Assertions for the arguments passed to the single call of stochastic_round_tensor_to_fp8
        torch.testing.assert_close(
            called_args[0],  # The tensor passed for rounding
            expected_scaled_weight_for_stochastic_round.to(mock_args.device),
        )
        assert called_args[1] == fp8_dtype_target  # The fp8_dtype passed

        mock_save_file.assert_called_once()
        saved_state_dict = mock_save_file.call_args[0][0]

        assert "layer.weight" in saved_state_dict
        assert saved_state_dict["layer.weight"].dtype == fp8_dtype_target

        expected_scale_key_weight = "layer.scale_weight"
        assert expected_scale_key_weight in saved_state_dict

        expected_scale_dtype = torch.float64
        assert saved_state_dict[expected_scale_key_weight].dtype == expected_scale_dtype
        torch.testing.assert_close(
            saved_state_dict[expected_scale_key_weight].to(scale_val_weight.dtype),
            scale_val_weight,
        )

        assert "layer.bias" in saved_state_dict
        assert saved_state_dict["layer.bias"].dtype == dummy_bias_tensor.dtype
        assert "layer.bias_comfyui_scale" not in saved_state_dict


# Further tests to consider:
# - Different fp8_type arguments (e5m2)
# - --keys_to_quantize_suffix variations (multiple suffixes, no match)
# - --complex_rounding, --shifturb, --owlshift flags set to True
# - --owlscale flag and its interaction with apply_owlscale_preprocess (needs mocking that too)
# - --plot flag and its interaction with plotting_utils (mock generate_comparison_plots)
# - Error handling for safetensors load/save failures (if possible to mock effectively)
# - Debug flag printing relevant messages
