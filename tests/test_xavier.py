import torch
import pytest
import argparse
import os
from unittest.mock import patch, MagicMock, mock_open

# Assuming xavier.py is in the root or PYTHONPATH is set up
import xavier


# Helper to create dummy safetensors file content
def create_dummy_safetensors_content(data_dict):
    from safetensors.torch import save
    import io

    bio = io.BytesIO()
    save(data_dict, bio)
    return bio.getvalue()


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
            "other_tensor": torch.randn(5, dtype=torch.int32),  # Non-float tensor
        }
        mock_load_file.return_value = dummy_state_dict

        # Mock stochastic_round_tensor_to_fp8 to return a tensor of the target dtype
        def mock_round_func(tensor, fp8_dtype, **kwargs):
            return tensor.to(fp8_dtype)  # Simplified mock

        mock_stochastic_round.side_effect = mock_round_func

        xavier.main()

        # Assertions
        mock_load_file.assert_called_once_with(mock_args.input_file, device="cpu")

        # Check that stochastic_round was called for the .weight tensor
        # It should be called once because only "layer1.weight" matches the suffix.
        assert mock_stochastic_round.call_count == 1
        call_args_list = mock_stochastic_round.call_args_list
        (
            (called_tensor_arg, called_fp8_dtype_arg),  # Positional args
            called_kwargs_arg,  # Keyword args
        ) = call_args_list[0]

        torch.testing.assert_close(
            called_tensor_arg, dummy_state_dict["layer1.weight"].to(mock_args.device)
        )
        assert called_fp8_dtype_arg == torch.float8_e4m3fn
        assert called_kwargs_arg["use_complex_method"] == mock_args.complex_rounding
        assert called_kwargs_arg["use_shift_perturb_method"] == mock_args.shifturb
        assert called_kwargs_arg["use_owlshift_method"] == mock_args.owlshift
        assert called_kwargs_arg["seed"] == mock_args.seed
        assert called_kwargs_arg["debug_mode"] == mock_args.debug

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
        assert (
            f"Successfully quantized and saved model to: {mock_args.output_file}"
            in captured.out
        )

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
        mock_stochastic_round.return_value = torch.randn(2, 2, dtype=torch.float8_e5m2)

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
        mock_stochastic_round.return_value = torch.randn(
            1, 1, dtype=torch.float8_e4m3fn
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
        assert mock_stochastic_round.call_args[0][0].equal(
            dummy_state_dict["layer1.bias"].to(mock_args.device)
        )

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
        mock_stochastic_round.return_value = torch.randn(
            2, 2, dtype=torch.float8_e4m3fn
        )

        xavier.main()

        assert mock_stochastic_round.call_count == 1
        kwargs_passed = mock_stochastic_round.call_args[1]
        if flag_to_test == "complex_rounding":
            assert kwargs_passed["use_complex_method"] == True
            assert kwargs_passed["use_shift_perturb_method"] == False
            assert kwargs_passed["use_owlshift_method"] == False
        elif flag_to_test == "shifturb":
            assert kwargs_passed["use_complex_method"] == False
            assert kwargs_passed["use_shift_perturb_method"] == True
            assert kwargs_passed["use_owlshift_method"] == False
        elif flag_to_test == "owlshift":
            assert kwargs_passed["use_complex_method"] == False
            assert kwargs_passed["use_shift_perturb_method"] == False
            assert kwargs_passed["use_owlshift_method"] == True

    @patch("xavier.load_file")
    @patch("xavier.save_file")
    @patch("xavier.stochastic_round_tensor_to_fp8")
    @patch("xavier.apply_owlscale_preprocess")  # Mock owlscale preprocess
    @patch("xavier.get_fp8_constants_for_owlscale")  # Mock constants getter
    @patch("argparse.ArgumentParser.parse_args")
    def test_owlscale_flag(
        self,
        mock_parse_args,
        mock_get_constants,
        mock_apply_owlscale,
        mock_stochastic_round,
        mock_save_file,
        mock_load_file,
        mock_args,
        capsys,
    ):
        mock_args.owlscale = True
        mock_args.fp8_type = "e4m3"  # for get_fp8_constants
        mock_parse_args.return_value = mock_args

        # Mock get_fp8_constants_for_owlscale to return some dummy values
        mock_get_constants.return_value = (-448.0, 448.0, 2**-9)

        dummy_weight_tensor = torch.randn(2, 2, dtype=torch.float32)
        dummy_bias_tensor = torch.randn(2, dtype=torch.float32)
        mock_load_file.return_value = {
            "layer.weight": dummy_weight_tensor,
            "layer.bias": dummy_bias_tensor,
        }

        # Mock apply_owlscale_preprocess behavior
        # It returns: processed_tensor, dequant_scale_factor
        scaled_weight_tensor = dummy_weight_tensor / 2.0  # dummy scaled tensor
        dequant_scale = torch.tensor(2.0, dtype=torch.float64)  # dummy dequant scale
        mock_apply_owlscale.return_value = (
            scaled_weight_tensor.to(
                xavier.OWLSCALE_COMPUTE_DTYPE
                if xavier.OWLSCALE_COMPUTE_DTYPE
                else torch.float64
            ),
            dequant_scale,
        )

        # Mock stochastic_round_tensor_to_fp8 to return a tensor of the target dtype
        def mock_round_func(tensor, fp8_dtype, **kwargs):
            return tensor.to(fp8_dtype)

        mock_stochastic_round.side_effect = mock_round_func

        xavier.main()

        # Assert get_fp8_constants was called
        mock_get_constants.assert_called_once_with(torch.float8_e4m3fn)

        # Assert apply_owlscale_preprocess was called for the weight tensor
        assert mock_apply_owlscale.call_count == 1
        apply_owlscale_call_args, apply_owlscale_call_kwargs = (
            mock_apply_owlscale.call_args_list[0]
        )
        torch.testing.assert_close(
            apply_owlscale_call_args[0], dummy_weight_tensor.to(mock_args.device)
        )
        assert apply_owlscale_call_kwargs["target_fp8_dtype"] == torch.float8_e4m3fn
        assert apply_owlscale_call_kwargs["fp8_min_val"] == -448.0
        assert apply_owlscale_call_kwargs["fp8_max_val"] == 448.0
        assert apply_owlscale_call_kwargs["fp8_min_pos_val"] == 2**-9
        assert apply_owlscale_call_kwargs["compute_dtype"] == (
            xavier.OWLSCALE_COMPUTE_DTYPE
            if xavier.OWLSCALE_COMPUTE_DTYPE
            else torch.float64
        )

        # Assert stochastic_round was called for BOTH tensors
        # 1. For the scaled weight tensor
        # 2. For the bias tensor (which skips owlscale preprocess)
        assert mock_stochastic_round.call_count == 2

        # Call 1: weight tensor (should be the output of apply_owlscale)
        args_weight_call, _ = mock_stochastic_round.call_args_list[0]
        torch.testing.assert_close(
            args_weight_call[0],
            scaled_weight_tensor.to(
                xavier.OWLSCALE_COMPUTE_DTYPE
                if xavier.OWLSCALE_COMPUTE_DTYPE
                else torch.float64
            ).to(mock_args.device),
        )
        assert args_weight_call[1] == torch.float8_e4m3fn

        # Call 2: bias tensor (should be the original bias tensor)
        args_bias_call, _ = mock_stochastic_round.call_args_list[1]
        torch.testing.assert_close(
            args_bias_call[0], dummy_bias_tensor.to(mock_args.device)
        )
        assert args_bias_call[1] == torch.float8_e4m3fn

        # Check that save_file was called with the dequant_scale for the weight tensor
        mock_save_file.assert_called_once()
        saved_state_dict = mock_save_file.call_args[0][0]
        assert "layer.weight" in saved_state_dict
        assert saved_state_dict["layer.weight"].dtype == torch.float8_e4m3fn
        assert "layer.weight_owlscale_dequant" in saved_state_dict
        # The scale saved should be on the original tensor's device, and in its original dtype (or compute_dtype if it was changed globally)
        # In the actual code, scale is cast to original_tensor_dtype_for_output_scale before saving.
        # Let's assume OWLSCALE_SCALE_DTYPE for the test consistency for now.
        expected_scale_dtype = (
            xavier.OWLSCALE_SCALE_DTYPE
            if xavier.OWLSCALE_SCALE_DTYPE
            else torch.float64
        )
        assert (
            saved_state_dict["layer.weight_owlscale_dequant"].dtype
            == expected_scale_dtype
        )
        torch.testing.assert_close(
            saved_state_dict["layer.weight_owlscale_dequant"].to(dequant_scale.dtype),
            dequant_scale,
        )
        assert "layer.bias" in saved_state_dict  # Bias should also be there
        assert saved_state_dict["layer.bias"].dtype == torch.float8_e4m3fn
        assert (
            "layer.bias_owlscale_dequant" not in saved_state_dict
        )  # No scale for bias


# Further tests to consider:
# - Different fp8_type arguments (e5m2)
# - --keys_to_quantize_suffix variations (multiple suffixes, no match)
# - --complex_rounding, --shifturb, --owlshift flags set to True
# - --owlscale flag and its interaction with apply_owlscale_preprocess (needs mocking that too)
# - --plot flag and its interaction with plotting_utils (mock generate_comparison_plots)
# - Error handling for safetensors load/save failures (if possible to mock effectively)
# - Debug flag printing relevant messages
