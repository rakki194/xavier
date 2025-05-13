import pytest
import torch
from safetensors.torch import save_file, load_file
import os
import sys
from unittest import mock

# Adjust sys.path to import xavier module from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import xavier and its components after sys.path adjustment
import xavier  # Main module to patch members like TORCHAO_AVAILABLE
from xavier import main as xavier_main  # The main function to call
from xavier import TORCHAO_AVAILABLE  # Import the flag

# Skip all tests in this file if torchao is not available
if not TORCHAO_AVAILABLE:
    pytest.skip(
        "TorchAO not available, skipping integration tests", allow_module_level=True
    )

# Now safe to import torchao components as we've skipped if not available
import torchao
from torchao.quantization import (  # Import actual types for spec
    Float8WeightOnlyConfig as ActualFloat8WeightOnlyConfig,  # Renamed for clarity
    Float8DynamicActivationFloat8WeightConfig as ActualFloat8DynamicActivationFloat8WeightConfig,  # Renamed
    PerTensor as ActualPerTensor,  # Renamed
)
from torchao.dtypes import (
    AffineQuantizedTensor as ActualAffineQuantizedTensor,
)  # Renamed
from torchao.float8.float8_linear import Float8Linear as ActualFloat8Linear  # Renamed


# Helper to run xavier.py main function
def run_xavier_script_main(argv_list, monkeypatch):
    full_argv = ["xavier.py"] + argv_list  # Script name is sys.argv[0]
    monkeypatch.setattr(sys, "argv", full_argv)
    try:
        xavier_main()
        return 0  # Indicates success or normal completion
    except SystemExit as e:
        return e.code  # Return exit code (e.g., from argparse errors)
    except Exception:  # Catch other exceptions for test stability
        pytest.fail(f"xavier_main raised an unhandled exception with args: {argv_list}")
        return 1  # Should not happen if SystemExit is always used on error


@pytest.fixture
def dummy_input_path(tmp_path):
    input_file = os.path.abspath(tmp_path / "input_torchao.safetensors")
    tensors = {
        "model.layer1.weight": torch.randn((64, 32), dtype=torch.float32),
        "model.layer1.bias": torch.randn((64,), dtype=torch.float32),
        "model.layer2.weight": torch.randn((32, 16), dtype=torch.float32),
        "model.another.tensor": torch.randn(
            (10,), dtype=torch.float32
        ),  # 1D, not weight/bias
        "model.large_weight": torch.randn((128, 256), dtype=torch.float32),
    }
    save_file(tensors, str(input_file))
    return str(input_file)


@pytest.fixture
def dummy_output_path(tmp_path):
    return os.path.abspath(str(tmp_path / "output_torchao.safetensors"))


@pytest.fixture
def dummy_plot_dir_torchao(tmp_path):
    return os.path.abspath(str(tmp_path / "plots_torchao_test"))


@pytest.fixture
def mock_torchao_integration(monkeypatch):
    """Mocks torchao components to simulate their behavior without actual quantization."""
    if (
        not TORCHAO_AVAILABLE
    ):  # Should not be reached due to module skip, but good practice
        return None

    mock_quantize_ = mock.Mock(name="quantize_")

    # Create mock *classes* using create_autospec. instance=False means it mocks the class itself.
    MockAQT_class = mock.create_autospec(
        ActualAffineQuantizedTensor, instance=False, name="MockAQT_class"
    )
    MockF8L_class = mock.create_autospec(
        ActualFloat8Linear, instance=False, name="MockF8L_class"
    )
    MockF8WOConfig_class = mock.create_autospec(
        ActualFloat8WeightOnlyConfig, instance=False, name="MockF8WOConfig_class"
    )
    MockF8DAWConfig_class = mock.create_autospec(
        ActualFloat8DynamicActivationFloat8WeightConfig,
        instance=False,
        name="MockF8DAWConfig_class",
    )
    MockPerTensor_class = mock.create_autospec(
        ActualPerTensor, instance=False, name="MockPerTensor_class"
    )

    # Side effect for F8WOConfig_class constructor
    def f8wo_constructor_side_effect(*args, **kwargs):
        # Create a mock *instance* that also conforms to the spec of the actual class instance
        instance = mock.create_autospec(ActualFloat8WeightOnlyConfig, instance=True)
        for k, v in kwargs.items():
            setattr(instance, k, v)
        # Ensure essential attributes like weight_dtype are set as xavier.py expects
        instance.weight_dtype = kwargs.get("weight_dtype")
        return instance

    MockF8WOConfig_class.side_effect = f8wo_constructor_side_effect

    # Instance for PerTensor mock
    mock_per_tensor_instance = (
        MockPerTensor_class()
    )  # Create an instance from the mocked PerTensor class

    # Side effect for F8DAWConfig_class constructor
    def f8daw_constructor_side_effect(*args, **kwargs):
        instance = mock.create_autospec(
            ActualFloat8DynamicActivationFloat8WeightConfig, instance=True
        )
        for k, v in kwargs.items():
            setattr(instance, k, v)
        instance.activation_dtype = kwargs.get("activation_dtype")
        instance.weight_dtype = kwargs.get("weight_dtype")
        # Default to the PerTensor mock instance if granularity isn't specified
        instance.granularity = kwargs.get("granularity", mock_per_tensor_instance)
        return instance

    MockF8DAWConfig_class.side_effect = f8daw_constructor_side_effect

    # Mocking the quantize_ function to simulate behavior:
    def quantize_side_effect(model, config):
        # config will be an instance of a mocked config class
        if isinstance(
            config, ActualFloat8WeightOnlyConfig
        ):  # Check against actual class for safety if mock is transparent
            # Simulate weight being replaced by an AffineQuantizedTensor-like mock
            mock_aqt_instance = (
                MockAQT_class()
            )  # Create instance from the mocked AQT class

            # Setup .tensor_impl on the instance (AQT instances have .tensor_impl)
            mock_aqt_instance.tensor_impl = mock.MagicMock(name="MockAQTImplInstance")
            mock_aqt_instance.tensor_impl.data = model.linear.weight.to(config.weight_dtype)  # type: ignore
            mock_aqt_instance.tensor_impl.scale = torch.tensor([0.123])
            model.linear.weight = mock_aqt_instance

        elif isinstance(config, ActualFloat8DynamicActivationFloat8WeightConfig):
            # Simulate the layer being replaced by a Float8Linear mock instance
            new_linear_layer_mock = (
                MockF8L_class()
            )  # Create instance from the mocked F8L class

            # Populate required attributes for Float8Linear that xavier.py might access
            new_linear_layer_mock.weight = model.linear.weight.to(config.weight_dtype)  # type: ignore
            new_linear_layer_mock.weight_scale = torch.tensor([0.456])
            # If xavier.py checks for other attributes like bias, set them up here too.
            # new_linear_layer_mock.bias = model.linear.bias # if bias is handled
            model.linear = new_linear_layer_mock  # Replace the layer in the dummy model
        return model

    mock_quantize_.side_effect = quantize_side_effect

    # Patch xavier.py's view of these classes with our mock *classes*
    monkeypatch.setattr("xavier.quantize_", mock_quantize_)
    monkeypatch.setattr("xavier.Float8WeightOnlyConfig", MockF8WOConfig_class)
    monkeypatch.setattr(
        "xavier.Float8DynamicActivationFloat8WeightConfig", MockF8DAWConfig_class
    )
    monkeypatch.setattr("xavier.AffineQuantizedTensor", MockAQT_class)
    monkeypatch.setattr("xavier.Float8Linear", MockF8L_class)
    monkeypatch.setattr("xavier.PerTensor", MockPerTensor_class)

    return {
        "quantize_": mock_quantize_,
        "F8WOConfigMock_class": MockF8WOConfig_class,  # Returning the mock class
        "F8DAWConfigMock_class": MockF8DAWConfig_class,  # Returning the mock class
        "AffineQuantizedTensor_Mock_class": MockAQT_class,  # Returning the mock class
        "Float8Linear_Mock_class": MockF8L_class,  # Returning the mock class
        "PerTensor_InstanceMock": mock_per_tensor_instance,  # Instance is fine for granularity
    }


# Test Cases
@mock.patch("xavier.TORCHAO_AVAILABLE", False)
def test_torchao_unavailable_error_message(
    dummy_input_path, dummy_output_path, monkeypatch, capsys
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only_aoscale",
    ]
    run_xavier_script_main(args, monkeypatch)
    captured = capsys.readouterr()
    assert (
        "Error: TorchAO method 'torchao_fp8_weight_only_aoscale' selected, but torchao library is not installed"
        in captured.out
    )
    assert not os.path.exists(dummy_output_path)


def test_torchao_fp8_weight_only_quantizes_weights(
    dummy_input_path, dummy_output_path, monkeypatch, mock_torchao_integration, capsys
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only_aoscale",
        "--fp8_type",
        "e4m3",
        "--keys_to_quantize_suffix",
        ".weight",
    ]
    exit_code = run_xavier_script_main(args, monkeypatch)
    assert exit_code == 0 or exit_code is None

    quantize_fn_mock = mock_torchao_integration["quantize_"]
    # We check the mock class for calls now, not an instance.
    f8wo_config_mock_class = mock_torchao_integration["F8WOConfigMock_class"]

    # 3 weights in dummy_input_path: model.layer1.weight, model.layer2.weight, model.large_weight
    assert quantize_fn_mock.call_count == 3

    assert (
        f8wo_config_mock_class.call_count == 3
    )  # Config class constructor called per tensor quantization attempt
    # .call_args_list[0][1] are kwargs of the first call to the constructor
    config_call_kwargs = f8wo_config_mock_class.call_args_list[0][1]
    assert config_call_kwargs["weight_dtype"] == torch.float8_e4m3fn

    assert os.path.exists(dummy_output_path)
    output_tensors = load_file(dummy_output_path)

    for key in ["model.layer1.weight", "model.layer2.weight", "model.large_weight"]:
        assert key in output_tensors
        assert output_tensors[key].dtype == torch.float8_e4m3fn
        # The mock AQT's data is the original tensor converted to FP8, scale is 0.123
        # The data saved is tensor_impl.data
        # For this test, we don't check specific values of the data due to mock complexity, just dtype and scale

        scale_key = key.replace(".weight", ".scale_weight")
        assert scale_key in output_tensors
        assert output_tensors[scale_key].item() == 0.123  # Mock scale

    assert "model.layer1.bias" in output_tensors  # Copied
    assert output_tensors["model.layer1.bias"].dtype == torch.float32
    assert "model.another.tensor" in output_tensors  # Copied (1D)
    assert output_tensors["model.another.tensor"].dtype == torch.float32

    assert "scaled_fp8" in output_tensors
    assert output_tensors["scaled_fp8"].dtype == torch.float8_e4m3fn


def test_torchao_fp8_dynamic_act_weight_quantizes_weights(
    dummy_input_path, dummy_output_path, monkeypatch, mock_torchao_integration
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_dynamic_act_weight_aoscale",
        "--fp8_type",
        "e5m2",
        "--keys_to_quantize_suffix",
        ".weight",
    ]
    exit_code = run_xavier_script_main(args, monkeypatch)
    assert exit_code == 0 or exit_code is None

    quantize_fn_mock = mock_torchao_integration["quantize_"]
    f8daw_config_mock_class = mock_torchao_integration["F8DAWConfigMock_class"]
    per_tensor_instance_mock = mock_torchao_integration["PerTensor_InstanceMock"]

    assert quantize_fn_mock.call_count == 3

    assert f8daw_config_mock_class.call_count == 3
    config_call_kwargs = f8daw_config_mock_class.call_args_list[0][1]
    assert config_call_kwargs["weight_dtype"] == torch.float8_e5m2
    assert config_call_kwargs["activation_dtype"] == torch.float8_e5m2
    assert config_call_kwargs["granularity"] == per_tensor_instance_mock

    output_tensors = load_file(dummy_output_path)
    # After quantize_ with F8DAWConfig, model.linear is replaced by a mock Float8Linear.
    # xavier.py extracts .weight and .weight_scale from this mock.
    for key in ["model.layer1.weight", "model.layer2.weight", "model.large_weight"]:
        assert key in output_tensors
        assert output_tensors[key].dtype == torch.float8_e5m2
        scale_key = key.replace(".weight", ".scale_weight")
        assert scale_key in output_tensors
        assert (
            output_tensors[scale_key].item() == 0.456
        )  # Mock scale from F8Linear mock

    assert "scaled_fp8" in output_tensors
    assert output_tensors["scaled_fp8"].dtype == torch.float8_e5m2


def test_torchao_warns_on_native_flags(
    dummy_input_path, dummy_output_path, monkeypatch, mock_torchao_integration, capsys
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only_aoscale",
        "--comfyscale",  # Native flag
    ]
    run_xavier_script_main(args, monkeypatch)
    captured = capsys.readouterr()
    assert "Warning: Native quantization flags" in captured.out
    assert "are ignored when a TorchAO method is selected" in captured.out


def test_torchao_skips_1d_matching_suffix(
    dummy_input_path, dummy_output_path, monkeypatch, mock_torchao_integration, capsys
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only_aoscale",
        "--keys_to_quantize_suffix",
        ".bias",  # Matches model.layer1.bias (1D)
        # ".weight", # Ensure weights are not targeted to isolate .bias behavior
    ]
    run_xavier_script_main(args, monkeypatch)

    # quantize_fn should NOT be called for the 1D .bias tensor
    # It would be called for .weight if that suffix was also present and matched.
    # Since only .bias is a suffix and it's 1D, quantize_ should not be called.
    assert mock_torchao_integration["quantize_"].call_count == 0

    output_tensors = load_file(dummy_output_path)
    assert "model.layer1.bias" in output_tensors
    assert (
        output_tensors["model.layer1.bias"].dtype == torch.float32
    )  # Original, copied

    captured = capsys.readouterr()
    assert "Warning: Tensor model.layer1.bias has dim < 2" in captured.out
    assert "skipping TorchAO" in captured.out


@mock.patch("xavier.MATPLOTLIB_AVAILABLE", True)
@mock.patch(
    "xavier.generate_comparison_plots"  # Mock the actual plotting function in xavier's namespace
)
def test_torchao_plotting_uses_dequantize_affine_floatx(
    mock_generate_plots_in_xavier,  # Name reflects where it's mocked
    dummy_input_path,
    dummy_output_path,
    dummy_plot_dir_torchao,  # Added plot dir fixture
    monkeypatch,
    mock_torchao_integration,
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only_aoscale",  # This will use the mock AQT path
        "--fp8_type",
        "e4m3",
        "--plot",
        "--plot_dir",
        dummy_plot_dir_torchao,  # Use the temp plot dir
        "--plot_max_tensors",
        "3",  # Ensure all 3 weights are plotted
        "--keys_to_quantize_suffix",
        ".weight",  # Target weights for quantization
    ]
    run_xavier_script_main(args, monkeypatch)

    # Check that generate_comparison_plots was called for the quantized weights
    # There are 3 tensors ending in .weight that are 2D+ and will be quantized by the mock
    assert mock_generate_plots_in_xavier.call_count == 3

    # Example check for one of the calls (e.g., model.layer1.weight)
    found_plot_call_for_layer1 = False
    for call_item in mock_generate_plots_in_xavier.call_args_list:
        kwargs = call_item.kwargs
        if kwargs.get("tensor_key") == "model.layer1.weight":
            found_plot_call_for_layer1 = True
            # In the mock, data is original.to(fp8), scale is 0.123
            # Dequant logic in xavier: data.to(float32) * scale.to(float32)
            original_tensor = load_file(dummy_input_path)["model.layer1.weight"].cpu()
            quantized_fp8_for_plot = original_tensor.to(torch.float8_e4m3fn)
            expected_dequant_val_for_plot = quantized_fp8_for_plot.to(
                torch.float32
            ) * torch.tensor([0.123], dtype=torch.float32)

            # Ensure dequantized tensor matches calculation based on mock scale and data
            torch.testing.assert_close(
                kwargs["dequantized_tensor_cpu"],
                expected_dequant_val_for_plot.to(original_tensor.dtype),
            )
            self.assertEqual(
                kwargs["quantized_fp8_tensor_cpu"].dtype, torch.float8_e4m3fn
            )
            break
    self.assertTrue(
        found_plot_call_for_layer1, "Plotting not called for model.layer1.weight"
    )
