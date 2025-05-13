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
    input_file = tmp_path / "input.safetensors"
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
    return str(tmp_path / "output.safetensors")


@pytest.fixture
def mock_torchao_integration(monkeypatch):
    monkeypatch.setattr("xavier.TORCHAO_AVAILABLE", True)

    mock_aqt_instances_created = []

    def mock_quantize_side_effect(module, config):
        original_tensor = module.linear.weight
        fp8_dtype = getattr(
            config, "weight_dtype", torch.float8_e4m3fn
        )  # Default if not on mock

        current_mock_aqt = mock.MagicMock(
            name=f"MockAQTInstance_{original_tensor.shape}"
        )
        current_mock_aqt.tensor_impl = mock.MagicMock()
        current_mock_aqt.tensor_impl.data = torch.full_like(
            original_tensor, 1.0, dtype=fp8_dtype
        )
        current_mock_aqt.tensor_impl.scale = torch.tensor([0.5], dtype=torch.float32)
        current_mock_aqt.dequantize = mock.MagicMock(
            return_value=torch.full_like(
                original_tensor, 0.5, dtype=original_tensor.dtype
            )
        )

        module.linear.weight = current_mock_aqt
        mock_aqt_instances_created.append(current_mock_aqt)

    mock_quantize_fn = mock.MagicMock(
        side_effect=mock_quantize_side_effect, name="quantize_fn_mock"
    )
    monkeypatch.setattr("xavier.quantize_", mock_quantize_fn)

    MockAQTClass = mock.MagicMock(name="AffineQuantizedTensorClassMock")
    monkeypatch.setattr("xavier.AffineQuantizedTensor", MockAQTClass)

    def config_init_behavior(self, *args, **kwargs):
        # self is the MagicMock instance representing the config class instance
        self._ctor_args = args
        self._ctor_kwargs = kwargs
        # Ensure essential attributes exist for the quantize_ side_effect
        self.weight_dtype = kwargs.get("weight_dtype", torch.float8_e4m3fn)
        if "activation_dtype" in kwargs:  # For dynamic config
            self.activation_dtype = kwargs.get("activation_dtype", torch.float8_e4m3fn)
        self.granularity = kwargs.get("granularity")

    MockF8WOConfig = mock.MagicMock(
        name="F8WOConfigMock", side_effect=config_init_behavior
    )
    monkeypatch.setattr("xavier.Float8WeightOnlyConfig", MockF8WOConfig)

    MockF8DAWConfig = mock.MagicMock(
        name="F8DAWConfigMock", side_effect=config_init_behavior
    )
    monkeypatch.setattr(
        "xavier.Float8DynamicActivationFloat8WeightConfig", MockF8DAWConfig
    )

    MockPerTensor = mock.MagicMock(name="PerTensorMock")
    # When PerTensor() is called, it should return an instance of itself (the mock)
    MockPerTensor_instance = mock.MagicMock(name="PerTensorInstance")
    MockPerTensor.return_value = MockPerTensor_instance
    monkeypatch.setattr("xavier.PerTensor", MockPerTensor)

    mock_dequant_fn = mock.MagicMock(
        return_value=torch.tensor([1.0]), name="dequantize_affine_floatx_mock"
    )

    # Patch for 'from torchao.quantization.quant_primitives import dequantize_affine_floatx'
    # This requires the module path to exist if create=True is not used or if it's a complex import.
    # We ensure sys.modules has the path so patch can find it.
    if "torchao" not in sys.modules:
        sys.modules["torchao"] = mock.MagicMock()
    if "torchao.quantization" not in sys.modules:
        sys.modules["torchao.quantization"] = mock.MagicMock()
    if "torchao.quantization.quant_primitives" not in sys.modules:
        sys.modules["torchao.quantization.quant_primitives"] = mock.MagicMock(
            dequantize_affine_floatx=mock_dequant_fn  # Set the attr for the import
        )
    else:  # Module exists, ensure attribute is set
        sys.modules[
            "torchao.quantization.quant_primitives"
        ].dequantize_affine_floatx = mock_dequant_fn

    # It's safer to patch where it's looked up if direct import in xavier.py allows it
    # However, xavier.py imports it inside a function: `from torchao.quantization.quant_primitives import dequantize_affine_floatx`
    # So the sys.modules approach above (or patching `torchao.quantization.quant_primitives.dequantize_affine_floatx`) is necessary.
    # The monkeypatch.setattr below is an alternative if the module was imported differently.
    # For now, relying on sys.modules manipulation for this specific import.

    return {
        "quantize_fn": mock_quantize_fn,
        "AffineQuantizedTensor_class_mock": MockAQTClass,
        "F8WOConfig_mock": MockF8WOConfig,
        "F8DAWConfig_mock": MockF8DAWConfig,
        "PerTensor_mock": MockPerTensor,  # The class mock
        "PerTensor_instance_mock": MockPerTensor_instance,  # The instance mock
        "dequantize_affine_floatx_mock": mock_dequant_fn,
        "mock_aqt_instances_created": mock_aqt_instances_created,
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
        "torchao_fp8_weight_only",
    ]
    run_xavier_script_main(args, monkeypatch)
    captured = capsys.readouterr()
    assert (
        "Error: TorchAO method 'torchao_fp8_weight_only' selected, but torchao library is not installed"
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
        "torchao_fp8_weight_only",
        "--fp8_type",
        "e4m3",
        "--keys_to_quantize_suffix",
        ".weight",
    ]
    exit_code = run_xavier_script_main(args, monkeypatch)
    assert exit_code == 0 or exit_code is None

    quantize_fn_mock = mock_torchao_integration["quantize_fn"]
    f8wo_config_mock = mock_torchao_integration["F8WOConfig_mock"]

    # 3 weights in dummy_input_path: model.layer1.weight, model.layer2.weight, model.large_weight
    assert quantize_fn_mock.call_count == 3

    assert (
        f8wo_config_mock.call_count == 3
    )  # Config instantiated per tensor quantization attempt
    config_call_kwargs = f8wo_config_mock.call_args_list[0][1]  # kwargs of first call
    assert config_call_kwargs["weight_dtype"] == torch.float8_e4m3fn

    assert os.path.exists(dummy_output_path)
    output_tensors = load_file(dummy_output_path)

    for key in ["model.layer1.weight", "model.layer2.weight", "model.large_weight"]:
        assert key in output_tensors
        assert output_tensors[key].dtype == torch.float8_e4m3fn
        assert torch.all(output_tensors[key] == 1.0)  # Mock data is all 1s

        scale_key = key.replace(".weight", ".scale_weight")
        assert scale_key in output_tensors
        assert output_tensors[scale_key].item() == 0.5  # Mock scale

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
        "torchao_fp8_dynamic_act_weight",
        "--fp8_type",
        "e5m2",
        "--keys_to_quantize_suffix",
        ".weight",
    ]
    exit_code = run_xavier_script_main(args, monkeypatch)
    assert exit_code == 0 or exit_code is None

    quantize_fn_mock = mock_torchao_integration["quantize_fn"]
    f8daw_config_mock = mock_torchao_integration["F8DAWConfig_mock"]
    per_tensor_instance_mock = mock_torchao_integration["PerTensor_instance_mock"]

    assert quantize_fn_mock.call_count == 3

    assert f8daw_config_mock.call_count == 3
    config_call_kwargs = f8daw_config_mock.call_args_list[0][1]
    assert config_call_kwargs["weight_dtype"] == torch.float8_e5m2
    assert config_call_kwargs["activation_dtype"] == torch.float8_e5m2
    assert config_call_kwargs["granularity"] == per_tensor_instance_mock

    output_tensors = load_file(dummy_output_path)
    assert "model.layer1.weight" in output_tensors  # Check one
    assert output_tensors["model.layer1.weight"].dtype == torch.float8_e5m2
    assert "scaled_fp8" in output_tensors
    assert output_tensors["scaled_fp8"].dtype == torch.float8_e5m2


def test_torchao_warns_on_native_flags(
    dummy_input_path, dummy_output_path, monkeypatch, mock_torchao_integration, capsys
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only",
        "--owlscale",  # Native flag
    ]
    run_xavier_script_main(args, monkeypatch)
    captured = capsys.readouterr()
    assert "Warning: Native quantization flags" in captured.out
    assert "are ignored when a TorchAO method is selected" in captured.out


def test_torchao_skips_1d_matching_suffix(
    dummy_input_path, dummy_output_path, monkeypatch, mock_torchao_integration, capsys
):
    args = [
        dummy_input_path,  # Contains model.layer1.bias (1D)
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only",
        "--keys_to_quantize_suffix",
        ".bias",
        ".weight",  # Try to quantize .bias
    ]
    run_xavier_script_main(args, monkeypatch)

    # quantize_fn should only be called for 3 2D .weight tensors
    assert mock_torchao_integration["quantize_fn"].call_count == 3

    output_tensors = load_file(dummy_output_path)
    assert "model.layer1.bias" in output_tensors
    assert output_tensors["model.layer1.bias"].dtype == torch.float32  # Original

    captured = capsys.readouterr()
    assert "Warning: Tensor model.layer1.bias has dim < 2" in captured.out
    assert "skipping TorchAO" in captured.out


@mock.patch("xavier.MATPLOTLIB_AVAILABLE", True)
@mock.patch(
    "xavier.generate_comparison_plots"
)  # Mock the actual plotting function in xavier's namespace
def test_torchao_plotting_uses_dequantize_affine_floatx(
    mock_generate_plots_in_xavier,  # Name reflects where it's mocked
    dummy_input_path,
    dummy_output_path,
    monkeypatch,
    mock_torchao_integration,
):
    args = [
        dummy_input_path,
        dummy_output_path,
        "--quant_method",
        "torchao_fp8_weight_only",
        "--fp8_type",
        "e4m3",
        "--plot",
        "--plot_max_tensors",
        "3",  # Ensure all 3 weights are plotted
    ]
    run_xavier_script_main(args, monkeypatch)

    dequant_mock = mock_torchao_integration["dequantize_affine_floatx_mock"]
    # Should be called for each of the 3 quantized weights plotted
    assert dequant_mock.call_count == 3

    assert mock_generate_plots_in_xavier.call_count == 3

    # Check args of the first dequantize_affine_floatx call
    first_call_args, first_call_kwargs = dequant_mock.call_args_list[0]
    quantized_tensor_arg = first_call_args[0]
    scale_arg = first_call_args[1]
    ebits_arg = first_call_args[2]
    mbits_arg = first_call_args[3]

    assert isinstance(quantized_tensor_arg, torch.Tensor)
    # Mocked AQT data is torch.float8_e4m3fn (based on --fp8_type e4m3 for this test)
    # The mock_quantize_side_effect uses config.weight_dtype.
    # The F8WOConfig_mock's side_effect (config_init_behavior) gets weight_dtype from kwargs.
    # The test test_torchao_fp8_weight_only_quantizes_weights correctly sets this.
    # This test needs to ensure that weight_dtype used by mock is e4m3
    assert quantized_tensor_arg.dtype == torch.float8_e4m3fn
    assert isinstance(scale_arg, torch.Tensor)
    assert scale_arg.item() == 0.5  # From mock data scale
    assert ebits_arg == 4  # for e4m3
    assert mbits_arg == 3  # for e4m3
    assert (
        first_call_kwargs["output_dtype"] == torch.float32
    )  # Original dtype assumed for plotting
