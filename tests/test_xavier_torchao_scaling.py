import unittest
import torch
import os
import subprocess
from safetensors.torch import save_file, load_file
import sys
import tempfile

# Adjust sys.path to import xavier module from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xavier  # For TORCHAO_AVAILABLE and type hints if needed
from xavier import (
    TORCHAO_AVAILABLE,
    AffineQuantizedTensor,
    Float8Linear,
)  # For isinstance checks

if not TORCHAO_AVAILABLE:
    # If torchao is not available, we define dummy versions of these classes
    # so that the tests can be parsed, but they will be skipped.
    class AffineQuantizedTensor:
        pass

    class Float8Linear:
        pass


# Test parameters
FP8_TYPES = ["e4m3", "e5m2"]
QUANT_METHODS_WO = [
    "torchao_fp8_weight_only_aoscale",
    "torchao_fp8_weight_only_comfyscale",
]
QUANT_METHODS_DYN = [
    "torchao_fp8_dynamic_act_weight_aoscale",
    "torchao_fp8_dynamic_act_weight_comfyscale",
]

# Determine if torchao is available to skip tests if not
SKIP_TORCHAO_TESTS = not TORCHAO_AVAILABLE


@unittest.skipIf(SKIP_TORCHAO_TESTS, "TorchAO not available, skipping scaling tests")
class TestXavierTorchAOScaling(unittest.TestCase):
    SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "xavier.py")

    @classmethod
    def setUpClass(cls):
        cls.temp_dir_manager = tempfile.TemporaryDirectory()
        cls.temp_dir = cls.temp_dir_manager.name

        cls.input_file = os.path.abspath(
            os.path.join(cls.temp_dir, "test_input.safetensors")
        )
        cls.output_file_pattern = os.path.abspath(
            os.path.join(cls.temp_dir, "test_output_{}_{}.safetensors")
        )
        cls.plot_dir = os.path.abspath(
            os.path.join(cls.temp_dir, "test_plots")
        )  # Temporary plot directory
        os.makedirs(cls.plot_dir, exist_ok=True)

        # Create a dummy input safetensor file
        # Using a weight that has a mix of positive/negative, and varied magnitudes
        # Also a bias and another non-target tensor
        cls.original_weight_tensor = torch.tensor(
            [[-1.0, 2.5, -0.1], [0.05, 10.0, -5.5]], dtype=torch.float32
        )
        cls.original_bias_tensor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        cls.other_tensor = torch.tensor([100.0, 200.0], dtype=torch.float16)

        tensors_to_save = {
            "model.layer.weight": cls.original_weight_tensor,
            "model.layer.bias": cls.original_bias_tensor,
            "model.other.tensor": cls.other_tensor,
        }
        save_file(tensors_to_save, cls.input_file)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir_manager.cleanup()

    def _run_xavier_and_verify(self, quant_method, fp8_type, is_comfyscale):
        output_file = self.output_file_pattern.format(quant_method, fp8_type)
        # Ensure plot_dir for this specific run is also specific if needed, or use class level one
        current_plot_dir = os.path.join(
            self.plot_dir, f"{quant_method}_{fp8_type}_plots"
        )
        os.makedirs(current_plot_dir, exist_ok=True)

        cmd = [
            sys.executable,  # Use the current Python interpreter
            self.SCRIPT_PATH,
            self.input_file,
            output_file,
            "--quant_method",
            quant_method,
            "--fp8_type",
            fp8_type,
            "--keys_to_quantize_suffix",
            ".weight",  # Only target weights
            "--device",
            "cpu",  # Force CPU for test consistency
            "--plot",  # Enable plotting for all scaling tests
            "--plot_dir",
            current_plot_dir,
            "--plot_max_tensors",
            "1",  # Only plot the one relevant tensor
        ]

        if (
            "comfyscale" in quant_method
        ):  # This implies native comfyscale flag might be relevant for other tests, but torchao ignores it.
            pass  # TorchAO comfyscale variants handle scaling internally

        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if process.returncode != 0:
            print(f"Xavier script stdout for {quant_method} {fp8_type}:")
            print(process.stdout)
            print(f"Xavier script stderr for {quant_method} {fp8_type}:")
            print(process.stderr)
        self.assertEqual(
            process.returncode,
            0,
            f"Xavier script failed for {quant_method} {fp8_type}. Error: {process.stderr}",
        )

        self.assertTrue(
            os.path.exists(output_file),
            f"Output file not created for {quant_method} {fp8_type}",
        )
        loaded_tensors = load_file(output_file)

        # ---- VERIFICATIONS ----
        target_key = "model.layer.weight"
        scale_key = "model.layer.scale_weight"

        self.assertIn(
            target_key, loaded_tensors, "Quantized tensor not found in output."
        )
        quantized_tensor = loaded_tensors[target_key]

        expected_torch_fp8_dtype = (
            torch.float8_e4m3fn if fp8_type == "e4m3" else torch.float8_e5m2
        )
        self.assertEqual(
            quantized_tensor.dtype,
            expected_torch_fp8_dtype,
            "Quantized tensor has incorrect dtype.",
        )
        self.assertEqual(
            quantized_tensor.shape,
            self.original_weight_tensor.shape,
            "Quantized tensor shape changed.",
        )

        self.assertIn(scale_key, loaded_tensors, "Scale tensor not found in output.")
        scale_tensor = loaded_tensors[scale_key]

        # Scale value/shape checks
        if is_comfyscale:
            if "weight_only" in quant_method:  # Per-channel for WO comfyscale
                expected_scale_shape = (self.original_weight_tensor.shape[0], 1)
                # For _comfyscale in WO, scale should be amax of original along dim 1
                # In xavier.py, the comfyscale path for torchao re-quantizes based on effective_hp_tensor's absmax.
                # So, we cannot directly compare with original_weight_tensor.amax here easily without replicating
                # the exact torchao quantization and then the comfyscale adaptation logic.
                # For now, we check shape and that it's a float.
                self.assertEqual(scale_tensor.shape, expected_scale_shape)
                self.assertTrue(
                    scale_tensor.is_floating_point()
                    or scale_tensor.dtype == torch.float32
                )

            else:  # Per-tensor for dynamic comfyscale
                self.assertEqual(
                    scale_tensor.numel(),
                    1,
                    "Comfyscale dynamic scale should be a scalar.",
                )
                # Similar to WO, exact value is hard to predict due to torchao internal scaling then adaptation.
                # We just check it's a float.
                self.assertTrue(
                    scale_tensor.is_floating_point()
                    or scale_tensor.dtype == torch.float32
                )
        else:  # aoscale
            # For aoscale, torchao determines the scale. It could be per-tensor or per-channel.
            # The mock in test_xavier_torchao.py provides scalar scales (0.123 or 0.456).
            # Actual torchao might do differently. For these tests, we are running actual torchao.
            # We just check it's a float tensor.
            self.assertTrue(
                scale_tensor.is_floating_point() or scale_tensor.dtype == torch.float32
            )
            # self.assertTrue(scale_tensor.numel() >= 1)

        # Check "scaled_fp8" marker
        self.assertIn("scaled_fp8", loaded_tensors, "scaled_fp8 marker not found.")
        self.assertEqual(loaded_tensors["scaled_fp8"].dtype, expected_torch_fp8_dtype)

        # Check other tensors are copied correctly
        self.assertTrue(
            torch.equal(loaded_tensors["model.layer.bias"], self.original_bias_tensor)
        )
        self.assertTrue(
            torch.equal(loaded_tensors["model.other.tensor"], self.other_tensor)
        )

        # Dequantization check (simplified)
        # Convert original to float32 for comparison baseline, as scale is float32
        original_tensor_fp32 = self.original_weight_tensor.to(torch.float32)
        dequantized_tensor = quantized_tensor.to(torch.float32) * scale_tensor.to(
            torch.float32
        )

        # Check if the dequantized tensor is close to the original (using a tolerance)
        # This tolerance might need adjustment based on FP8 precision and scaling method
        # For E4M3, atol might need to be higher. Max value of E4M3 is 448.
        # If original values are large, absolute tolerance will be more sensitive.
        # Using a relative tolerance might be better, but torch.allclose handles both.
        self.assertTrue(
            torch.allclose(
                dequantized_tensor, original_tensor_fp32, atol=0.8, rtol=0.1
            ),
            f"Dequantized tensor significantly different from original for {quant_method} {fp8_type}.\nOriginal:\n{original_tensor_fp32}\nDequantized:\n{dequantized_tensor}\nDiff:\n{original_tensor_fp32 - dequantized_tensor}",
        )

        # Check plot was created
        # Construct expected plot filename based on xavier.py's logic
        safe_tensor_key_for_filename = target_key.replace("/", "_").replace(".", "_")
        expected_plot_filename = os.path.join(
            current_plot_dir, f"{safe_tensor_key_for_filename}_comparison.png"
        )
        self.assertTrue(
            os.path.exists(expected_plot_filename),
            f"Plot file not created: {expected_plot_filename}",
        )

    # -- Weight Only Tests --
    def test_torchao_wo_aoscale_e4m3(self):
        self._run_xavier_and_verify(
            "torchao_fp8_weight_only_aoscale", "e4m3", is_comfyscale=False
        )

    def test_torchao_wo_aoscale_e5m2(self):
        self._run_xavier_and_verify(
            "torchao_fp8_weight_only_aoscale", "e5m2", is_comfyscale=False
        )

    def test_torchao_wo_comfyscale_e4m3(self):
        self._run_xavier_and_verify(
            "torchao_fp8_weight_only_comfyscale", "e4m3", is_comfyscale=True
        )

    def test_torchao_wo_comfyscale_e5m2(self):
        self._run_xavier_and_verify(
            "torchao_fp8_weight_only_comfyscale", "e5m2", is_comfyscale=True
        )

    # -- Dynamic Activation / Weight Tests --
    def test_torchao_dyn_aoscale_e4m3(self):
        self._run_xavier_and_verify(
            "torchao_fp8_dynamic_act_weight_aoscale", "e4m3", is_comfyscale=False
        )

    def test_torchao_dyn_aoscale_e5m2(self):
        self._run_xavier_and_verify(
            "torchao_fp8_dynamic_act_weight_aoscale", "e5m2", is_comfyscale=False
        )

    def test_torchao_dyn_comfyscale_e4m3(self):
        self._run_xavier_and_verify(
            "torchao_fp8_dynamic_act_weight_comfyscale", "e4m3", is_comfyscale=True
        )

    def test_torchao_dyn_comfyscale_e5m2(self):
        self._run_xavier_and_verify(
            "torchao_fp8_dynamic_act_weight_comfyscale", "e5m2", is_comfyscale=True
        )


if __name__ == "__main__":
    unittest.main()
