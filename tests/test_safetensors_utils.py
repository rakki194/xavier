import pytest
import torch
from safetensors.torch import save_file
from safetensors import SafetensorError
import tempfile
import os

# Assuming safetensors_utils.py is in the parent directory or PYTHONPATH is set up
from safetensors_utils import (
    get_safetensors_metadata,
    load_tensor_from_safetensors,
    get_safetensors_tensor_keys,
)


@pytest.fixture
def dummy_safetensors_file(tmp_path):
    """Creates a dummy .safetensors file and returns its path and expected metadata."""
    file_path = tmp_path / "test_model.safetensors"
    data = {
        "tensor_fp32": torch.randn(2, 3, dtype=torch.float32),
        "tensor_bf16": torch.randn(5, dtype=torch.bfloat16),
        "tensor_int8": torch.randint(-10, 10, (4, 4), dtype=torch.int8),
    }
    save_file(data, file_path)

    expected_metadata = {
        "tensor_fp32": {"shape": [2, 3], "dtype": "F32"},
        "tensor_bf16": {"shape": [5], "dtype": "BF16"},
        "tensor_int8": {"shape": [4, 4], "dtype": "I8"},
    }
    return str(file_path), expected_metadata


class TestSafetensorsUtils:
    def test_get_safetensors_metadata(self, dummy_safetensors_file):
        file_path, expected_metadata = dummy_safetensors_file

        metadata = get_safetensors_metadata(file_path)

        assert metadata == expected_metadata

    def test_get_safetensors_tensor_keys(self, dummy_safetensors_file):
        file_path, expected_metadata = dummy_safetensors_file
        keys = get_safetensors_tensor_keys(file_path)
        assert sorted(keys) == sorted(list(expected_metadata.keys()))

    def test_load_tensor_from_safetensors(self, dummy_safetensors_file):
        file_path, _ = dummy_safetensors_file
        # Test loading one of the tensors
        tensor_name = "tensor_fp32"
        original_tensor = torch.randn(
            2, 3, dtype=torch.float32
        )  # Need to re-create to save for comparison

        # Re-save the file with a known tensor to compare against
        # (the one in dummy_safetensors_file is random each time)
        save_data = {tensor_name: original_tensor, "other": torch.tensor(1)}
        save_file(save_data, file_path)

        loaded_tensor = load_tensor_from_safetensors(
            file_path, tensor_name, device="cpu"
        )
        assert torch.equal(loaded_tensor, original_tensor)
        assert (
            str(loaded_tensor.dtype).split(".")[-1] == "float32"
        )  # Check dtype on loaded tensor
        assert loaded_tensor.device.type == "cpu"

    def test_load_tensor_to_device(self, dummy_safetensors_file):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device load test")

        file_path, _ = dummy_safetensors_file
        tensor_name = "tensor_bf16"
        original_tensor = torch.randn(5, dtype=torch.bfloat16)
        save_data = {tensor_name: original_tensor}
        save_file(save_data, file_path)

        loaded_tensor = load_tensor_from_safetensors(
            file_path, tensor_name, device="cuda"
        )
        assert torch.equal(
            loaded_tensor.cpu(), original_tensor
        )  # Compare by moving to CPU
        assert str(loaded_tensor.dtype).split(".")[-1] == "bfloat16"
        assert loaded_tensor.device.type == "cuda"

    def test_get_safetensors_metadata_empty_file(self, tmp_path):
        file_path = tmp_path / "empty.safetensors"
        save_file({}, file_path)
        metadata = get_safetensors_metadata(file_path)
        assert metadata == {}

    def test_get_safetensors_tensor_keys_empty_file(self, tmp_path):
        file_path = tmp_path / "empty.safetensors"
        save_file({}, file_path)
        keys = get_safetensors_tensor_keys(file_path)
        assert keys == []

    def test_load_tensor_non_existent_key(self, dummy_safetensors_file):
        file_path, _ = dummy_safetensors_file
        with pytest.raises(
            SafetensorError
        ):  # Or a more specific SafetensorError if defined and expected
            load_tensor_from_safetensors(file_path, "non_existent_key")
