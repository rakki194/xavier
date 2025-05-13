from safetensors import safe_open


def load_tensor_from_safetensors(file_path: str, tensor_name: str, device: str = "cpu"):
    """
    Loads a specific tensor from a .safetensors file on the fly.

    Args:
        file_path (str): Path to the .safetensors file.
        tensor_name (str): The name of the tensor to load.
        device (str): The device to load the tensor onto (e.g., "cpu", "cuda").

    Returns:
        torch.Tensor: The loaded tensor.
    """
    with safe_open(file_path, framework="pt", device=device) as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


def get_safetensors_metadata(file_path: str):
    """
    Retrieves the metadata (list of tensor keys and their shapes/dtypes) from a .safetensors file.

    Args:
        file_path (str): Path to the .safetensors file.

    Returns:
        dict: A dictionary where keys are tensor names and values are dicts containing their shape (list of int) and dtype (str).
    """
    metadata = {}
    with safe_open(
        file_path, framework="pt", device="cpu"
    ) as f:  # Read metadata on CPU
        for key in f.keys():
            try:
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                dtype_str = tensor_slice.get_dtype()
                metadata[key] = {"shape": shape, "dtype": dtype_str}
            except Exception as e:
                # Fallback or error logging if a specific key fails, though unlikely with f.keys()
                metadata[key] = {"error": f"Could not retrieve metadata: {str(e)}"}
    return metadata


def get_safetensors_tensor_keys(file_path: str):
    """
    Retrieves the list of tensor keys from a .safetensors file.

    Args:
        file_path (str): Path to the .safetensors file.

    Returns:
        list[str]: A list of tensor names (keys) in the file.
    """
    with safe_open(file_path, framework="pt", device="cpu") as f:  # Read keys on CPU
        keys = f.keys()
    return keys
