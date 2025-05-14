import sys
import torch
from safetensors.torch import safe_open


def verify(
    file_path,
    expected_fp8_dtype_str,
    expected_weight_key="layer.weight",
    expected_bias_key="layer.bias",
):
    """
    Verifies a .safetensors file for expected quantization.

    Args:
        file_path (str): Path to the .safetensors file.
        expected_fp8_dtype_str (str): Expected FP8 type as a string ('e4m3' or 'e5m2').
        expected_weight_key (str): Key of the tensor expected to be FP8.
        expected_bias_key (str): Key of the bias tensor (typically remains FP32).

    Returns:
        bool: True if verification passes, False otherwise.
    """
    try:
        if expected_fp8_dtype_str == "e4m3":
            expected_fp8_dtype = torch.float8_e4m3fn
        elif expected_fp8_dtype_str == "e5m2":
            expected_fp8_dtype = torch.float8_e5m2
        else:
            print(f"Error: Unknown expected_fp8_dtype_str: {expected_fp8_dtype_str}")
            return False

        tensor_dtypes = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor_dtypes[k] = f.get_tensor(k).dtype

        print(
            f"Verifying {file_path} (expected FP8 for weight: {expected_fp8_dtype_str}):"
        )

        weight_correct = False
        if expected_weight_key in tensor_dtypes:
            if tensor_dtypes[expected_weight_key] == expected_fp8_dtype:
                print(
                    f"  OK: '{expected_weight_key}' is {tensor_dtypes[expected_weight_key]}."
                )
                weight_correct = True
            else:
                print(
                    f"  FAIL: '{expected_weight_key}' is {tensor_dtypes[expected_weight_key]}, expected {expected_fp8_dtype}."
                )
        else:
            print(f"  FAIL: Expected key '{expected_weight_key}' not found.")

        # Bias check (optional, typically stays fp32 for this model)
        if expected_bias_key in tensor_dtypes:
            # The original model's bias is torch.float32
            if tensor_dtypes[expected_bias_key] == torch.float32:
                print(
                    f"  OK: '{expected_bias_key}' is {tensor_dtypes[expected_bias_key]}."
                )
            else:
                # This is not a hard fail for the script's return status, but good to note.
                print(
                    f"  NOTE: '{expected_bias_key}' is {tensor_dtypes[expected_bias_key]} (expected {torch.float32})."
                )
        # else: Bias might not be present, which is fine.

        # Check for 'scaled_fp8' marker if comfyscale or relevant torchao methods were used
        scaled_marker_key = "scaled_fp8"
        if scaled_marker_key in tensor_dtypes:
            if tensor_dtypes[scaled_marker_key] == expected_fp8_dtype:
                print(
                    f"  OK: '{scaled_marker_key}' marker found with dtype {tensor_dtypes[scaled_marker_key]}."
                )
            else:
                print(
                    f"  WARN: '{scaled_marker_key}' marker found with dtype {tensor_dtypes[scaled_marker_key]}, expected {expected_fp8_dtype}."
                )

        # Check for scale keys (e.g. weight.scale_weight or weight.scale_absmax)
        # This is an informational check. Dtype of scale can vary.
        found_scale_key = False
        for key in tensor_dtypes:
            if key.endswith(".scale_weight") or key.endswith(".scale_absmax"):
                print(
                    f"  INFO: Found scale key '{key}' with dtype {tensor_dtypes[key]}."
                )
                found_scale_key = True

        if not weight_correct:
            print(
                f"Verification Summary for {file_path}: FAILED (weight not correctly quantized)."
            )
            return False

        print(f"Verification Summary for {file_path}: PASSED.")
        return True

    except Exception as e:
        print(f"Error during verification of {file_path}: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python verify_quantization.py <safetensors_file_path> <expected_fp8_type_str e.g. e4m3 or e5m2>"
        )
        sys.exit(1)

    file_to_verify = sys.argv[1]
    fp8_type_to_check = sys.argv[2]

    if not verify(file_to_verify, fp8_type_to_check):
        sys.exit(1)  # Indicate failure to the calling script
    sys.exit(0)  # Indicate success
