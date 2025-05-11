import torch
import pytest

from scaling.get_fp8_constants_for_owlscale import get_fp8_constants_for_owlscale

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2


def test_get_fp8_constants_e4m3fn():
    fp8_dtype = FP8_E4M3
    finfo = torch.finfo(fp8_dtype)
    expected_min = float(finfo.min)
    expected_max = float(finfo.max)
    expected_min_pos = 2**-9  # As per function's hardcoding for E4M3FN

    actual_min, actual_max, actual_min_pos = get_fp8_constants_for_owlscale(fp8_dtype)

    assert actual_min == expected_min
    assert actual_max == expected_max
    assert actual_min_pos == expected_min_pos


def test_get_fp8_constants_e5m2():
    fp8_dtype = FP8_E5M2
    finfo = torch.finfo(fp8_dtype)
    expected_min = float(finfo.min)
    expected_max = float(finfo.max)
    expected_min_pos = 2**-16  # As per function's hardcoding for E5M2

    actual_min, actual_max, actual_min_pos = get_fp8_constants_for_owlscale(fp8_dtype)

    assert actual_min == expected_min
    assert actual_max == expected_max
    assert actual_min_pos == expected_min_pos


# Optional: Test with a hypothetical other fp8 type to check fallback
# This requires a dummy fp8 type or a real one if more become available in torch.
# For now, we will skip this as the primary types are covered.

# Test for unsupported type (though current function doesn't raise error, uses finfo.tiny)
# We could add a check for a specific warning if that were the behavior, or ensure fallback is reasonable.
# Given the current implementation, we just test the main paths.
