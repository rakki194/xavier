import torch
from .get_fp8_neighbor import get_fp8_neighbor
from .get_fp8_bracketing_candidates_complex import get_fp8_bracketing_candidates_complex
from .stochastic_round_owlshift_method import stochastic_round_owlshift_method
from .stochastic_round_shift_perturb import stochastic_round_shift_perturb


def stochastic_round_tensor_to_fp8(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    use_complex_method: bool,
    use_shift_perturb_method: bool,
    use_owlshift_method: bool,
    seed: int = 0,
    debug_mode: bool = False,
) -> torch.Tensor:
    """
    Applies stochastic rounding to convert a tensor to the specified FP8 dtype.
    Note: This is a challenging operation due to the nature of FP8 and relies on
    approximations for finding neighboring FP8 values.
    """
    if debug_mode:
        print(
            f"DEBUG stochastic_round_tensor_to_fp8: use_complex={use_complex_method}, use_shifturb={use_shift_perturb_method}, use_owlshift={use_owlshift_method}"
        )

    # If the tensor is complex, proceed with its real part for all methods.
    # This makes the conversion explicit and silences PyTorch warnings.
    # Child methods (owlshift, shift_perturb) also have this check for direct calls,
    # but it's good to prepare tensor_to_process here for the default logic path.
    if tensor.is_complex():
        tensor_to_process = tensor.real
    else:
        tensor_to_process = tensor

    if not tensor_to_process.is_floating_point():
        try:
            return tensor_to_process.to(fp8_dtype)
        except Exception:
            print(
                f"Warning: Could not convert non-float tensor of dtype {tensor_to_process.dtype} to {fp8_dtype}. Returning original."
            )
            return tensor_to_process

    if tensor_to_process.numel() == 0:
        return tensor_to_process.to(fp8_dtype)

    if use_owlshift_method:
        if debug_mode:
            print("DEBUG: Using Owlshift Method")
        return stochastic_round_owlshift_method(tensor_to_process, fp8_dtype, seed=seed)
    elif use_shift_perturb_method:
        if debug_mode:
            print("DEBUG: Using Shift-Perturb Method")
        return stochastic_round_shift_perturb(tensor_to_process, fp8_dtype)
    elif use_complex_method:
        if debug_mode:
            print("DEBUG: Using Complex Bracketing Method for candidate search")
        low_candidate, high_candidate = get_fp8_bracketing_candidates_complex(
            tensor_to_process, fp8_dtype
        )
    else:
        if debug_mode:
            print(
                "DEBUG: Using Default (Simple) Bracketing Method for candidate search"
            )
        x_rne_orig_prec = tensor_to_process.to(fp8_dtype).to(tensor_to_process.dtype)
        direction_to_tensor = torch.sign(tensor_to_process - x_rne_orig_prec)
        direction_for_neighbor_search = torch.where(
            direction_to_tensor == 0,
            torch.ones_like(direction_to_tensor),
            direction_to_tensor,
        )
        x_neighbor_orig_prec = get_fp8_neighbor(
            x_rne_orig_prec, direction_for_neighbor_search, fp8_dtype
        )
        low_candidate = torch.min(x_rne_orig_prec, x_neighbor_orig_prec)
        high_candidate = torch.max(x_rne_orig_prec, x_neighbor_orig_prec)

    denominator = high_candidate - low_candidate
    prob_high = torch.zeros_like(tensor_to_process)
    non_degenerate_mask = denominator > torch.finfo(denominator.dtype).eps

    if torch.any(non_degenerate_mask):
        prob_high[non_degenerate_mask] = (
            (tensor_to_process - low_candidate) / denominator
        )[non_degenerate_mask]

    prob_high = torch.clamp(prob_high, 0.0, 1.0)
    random_draw = torch.rand_like(tensor_to_process)
    chosen_value_orig_prec = torch.where(
        random_draw < prob_high, high_candidate, low_candidate
    )

    if (
        debug_mode
        and not use_complex_method
        and not use_shift_perturb_method
        and not use_owlshift_method
    ):
        print(
            f"DEBUG Default Bracketing: tensor={tensor_to_process}, low_cand={low_candidate}, high_cand={high_candidate}, prob_high={prob_high}, rand_draw={random_draw}, chosen_orig_prec={chosen_value_orig_prec}"
        )

    return chosen_value_orig_prec.to(fp8_dtype)
