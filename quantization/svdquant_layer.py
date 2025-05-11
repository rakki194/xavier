import torch
import torch.nn as nn
import math
from quantization.smoothing_utils import (
    apply_smoothing_to_activations,
    apply_smoothing_to_weights,
    calculate_smoothing_factors,
)
from .svd_utils import decompose_weights_svd
from .group_quant_utils import (
    quantize_symmetric_per_group,
    dequantize_symmetric_per_group,
)

# Add new imports
from .calibration_utils import find_optimal_alpha_svdquant
from .gptq_utils import gptq_quantize_layer_residual_refined


class SVDQuantLinear(nn.Module):
    def __init__(
        self,
        original_in_features,
        original_out_features,
        rank,
        num_bits,
        group_size,
        alpha,
        bias,
        low_rank_dtype,
        scale_dtype,
        device,
        dtype=None,
    ):
        super(SVDQuantLinear, self).__init__()
        self.in_features = original_in_features
        self.out_features = original_out_features
        self.rank = rank
        self.num_bits = num_bits
        self.group_size = group_size
        self.alpha = alpha
        self.bias = bias
        self.low_rank_dtype = low_rank_dtype
        self.scale_dtype = scale_dtype
        self.device = device
        self.dtype = dtype

        self.L1 = None  # Shape: (in_features, rank), dtype: low_rank_dtype
        self.L2 = None  # Shape: (rank, out_features), dtype: low_rank_dtype

        # For residual R = hat_W - L1 @ L2
        self.Q_R_int8 = None  # Shape: (in_features, out_features), dtype: torch.int8
        self.scales_R = (
            None  # Shape: (in_features, out_features // group_size), dtype: scale_dtype
        )

        self.lambda_val = None  # Shape: (in_features,), for smoothing
        # self.alpha is an input, but will be overridden if alpha_search is True

        # Store original bias if it exists
        if bias:
            # nn.Parameter ensures it's part of state_dict, etc.
            self.original_bias = nn.Parameter(
                torch.zeros(
                    original_out_features,
                    device=device,
                    dtype=dtype if dtype is not None else torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter(
                "original_bias", None
            )  # Correct way to register no bias

        self.prepared = False  # Flag to check if 'prepare' has been called

    def prepare(
        self,
        original_weight_tensor,
        calibration_activations,
        original_bias_tensor=None,
        perform_alpha_search=True,  # New parameter to control alpha search
        alpha_search_values=None,  # Optional list of alphas for search
        verbose_alpha_search=False,
        # GPTQ specific parameters to pass through
        gptq_percdamp=0.01,
        gptq_act_order=False,
        gptq_compensation_strength=0.1,
        gptq_verbose=False,
    ):
        """
        Prepares the layer for SVDQuant.
        Args:
            original_weight_tensor (torch.Tensor): The weight tensor of the original layer.
                                                  Shape: (out_features, in_features).
            calibration_activations (torch.Tensor): The calibration activations.
                                                    Shape: (batch_size, in_features).
            original_bias_tensor (torch.Tensor, optional): The bias tensor of the original layer.
                                                          Shape: (out_features,).
            perform_alpha_search (bool): If True, search for optimal alpha.
                                         Otherwise, use self.alpha.
            alpha_search_values (list, optional): Specific alpha values for search.
            verbose_alpha_search (bool): Verbosity for alpha search.
            gptq_percdamp (float): Dampening for Hessian in GPTQ.
            gptq_act_order (bool): Column reordering for GPTQ.
            gptq_compensation_strength (float): Error compensation strength in GPTQ.
            gptq_verbose (bool): Verbosity for GPTQ process.
        """
        if original_weight_tensor.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Weight tensor shape ({original_weight_tensor.shape}) does not match expected ({self.out_features}, {self.in_features})"
            )
        # Our utilities (smoothing, SVD) expect weights as (in_features, out_features)
        # nn.Linear has weights as (out_features, in_features)
        W_for_processing = (
            original_weight_tensor.T.contiguous().to(self.device).float()
        )  # Shape: (in_features, out_features)

        if perform_alpha_search:
            print(
                f"Performing alpha search for SVDQuantLinear layer ({self.in_features}x{self.out_features})..."
            )
            alpha_search_result = find_optimal_alpha_svdquant(
                original_weight_tensor,  # Pass original nn.Linear format
                calibration_activations.to(self.device),
                self.rank,
                self.num_bits,
                self.group_size,
                alpha_values=alpha_search_values,
                low_rank_dtype=self.low_rank_dtype,
                scale_dtype=self.scale_dtype,
                verbose=verbose_alpha_search,
            )
            self.alpha = alpha_search_result["best_alpha"]  # Update layer's alpha
            self.lambda_val = alpha_search_result["best_lambda_val"].to(self.device)
            L1_hp = alpha_search_result["best_L1"].to(self.device)
            L2_hp = alpha_search_result["best_L2"].to(self.device)
            # Q_R and scales_R from alpha search are based on round-to-nearest.
            # If we want to use GPTQ on the R derived from best_alpha's L1/L2, we need R_hp first.
            # The alpha search's L1/L2 were derived from hat_W for that alpha.
            # So, re-calculate hat_W for the best_alpha, then R_hp.
            hat_W_best_alpha = apply_smoothing_to_weights(
                W_for_processing, self.lambda_val
            )
            # R_hp for GPTQ should be hat_W_best_alpha - (L1_hp @ L2_hp)
            # Note: L1_hp, L2_hp from alpha search are already the high-precision ones for the best alpha.
            R_hp_for_gptq = hat_W_best_alpha - (
                L1_hp.to(hat_W_best_alpha.dtype) @ L2_hp.to(hat_W_best_alpha.dtype)
            )

            print(f"Alpha search complete. Best alpha: {self.alpha:.3f}")

        else:  # Use the provided self.alpha
            print(f"Using pre-set alpha: {self.alpha:.3f}")
            # 1. Calculate Smoothing Factors (lambda_val)
            self.lambda_val = calculate_smoothing_factors(
                calibration_activations.to(self.device), W_for_processing, self.alpha
            ).to(self.device)

            # 2. Apply Smoothing to Weights to get hat_W
            hat_W = apply_smoothing_to_weights(W_for_processing, self.lambda_val)

            # 3. Decompose hat_W into L1, L2, and R_hp (high precision)
            L1_hp, L2_hp, R_hp_for_gptq = decompose_weights_svd(hat_W, self.rank)

        # Store L1, L2 (cast to low_rank_dtype)
        self.L1 = nn.Parameter(L1_hp.to(self.low_rank_dtype), requires_grad=False)
        self.L2 = nn.Parameter(L2_hp.to(self.low_rank_dtype), requires_grad=False)

        # 4. Quantize Residual R_hp_for_gptq using GPTQ (placeholder for now)
        print(
            f"Quantizing residual R (shape {R_hp_for_gptq.shape}) using GPTQ placeholder..."
        )
        # Smoothed calibration activations are needed for actual GPTQ
        hat_X_cal_for_gptq = apply_smoothing_to_activations(
            calibration_activations.to(self.device).float(), self.lambda_val
        )

        Q_R_int8_temp, scales_R_temp = gptq_quantize_layer_residual_refined(
            R_hp_for_gptq,
            hat_X_cal_for_gptq,  # Pass smoothed activations
            self.num_bits,
            self.group_size,
            scale_dtype=self.scale_dtype,
            # Pass through GPTQ specific parameters
            percdamp=gptq_percdamp,
            act_order=gptq_act_order,
            compensation_strength=gptq_compensation_strength,
            verbose=gptq_verbose,
        )
        self.Q_R_int8 = nn.Parameter(Q_R_int8_temp, requires_grad=False)
        self.scales_R = nn.Parameter(scales_R_temp, requires_grad=False)

        # Store original bias if provided
        if original_bias_tensor is not None:
            if original_bias_tensor.shape != (self.out_features,):
                raise ValueError(
                    f"Bias tensor shape ({original_bias_tensor.shape}) does not match expected ({self.out_features},)"
                )
            self.bias = nn.Parameter(
                original_bias_tensor.to(self.device), requires_grad=False
            )

    def forward(self, x):
        # Implementation of forward pass
        pass

    # --- Example Usage (Conceptual) ---
    if __name__ == "__main__":
        # This is a conceptual example.
        # In a real scenario, you'd iterate over layers of a model.

        # Configuration
        _in_features = 128
        _out_features = 256
        _rank = 32
        _num_bits = 4
        _group_size = 64
        _default_alpha = 0.5  # For the no-search case
        _low_rank_dt = torch.float16
        _scale_dt = torch.float16
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dtype_module = (
            torch.float32
        )  # Default dtype for nn.Module params if not specified
        _num_calibration_samples = 128  # Number of samples for calibration

        # 1. Create an instance of the SVDQuantLinear layer
        svd_layer_alpha_search = SVDQuantLinear(
            original_in_features=_in_features,
            original_out_features=_out_features,
            rank=_rank,
            num_bits=_num_bits,
            group_size=_group_size,
            alpha=_default_alpha,  # This will be overridden by search
            bias=True,
            low_rank_dtype=_low_rank_dt,
            scale_dtype=_scale_dt,
            device=_device,
            dtype=_dtype_module,
        ).to(_device)

        # 2. Prepare dummy original weights, bias, and calibration data
        # Original nn.Linear stores weight as (out_features, in_features)
        dummy_original_weight = (
            torch.randn(
                _out_features, _in_features, device=_device, dtype=torch.float32
            )
            * 0.1
        )
        dummy_original_bias = torch.randn(
            _out_features, device=_device, dtype=torch.float32
        )
        dummy_calibration_activations = torch.randn(
            _num_calibration_samples, _in_features, device=_device, dtype=torch.float32
        )

        print(f"SVD Layer (for alpha search) before prepare: {svd_layer_alpha_search}")

        # 3. Call the prepare method - WITH alpha search
        svd_layer_alpha_search.prepare(
            dummy_original_weight,
            dummy_calibration_activations,
            dummy_original_bias,
            perform_alpha_search=True,
            verbose_alpha_search=True,
            # GPTQ params for the call inside prepare
            gptq_percdamp=0.01,
            gptq_act_order=False,  # Try True to test this path as well
            gptq_compensation_strength=0.05,  # Example value
            gptq_verbose=True,
        )

        print(f"SVD Layer (with alpha search) after prepare: {svd_layer_alpha_search}")
        print(f"  Best Alpha found: {svd_layer_alpha_search.alpha:.3f}")
        print(
            f"  L1 shape: {svd_layer_alpha_search.L1.shape}, dtype: {svd_layer_alpha_search.L1.dtype}"
        )

        # Call prepare method - WITHOUT alpha search (using default alpha=0.5)
        svd_layer_no_search = SVDQuantLinear(
            original_in_features=_in_features,
            original_out_features=_out_features,
            rank=_rank,
            num_bits=_num_bits,
            group_size=_group_size,
            alpha=0.6,  # Fixed alpha for this test
            bias=True,
            low_rank_dtype=_low_rank_dt,
            scale_dtype=_scale_dt,
            device=_device,
            dtype=_dtype_module,
        ).to(_device)
        print("\nPreparing layer with fixed alpha (0.6)...")
        svd_layer_no_search.prepare(
            dummy_original_weight,
            dummy_calibration_activations,
            dummy_original_bias,
            perform_alpha_search=False,
            # GPTQ params for this call
            gptq_percdamp=0.01,
            gptq_act_order=False,
            gptq_compensation_strength=0.05,
            gptq_verbose=True,
        )
        print(
            f"SVD Layer (no search, alpha={svd_layer_no_search.alpha}) after prepare: {svd_layer_no_search}"
        )

        # 4. Perform a forward pass (using the layer with alpha search)
        print("\nTesting forward pass...")
        batch_size = 16
        dummy_input_activations = torch.randn(
            batch_size, _in_features, device=_device, dtype=torch.float32
        )

        try:
            # Test with the layer that performed alpha search
            output_alpha_search = svd_layer_alpha_search(dummy_input_activations)
            print(
                f"Output shape (alpha search layer): {output_alpha_search.shape}, dtype: {output_alpha_search.dtype}"
            )

            # Test with the layer that used fixed alpha
            output_no_search = svd_layer_no_search(dummy_input_activations)
            print(
                f"Output shape (fixed alpha layer): {output_no_search.shape}, dtype: {output_no_search.dtype}"
            )

            # For comparison: output of an equivalent original linear layer (float32)
            ref_linear = nn.Linear(
                _in_features,
                _out_features,
                bias=True,
                device=_device,
                dtype=torch.float32,
            )
            ref_linear.weight.data.copy_(dummy_original_weight)
            ref_linear.bias.data.copy_(dummy_original_bias)
            reference_output = ref_linear(dummy_input_activations)

            mse_vs_original_alpha_search = torch.mean(
                (output_alpha_search.float() - reference_output.float()) ** 2
            )
            print(
                f"MSE vs original (alpha search layer): {mse_vs_original_alpha_search.item():.6e}"
            )
            mse_vs_original_no_search = torch.mean(
                (output_no_search.float() - reference_output.float()) ** 2
            )
            print(
                f"MSE vs original (fixed alpha layer): {mse_vs_original_no_search.item():.6e}"
            )
            print("(Note: These MSEs reflect overall SVDQuant error vs FP32 original)")

        except Exception as e:
            print(f"Error in forward pass: {e}")
