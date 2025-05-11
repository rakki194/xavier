# FP8 Scaling Implementation in ComfyUI

ComfyUI incorporates 8-bit floating-point (FP8) capabilities to optimize model performance and reduce memory usage. This document details how FP8 scaling is implemented, focusing on the key code components and their interactions.

## 1. Core FP8 Operations (`comfy/ops.py`)

The primary logic for FP8 operations, especially for linear layers, resides in `comfy/ops.py`. This file defines custom operational classes that override standard PyTorch layer behaviors to enable FP8 computation and weight handling.

### 1.1. `fp8_linear(self, input)`

This function is a specialized implementation for performing linear operations (matrix multiplication) when weights are in FP8 format, specifically `torch.float8_e4m3fn`.

```python
# Relevant section from comfy/ops.py
def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]: # Only handles e4m3fn directly
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1) # Ensure 3D input for processing

    input_shape = input.shape
    input_dtype = input.dtype # Original input dtype for output
    if len(input.shape) == 3:
        # cast_bias_weight handles fetching and casting weights/biases
        # Weights (w) are cast to the FP8 dtype here
        w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
        w = w.t() # Transpose weight for matrix multiplication

        scale_weight = self.scale_weight # Per-tensor scale for weights
        scale_input = self.scale_input   # Per-tensor scale for activations

        if scale_weight is None:
            # Default to no scaling if not provided
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            # If no input scale, clamp input to a range suitable for e4m3fn
            # then cast to FP8. Clamping helps prevent overflow/underflow.
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            input = input.reshape(-1, input_shape[2]).to(dtype)
        else:
            # If input scale is provided, scale the input before casting to FP8
            scale_input = scale_input.to(input.device)
            input = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype)

        # torch._scaled_mm performs the scaled matrix multiplication:
        # (input_fp8 * scale_input) @ (weight_fp8 * scale_weight)
        # Output is cast to original_input_dtype.
        if bias is not None:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple): # _scaled_mm might return a tuple
            o = o[0]

        if tensor_2d:
            return o.reshape(input_shape[0], -1) # Reshape to original 2D if needed

        return o.reshape((-1, input_shape[1], self.weight.shape[0]))

    return None # Fallback if input is not 3D (after potential unsqueeze) or other issues
```

**Explanation:**

* **Purpose**: Executes $Y = (X @ W^T) + B$ where $X$ (input/activations) and $W$ (weights) can be FP8, and $X$ and $W$ have associated scaling factors.
* **Weight Type**: Explicitly checks if `self.weight.dtype` is `torch.float8_e4m3fn`. If not, it returns `None`, signaling a fallback to other computation paths.
* **Scaling Factors**: It expects `self.scale_weight` and `self.scale_input` to be attributes of the layer instance (`self`). These are float32 scalars.
  * `scale_weight`: Factor for the FP8 weights.
  * `scale_input`: Factor for the FP8 activations.
* **Input Processing**:
  * If `scale_input` is missing, input is clamped to `[-448, 448]` (empirically chosen for `e4m3fn`) and then directly cast to the FP8 `dtype`.
  * If `scale_input` is present, the input is multiplied by `(1.0 / scale_input)` *before* casting to FP8. This effectively means `scale_input` is the value by which the FP8 activations need to be multiplied to approximate their original scale.
* **Core Computation**: `torch._scaled_mm` is used. This function takes the FP8 input, FP8 weight, their respective scaling factors (`scale_a` for input, `scale_b` for weight), and computes the matrix multiplication, outputting in the original `input_dtype`.
* **Fallback**: Returns `None` if it cannot handle the operation (e.g., wrong weight type), allowing a more general path to be taken.

#### Mathematical Formulation (`fp8_linear`)

Let:

* $X_{orig}$ be the original input tensor (e.g., float16 or float32).
* $W_{f8}$ be the weight tensor, already in FP8 format (`self.weight`, e.g., `torch.float8_e4m3fn`).
* $B$ be the bias tensor (in original input dtype).
* $S_W$ be `self.scale_weight` (a float32 scalar).
* $S_X$ be `self.scale_input` (a float32 scalar).

1. **Input Tensor Preparation (Activation $A_{f8}$ for `_scaled_mm`)**:
    * If $S_X$ is provided (not `None`):
        The original input `X_orig` is scaled by the reciprocal of `S_X` to get `X_scaled` (i.e., `X_scaled = X_orig * (1.0 / S_X)`).
        `X_scaled` is then cast to an FP8 tensor, resulting in `A_f8` (i.e., `A_f8 = CastToFP8(X_scaled)`).
        The `scale_a` argument for `_scaled_mm` will be $S_X$.
    * If $S_X$ is `None`:
        The original input `X_orig` is clamped to the range `[-448, 448]`, resulting in `X_clamped`.
        `X_clamped` is then cast to an FP8 tensor, resulting in `A_f8`.
        The `scale_a` argument for `_scaled_mm` will effectively be $1.0$ (as `scale_input` defaults to `1.0` in `fp8_linear` if `self.scale_input` was None).

2. **Weight Tensor Preparation (Weight $W_{f8}$ for `_scaled_mm`)**:
    * The weight tensor $W_{f8}$ is used directly as it's already in FP8.
    * The `scale_b` argument for `_scaled_mm` will be $S_W$ (or $1.0$ if `self.scale_weight` was `None`).

3. **Scaled Matrix Multiplication (`torch._scaled_mm`)**:
    The function `torch._scaled_mm(a, b, out_dtype, bias, scale_a, scale_b)` conceptually computes the output `O` by first casting its inputs `a` and `b` to `out_dtype`, then multiplying them by their respective scales (`scale_a` and `scale_b`), performing a matrix multiplication of these scaled tensors (with `b` transposed), and finally adding the bias (also cast to `out_dtype`).
    However, the typical interpretation for FP8 GEMM is that the scaling happens *before* the multiplication accumulates in higher precision, and `a` and `b` are FP8.
    PyTorch's `_scaled_mm` with FP8 inputs and float32 scales likely performs an operation conceptually equivalent to: first, upcasting the FP8 activation `A_f8` to float32 and multiplying by its scale `S_X`; second, upcasting the FP8 weight `W_f8` to float32 and multiplying by its scale `S_W`; then, performing a matrix multiplication of these two resulting float32 tensors (with the scaled weight tensor transposed); and finally, adding the bias `B`. The result `O` is then cast to the desired `out_dtype`.
    Or, more directly, it computes $ (A_{f8} \cdot S_X) @ (W_{f8} \cdot S_W)^T + B $ where the products $A_{f8} \cdot S_X$ and $W_{f8} \cdot S_W$ are performed with appropriate precision handling (e.g., upcasting FP8 values before multiplying by float32 scales) and the matrix multiplication result is accumulated in float32 before being cast to `out_dtype` (which is `input_dtype`). The output $O$ is in the original input precision (`input_dtype`).

    It's important to note that `input` (which becomes $A_{f8}$) is cast to FP8 *after* being divided by $S_X$ (if $S_X$ is present). So, the scaling effectively brings the original input values into a range that is then quantized to FP8. The `_scaled_mm` then "undoes" this scaling by multiplying by $S_X$ again, but on the FP8 values.

### 1.2. `class fp8_ops(manual_cast)`

This class provides a set of FP8-aware neural network layers (currently focusing on `Linear`) by inheriting from `manual_cast`. `manual_cast` itself ensures that weights are cast to the appropriate type during the forward pass.

```python
# Relevant section from comfy/ops.py
class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            # Initialize scales to None; they are expected to be loaded
            # if the model is an FP8 model.
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            # Attempt to use the specialized fp8_linear function first
            out = fp8_linear(self, input)
            if out is not None:
                return out # Return if fp8_linear handled it

            # Fallback: Cast weights and use standard linear operation
            # This path is taken if fp8_linear returned None (e.g., weight not e4m3fn)
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)
```

**Explanation:**

* The `Linear` layer within `fp8_ops` overrides `forward_comfy_cast_weights`.
* It first tries to use the `fp8_linear` function.
* If `fp8_linear` returns a result (not `None`), that result is used.
* Otherwise (e.g., if weights are not `torch.float8_e4m3fn`), it falls back to the behavior of `manual_cast.Linear`, which typically involves casting weights to the input's data type and then performing a standard `torch.nn.functional.linear` operation.
* `reset_parameters` ensures `scale_weight` and `scale_input` are `None` initially. These would be populated if an FP8 model with these scales is loaded.

### 1.3. `scaled_fp8_ops(fp8_matrix_mult=False, scale_input=False, override_dtype=None)`

This is a factory function that returns a different FP8 operations class, `scaled_fp8_op`. This class is more versatile and manages its own scaling factors, particularly `scale_weight`, as part of the layer's parameters. It's designed for scenarios where weights are stored in FP8, and the scaling factor is intrinsic to the layer for on-the-fly dequantization or for use with `torch._scaled_mm`.

```python
# Relevant section from comfy/ops.py
def scaled_fp8_ops(fp8_matrix_mult=False, scale_input=False, override_dtype=None):
    logging.info("Using scaled fp8: fp8 matrix mult: {}, scale input: {}".format(fp8_matrix_mult, scale_input))
    class scaled_fp8_op(manual_cast):
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                if override_dtype is not None:
                    # Allows specifying the FP8 dtype (e.g., e4m3fn, e5m2) for weights
                    kwargs['dtype'] = override_dtype
                super().__init__(*args, **kwargs)

            def reset_parameters(self):
                # Initialize scale_weight as a non-trainable Parameter (float32)
                if not hasattr(self, 'scale_weight'):
                    self.scale_weight = torch.nn.parameter.Parameter(
                        data=torch.ones((), device=self.weight.device, dtype=torch.float32), 
                        requires_grad=False
                    )

                if not scale_input: # scale_input is from the factory function
                    self.scale_input = None
                elif not hasattr(self, 'scale_input'): # Initialize if scale_input is True
                     self.scale_input = torch.nn.parameter.Parameter(
                        data=torch.ones((), device=self.weight.device, dtype=torch.float32),
                        requires_grad=False
                    )
                return None

            def forward_comfy_cast_weights(self, input):
                if fp8_matrix_mult: # If True, try fp8_linear (which uses _scaled_mm)
                    out = fp8_linear(self, input) # fp8_linear will use self.scale_weight and self.scale_input
                    if out is not None:
                        return out

                # Fallback or primary path if fp8_matrix_mult is False
                weight, bias = cast_bias_weight(self, input) # Weight is FP8 here

                # Dequantize on-the-fly: Multiply either weight or input by scale_weight
                # This assumes weight is FP8 and needs to be scaled to a higher effective precision
                # for the F.linear operation if _scaled_mm is not used.
                if weight.numel() < input.numel(): # Heuristic for optimization
                    return torch.nn.functional.linear(input, 
                                                      weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype), 
                                                      bias)
                else:
                    return torch.nn.functional.linear(input * self.scale_weight.to(device=input.device, dtype=input.dtype), 
                                                      weight, 
                                                      bias)

            # convert_weight is likely for model conversion/saving
            def convert_weight(self, weight, inplace=False, **kwargs):
                # Scales a given weight by the layer's scale_weight
                scaled_w = weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                if inplace:
                    weight.copy_(scaled_w)
                    return weight
                return scaled_w

            # set_weight is crucial for loading weights into this layer
            def set_weight(self, weight, inplace_update=False, seed=None, **kwargs):
                # Input 'weight' is higher precision (e.g., float16/32)
                # De-scale it first: Original_Value / Scale = Value_to_Quantize
                weight_to_quantize = weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                
                # Quantize to the layer's FP8 dtype using stochastic rounding
                fp8_w = comfy.float.stochastic_rounding(weight_to_quantize, self.weight.dtype, seed=seed)
                
                if inplace_update:
                    self.weight.data.copy_(fp8_w)
                else:
                    self.weight = torch.nn.Parameter(fp8_w, requires_grad=False)

    return scaled_fp8_op
```

**Explanation:**

* **`scaled_fp8_op.Linear`**:
  * `__init__`: Can take an `override_dtype` to set the FP8 type for its weights (e.g., `torch.float8_e4m3fn` or `torch.float8_e5m2`).
  * `reset_parameters()`: Initializes `self.scale_weight` as a non-trainable `torch.nn.Parameter` (defaulting to 1.0, stored in float32). This scaling factor is considered part of the layer. `self.scale_input` is also initialized if the `scale_input` flag (from the factory function) is true.
  * `set_weight(self, weight, ...)`: This is a critical method. When weights are loaded (e.g., from a checkpoint) into a layer of this type:
        1. The incoming `weight` (assumed to be in a higher precision like float16 or float32) is *divided* by `self.scale_weight`. This step effectively prepares the weight for quantization by scaling it to the range that the FP8 type can represent, given `self.scale_weight`.
        2. `comfy.float.stochastic_rounding` is called to convert this de-scaled weight into the layer's FP8 data type (`self.weight.dtype`).
        3. The resulting FP8 weight is stored in `self.weight`.
  * `forward_comfy_cast_weights(self, input)`:
    * If `fp8_matrix_mult` is `True` (an argument to the factory function), it attempts to use `fp8_linear`. `fp8_linear` will inherently use the `self.scale_weight` and `self.scale_input` from this layer instance.
    * If `fp8_linear` is not used (either `fp8_matrix_mult` is false or `fp8_linear` returned `None`), it performs a standard `torch.nn.functional.linear`. Crucially, before this operation, it scales either the FP8 `weight` or the `input` by `self.scale_weight`.
      * `weight * self.scale_weight.to(...)`: This step de-quantizes the FP8 weight on-the-fly to its effective higher-precision value before the matrix multiplication if `_scaled_mm` is not being used.

#### Mathematical Formulation (`scaled_fp8_ops.Linear`)

Let:

* $X_{orig}$ be the original input tensor (e.g., float16 or float32).
* $W_{f8}^{\text{stored}}$ be `self.weight`, the FP8 weight stored in the layer.
* $S_W$ be `self.scale_weight` (a float32 scalar Parameter of the layer).
* $S_X$ be `self.scale_input` (a float32 scalar Parameter, if `scale_input=True`).
* $B$ be the bias tensor.

**A. `set_weight(self, weight, ...)` Method:**
This method is called when loading higher-precision weights ($W_{hp}$, e.g., float32) into the layer.

1. **De-scaling for Quantization**: The goal is to find an FP8 tensor $W_{f8}^{\text{stored}}$ such that $W_{hp} \approx W_{f8}^{\text{stored}} \cdot S_W$.

    $$
    W_{quantizable} = \frac{W_{hp}}{S_W}
    $$

2. **Stochastic Rounding to FP8**:
    The `W_quantizable` value is then converted to the target FP8 data type using the stochastic rounding process. This can be represented as:
    `W_f8_stored = StochasticRoundToFP8(W_quantizable, target_fp8_dtype)`
    where `W_f8_stored` is the resulting FP8 weight, and `target_fp8_dtype` is the desired FP8 format (e.g., E4M3 or E5M2).

    This $W_{f8}^{\text{stored}}$ is then assigned to `self.weight`.

**B. `forward_comfy_cast_weights(self, input)` Method:**

* **Path 1: Using `fp8_linear` (if `fp8_matrix_mult=True`)**
  * This path is similar to the `fp8_linear` explanation above. `fp8_linear` will use `self.scale_weight` (which is $S_W$) and `self.scale_input` (which is $S_X$, if available) from this layer instance.
  * Input $X_{orig}$ is processed:
    * If $S_X$ exists: $A_{f8} = \text{CastToFP8}(X_{orig} / S_X)$. `scale_a` for `_scaled_mm` is $S_X$.
    * If $S_X$ is `None`: $A_{f8} = \text{CastToFP8}(\text{clamp}(X_{orig}, -448, 448))$. `scale_a` is $1.0$.
  * Weight $W_{f8}^{\text{stored}}$ is used. `scale_b` for `_scaled_mm` is $S_W$.
  * Output $O = ( (A_{f8} \cdot S_X) @ (W_{f8}^{\text{stored}} \cdot S_W)^T ) + B$ (conceptual view of `_scaled_mm`).

* **Path 2: Fallback (if `fp8_linear` is not used)**
    The code executes `torch.nn.functional.linear(input_arg, weight_arg, bias)`.
    Let `W_f8` be `self.weight`. The crucial part is how `input_arg` and `weight_arg` are formed:
    1. `weight_fp8_from_cast, bias_input_dtype = cast_bias_weight(self, input)`: Here `weight_fp8_from_cast` is `W_f8_stored`.
    2. If `weight_fp8_from_cast.numel() < input.numel()`:
        * First, the `weight_arg` for the `torch.nn.functional.linear` operation is determined. It is calculated by multiplying the stored FP8 weight (denoted `W_f8_stored`) with the layer's scale factor `S_W`. For this multiplication, `S_W` is first cast to an FP8 data type (denoted `S_W_casted_to_fp8`). The product of `W_f8_stored` and `S_W_casted_to_fp8` results in an FP8 tensor that serves as `weight_arg`.
        * The `input_arg` for the `torch.nn.functional.linear` operation is simply the original input tensor, `X_orig`.
        * Note: `S_W_casted_to_fp8` is equivalent to `CastToFP8(S_W)`.
        * The overall linear operation to get the output `O` is then performed as: `O = X_orig @ weight_arg.T + B_input_dtype` (where `@` denotes matrix multiplication and `.T` denotes transpose).
        * Substituting the expression for `weight_arg`, this step effectively computes: `O = X_orig @ (W_f8_stored * S_W_casted_to_fp8).T + B_input_dtype`.
    3. Else:
        * In this case, the `input_arg` for the `torch.nn.functional.linear` operation is modified. It is calculated by multiplying the original input tensor (`X_orig`) with the layer's scale factor `S_W`. For this multiplication, `S_W` is first cast to the data type of the original input (denoted `S_W_casted_to_input_dtype`). The product of `X_orig` and `S_W_casted_to_input_dtype` serves as `input_arg`.
        * The `weight_arg` for the `torch.nn.functional.linear` operation is the stored FP8 weight, `W_f8_stored`, used directly.
        * Note: `S_W_casted_to_input_dtype` is equivalent to `CastToInputDtype(S_W)`.
        * The overall linear operation to get the output `O` is then performed as: `O = input_arg @ W_f8_stored.T + B_input_dtype`.
        * Substituting the expression for `input_arg`, this step effectively computes: `O = (X_orig * S_W_casted_to_input_dtype) @ W_f8_stored.T + B_input_dtype`.

    **Note on Precision in Fallback Path**:
    Casting the float32 scale `S_W` to FP8 (as in `S_W_casted_to_fp8`) before multiplying with `W_f8_stored` means the de-quantization step itself is performed using low-precision arithmetic for the scale. This can lead to a loss of precision for the scaling factor `S_W` compared to if the product `W_f8_stored` * `S_W` was computed by first upcasting `W_f8_stored` to float32.
    When `S_W` is cast to `input.dtype` (higher precision) and multiplied with `input`, the scale maintains its precision, but it's applied to the activations instead of the weights. The choice between these two sub-paths is a heuristic based on tensor element counts.

## 2. Stochastic Rounding (`comfy/float.py`)

Converting numbers from higher precision (like float32 or float16) to FP8 involves quantization, which can introduce errors. Stochastic rounding is a technique that helps to make these errors unbiased on average.

### 2.1. `stochastic_rounding(value, dtype, seed=0)`

This is the main interface for applying stochastic rounding.

```python
# Relevant section from comfy/float.py
def stochastic_rounding(value, dtype, seed=0):
    if dtype == torch.float32: # Passthrough for non-FP8 common types
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
        
    # Handle FP8 types
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed) # For reproducibility
        output = torch.empty_like(value, dtype=dtype)
        
        # Process in slices for potentially large tensors to manage memory
        num_slices = max(1, (value.numel() / (4096 * 4096))) 
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(
                manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator)
            )
        return output

    return value.to(dtype=dtype) # Default cast for other types
```

### 2.2. `manual_stochastic_round_to_float8(x, dtype, generator=None)`

This function implements the core logic for FP8 conversion with stochastic rounding.

```python
# Relevant section from comfy/float.py
def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    # The 'sign' variable will be progressively modified:
    # 1. It starts as the sign of x.
    # 2. It's updated for the case where x is zero.
    # 3. It's multiplied by the reconstructed magnitude (using the updated abs_x).
    # 4. It's clamped in-place.
    # 5. It's returned as the result (still in x.half() dtype).
    sign = torch.sign(x)
    # The 'abs_x' variable is also modified:
    # 1. It starts as the absolute value of x.
    # 2. Its content is overwritten by the new fractional mantissa from calc_mantissa.
    abs_x = x.abs()
    # Update 'sign' for true zero, ensuring it's a tensor of the same dtype/device.
    sign = torch.where(abs_x == 0, torch.tensor(0.0, device=sign.device, dtype=sign.dtype), sign)

    # Calculate and clamp exponent to FP8 range, using the original abs_x value.
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    normal_mask = ~(exponent == 0) # Mask for normal vs. subnormal numbers

    # CRITICAL: abs_x is updated in-place.
    # The first argument to calc_mantissa is the original value of abs_x.
    # The result of calc_mantissa (new fractional mantissa) then overwrites the content of 'abs_x'.
    # This is done in the source code via:
    # abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)
    # After this line, 'abs_x' holds the new stochastically rounded fractional mantissa.
    # To reflect this for the snippet structure while maintaining clarity for the next step:
    _abs_x_original_value = abs_x.clone() # Value of abs_x before it's overwritten by calc_mantissa's result
    _new_fractional_mantissa = calc_mantissa(_abs_x_original_value, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)
    abs_x = _new_fractional_mantissa # In the source, this update happens via abs_x[:] = ...

    # Reconstruct the number: 'sign' (original sign of x) is multiplied by the reconstructed magnitude.
    # The 'abs_x' used in torch.where below now refers to the *updated* fractional mantissa.
    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x), # abs_x is the new fractional mantissa
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x                # abs_x is the new fractional mantissa
    )

    # Clamp 'sign' (which now holds the signed reconstructed value) to the FP8 representable range.
    # Note: original source uses 'inf = torch.finfo(dtype)'
    finfo = torch.finfo(dtype)
    torch.clamp(sign, min=finfo.min, max=finfo.max, out=sign) # In-place clamp
    
    # Returns 'sign'. It is still in x.half() dtype (e.g., float16).
    # The final cast to target FP8 dtype is handled by the calling `stochastic_rounding` function
    # when it copies this result into an FP8 tensor (e.g., output.copy_(result)).
    return sign
```

**Explanation of Stochastic Rounding:**

1. **Decomposition**: The input number `x` is decomposed into sign, exponent, and mantissa.
2. **Exponent Handling**: The exponent is calculated and clamped to fit within the target FP8 format's exponent bits.
3. **Mantissa Calculation (`calc_mantissa`)**:
    * The mantissa is isolated and scaled to an integer range (e.g., for 3 mantissa bits, scaled by `2^3=8`).
    * **Crucially, a random number uniformly distributed between 0 and 1 is added to this scaled mantissa.**
    * The result is then floored (truncated). This addition of random noise means that, on average, the rounding is not biased towards zero (like truncation) or towards the nearest even number.
    * The floored value is scaled back to a fractional mantissa.
4. **Reconstruction**: The number is rebuilt using the original sign, the clamped exponent, and the stochastically rounded mantissa.
5. **Clamping**: The final reconstructed number is clamped to the minimum and maximum representable values of the target FP8 `dtype`.

This process is applied by `scaled_fp8_ops.Linear.set_weight` when converting higher-precision weights to FP8 for storage.

```python
# calc_mantissa (called by manual_stochastic_round_to_float8
def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    # Scale to isolate mantissa and then scale to integer range
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 **(exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS), # Normal
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS))) # Subnormal
    )

    # STOCHASTIC PART: Add random noise [0, 1) before flooring
    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, 
                                  layout=mantissa_scaled.layout, device=mantissa_scaled.device, 
                                  generator=generator)
    # Floor and scale back to fractional mantissa
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)
```

#### Mathematical Formulation (`manual_stochastic_round_to_float8` & `calc_mantissa`)

Let:

* $x_{in}$ be the input value to `manual_stochastic_round_to_float8` (after `.half()` call, so it's effectively float16).
* $E_{bits}, M_{bits}, \text{Bias}$ be the parameters for the target FP8 format (e.g., E4M3: 4, 3, 7; E5M2: 5, 2, 15).
* $s = \text{sign}(x_{in})$.
* $x_{abs} = |x_{in}|$.

1. **Exponent Calculation**:
    * Unbiased exponent: $E_{unbiased} = \lfloor \log_2(x_{abs}) \rfloor$
    * Biased exponent: $E_{biased} = E_{unbiased} + \text{Bias}$
    * Clamped final exponent: $E_{f} = \text{clamp}(E_{biased}, 0, 2^{E_{bits}} - 1)$
        * The value $0$ for $E_f$ can signify zero or subnormal numbers.
        * The value $2^{E_{bits}} - 1$ can signify infinity or NaN.

2. **Mantissa Calculation (via `calc_mantissa(x_{abs}, E_f, \text{normal_mask}, M_{bits}, \text{Bias}, \text{generator})` returning $M_{new\\_frac}$)**:
    * `normal_mask` is true if $E_f \neq 0$ (simplified, actual hardware check is more complex for all special values).

    * **If Normal (`normal_mask` is true)**:
        * The value is $x_{abs} = (1.m_1m_2...m_k) \cdot 2^{E_f - \text{Bias}}$.
        * Normalized significand (should be in $[1, 2)$ ): $x_{significand} = x_{abs} / (2^{E_f - \text{Bias}})$.
        * Fractional part of significand: $m_{original\\_frac} = x_{significand} - 1.0$. This is in $[0, 1)$.
        * Scaled mantissa (integer part of the mantissa if directly represented):
            $$
            m_{scaled} = m_{original\\_frac} \cdot 2^{M_{bits}}
            $$
    * **If Subnormal (`normal_mask` is false, assuming $E_f = 0$ and $x_{abs} \neq 0$)**:
        * The value is $x_{abs} = (0.m_1m_2...m_k) \cdot 2^{E_{min}}$, where $E_{min} = 1 - \text{Bias}$.
        * The scaling factor for subnormals in `calc_mantissa` is $2^{(1 - \text{Bias} - M_{bits})} = 2^{E_{min} - M_{bits}}$.
            This factor represents the value of the least significant bit (LSB) of the smallest normal number's mantissa, scaled to the subnormal exponent.
        * Scaled mantissa for subnormals:
            $$
            m_{scaled} = x_{abs} / (2^{1 - \text{Bias} - M_{bits}})
            $$
            This effectively scales the subnormal value so that its bits align with the positions of the normal mantissa bits for the rounding operation.

    * **Stochastic Rounding Step (common for normal and subnormal paths after `m_scaled` calculation)**:
        * Generate a random number: $r \sim U(0, 1)$ using `generator`.
        * Add randomness:
            $$
            m_{stochastic} = m_{scaled} + r
            $$
        * Floor to get new integer mantissa bits:
            $$
            m_{final\\_int} = \lfloor m_{stochastic} \rfloor
            $$
            This $m_{final\\_int}$ will be an integer between $0$ and $2^{M_{bits}}$ (it can reach $2^{M_{bits}}$ if $m_{scaled} + r \ge 2^{M_{bits}}$, leading to rounding up, potentially carrying over to the exponent, which is handled by clamping later or by the reconstruction).
        * New fractional mantissa (for reconstruction):
            $$
            M_{new\\_frac} = m_{final\\_int} / 2^{M_{bits}}
            $$
            Note: `calc_mantissa` returns this $M_{new\\_frac}$.

3. **Reconstruction of the FP8 Value (in `manual_stochastic_round_to_float8` using $M_{new\\_frac}$ from `calc_mantissa`)**:
    * The variable `abs_x` (which initially held $|x_{in}|$) is updated in-place to store $M_{new\\_frac}$ (the stochastically rounded fractional mantissa from `calc_mantissa`).
    * The variable `sign` (which initially held $s = \text{sign}(x_{in})$) is then updated by multiplying it with the reconstructed magnitude. This effectively computes the signed reconstructed value:
        * **If Normal**:
            $$
            \text{sign}_{\text{variable}} = s \cdot (2^{E_f - \text{Bias}}) \cdot (1.0 + M_{new\\_frac})
            $$
        * **If Subnormal ($E_f = 0$)**:
            $$
            \text{sign}_{\text{variable}} = s \cdot (2^{1 - \text{Bias}}) \cdot M_{new\\_frac}
            $$
            (Here, $2^{1 - \text{Bias}}$ is $2^{E_{min}}$, the smallest normal exponent scale).
        Let $v_{signed\_reconstructed}$ be the value of the `sign` variable after this step.

4. **Final Clamping and Casting**:
    * The `sign` variable (now containing $v_{signed\_reconstructed}$) is clamped in-place to the min/max representable values of the target FP8 `dtype`.
        $$
        \text{sign}_{\text{variable\_clamped}} = \text{clamp}(v_{signed\_reconstructed}, \text{FP8}_{\\text{min_val}}, \text{FP8}_{\\text{max_val}})
        $$
    * The `manual_stochastic_round_to_float8` function returns this clamped `sign` variable. It is still of `x.half()` dtype (e.g., float16).
    * The final cast to the target FP8 `dtype` is performed by the calling `stochastic_rounding` function when it copies this returned result into an output tensor that is already initialized with the target FP8 `dtype`.
        $$
        x_{f8} = \text{CastToFP8}(\text{sign}_{\text{variable\_clamped}}) \quad \text{(conceptually, done by caller)}
        $$

This detailed breakdown shows how stochastic rounding introduces unbiased rounding during the quantization to FP8, which is critical for maintaining model accuracy when using `scaled_fp8_ops`.

## 3. Model Configuration and FP8 Selection

Several components work together to determine if and how FP8 operations should be used for a model.

### 3.1. Command-Line Arguments for FP8 Control

ComfyUI provides several command-line arguments to directly influence the use of FP8 data types for specific model components, overriding some of the automatic detection logic. These are particularly useful for experimentation or forcing specific configurations:

* **UNet Weights:**
  * `--fp8_e4m3fn-unet`: Forces UNet weights to be stored and used in `torch.float8_e4m3fn` format.
  * `--fp8_e5m2-unet`: Forces UNet weights to be stored and used in `torch.float8_e5m2` format.
  * `--fp8_e8m0fnu-unet`: Forces UNet weights to be stored and used in `torch.float8_e8m0fnu` format. (Note: While this type is recognized, core functions like `fp8_linear` are primarily optimized for `e4m3fn`.)

* **Text Encoder Weights:**
  * `--fp8_e4m3fn-text-enc`: Forces Text Encoder weights to be stored and used in `torch.float8_e4m3fn`.
  * `--fp8_e5m2-text-enc`: Forces Text Encoder weights to be stored and used in `torch.float8_e5m2`.

Using these arguments will generally lead to `unet_dtype()` or `text_encoder_dtype()` (in `comfy.model_management`) returning the specified FP8 type, which then influences `pick_operations`.

* **`comfy.model_management.unet_dtype(...)`**:
  * It chooses the appropriate set of layer operations (e.g., `fp8_ops`, `scaled_fp8_ops`, `manual_cast`) based on:
    * `weight_dtype` (from `unet_dtype`), `compute_dtype`.
    * `model_config.scaled_fp8`: If a model's configuration explicitly specifies an FP8 type for scaled operations.
    * `fp8_optimizations` flag from the model config.
    * Device FP8 compute support (`comfy.model_management.supports_fp8_compute`): This utility function checks for hardware and software prerequisites for efficient FP8 computation. Key checks include:
      * The device must be an NVIDIA GPU.
      * The GPU must have sufficient compute capability (typically Hopper architecture - compute capability 9.x, or Ada Lovelace - compute capability 8.9).
      * A compatible PyTorch version is required (generally 2.3+; on Windows, 2.4+ might be necessary for full support).
  * **Logic**:
    * If `model_config.scaled_fp8` is set (e.g., `torch.float8_e4m3fn`), it returns `scaled_fp8_ops` configured with this `override_dtype`.
      * The `fp8_matrix_mult` argument for `scaled_fp8_ops` (enabling `torch._scaled_mm` usage via `fp8_linear`) is activated if the device supports FP8 compute *and* the `fp8_optimizations` flag from the model config is `True`.
      * The `scale_input` argument for `scaled_fp8_ops` (allowing the layer to manage an input scaling factor) is enabled if `fp8_optimizations` is `True`.
    * Else, if general FP8 compute is supported *and* (`fp8_optimizations` is `True` or `PerformanceFeature.Fp8MatrixMultiplication` is forced via CLI), it returns `fp8_ops`.
    * Otherwise, it falls back to other op types like `cublas_ops` or `manual_cast`.

* **`comfy.model_base.BaseModel.__init__(...)`**:
  * When a UNet model (which inherits from `BaseModel`) is initialized:
        1. It retrieves `fp8 = model_config.optimizations.get("fp8", False)` and `scaled_fp8_dtype = model_config.scaled_fp8`.
        2. It calls `comfy.ops.pick_operations(...)` with these flags, the UNet's weight dtype, and manual cast dtype.
        3. The returned `operations` class (e.g., `fp8_ops` or `scaled_fp8_ops`) is then passed to the `UNetModel` constructor: `self.diffusion_model = unet_model(..., operations=operations)`.
        4. This ensures that `Linear` (and other) layers within the UNet are instantiated using the custom FP8-aware classes if selected.

* **`comfy.model_detection.model_config_from_unet(...)`**:
  * When loading a model's state dictionary:
    * It checks for a special key, e.g., `model.diffusion_model.scaled_fp8` (the prefix varies).
    * If this key exists:
      * The associated tensor's `dtype` is used to set `model_config.scaled_fp8` (e.g., `torch.float8_e4m3fn`). This informs `pick_operations` to use `scaled_fp8_ops`.
      * `model_config.optimizations["fp8"]` is set to `True`. (The condition `scaled_fp8_weight.nelement() == 2` seems to be a specific check, possibly for how some early FP8 models might have stored just scale values, but generally, its presence indicates FP8 usage).
    * The `"scaled_fp8"` key is then removed from the state dictionary before loading into the model.

* **Saving Models with FP8 Info**:
  * In `comfy.model_base.BaseModel.state_dict_for_saving`, if `self.model_config.scaled_fp8` is not `None` (meaning `scaled_fp8_ops` were likely used), an empty tensor with this FP8 dtype is saved into the state dictionary under the key `"scaled_fp8"` (prefixed appropriately). This acts as a marker for `model_detection` when the model is reloaded.

## 4. Overall FP8 Scaling Workflow

ComfyUI's FP8 implementation primarily targets `Linear` layers and uses two main strategies, selected by `pick_operations`:

**Scenario 1: Using `fp8_ops` (often with `fp8_linear` and `torch._scaled_mm`)**

* **Weights**: Expected to be already in `torch.float8_e4m3fn`.
* **Scales**: `scale_weight` and `scale_input` are attributes of the layer, typically loaded from a model checkpoint if the model was trained/converted to use this scheme. These are float32 scalars.
* **Activation Handling**: Input activations are either clamped (if `scale_input` is None) or scaled by `1.0 / scale_input`, then cast to `torch.float8_e4m3fn`.
* **Computation**: `torch._scaled_mm` is the preferred method.
  * `Output = ((Activation_fp8 * scale_input) @ (Weight_fp8 * scale_weight)) + Bias`
  * The result is cast back to the original activation data type.

**Scenario 2: Using `scaled_fp8_ops`**

* **Weights**: Stored in an FP8 format (e.g., `e4m3fn` or `e5m2`, determined by `override_dtype`). The layer's `set_weight` method handles the conversion from a higher precision to this FP8 format using `stochastic_rounding`.
* **Scales**:
  * `self.scale_weight`: A float32 `torch.nn.Parameter` (non-trainable) intrinsic to the layer. It represents the factor by which the FP8 weights should be multiplied to approximate their original higher-precision values.
    * During `set_weight(higher_prec_w, ...)`: `fp8_w = stochastic_round(higher_prec_w / self.scale_weight)`.
  * `self.scale_input`: Also a float32 Parameter if `scale_input=True` in the factory.
* **Computation**:
  * **If `fp8_matrix_mult` is True and `fp8_linear` is used**: Same as Scenario 1, `torch._scaled_mm` uses `self.scale_weight` and `self.scale_input`.
  * **If `fp8_linear` is NOT used (fallback or `fp8_matrix_mult=False`)**:
    * The FP8 weights (`weight`) are de-quantized on-the-fly before the standard `torch.nn.functional.linear` call:
            `Effective_Weight = weight * self.scale_weight.to(weight.dtype)`
    * `Output = (Activation @ Effective_Weight) + Bias` (or input is scaled if `weight.numel() >= input.numel()`).

## 5. Key Takeaways for Experimentation

Understanding this implementation allows for targeted experimentation:

* **Scaling Factor Derivation**: The origin and calculation of `scale_weight` and `scale_input` are critical. `scaled_fp8_ops` makes `scale_weight` an explicit layer parameter. You could explore different ways to determine/learn these scales.
* **Quantization Methods**: `comfy.float.stochastic_rounding` is the chosen method for `scaled_fp8_ops`. Alternative quantization-aware training or post-training quantization methods and different rounding schemes (e.g., round-to-nearest) could be tested.
* **Granularity**: The current system uses per-tensor scaling factors. More advanced techniques might involve per-channel or per-group scaling for weights or activations.
* **Activation Clamping/Scaling**: The clamping range `[-448, 448]` in `fp8_linear` for `e4m3fn` is a heuristic. The impact of this range, or dynamic calibration of activation scales, could be an area of study.
* **FP8 Formats**: While `torch.float8_e4m3fn` is prominent in `fp8_linear`, `scaled_fp8_ops` supports `torch.float8_e5m2` via `override_dtype` and `stochastic_rounding`. The choice between E4M3 (wider range, less precision) and E5M2 (narrower range, more precision) depends on the model and data.
* **Broader FP8 Type Awareness**: ComfyUI, through `comfy.model_management.get_supported_float8_types()`, can recognize a range of FP8 formats if supported by the underlying PyTorch installation. These include `torch.float8_e4m3fn`, `torch.float8_e4m3fnuz`, `torch.float8_e5m2`, `torch.float8_e5m2fnuz`, and `torch.float8_e8m0fnu`. While direct optimized paths like `fp8_linear` primarily target `e4m3fn`, the `scaled_fp8_ops` with `override_dtype` and custom nodes can potentially leverage these other types.
* **Broader Layer Support**: The current deep FP8 integration is mainly for `Linear` layers. Extending similar scaled operations to Convolutional layers would be a significant undertaking.

This framework provides a flexible way to integrate and utilize FP8 precision in ComfyUI, balancing performance gains with the complexities of low-precision numerical representation.

## 6. Creating Custom Nodes for Advanced FP8 Scaling Strategies

While ComfyUI provides robust FP8 integration through `fp8_ops` and `scaled_fp8_ops`, users might want to experiment with novel FP8 scaling methodologies or quantization techniques. Custom ComfyUI nodes offer a powerful way to implement and test such custom strategies.

This section outlines how to build a custom node that can modify a model's layers to use custom-derived FP8 scales and re-quantize weights accordingly.

### 6.1. Purpose of a Custom FP8 Scaling Node

A custom FP8 scaling node allows users to:

* Implement and apply various algorithms for calculating `scale_weight` (and potentially `scale_input`) for model layers (e.g., `Linear` layers). Examples include statistics-based scaling (min/max, percentile), or more complex heuristics.
* Experiment with different FP8 target data types (`torch.float8_e4m3fn`, `torch.float8_e5m2`).
* Potentially integrate custom quantization functions beyond the default `comfy.float.stochastic_rounding`.
* Iteratively refine scaling strategies by observing their impact on model output quality and performance.

### 6.2. Conceptual Structure of the Custom Node

A typical custom node for this purpose (`CustomFP8ScalerNode.py`) would have the following structure:

* **`INPUT_TYPES`**:
  * `model`: The input model (`MODEL` type).
  * `scaling_method`: A string or dropdown to select the custom scaling algorithm (e.g., "custom_absmax", "percentile_scale").
  * `target_fp8_dtype`: A string or dropdown for the target FP8 format (e.g., "torch.float8_e4m3fn").
  * Additional parameters specific to the chosen scaling methods (e.g., percentile value, clamping thresholds).
* **`RETURN_TYPES`**:
  * `model`: The modified model (`MODEL` type).
* **`FUNCTION` (e.g., `apply_custom_scaling`)**:
  * This method contains the core logic.

### 6.3. Core Logic within the Node's Function

1. **Iterate Through Model Layers**: Access the underlying PyTorch model (e.g., `model.model`) and iterate through its layers using `named_modules()`.
2. **Identify Target Layers**: Filter for layers intended for FP8 conversion, typically `torch.nn.Linear` or existing FP8-aware layers like `comfy.ops.scaled_fp8_ops.Linear`.
3. **Ensure Layer Compatibility**:
    * The target layer must be of a type that can utilize `self.scale_weight` and an FP8 `self.weight` in its forward pass. The `comfy.ops.scaled_fp8_ops.Linear` class is designed for this.
    * If a layer (e.g., a standard `torch.nn.Linear`) is not already of a compatible type, the custom node may need to **replace it** with an instance of a compatible class (e.g., instantiate a `scaled_fp8_ops.Linear` with the original layer's dimensions and weights, configured for the `target_fp8_dtype`). This is a critical step for ensuring the scaling is effective.
4. **Access/Prepare High-Precision Weights**:
    * Obtain the weights of the layer in a high-precision format (e.g., FP32, FP16).
    * If the layer's weights are already in FP8 (e.g., if it's a `scaled_fp8_ops.Linear` that was previously processed), they must first be de-quantized: `hp_weight = layer.weight.float() * layer.scale_weight.float()`.
    * If the layer's weights are already in a suitable high precision (e.g. FP16, BFloat16, FP32), they can be used directly after cloning.
5. **Calculate Custom `scale_weight`**:
    * Implement one or more Python functions that take the high-precision weights and any method-specific parameters, returning the calculated `scale_weight` (as a float32 scalar tensor).
    * Example: `new_scale = my_absmax_scaling_function(hp_weight, threshold=0.99)`.
6. **Apply `scale_weight` to Layer**:
    * Set the calculated scale on the layer: `layer.scale_weight = torch.nn.parameter.Parameter(new_scale, requires_grad=False)`. (Ensure `scale_weight` is a `Parameter` if the ops class expects it).
    * If the custom method also determines `scale_input`, set `layer.scale_input` similarly.
7. **Re-Quantize Layer Weights**:
    * Prepare weights for quantization: `weight_to_quantize = hp_weight / layer.scale_weight.to(hp_weight.device, hp_weight.dtype)`.
    * Convert to the target FP8 dtype using `comfy.float.stochastic_rounding` (or a custom quantization function):
        `fp8_w = comfy.float.stochastic_rounding(weight_to_quantize, target_fp8_dtype, seed=...)`.
    * Update the layer's weight parameter: `layer.weight.data.copy_(fp8_w)`.
8. **Return Modified Model**: The node returns the `model` object containing the modified layers.

### 6.4. Example Snippet for a Custom Scaling Function

```python
# In your custom node file or an imported utility
import torch

def calculate_percentile_scale(weights_hp, percentile=99.9):
    if weights_hp.numel() == 0:
        return torch.tensor(1.0, device=weights_hp.device, dtype=torch.float32)
    
    flat_abs_weights = torch.abs(weights_hp.float()).flatten()
    if flat_abs_weights.numel() == 0: # Should be caught by weights_hp.numel() but good for safety
        return torch.tensor(1.0, device=weights_hp.device, dtype=torch.float32)
        
    k = int(round((percentile / 100.0) * flat_abs_weights.numel()))
    k = min(k, flat_abs_weights.numel() -1) # Ensure k is a valid index
    k = max(k, 0) # Ensure k is non-negative

    value_at_percentile = torch.kthvalue(flat_abs_weights, k + 1).values 
    
    # Ensure scale is not zero, and provide a sensible default
    return value_at_percentile if value_at_percentile > 1e-6 else torch.tensor(1.0, device=weights_hp.device, dtype=torch.float32)

```

### 6.5. Important Considerations

* **Layer Replacement**: The most robust way to ensure compatibility is often to replace targeted `torch.nn.Linear` layers with instances of `comfy.ops.scaled_fp8_ops(override_dtype=target_fp8_dtype).Linear` (or a custom derivative). The custom scaling and quantization are then applied to this new layer instance. This ensures the forward pass correctly utilizes `scale_weight`.
* **Accessing Original Weights**: Carefully manage how original high-precision weights are obtained, especially if the model might have undergone previous transformations.
* **`comfy.ops.pick_operations`**: While this node focuses on modifying layers directly, be aware that ComfyUI uses `pick_operations` during model loading to choose ops classes. Advanced users might explore creating entirely new ops classes that integrate their scaling directly and having `pick_operations` select them based on model config flags set by another custom node.
* **Seed Management**: For reproducible quantization with `stochastic_rounding`, manage the `seed` parameter consistently.
* **Performance**: Extensive Python iteration over layers can be slow for very large models.

This approach provides a flexible framework for research and experimentation with FP8 scaling directly within the ComfyUI environment.
