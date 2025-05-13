# TorchAO Integration for Xavier Project

This document provides an overview of `torchao` with a focus on its FP8 quantization capabilities and how they can be integrated into the `xavier.py` script. `torchao` is a PyTorch-native library for optimizing models using techniques like quantization and sparsity.

## Core Concepts in `torchao`

`torchao` aims to make models faster and smaller. Key components relevant to `xavier.py` include:

1. **Quantization API (`torchao.quantization`):**
    * **`quantize_(model, config, ...)`:** The main function to apply quantization to an `nn.Module`. It takes a model and a configuration object.
    * **`autoquant(model, ...)`:** An automated quantization tool that attempts to find optimal quantization strategies for different parts of a model.
    * **Configuration Objects (e.g., `Float8WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`):** These dataclasses define the parameters for a specific quantization scheme (target dtype, granularity, layout, etc.).
    * **Convenience Functions (e.g., `float8_weight_only(model, ...)`):** Wrappers around `quantize_` for common configurations.

2. **FP8 Support (`torchao.float8` and `torchao.dtypes`):**
    * `torchao` provides extensive support for FP8 (E4M3 and E5M2) data types.
    * **Scaling:** Dynamic scaling is a core part of its FP8 implementation. Scales are typically derived from the `amax` (absolute maximum) of the tensor, similar to the "owlscale" method in `xavier.py`.
        * `torchao.float8.float8_utils.tensor_to_scale(...)`: Computes scaling factor.
        * `torchao.float8.float8_scaling_utils.hp_tensor_to_float8_dynamic(...)`: Converts a high-precision tensor to a `Float8Tensor` with dynamic scaling.
    * **Quantization Primitives:**
        * `torchao.dtypes.to_affine_quantized_floatx(...)`: A low-level function that converts a float tensor to an "affine quantized floatx" tensor (which includes FP8). It takes the input tensor, mapping type (symmetric/asymmetric), target FP8 dtype, scale, block size (for scaling granularity), etc.
        * `torchao.quantization.quant_primitives.choose_qparams_affine_floatx(...)`: Calculates the scale and zero-point for floatx types.
        * `torchao.quantization.quant_primitives.quantize_affine_floatx(...)`: Applies the actual quantization using the chosen parameters.
    * **FP8 Linear Layers (`torchao.float8.float8_linear.Float8Linear`):** When quantizing both weights and activations to FP8 (e.g., using `Float8DynamicActivationFloat8WeightConfig`), `torchao` can replace existing `nn.Linear` layers with its own `Float8Linear` layer, which is designed to perform FP8 matrix multiplications.

3. **Data Types and Layouts (`torchao.dtypes`):**
    * `AffineQuantizedTensor`: A tensor subclass used to represent various quantized tensors, including those produced by FP8 quantization. It wraps the quantized integer/float8 data along with scales and zero-points.
    * `Layouts` (e.g., `PlainLayout`, `Float8Layout`): Define how the quantized data is stored and processed.

## Key FP8 Quantization Configurations

For `xavier.py`, the following configurations from `torchao.quantization.quant_api` are most relevant:

1. **`Float8WeightOnlyConfig`:**
    * **Purpose:** Quantizes only the weights of linear layers to FP8. Activations remain in higher precision. The matrix multiplication itself might still happen in the original precision after dequantizing weights on-the-fly, unless further compiled with specific kernel support.
    * **Key Parameters:**
        * `weight_dtype`: `torch.float8_e4m3fn` or `torch.float8_e5m2`.
    * **Usage:** `quantize_(model, Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn))` or the convenience function `float8_weight_only(model, weight_dtype=torch.float8_e4m3fn)`.

2. **`Float8DynamicActivationFloat8WeightConfig`:**
    * **Purpose:** Quantizes both weights and activations to FP8 dynamically. This is more likely to leverage actual FP8 GEMM kernels for speedup if the hardware and compiler (e.g., Inductor) support it.
    * **Key Parameters:**
        * `activation_dtype`: Target FP8 type for activations.
        * `weight_dtype`: Target FP8 type for weights.
        * `granularity`: Scaling granularity (e.g., `PerTensor()`, `PerRow()`). Can be specified separately for activations and weights.
        * `mm_config`: `Float8MMConfig` to control aspects of the FP8 matrix multiplication, like using fast accumulation.
    * **Usage:** `quantize_(model, Float8DynamicActivationFloat8WeightConfig(...))` or `float8_dynamic_activation_float8_weight(model, ...)`.
    * **Behavior:** This typically replaces `nn.Linear` modules with `torchao.float8.float8_linear.Float8Linear`.

## How `torchao` Handles Scaling

`torchao`'s FP8 quantization intrinsically handles scaling.

* For dynamic quantization, the scale is usually calculated as: `scale = torch.finfo(fp8_dtype).max / torch.max(torch.abs(input_tensor))`. This is applied per block as defined by the `block_size` or `granularity` parameter.
* The `block_size` in `to_affine_quantized_floatx` and the `granularity` in higher-level configs determine if scaling is per-tensor, per-channel/row, or per-group.
* The calculated scales are stored within the `AffineQuantizedTensor` subclass (usually in its `tensor_impl` attribute).

## Stochastic Rounding

The current `xavier.py` script implements custom stochastic rounding. `torchao`'s default quantization primitives (`quantize_affine_floatx`, etc.) typically use deterministic rounding (round-to-nearest-even when converting to the target format after scaling).

If stochastic rounding remains a critical requirement, you would need to:

1. **Option A (Not directly supported by `torchao`'s high-level API):** Manually use `torchao`'s scale calculation (`choose_qparams_affine_floatx`) and then apply your existing stochastic rounding logic to the scaled tensor before casting to the FP8 dtype. This would bypass `torchao`'s `quantize_affine_floatx`.
2. **Option B (Potentially extending `torchao`):** Modify or create custom quantization primitives within `torchao` to incorporate stochastic rounding. This is a more advanced approach.
3. **Option C (Evaluate necessity):** Assess if `torchao`'s deterministic rounding (with its optimized kernels and `torch.compile` compatibility) provides sufficient accuracy and performance, potentially negating the need for custom stochastic rounding. The benefits of `torchao`'s integration with `torch.compile` might outweigh the nuanced benefits of stochastic rounding for some models.

## Integration Plan for `xavier.py`

Here's a suggested approach to integrate `torchao` into `xavier.py`:

### 1. Argument Parsing

* **Add `torchao` specific arguments:**
  * `--quant_method` (e.g., choices: `torchao_fp8_weight_only`, `torchao_fp8_dynamic_act_weight`).
  * Keep `--fp8_type` (`e4m3`, `e5m2`) as this will be passed to `torchao` configs.
  * The existing `--complex_rounding`, `--shifturb`, `--owlshift` might become irrelevant if using `torchao`'s core quantization, unless you opt for Option A above for stochastic rounding.
  * `--owlscale`'s direct functionality will be replaced by `torchao`'s internal scaling. However, the *concept* of per-tensor max-abs scaling is what `torchao` uses by default (for per-tensor granularity).
* **Deprecate or adapt existing rounding/scaling flags** when a `torchao` method is chosen.

### 2. Model Loading and Preparation

* No significant changes here, but instead of iterating through tensors and quantizing them individually with custom logic, you will apply `torchao.quantization.quantize_` (or a convenience wrapper) to entire `nn.Module` instances or to the whole model if appropriate.
* **Challenge:** `xavier.py` currently loads tensors one by one from a `.safetensors` file and quantizes them. `torchao.quantization.quantize_` expects an `nn.Module`.
  * **Solution 1 (Per-Tensor Quantization with `torchao` primitives):**
        If you must continue quantizing tensors individually (e.g., because you don't have the full model definition as an `nn.Module` at quantization time, or only want to quantize weights without affecting module structure initially):
    * You would not use `torchao.quantization.quantize_` directly.
    * Instead, for each tensor, you would use `torchao.dtypes.to_affine_quantized_floatx`.
    * You'd need to define the `block_size` for scaling (e.g., `(1, tensor.numel())` for per-tensor scaling).
    * You'd select a `Layout`, e.g., `torchao.dtypes.Float8Layout(ebits=4, mbits=3)` for E4M3.
    * The result of `to_affine_quantized_floatx` is an `AffineQuantizedTensor`. You'd store its `tensor_impl.data` (the quantized FP8 values) and `tensor_impl.scale`.
  * **Solution 2 (Wrapping tensors in a dummy module for `quantize_`):**
        This is more aligned with `torchao`'s intended use if you want to leverage its higher-level APIs like `Float8WeightOnlyConfig`.

        ```python
        class DummyModule(torch.nn.Module):
            def __init__(self, weight_tensor, bias_tensor=None):
                super().__init__()
                self.linear = torch.nn.Linear(weight_tensor.shape[1], weight_tensor.shape[0], bias=(bias_tensor is not None))
                self.linear.weight.data = weight_tensor
                if bias_tensor is not None:
                    self.linear.bias.data = bias_tensor

            # No forward needed if only quantizing weights and not replacing module
        ```

        Then, for each layer from the safetensors file, create `DummyModule(weight, bias)`, apply `torchao.quantization.quantize_(dummy_module, config)`, and extract the quantized weight (and bias) from `dummy_module.linear.weight`.

### 3. Quantization Logic (Main Loop)

* **Replace `stochastic_round_tensor_to_fp8` and Owlscale logic.**

* **If using Solution 1 (Per-Tensor with `to_affine_quantized_floatx`):**

    ```python
    # Inside the loop for each tensor_key
    # from torchao.dtypes import to_affine_quantized_floatx, Float8Layout
    # from torchao.quantization.quant_primitives import MappingType

    # ... load tensor_to_process ...
    fp8_torch_dtype = torch.float8_e4m3fn if args.fp8_type == "e4m3" else torch.float8_e5m2
    ebits = 4 if args.fp8_type == "e4m3" else 5
    mbits = 3 if args.fp8_type == "e4m3" else 2

    # Determine block_size for scaling (e.g., per-tensor)
    # For per-tensor scaling of a 2D weight matrix:
    # block_size = tensor_to_process.shape
    # Or more generally for per-tensor: block_size = tuple(1 for _ in tensor_to_process.shape) then ensure it's broadcastable
    # For simple per-tensor, it's often (1, tensor_to_process.numel()) if flattened, or handled by choose_qparams_affine_floatx if block_size is entire shape.
    # Let's assume per-tensor scaling for simplicity here. choose_qparams_affine_floatx handles this if block_size matches tensor shape.
    block_size = tensor_to_process.shape # For per-tensor scale

    layout = Float8Layout(ebits=ebits, mbits=mbits) # Or other layouts if needed

    # This will calculate scale internally using choose_qparams_affine_floatx
    # and then quantize using quantize_affine_floatx.
    # mapping_type needs to be chosen, SYMMETRIC is common for weights.
    aq_tensor = to_affine_quantized_floatx(
        input_float=tensor_to_process.to(torch.float32), # Ensure consistent input dtype for qparam selection
        mapping_type=MappingType.SYMMETRIC, # Or ASYMMETRIC
        block_size=block_size,
        target_dtype=fp8_torch_dtype,
        _layout=layout,
        # scale_dtype can be specified if needed, defaults usually fine
    )

    quantized_tensor = aq_tensor.tensor_impl.data # This is the FP8 data
    scale_factor_for_comfyui_to_save = aq_tensor.tensor_impl.scale

    # Store quantized_tensor and scale_factor_for_comfyui_to_save
    # ...
    ```

  * **Stochastic Rounding with this approach:** After `choose_qparams_affine_floatx` (which `to_affine_quantized_floatx` calls internally to get the scale) and before `quantize_affine_floatx` (which `to_affine_quantized_floatx` also calls), you'd need to intercept. This means you can't use `to_affine_quantized_floatx` directly.
        You'd do:
        1. `scale = choose_qparams_affine_floatx(...)`
        2. `scaled_tensor = tensor_to_process / scale`
        3. `stochastically_rounded_scaled_tensor = your_stochastic_round_function(scaled_tensor)`
        4. `quantized_tensor = stochastically_rounded_scaled_tensor.to(fp8_torch_dtype)` (with saturation if needed, see `torchao.float8.float8_utils.to_fp8_saturated`)

* **If using Solution 2 (Dummy Module with `quantize_`):**
    This is preferable if you want to use `Float8WeightOnlyConfig` or `Float8DynamicActivationFloat8WeightConfig` as they handle module transformations.

    ```python
    # from torchao.quantization import quantize_, Float8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig

    fp8_torch_dtype = torch.float8_e4m3fn if args.fp8_type == "e4m3" else torch.float8_e5m2

    # Example for Float8WeightOnlyConfig
    if args.quant_method == "torchao_fp8_weight_only":
        config = Float8WeightOnlyConfig(weight_dtype=fp8_torch_dtype)
    elif args.quant_method == "torchao_fp8_dynamic_act_weight":
        # Per-tensor granularity example
        from torchao.quantization.granularity import PerTensor
        config = Float8DynamicActivationFloat8WeightConfig(
            activation_dtype=fp8_torch_dtype,
            weight_dtype=fp8_torch_dtype,
            granularity=PerTensor() # Or PerRow(), or tuple for different act/weight
        )
    else:
        # ... handle other methods or raise error

    # Create dummy module for each layer
    # Assuming 'tensor' is the weight tensor for the current key
    # and 'bias_tensor' might exist for some keys
    dummy_model = DummyModule(tensor_to_process, bias_tensor_if_any).to(main_device)
    quantize_(dummy_model, config)

    # Extract the quantized weight. The weight is now an AffineQuantizedTensor.
    quantized_weight_aqt = dummy_model.linear.weight
    quantized_tensor = quantized_weight_aqt.tensor_impl.data
    scale_factor_for_comfyui_to_save = quantized_weight_aqt.tensor_impl.scale
    # Handle bias similarly if quantized

    # Store quantized_tensor and scale_factor_for_comfyui_to_save
    # ...
    ```

  * **Stochastic Rounding with this approach:** `quantize_` applies transformations that internally call `to_affine_quantized_floatx`. Modifying this for stochastic rounding would mean altering `torchao`'s internal transform handlers or primitives, which is complex. It's less straightforward than Option A for per-tensor.

### 4. Saving Output and ComfyUI Compatibility

* The `AffineQuantizedTensor` (if you get one directly) or the underlying data and scale (if extracted) needs to be saved.
* **Scale Factor:** `torchao` provides the scale. Your current logic for deriving `_scale_key_to_use_` and saving `scale_factor_for_comfyui_to_save` can be adapted. The scale from `torchao` (e.g., `aq_tensor.tensor_impl.scale`) should be used.
* **FP8 Marker:** The `scaled_fp8` marker key should still be added if `owlscale`-like behavior (where scales are saved separately) is achieved. `torchao`'s `Float8WeightOnlyConfig` and `Float8DynamicActivationFloat8WeightConfig` manage scales alongside the quantized weights, so the concept aligns.

### 5. Plotting

* Your plotting logic can largely remain the same.
* You'll need to dequantize the `torchao`-quantized tensor to its original precision for comparison. `AffineQuantizedTensor` has a `.dequantize()` method.

    ```python
    # If you have an AffineQuantizedTensor instance:
    # dequantized_tensor_cpu_for_plot = quantized_weight_aqt.dequantize().cpu()
    # If you only have the int_data and scale:
    # from torchao.quantization.quant_primitives import dequantize_affine_floatx
    # dequantized_tensor_cpu_for_plot = dequantize_affine_floatx(quantized_tensor.cpu(), scale_factor_for_comfyui_to_save.cpu(), ebits, mbits, output_dtype=original_dtype).
    ```

### Example: Choosing `torchao_fp8_weight_only`

1. User specifies `--quant_method torchao_fp8_weight_only --fp8_type e4m3`.
2. In `main()`, you detect this.
3. For each tensor that matches `keys_to_quantize_suffix`:
    a.  Load the float tensor (`tensor_to_process`).
    b.  (If using Solution 2) Create `dummy_model = DummyModule(tensor_to_process, ...).to(main_device)`.
    c.  `fp8_dtype = torch.float8_e4m3fn`.
    d.  `config = Float8WeightOnlyConfig(weight_dtype=fp8_dtype)`.
    e.  `quantize_(dummy_model, config)`.
    f.  `quantized_weight_aqt = dummy_model.linear.weight`.
    g.  `quantized_fp8_data = quantized_weight_aqt.tensor_impl.data`.
    h.  `scale = quantized_weight_aqt.tensor_impl.scale`.
    i.  Save `quantized_fp8_data` under the original key.
    j.  Save `scale` under the derived `_scale_key_to_use_`.
4. Add the `scaled_fp8` marker to `quantized_state_dict`.

## Summary of `torchao` Benefits for `xavier.py`

* **Standardization:** Uses a library increasingly adopted in the PyTorch ecosystem.
* **Potential for Optimized Kernels:** Especially when using configurations like `Float8DynamicActivationFloat8WeightConfig` and `torch.compile`, `torchao` can enable highly optimized FP8 computations.
* **Support for Various Granularities:** Easily switch between per-tensor, per-row/channel, or per-group scaling.
* **Maintained Codebase:** Benefits from ongoing development and bug fixes from the PyTorch team and community.

The primary challenge will be adapting the per-tensor processing flow of `xavier.py` to `torchao`'s module-centric `quantize_` API (Solution 2) or carefully using its lower-level primitives for per-tensor quantization (Solution 1). If stochastic rounding is non-negotiable, Solution 1 with manual scale application and custom rounding is more feasible but misses out on some of `torchao`'s higher-level abstractions and potential kernel fusions.
