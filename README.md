# Xavier FP8 Quantization 🧠

`xavier.py` is a Python script designed to quantize `.safetensors` model files to FP8 (8-bit floating-point) precision. It supports its own suite of stochastic rounding techniques and can also leverage the `torchao` library for optimized quantization.

Stochastic rounding, as opposed to deterministic methods like Round-to-Nearest-Even (RNE), introduces a controlled amount of noise during quantization. This can help preserve model performance by ensuring that rounding errors do not systematically bias values. `torchao` typically uses deterministic rounding but benefits from optimized kernels and integration with `torch.compile`.

The script supports multiple FP8 formats and, for its native methods, several distinct stochastic rounding algorithms, along with pre-processing options like per-tensor scaling.

## Key Features

* **FP8 Quantization**: Converts tensors (typically FP32, FP16, or BF16) to FP8.
* **Multiple Quantization Backends**:
  * **Native Methods**: Employs different stochastic rounding techniques to probabilistically round values.
  * **TorchAO Integration**: Leverages the `torchao` library for quantization, offering access to potentially more optimized routines and `torch.compile` compatibility.
* **Stochastic Rounding (Native Methods)**: See details below for specific algorithms.
* **Per-Tensor Max-Absolute Scaling (comfyscale - Native Method)**:
  * A pre-processing step that scales each tensor to map its maximum absolute value into the representable range of the target FP8 format.
  * When used with native methods, it saves dequantization scales alongside the quantized tensors, compatible with ComfyUI-style scaled FP8 loading.
* **TorchAO Scaling**: When using `torchao`, scaling is handled internally by the library, typically based on `amax` (absolute maximum) of the tensor or tensor groups.
* **Target FP8 Formats**: `e4m3` (`torch.float8_e4m3fn`) and `e5m2` (`torch.float8_e5m2`).
* **Selective Quantization**: Allows specifying which tensors to quantize based on key name suffixes.
* **Comparison Plotting**: Generates plots comparing original, quantized, and dequantized tensor distributions and values (requires `matplotlib`).
* **Debug Mode**: Provides verbose output for tracing execution flow.

## Native Stochastic Rounding Techniques

These methods are used when `--quant_method native` (default) is selected.

* **Default Stochastic Rounding**:
  * This method first determines two FP8-representable candidates, $x_L$ (low) and $x_H$ (high), that are likely to bracket the input value $v$.
  * One candidate is obtained by PyTorch's default Round-to-Nearest-Even (RNE) cast of $v$ to the target FP8 type, let's call this $v_{RNE}$.
  * The other candidate is its FP8 neighbor, $v_{neighbor}$, found in the direction of $v$ relative to $v_{RNE}$. This neighbor search is an approximation.
  * So, $x_L = \min(v_{RNE}, v_{neighbor})$ and $x_H = \max(v_{RNE}, v_{neighbor})$.
  * The probability of rounding to the higher candidate $x_H$ is proportional to the input value's position between these two candidates:
        $$ P(\text{round to } x_H) = \frac{v - x_L}{x_H - x_L} $$
  * If $x_H = x_L$ (e.g., $v$ is exactly representable or at the boundary of the FP8 range), the probability is clamped to ensure deterministic rounding to that value. A random number $u \sim U[0,1]$ is drawn; if $u < P(\text{round to } x_H)$, $v$ is rounded to $x_H$, otherwise to $x_L$.
* **Complex Neighbor Stochastic Rounding** (`--complex_rounding`):
  * This method uses a more sophisticated approach (`get_fp8_bracketing_candidates_complex`) to find two FP8-representable values, $x_L$ and $x_H$, that strictly bracket the input tensor values $v$.
  * It considers three cases for each element in $v$:
        1. If $v > v_{RNE}$: $x_L = v_{RNE}$, and $x_H$ is the next FP8 value greater than $v_{RNE}$.
        2. If $v < v_{RNE}$: $x_H = v_{RNE}$, and $x_L$ is the next FP8 value smaller than $v_{RNE}$.
        3. If $v = v_{RNE}$: $x_L = v_{RNE}$, and $x_H$ is the next FP8 value greater than $v_{RNE}$ (by convention).
  * The `torch.nextafter` function is used (approximated on the FP8 grid by casting back and forth from the original precision) to find these neighbors.
  * The probabilistic choice between $x_L$ and $x_H$ is the same as the default method: $P(\text{round to } x_H) = \frac{v - x_L}{x_H - x_L}$
* **Shift-and-Perturb (Shifturb)** (`--shifturb`):
  * This method implements stochastic rounding by adding carefully scaled uniform random noise to the input tensor *before* quantizing it with standard Round-to-Nearest-Even (RNE).
  * **Shift Implementation**:
        1. First, it determines the bracketing FP8 candidates $x_L$ and $x_H$ for the input value $v$ (typically using the `get_fp8_bracketing_candidates_complex` method for a good local estimate).
        2. The difference $\Delta = x_H - x_L$ approximates the quantization step size around $v$.
        3. Uniform random noise $n$ is generated from the distribution $U[-\Delta/2, +\Delta/2]$.
        4. The perturbed input is $v' = v + n$.
        5. The final FP8 value is obtained by applying RNE to the perturbed value: $v_{FP8} = \text{RNE}(v')$.
  * The idea is that adding noise centered at zero with a range equal to the quantization step effectively randomizes which side of the rounding boundary the value falls on, simulating stochastic rounding.
* **Owlshift** (`--owlshift`):
  * This method directly implements stochastic rounding by manipulating the mantissa bits of the floating-point numbers. It typically operates on an intermediate `.half()` (FP16) representation of the input.
  * For a given input value $v$:
        1. The sign, exponent ($e$), and mantissa ($m$) are extracted.
        2. The mantissa is scaled based on the target FP8 format's mantissa bits (MANTISSA_BITS). For normal numbers, this involves taking the absolute value of the input `v`, dividing it by 2 raised to the power of (the exponent `e` minus the `EXPONENT_BIAS`), subtracting 1.0 (to isolate the fractional part of the mantissa), and then multiplying by 2 raised to the power of `MANTISSA_BITS` (the number of mantissa bits in the target FP8 format). This effectively scales the mantissa to an integer range suitable for stochastic addition.
            A similar calculation is done for subnormal numbers.
        3. A uniform random number $u \sim U[0,1)$ is added to $m_{\mathrm{scaled}}$: $m_{\mathrm{stoch}} = \lfloor m_{\mathrm{scaled}} + u \rfloor$.
        4. The stochastically rounded integer mantissa, `m_stoch`, is then scaled back to its fractional form by dividing it by 2 raised to the power of `MANTISSA_BITS`. This `m_final` is then used to reconstruct the number.
        5. The number is then reconstructed from the sign, original exponent, and the new stochastically rounded mantissa, and finally cast to the target FP8 type.
  * This method includes logic for handling normal and subnormal numbers, and tensor slicing for large tensors to manage memory/computation. It uses a dedicated random number generator seeded by the `--seed` argument.

## TorchAO Quantization

When a `--quant_method` starting with `torchao_` is selected (e.g., `torchao_fp8_weight_only_aoscale`), `xavier.py` uses the `torchao` library for quantization.

* **Mechanism**: `torchao` operates on `nn.Module` instances. `xavier.py` wraps individual tensors (typically weights from linear layers) in a temporary `DummyModule` to make them compatible with `torchao`'s `quantize_` API.
* **Scaling**:
  * **`_aoscale` variants (e.g., `torchao_fp8_weight_only_aoscale`)**: `torchao` handles scaling internally. For FP8, this is generally dynamic scaling based on `amax` (absolute maximum value of the tensor or sub-groups of the tensor). The scale is calculated typically as `scale_ao = torch.finfo(fp8_dtype).max / amax(original_tensor)`. The quantized tensor `fp8_tensor_ao` then represents `original_tensor / scale_ao`. Both `fp8_tensor_ao` and `scale_ao` are saved. This is TorchAO's native scaling convention.
  * **`_comfyscale` variants (e.g., `torchao_fp8_weight_only_comfyscale`)**: After `torchao` performs its quantization (obtaining `fp8_tensor_ao` and `scale_ao`), `xavier.py` adapts these to be compatible with ComfyUI's "comfyscale" convention.
    * The ComfyUI-compatible scale is calculated as `scale_comfy = scale_ao * torch.finfo(fp8_dtype).max`. This `scale_comfy` is equivalent to `amax(original_tensor)`.
    * The ComfyUI-compatible FP8 tensor is calculated as `fp8_tensor_comfy = to_fp8_saturated(fp8_tensor_ao.float() / torch.finfo(fp8_dtype).max, target_dtype=fp8_dtype)`. This `fp8_tensor_comfy` now represents `original_tensor / scale_comfy` (i.e., values normalized to `[-1, 1]` before FP8 conversion).
    * Both `fp8_tensor_comfy` and `scale_comfy` are saved. This ensures that the saved FP8 tensor and scale have the same semantic meaning as those produced by the native `--comfyscale` method.
* **Rounding**: `torchao` typically uses deterministic rounding (e.g., Round-to-Nearest-Even) after scaling. For `_comfyscale` variants, `torchao.float8.float8_utils.to_fp8_saturated` is used for the final conversion, which also implies deterministic rounding.
* **Configurations**: `xavier.py` uses `torchao`'s `Float8WeightOnlyConfig` or `Float8DynamicActivationFloat8WeightConfig` to control the quantization process.
  * `Float8WeightOnlyConfig`: Quantizes only weights to FP8. Activations remain in higher precision. Matmul might dequantize weights on the fly unless compiled for FP8 kernels.
  * `Float8DynamicActivationFloat8WeightConfig`: Quantizes both weights and activations to FP8, potentially leveraging FP8 GEMM kernels if supported by hardware and `torch.compile`.
* **Output**: The quantized tensor data and its associated scale are saved. Scales are stored with a `scale_weight` suffix (e.g., `layer.weight` becomes `layer.quantized_weight` and `layer.scale_weight`) for ComfyUI compatibility.

## Usage

```bash
python xavier.py <input.safetensors> <output.safetensors> [OPTIONS]
```

### Required Arguments

* `input_file`: Path to the input `.safetensors` model file.
* `output_file`: Path to save the quantized `.safetensors` model file.

### Options

* `--quant_method <native|torchao_fp8_weight_only_aoscale|torchao_fp8_weight_only_comfyscale|torchao_fp8_dynamic_act_weight_aoscale|torchao_fp8_dynamic_act_weight_comfyscale>`
  * Specifies the quantization method.
  * `native` (default): Uses `xavier.py`'s original stochastic rounding methods.
  * `torchao_fp8_weight_only_aoscale`: Uses `torchao` to quantize weights only to FP8, saving with TorchAO's native scaling.
  * `torchao_fp8_weight_only_comfyscale`: Uses `torchao` to quantize weights only to FP8, adapting the output to be ComfyUI/`comfyscale` compatible.
  * `torchao_fp8_dynamic_act_weight_aoscale`: Uses `torchao` for dynamic activation and weight FP8 quantization, saving with TorchAO's native scaling.
  * `torchao_fp8_dynamic_act_weight_comfyscale`: Uses `torchao` for dynamic activation and weight FP8 quantization, adapting the output to be ComfyUI/`comfyscale` compatible.
* `--fp8_type <e4m3|e5m2>`
  * Specifies the target FP8 data type.
  * `e4m3` (default): Uses `torch.float8_e4m3fn`.
  * `e5m2`: Uses `torch.float8_e5m2`.
* `--device <cpu|cuda|mps|...>`
  * Device to use for computations (e.g., `cuda`, `cpu`).
  * Defaults to `cuda` if available, otherwise `cpu`.
* `--keys_to_quantize_suffix .suffix1 .suffix2 ...`
  * Suffixes of tensor keys to identify for quantization.
  * Default: `.weight`. This is particularly relevant for `torchao` methods which target linear layer weights via the `DummyModule`.
  * Example: `--keys_to_quantize_suffix .weight .bias` will target tensors ending with `.weight` or `.bias`.

#### Native Method Specific Options

(These are ignored if a `torchao_` method is selected)

* `--complex_rounding`
  * Activates the complex neighbor finding method for stochastic rounding.
* `--shifturb`
  * Activates the shift-and-perturb (additive noise) stochastic rounding method.
* `--owlshift`
  * Activates the Owlshift method (manual stochastic mantissa rounding).
* `--comfyscale`
  * Applies per-tensor max-abs scaling (ComfyUI-compatible) before native stochastic rounding.
  * If used, dequantization scale tensors are saved (e.g., `layer.weight` -> `layer.scale_weight`).
  * Bias terms are typically excluded from this specific scaling path.
* `--seed <int>`
  * Random seed for native stochastic rounding methods that use it (e.g., Owlshift).
  * Default: `0`.

#### Common Options

* `--plot`
  * Enables generation of comparison plots. Requires `matplotlib` to be installed.
  * Plots are saved to the directory specified by `--plot_dir`.
* `--plot_dir <path/to/directory/>`
  * Directory to save generated plots.
  * Default: `./quant_plots/`.
* `--plot_max_tensors <int>`
  * Maximum number of tensors for which to generate plots.
  * Default: `5`.
* `--plot_sample_size <int>`
  * Number of data points to sample for scatter plots if a tensor is very large.
  * Default: `5000`.
* `--debug`
  * Enables debug print statements to trace execution flow and intermediate tensor states.

## Combining Flags

* **Quantization Method**: The `--quant_method` flag dictates the primary quantization engine. If a `torchao_` method is chosen, native stochastic rounding flags (`--complex_rounding`, `--shifturb`, `--owlshift`) and `--comfyscale` are ignored, as `torchao` handles its own rounding (typically deterministic) and scaling.
* **Native Rounding Methods**: If `--quant_method native` is used, `--complex_rounding`, `--shifturb`, and `--owlshift` are mutually exclusive regarding the core rounding logic. The script has an internal order of preference: Owlshift > Shifturb > Complex Rounding > Default.
* **Native Scaling**: If `--quant_method native` is used, `--comfyscale` can be combined with any of the native rounding methods. The scaling is applied *before* the chosen rounding method operates on the scaled tensor.
* **TorchAO Scaling Variants**:
  * `_aoscale` methods save tensors using TorchAO's direct output scaling.
  * `_comfyscale` methods adapt TorchAO's output to match the `comfyscale` convention (FP8 tensor normalized to `[-1,1]`, scale is `amax`). This ensures compatibility with systems expecting `comfyscale` formatted FP8 tensors.

## Plotting

If `--plot` is enabled and `matplotlib` is installed, the script will generate:

* **Histograms**: Comparing the value distributions of the original tensor, the FP8 quantized tensor (values cast to float for visualization), and the dequantized tensor.
* **Scatter Plots**: Showing original values vs. dequantized values to visualize the quantization effect.

These plots are helpful for qualitatively assessing the impact of the quantization process on different tensors within the model.

## ComfyUI-Style Scaled FP8

* **Native Method with `--comfyscale`**: Implements per-tensor scaling compatible with ComfyUI. Scales are saved separately (e.g., `layer.weight` and `layer.scale_weight`).
* **TorchAO Methods**:
  * `torchao_..._aoscale`: Saves scales according to TorchAO's native convention.
  * `torchao_..._comfyscale`: Saves scales adapted to be equivalent to the `comfyscale` convention (`scale = amax(original_tensor)`).
  In all `torchao` cases and native `--comfyscale`, scales are saved with a `scale_weight` suffix if the original key was `*.weight`.

In both cases (native `--comfyscale` or any `torchao_` method), a marker tensor `scaled_fp8` (with the target `fp8_dtype`) is added to the state dictionary to indicate this scaling convention.

## Examples

### Native Quantization (comfyscale + Owlshift)

```bash
python xavier.py model_fp32.safetensors model_fp8_native_owl.safetensors \
    --quant_method native \
    --fp8_type e4m3 \
    --comfyscale \
    --owlshift \
    --keys_to_quantize_suffix .weight \
    --plot \
    --device cuda
```

This command will:

1. Load `model_fp32.safetensors`.
2. Use the `native` quantization method.
3. Apply per-tensor max-absolute scaling (`--comfyscale`) to tensors ending in `.weight`.
4. Quantize these scaled tensors to `torch.float8_e4m3fn` (`--fp8_type e4m3`) using the Owlshift stochastic rounding method (`--owlshift`).
5. Save the resulting model (with quantized tensors and scales) to `model_fp8_native_owl.safetensors`.
6. Generate comparison plots.

### TorchAO Weight-Only Quantization (Native TorchAO Scaling)

```bash
python xavier.py model_fp32.safetensors model_fp8_torchao_wo_aoscale.safetensors \
    --quant_method torchao_fp8_weight_only_aoscale \
    --fp8_type e4m3 \
    --keys_to_quantize_suffix .weight \
    --plot \
    --device cuda
```

This command will:

1. Load `model_fp32.safetensors`.
2. Use `torchao` for weight-only FP8 quantization with its native scaling convention.
3. Target `torch.float8_e4m3fn`.
4. Quantize tensors whose keys end in `.weight`.
5. `torchao` will handle scaling internally. Scales will be saved alongside the quantized weights according to TorchAO's default.
6. Save to `model_fp8_torchao_wo_aoscale.safetensors`.

### TorchAO Weight-Only Quantization (ComfyUI-Compatible Scaling)

```bash
python xavier.py model_fp32.safetensors model_fp8_torchao_wo_comfyscale.safetensors \
    --quant_method torchao_fp8_weight_only_comfyscale \
    --fp8_type e4m3 \
    --keys_to_quantize_suffix .weight \
    --plot \
    --device cuda
```

This command will:

1. Load `model_fp32.safetensors`.
2. Use `torchao` for weight-only FP8 quantization.
3. The output FP8 tensor and scale will be adapted to match the ComfyUI/`comfyscale` convention.
4. Target `torch.float8_e4m3fn`.
5. Quantize tensors whose keys end in `.weight`.
6. Save to `model_fp8_torchao_wo_comfyscale.safetensors`.

### TorchAO Dynamic Activation and Weight Quantization (Native TorchAO Scaling)

```bash
python xavier.py model_fp32.safetensors model_fp8_torchao_dyn_aoscale.safetensors \
    --quant_method torchao_fp8_dynamic_act_weight_aoscale \
    --fp8_type e5m2 \
    --keys_to_quantize_suffix .weight \
    --device cuda
```

This command will:

1. Use `torchao` for dynamic activation and weight FP8 quantization with TorchAO's native scaling.
2. Target `torch.float8_e5m2` for both activations and weights.
3. This method is more likely to utilize optimized FP8 computation kernels if `torch.compile` is used later and hardware supports it.
4. Save to `model_fp8_torchao_dyn_aoscale.safetensors`.

### TorchAO Dynamic Activation and Weight Quantization (ComfyUI-Compatible Scaling)

```bash
python xavier.py model_fp32.safetensors model_fp8_torchao_dyn_comfyscale.safetensors \
    --quant_method torchao_fp8_dynamic_act_weight_comfyscale \
    --fp8_type e5m2 \
    --keys_to_quantize_suffix .weight \
    --device cuda
```

This command will:

1. Use `torchao` for dynamic activation and weight FP8 quantization.
2. The output FP8 tensor and scale will be adapted to match the ComfyUI/`comfyscale` convention.
3. Target `torch.float8_e5m2` for both activations and weights.
4. Save to `model_fp8_torchao_dyn_comfyscale.safetensors`.

## Notes

* FP8 quantization is a lossy process. The effectiveness can vary depending on the model, task, and quantization method.
* Always evaluate the performance of the quantized model on downstream tasks.
* The `--debug` flag can be very helpful for understanding intermediate steps.
* Ensure `torchao` is installed (`pip install torchao`) if using `torchao_` quantization methods.
