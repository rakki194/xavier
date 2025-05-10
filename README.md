# Xavier FP8 Quantization Script (xavier.py)

`xavier.py` is a Python script designed to quantize `.safetensors` model files to FP8 (8-bit floating-point) precision using various stochastic rounding techniques. Stochastic rounding, as opposed to deterministic methods like Round-to-Nearest-Even (RNE), introduces a controlled amount of noise during quantization. This can help preserve model performance by ensuring that rounding errors do not systematically bias values.

The script supports multiple FP8 formats and several distinct stochastic rounding algorithms, along with pre-processing options like per-tensor scaling.

## Key Features

* **FP8 Quantization**: Converts tensors (typically FP32, FP16, or BF16) to FP8.
* **Stochastic Rounding**: Employs different methods to probabilistically round values, aiming to maintain model fidelity better than deterministic rounding.
* **Multiple Rounding Techniques**:
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
      2. The mantissa is scaled based on the target FP8 format's mantissa bits (MANTISSA_BITS). For normal numbers:
        $$ m_{\text{scaled}} = \left( \frac{|v|}{2^{e - \text{EXPONENT\_BIAS}}} - 1.0 \right) \times 2^{\text{MANTISSA\_BITS}} $$
        A similar calculation is done for subnormal numbers.
      3. A uniform random number $u \sim U[0,1)$ is added to $m_{\text{scaled}}$: $m_{\text{stoch}} = \lfloor m_{\text{scaled}} + u \rfloor$.
      4. The stochastically rounded mantissa $m_{\text{final}} = m_{\text{stoch}} / 2^{\text{MANTISSA\_BITS}}$ is used to reconstruct the number.
      5. The number is then reconstructed from the sign, original exponent, and the new stochastically rounded mantissa, and finally cast to the target FP8 type.
    * This method includes logic for handling normal and subnormal numbers, and tensor slicing for large tensors to manage memory/computation. It uses a dedicated random number generator seeded by the `--seed` argument.
* **Per-Tensor Max-Absolute Scaling (Owlscale)** (`--owlscale`):
  * A pre-processing step that scales each tensor to map its maximum absolute value into the representable range of the target FP8 format.
  * When used, it saves dequantization scales alongside the quantized tensors, compatible with ComfyUI-style scaled FP8 loading (e.g., `layer.weight` gets `layer.scale_weight`).
  * Bias terms are typically excluded from this specific scaling path.
* **Target FP8 Formats**:
  * `e4m3`: `torch.float8_e4m3fn` (4 exponent bits, 3 mantissa bits, NaN-preserving).
  * `e5m2`: `torch.float8_e5m2` (5 exponent bits, 2 mantissa bits).
* **Selective Quantization**: Allows specifying which tensors to quantize based on key name suffixes (e.g., `.weight`, `.bias`).
* **Comparison Plotting**: Generates plots comparing original, quantized, and dequantized tensor distributions and values (requires `matplotlib`).
* **Debug Mode**: Provides verbose output for tracing execution flow.

## Usage

The script is run from the command line.

```bash
python xavier.py <input.safetensors> <output.safetensors> [OPTIONS]
```

### Required Arguments

* `input_file`: Path to the input `.safetensors` model file.
* `output_file`: Path to save the quantized `.safetensors` model file.

### Options

* `--fp8_type <e4m3|e5m2>`
  * Specifies the target FP8 data type.
  * `e4m3` (default): Uses `torch.float8_e4m3fn`.
  * `e5m2`: Uses `torch.float8_e5m2`.
* `--device <cpu|cuda|mps|...>`
  * Device to use for computations (e.g., `cuda`, `cpu`).
  * Defaults to `cuda` if available, otherwise `cpu`.
* `--keys_to_quantize_suffix .suffix1 .suffix2 ...`
  * Suffixes of tensor keys to identify for quantization.
  * Default: `.weight` `.bias`.
  * Example: `--keys_to_quantize_suffix .weight` will only quantize tensors whose names end with ".weight".
* `--complex_rounding`
  * Activates the complex neighbor finding method for stochastic rounding.
* `--shifturb`
  * Activates the shift-and-perturb (additive noise) stochastic rounding method.
* `--owlshift`
  * Activates the Owlshift method (manual stochastic mantissa rounding).
* `--owlscale`
  * Applies per-tensor max-abs scaling (ComfyUI-compatible) before stochastic rounding.
  * If used, dequantization scale tensors are saved (e.g., `layer.weight` -> `layer.scale_weight`).
  * Bias terms are typically excluded from this specific scaling path.
* `--seed <int>`
  * Random seed for stochastic rounding methods that use it (e.g., Owlshift).
  * Default: `0`.
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

* **Rounding Methods**: `--complex_rounding`, `--shifturb`, and `--owlshift` are mutually exclusive regarding the core rounding logic. If multiple are specified, the script has an internal order of preference: Owlshift > Shifturb > Complex Rounding > Default.
* **Scaling**: `--owlscale` can be combined with any of the rounding methods (or the default). The scaling is applied *before* the chosen rounding method operates on the scaled tensor.

## Plotting

If `--plot` is enabled and `matplotlib` is installed, the script will generate:

* **Histograms**: Comparing the value distributions of the original tensor, the FP8 quantized tensor (values cast to float for visualization), and the dequantized tensor.
* **Scatter Plots**: Showing original values vs. dequantized values to visualize the quantization effect.

These plots are helpful for qualitatively assessing the impact of the quantization process on different tensors within the model.

## ComfyUI-Style Scaled FP8

When `--owlscale` is used, the script implements a per-tensor scaling strategy that is compatible with how ComfyUI handles scaled FP8 models.

1. For each eligible tensor (typically weights, not biases), a scale factor is computed (usually the absolute maximum value of the tensor).
2. The original tensor is divided by this scale factor.
3. The now-normalized tensor (roughly in the `[-1, 1]` range) is quantized to FP8 using the chosen stochastic rounding method.
4. The quantized FP8 tensor and its corresponding scale factor (e.g., `layer.weight` and `layer.scale_weight`) are saved in the output file.
5. A marker tensor `scaled_fp8` (with the target `fp8_dtype`) is added to the state dictionary to indicate this scaling convention.

This allows an inference framework like ComfyUI to load the FP8 tensor and its scale, then dequantize it correctly during model loading or execution by multiplying the FP8 tensor (cast to a higher precision) by its scale.

## Example

```bash
python xavier.py model_fp32.safetensors model_fp8_e4m3_owlscaled.safetensors \
    --fp8_type e4m3 \
    --owlscale \
    --owlshift \
    --keys_to_quantize_suffix .weight \
    --plot \
    --plot_dir ./my_model_plots/ \
    --device cuda
```

This command will:

1. Load `model_fp32.safetensors`.
2. Apply per-tensor max-absolute scaling (`--owlscale`) to tensors ending in `.weight`.
3. Quantize these scaled tensors to `torch.float8_e4m3fn` (`--fp8_type e4m3`) using the Owlshift stochastic rounding method (`--owlshift`).
4. Save the resulting model (with quantized tensors and scales) to `model_fp8_e4m3_owlscaled.safetensors`.
5. Generate comparison plots for the first few quantized tensors and save them in `./my_model_plots/`.
6. Perform computations on a CUDA-enabled GPU (`--device cuda`).

## Notes

* FP8 quantization is a lossy process. The effectiveness of stochastic rounding and the choice of FP8 format can vary depending on the model architecture and task.
* Always evaluate the performance of the quantized model on downstream tasks to ensure it meets requirements.
* The `--debug` flag can be very helpful for understanding the intermediate steps and values, especially when experimenting with different rounding methods or troubleshooting.
