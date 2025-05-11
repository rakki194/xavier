# Review of FP8 Quantization in PyTorch: Methods, Implementation, and Dynamics

## 1. Introduction

The demand for larger and more complex deep learning models has driven significant research into reducing their computational and memory footprints. While 16-bit floating-point formats (FP16, BF16) have become standard, 8-bit floating-point (FP8) formats are emerging as the next frontier for further optimization, particularly for inference and, increasingly, for training. FP8 offers the potential for up to 2x speedup and memory reduction compared to 16-bit formats. This document reviews FP8 quantization, focusing on its mathematical underpinnings, statistical implications, PyTorch implementation details, the role of stochastic rounding, and considerations for network dynamics, drawing insights from proposals like PyTorch RFC-0030 and practical implementations like `fp8-auto` and FluxMod.

The primary motivation, as highlighted in `RFC-0030-native-fp8-dtype.md`, is to standardize FP8 to simplify research, enable high-level libraries (e.g., automatic mixed-precision frameworks), and improve performance by avoiding emulation overhead.

## 2. FP8 Data Formats

FP8 is not a single standard but a family of formats. The most prominent ones, and those proposed for PyTorch in RFC-0030, are E4M3 and E5M2, largely based on the Nvidia/Arm/Intel specification.

* **E4M3 (HF8 - Hybrid Float 8):**
  * 1 sign bit, 4 exponent bits, 3 mantissa bits.
  * **Bias:** Typically 7 (can vary, but 7 is common for this configuration if following a pattern).
  * **Dynamic Range:** Narrower. Max normal value around 448 (if using non-standard NaN/Inf encoding) or 240 (standard IEEE encoding).
  * **Precision:** Higher for its range compared to E5M2.
  * **Special Values:** The Nvidia/Arm/Intel proposal suggests deviating from strict IEEE 754 for E4M3 to extend its dynamic range:
    * No explicit infinities.
    * Reduced NaN encodings (e.g., only all-ones mantissa).
  * **Use Cases:** Often preferred for weights and activations where values might be well-scaled within a more limited range but benefit from slightly higher precision within that range.

* **E5M2 (BF8 - Brain Float 8):**
  * 1 sign bit, 5 exponent bits, 2 mantissa bits.
  * **Bias:** Typically 15.
  * **Dynamic Range:** Wider than E4M3, similar to BF16's exponent range but scaled down. Max normal value around 57344.
  * **Precision:** Lower than E4M3.
  * **Special Values:** Can adhere more closely to IEEE 754 for NaN/Inf.
  * **Use Cases:** Often preferred for gradients during training due to its wider dynamic range, which helps prevent underflow/overflow. Also suitable for activations if a wider range is necessary.

The choice between E4M3 and E5M2 depends on the specific application and the distribution of values being quantized.

## 3. Mathematical Basis of FP8 Quantization

Quantizing a higher-precision floating-point number (e.g., FP32 or BF16) to FP8 involves several steps, conceptually similar to the process outlined in `fp8-auto/main.py`. Let $x$ be the original high-precision number.

**Step 1: Handle Special Cases**
Zero, NaN, and Infinity have specific representations in FP8.

* $x = 0 \implies \text{fp8_value} = 0\text{b00000000}$ (positive zero) or $0\text{b100000000}$ (negative zero).
* $x = \text{NaN} \implies \text{fp8_value} = \text{predefined NaN pattern}$ (e.g., $0\text{b01111111}$ or $0\text{b11111111}$ for E4M3 as per RFC-0030; for E5M2, it could be $0\text{b01111110}$ or $0\text{b01111111}$).
* $x = \pm\infty \implies \text{fp8_value} = \text{predefined Inf pattern}$ (e.g., E4M3 might map overflow to max value, E5M2 might have $0\text{b01111100}$ for $+\infty$).

The `fp8-auto/main.py` script handles this:

```python
# Conceptual from fp8-auto/main.py
def handle_special_cases(value: float) -> Optional[int]:
    if abs(value) < 1e-7: return 0 # zero
    if torch.isnan(torch.tensor(value)): return 0xFF # Example NaN
    if torch.isinf(torch.tensor(value)): return 0x7F # Example Inf
    return None
```

**Step 2: Pre-Quantization Scaling (Crucial for FP8)**
Due to FP8's limited range, direct quantization often leads to significant clamping (overflow) or loss of precision (underflow). A scaling factor, $S$, is applied: $x' = x \cdot S$. The choice of $S$ is critical and discussed in Section 5.

**Step 3: Sign Extraction**
The sign bit $s$ is 0 for positive $x'$, 1 for negative $x'$. Let $x'' = |x'|$.

**Step 4: Exponent and Mantissa Calculation**
For a normalized number $x'' = (1.m_1m_2...m_k)_2 \times 2^{e_{\text{actual}}}$, the stored exponent $e_{\text{stored}}$ is $e_{\text{actual}} + \text{bias}$.
The number of exponent bits ($N_e$) and mantissa bits ($N_m$) defines the format (e.g., $N_e=4, N_m=3$ for E4M3).

* The exponent $E = \lfloor \log_2 x'' \rfloor$.
* The stored exponent $e_s = E + \text{bias}$.
  * If $e_s < 0$ (or $e_s=0$ for subnormals), it becomes a subnormal number or zero.
  * If $e_s \ge 2^{N_e}-1$, it's an overflow (map to Inf or max value).
* The mantissa $M = (x'' / 2^E) - 1$. This $M$ is then rounded to $N_m$ bits.

The `fp8-auto/main.py` script details this:

```python
# Conceptual from fp8-auto/main.py
def calculate_exponent_mantissa(
    value_scaled: float, n_mantissa: int, bias: int
) -> Tuple[int, float]:
    # ...
    exponent = int(torch.floor(torch.log2(torch.tensor(value_scaled))).item()) + bias
    mantissa = value_scaled / (2 ** (exponent - bias)) - 1
    return exponent, mantissa
```

**Step 5: Rounding the Mantissa**
The calculated mantissa typically has more precision than available $N_m$ bits. It must be rounded.

* **Round-to-Nearest-Even (RNE):** Standard default.
* **Stochastic Rounding (SR):** Discussed in Section 4.

**Step 6: Handling Subnormals**
If $E < E_{\text{min}}$ (where $E_{\text{min}} = 1 - \text{bias}$ for normalized numbers), the number is subnormal.
The value is $x'' = (0.m_1m_2...m_{N_m})_2 \times 2^{E_{\text{min}}}$. The exponent field is set to 0. The mantissa is $x'' / 2^{E_{\text{min}}}$, then rounded.

**Step 7: Packing Bits**
Combine sign $s$, stored exponent $e_s$, and rounded $N_m$-bit mantissa into an 8-bit word.

**De-quantization (FP8 to Float):**
This reverses the process:

1. Unpack sign, exponent, mantissa.
2. If exponent field is all 1s: NaN or Inf.
3. If exponent field is all 0s: subnormal. Value is $(-1)^s \times (0.m_1...m_{N_m})_2 \times 2^{E_{\text{min}}}$.
4. Else (normalized): Value is $(-1)^s \times (1.m_1...m_{N_m})_2 \times 2^{e_s - \text{bias}}$.
5. Apply inverse scaling: $x_{\text{dequant}} = x'_{\text{dequant}} / S$.

## 4. Stochastic Rounding (SR)

**Motivation:**
When quantizing, RNE can introduce systematic bias if the distribution of values being truncated is not symmetric around the midpoint of representable numbers. For very low precision formats like FP8, this bias can accumulate and degrade model accuracy. SR aims to make the rounding error an unbiased random variable in expectation.

**Mathematical Principle:**
Given a value $v$ that lies between two representable FP8 values $v_{\text{low}}$ and $v_{\text{high}}$.
$v_{\text{quantized}} = \begin{cases} v_{\text{high}} & \text{with probability } p = \frac{v - v_{\text{low}}}{v_{\text{high}} - v_{\text{low}}} \\ v_{\text{low}} & \text{with probability } 1-p \end{cases}$
In expectation, $E[v_{\text{quantized}}] = p \cdot v_{\text{high}} + (1-p) \cdot v_{\text{low}} = v$.

**Implementation:**
The `fp8-auto/main.py` script provides a practical implementation:

```python
# From fp8-auto/main.py
def stochastic_rounding(value: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    floor_value = torch.floor(value / epsilon) * epsilon
    ceil_value = floor_value + epsilon # epsilon is the smallest representable step
    probability = (value - floor_value) / epsilon
    rand_value = torch.rand_like(value) # Uniform random number [0, 1)
    rounded_value = torch.where(rand_value < probability, ceil_value, floor_value)
    return rounded_value

# Applied in float_to_fp8:
# mantissa_quantized = apply_stochastic_rounding(mantissa, n_mantissa)
# where epsilon for mantissa is 1.0 / (2**n_mantissa)
```

This `epsilon` is the quantization step size for the mantissa (which is in range [0, 1) before adding the implicit 1).

**Pros:**

* Reduces quantization bias, potentially improving accuracy, especially in training or for sensitive activations.
* `RFC-0030` mentions its potential importance for FP8 training and suggests exposing it as a casting option.

**Cons:**

* Introduces randomness, making runs non-deterministic unless seeded carefully.
* May require hardware support for efficient execution; software emulation can be slow.
* `RFC-0030` notes that SR "may cause harm when used in wrong places, e.g. in optimizer step."
* The benefits are not always guaranteed and can be model/task-dependent.

**PyTorch `RFC-0030` Considerations:**
The RFC discusses exposing SR:

* Python level: `tensor.to(dtype=torch.fp8e4m3fn, rounding_mode="stochastic", seed=...)`.
* Higher-level control: enabling SR for a whole model or parts via a mixed-precision module.

The `simulate_stochastic_rounding_test` in `fp8-auto/main.py` effectively demonstrates that over many runs, the average of stochastically rounded values converges to the original mean, validating its unbiased nature.

## 5. Scaling Strategies

FP8's limited dynamic range makes scaling paramount. The goal is to map the original tensor's values into FP8's representable range with minimal information loss.

**Types of Scaling:**

1. **Per-Tensor Scaling:** A single scaling factor $S$ is computed and applied to the entire tensor.
    * **Static:** $S$ is determined offline using a calibration dataset (e.g., min/max, L2 norm, KL-divergence).
    * **Dynamic:** $S$ is computed on-the-fly for each tensor (typically for activations). Common dynamic methods use `max(abs(tensor))`.
    $S = \frac{\text{FP8_MAX_VAL}}{\max(|X|)}$, where $X$ is the tensor.

2. **Per-Channel/Group/Token Scaling:** More granular scaling, applying different $S$ values to subsets of the tensor (e.g., per output channel for weights). This can improve accuracy but increases overhead.

**`RFC-0030` on Scaling:**

* Recognizes that "tensors should be scaled to fp8 range."
* Suggests adapting `torch.cuda.amp.GradScaler` for FP8 gradients, potentially with per-tensor scale factors due to the severe range/precision limits.
    > "Considering bigger models, it’s easy to imagine a set of gradient tensors with values that cannot all be scaled into fp8 format without part of it being clamped to zero. The proper, more complex solution should be used for more efficient scaling, collecting maximum values statistics from few iterations back and keeping separate scale factor for each gradient tensor."

**FluxMod Context (`ComfyUI_FluxMod/flux_mod/loader.py`):**
FluxMod's loader (`load_flux_mod`) quantizes linear layers:

```python
# From ComfyUI_FluxMod/flux_mod/loader.py
def cast_layers(module, layer_type, dtype, exclude_keywords=()):
    # ...
    for child_name, child in module.named_children():
        if isinstance(child, layer_type) and not any(keyword in child_name for keyword in exclude_keywords):
            child.weight.data = child.weight.data.to(dtype=dtype) # Bias often kept in higher precision
        # ...
# Called as:
# cast_layers(model.diffusion_model, nn.Linear, dtype=linear_dtypes, exclude_keywords={"img_in", "final_layer", "scale"})
```

This implies weights of `nn.Linear` layers are cast to the specified `linear_dtypes` (which can be `torch.float8_e4m3fn` or `torch.float8_e5m2`). The scaling method for these weights isn't explicitly defined in the snippet but would typically be static, pre-calculated when the model was trained or fine-tuned with quantization in mind, or determined via PTQ calibration. Activations might use dynamic scaling or a scheme learned by the `Approximator` within the `distilled_guidance_layer`.

The `Approximator` layer in FluxMod (`model.diffusion_model.distilled_guidance_layer`) is a neural network itself. If its inputs or internal computations are in FP8, it implies a learned or dynamic scaling mechanism to handle the data transformations effectively in low precision.

**Scaled FP8 (`model_conf.scaled_fp8`):**
FluxMod's loader also mentions `scaled_fp8`. This typically refers to a specific FP8 quantization scheme where scale factors are handled explicitly by the hardware or kernel, often per-tensor or per-axis. The scale factors themselves are usually in a higher precision format (e.g., FP16/FP32). This is a common approach in hardware accelerators that support FP8. The quantization would be $X_{\text{fp8}} = \text{round}(X_{\text{fp32}} / \text{scale}) \text{ and dequantization is } X_{\text{fp32}} = X_{\text{fp8}} \times \text{scale}$. The `scale` itself is chosen to maximize the representational power of FP8 for the given tensor $X_{\text{fp32}}$.

## 6. PyTorch Implementation Aspects

`RFC-0030` outlines the proposal for native FP8 types in PyTorch:

**Native DTypes:**

* `torch.fp8e4m3fn` (for E4M3, `fn` indicates potential for finite-values-only interpretation if Inf/NaN are remapped to extend range, or specific NaN behavior).
* `torch.fp8e5m2` (for E5M2).
* These would be first-class citizens, similar to `torch.float16` and `torch.bfloat16`.

**Type Promotion:**
FP8 would participate in PyTorch's type promotion system. Generally:

* Floating point types take precedence over integer types.
* Types with more bits take precedence.

    ```python
    # Example from RFC-0030
    input_fp32 = torch.rand((3, 3))
    input_fp8 = torch.rand((3, 3)).to(torch.fp8e5m2)
    res_fp32_fp8 = torch.add(input_fp32, input_fp8) # dtype == torch.fp32
    res_bf16_fp8 = torch.add(input_bf16, input_fp8) # dtype == torch.bf16
    ```

**CPU Support:**
Basic CPU support would be via casting to `float` for operations, mainly for testing and prototyping as widespread native CPU hardware support for FP8 is not yet common.

```cpp
// Conceptual from RFC-0030
inline C10_HOST_DEVICE float operator+(FP8_TYPE a, float b) {
  return static_cast<float>(a) + b;
}
```

**Autocast (`torch.amp.autocast`):**
The existing `autocast` is likely too simple for robust FP8 mixed-precision training. `RFC-0030` proposes enhancements:

* A `second_dtype` parameter for more fine-grained control.
* New `CastPolicy` options.

    ```python
    # Conceptual from RFC-0030
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, second_dtype=torch.fp8e5m2):
        # Ops might go to bf16 or fp8e5m2 based on policy
        mm = torch.matmul(a, b) # could be fp8e5m2
        addmm_result = torch.addmm(a, b, c) # could be bfloat16
    ```

FluxMod's KSampler wrapper uses `torch.autocast` for `bf16` or `fp16` activations, but notes that if `using_scaled_fp8` (a specific hardware/kernel optimization) is active, autocast might be bypassed or handled differently.

```python
# From ComfyUI_FluxMod/flux_mod/nodes.py KSamplerMod
if using_scaled_fp8(model): # A specific optimized path
    return common_ksampler(...)
else: # Generic path with autocast
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    with torch.autocast(device_type="cuda", dtype=dtypes[activation_casting]):
        return common_ksampler(...)
```

**Higher-Level Libraries:**
`RFC-0030` anticipates that more sophisticated automatic mixed-precision (AMP) frameworks (like NVIDIA's TransformerEngine or future PyTorch modules) would be built on top of native FP8 types to manage scaling and precision selection more effectively.

## 7. Dynamics of Quantized Networks

Quantizing a neural network to FP8 introduces several challenges to its dynamics, especially during training:

**Reduced Range and Precision:**

* **Activations:** Can easily overflow or underflow if not properly scaled. Outliers are particularly problematic. Gradients of activation functions (e.g., ReLU's gradient being 0 or 1) can become sparse or lose fidelity.
* **Weights:** Small weights may be quantized to zero, effectively pruning connections. Large weights may be clamped.
* **Gradients:** Extremely susceptible to underflow (vanishing gradients) or overflow (exploding gradients) due to the wide range of values they can take, especially in deep networks. E5M2 is generally preferred here.
* **Optimizer States:** Accumulators in optimizers (e.g., Adam's momentum and variance) are usually kept in FP32 to maintain precision over many updates. Quantizing these is highly risky.

**Impact on Learning:**

* **Quantization Noise:** The error introduced by quantization acts as a noise source. While sometimes beneficial (regularization), it can also hinder convergence if too large. SR can help make this noise unbiased.
* **Gradient Mismatch:** If forward pass uses FP8 weights/activations but backward pass computes gradients in FP32, there's a mismatch. Quantization-Aware Training (QAT) attempts to mitigate this by simulating quantization effects during training.
* **Need for QAT:** For aggressive quantization like FP8, Post-Training Quantization (PTQ) often yields insufficient accuracy. QAT, where the model is (re)trained with simulated quantization in the loop (using Straight-Through Estimators for gradients of non-differentiable quantization ops), is usually necessary.
* **Layer Sensitivity:** Different layers exhibit varying sensitivity to quantization. Output layers or layers with critical information flow might need higher precision or careful scaling. The `exclude_keywords` in FluxMod's `cast_layers` (e.g., `"img_in"`, `"final_layer"`, `"scale"`) hints at this, keeping sensitive layers in higher precision.

**Statistical Shifts:**
The distribution of weights and activations can shift during training. A static scaling factor determined early in training might become suboptimal later. Adaptive scaling or periodic re-calibration can be important.

## 8. Hardware Considerations

The practical benefits of FP8 are realized through hardware support.

* **NVIDIA Hopper (H100):** Supports E4M3 and E5M2 FP8 matrix multiplications with FP32 or FP16 accumulation.
* **Intel Gaudi2/3:** Supports E4M3 and E5M2 FP8 with FP32 accumulation.
* These accelerators often have specialized Tensor Cores or Matrix Math Engines that perform FP8 operations much faster than emulated ones.
* Hardware might also offer accelerated support for specific scaling schemes (e.g., per-tensor scaled FP8) or stochastic rounding.
* Differences exist in how special values (NaN, Inf) are handled or if certain exponent ranges are reserved, which can affect numerical equivalence across platforms.

## 9. Challenges and Open Points

From `RFC-0030` and general experience:

1. **Standardization:** While E4M3/E5M2 are gaining traction, minor variations can exist.
2. **Optimal Scaling:** Finding robust and efficient scaling strategies for diverse models and data distributions remains an active research area. Per-tensor dynamic scaling is common for activations, but more advanced methods might be needed.
3. **Stochastic Rounding Utility:** When and where to apply SR for maximal benefit without performance or determinism penalties is still being explored.
4. **Software Ecosystem:** Tooling for debugging, profiling, and deploying FP8 models needs to mature.
5. **User-Friendliness:** Making FP8 easy to use (e.g., via improved `autocast` or dedicated AMP frameworks) is key for broader adoption.
6. **Interoperability:** Ensuring models quantized on one hardware/software stack work correctly on another.

## 10. Conclusion

FP8 quantization represents a significant step towards more efficient deep learning. The proposed introduction of native E4M3 and E5M2 types into PyTorch, as detailed in RFC-0030, will provide a foundational layer for research and development. Key to successful FP8 deployment are:

* **Appropriate Format Selection:** E4M3 for precision-sensitive values within a controlled range (often weights/activations), E5M2 for range-sensitive values (often gradients).
* **Effective Scaling:** This is the most critical aspect, requiring careful per-tensor or more granular, static or dynamic strategies to fit values into FP8's narrow confines.
* **Stochastic Rounding:** A valuable tool to combat quantization bias, particularly in training, though its application needs careful consideration.
* **Quantization-Aware Training:** Often necessary to recover accuracy lost due to aggressive quantization.
* **Hardware Support:** Essential for realizing the performance benefits.
