import torch

# Attempt to import matplotlib for plotting
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


def generate_comparison_plots(
    original_tensor_cpu: torch.Tensor,
    quantized_fp8_tensor_cpu: torch.Tensor,  # This is still in FP8 format, but on CPU
    dequantized_tensor_cpu: torch.Tensor,  # This is in original_tensor_cpu's dtype
    tensor_key: str,
    plot_filename: str,
    original_dtype_str: str,
    fp8_dtype_str: str,
    sample_size: int,
):
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        original_np = original_tensor_cpu.float().numpy().flatten()
        # For histogram, cast FP8 to float to see its quantized levels
        quantized_for_hist_np = quantized_fp8_tensor_cpu.float().numpy().flatten()
        dequantized_np = dequantized_tensor_cpu.float().numpy().flatten()

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(
            f"Quantization Comparison: {tensor_key}\nOriginal ({original_dtype_str}) vs. Quantized ({fp8_dtype_str})",
            fontsize=16,
        )

        # Subplot 1: Histograms
        axs[0].hist(
            original_np,
            bins="auto",
            alpha=0.6,
            label=f"Original Values\n(min: {original_np.min():.3g}, max: {original_np.max():.3g})",
            color="blue",
            density=True,
        )
        axs[0].hist(
            quantized_for_hist_np,
            bins="auto",
            alpha=0.7,
            label=f"FP8 Values (cast to float)\n(min: {quantized_for_hist_np.min():.3g}, max: {quantized_for_hist_np.max():.3g})",
            color="red",
            density=True,
        )
        if dequantized_tensor_cpu is not None:  # Check if it's actually available
            axs[0].hist(
                dequantized_np,
                bins="auto",
                alpha=0.5,
                label=f"Dequantized Values\n(min: {dequantized_np.min():.3g}, max: {dequantized_np.max():.3g})",
                color="green",
                density=True,
            )
        axs[0].set_title("Value Distributions (Density)")
        axs[0].set_xlabel("Value")
        axs[0].set_ylabel("Density")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Use a symmetric log scale if data spans positive and negative with wide range, else linear
        if (
            original_np.size > 0 and dequantized_np.size > 0
        ):  # Ensure not empty before min/max
            if original_np.min() < 0 < original_np.max() and (
                original_np.max() / original_np.min() < -0.01
                or np.log10(np.abs(original_np.max()))
                - np.log10(np.abs(original_np.min()))
                > 2
            ):
                axs[0].set_yscale("symlog", linthresh=1e-5)
            else:
                axs[0].set_yscale("linear")  # or 'log' if appropriate for all positive
        else:
            axs[0].set_yscale("linear")  # Default if arrays are empty

        # Subplot 2: Scatter Plot (Original vs. Dequantized/Quantized)
        # Sample data for scatter plot if tensors are too large
        num_elements = original_np.shape[0]
        indices = np.arange(num_elements)
        if num_elements > sample_size:
            indices = np.random.choice(num_elements, size=sample_size, replace=False)

        scatter_x = original_np[indices]
        scatter_y = dequantized_np[indices]

        axs[1].scatter(
            scatter_x,
            scatter_y,
            alpha=0.3,
            s=5,
            label="Original vs. Dequantized/FP8-as-float",
            color="purple",
        )
        # Add a y=x line for reference
        if scatter_x.size > 0 and scatter_y.size > 0:  # Ensure not empty before min/max
            lims = [
                min(scatter_x.min(), scatter_y.min()),
                max(scatter_x.max(), scatter_y.max()),
            ]
            axs[1].plot(lims, lims, "k--", alpha=0.75, zorder=0, label="y=x (Ideal)")
        else:  # Provide default lims if data is empty to avoid errors
            axs[1].plot(
                [0, 1], [0, 1], "k--", alpha=0.75, zorder=0, label="y=x (Ideal)"
            )

        axs[1].set_title(f"Scatter Plot (Sampled {len(scatter_x)} points)")
        axs[1].set_xlabel(f"Original Value ({original_dtype_str})")
        axs[1].set_ylabel(f"Dequantized/FP8-as-float Value")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        axs[1].axhline(0, color="black", linewidth=0.5)
        axs[1].axvline(0, color="black", linewidth=0.5)

        plt.savefig(plot_filename)
        plt.close(fig)  # Close the figure to free memory
        print(f"    Plot saved to {plot_filename}")

    except Exception as e:
        print(f"Error generating plot for {tensor_key}: {e}")
        if "plt" in locals() and plt.gcf().get_axes():  # Check if a figure is open
            plt.close("all")  # Close all figures in case of an error during plotting
