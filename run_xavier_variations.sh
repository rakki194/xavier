#!/bin/bash

# Script to generate all possible rounding and scaling variations with xavier.py

# --- Configuration ---
# Path to the input safetensors file
INPUT_FILE="/home/kade/toolkit/diffusion/comfy/models/unet/chroma-unlocked-v29.safetensors"

# Base name for output files (derived from input file, without extension)
MODEL_BASENAME=$(basename "$INPUT_FILE" .safetensors)

# Directory where all outputs (models and plots) will be saved
BASE_OUTPUT_DIR="/home/kade/toolkit/diffusion/comfy/models/unet/"

# Path to the xavier.py script
XAVIER_SCRIPT="./xavier.py" # Assumes xavier.py is in the same directory as this script

# Python command
PYTHON_CMD="python3" # Use python3, or change to "python" if preferred

# Device for PyTorch computations
DEVICE="cuda" # Or "cpu" if cuda is not available/preferred
# --- End Configuration ---

# --- Option Definitions ---
# FP8 types
fp8_types=("e4m3" "e5m2") # Generate for both e4m3 and e5m2

# Quantization methods
quant_methods=(
    "native"
    "torchao_fp8_weight_only_aoscale"
    "torchao_fp8_weight_only_comfyscale"
    "torchao_fp8_dynamic_act_weight_aoscale"
    "torchao_fp8_dynamic_act_weight_comfyscale"
)

# Native specific rounding methods:
# Key is the flag to pass to xavier.py ("" for default)
# Value is the name for file/directory naming
declare -A native_rounding_methods
native_rounding_methods[""]="default"
native_rounding_methods["--complex_rounding"]="complex"
native_rounding_methods["--shifturb"]="shifturb"
native_rounding_methods["--owlshift"]="owlshift"

# Native specific scaling options:
# Key is the flag to pass to xavier.py ("" for no scaling)
# Value is the name for file/directory naming
declare -A native_scaling_options
native_scaling_options[""]="noscale"
native_scaling_options["--comfyscale"]="comfyscale"
# --- End Option Definitions ---


# --- Script Logic ---
echo "Starting Xavier.py quantization variations..."
echo "Input Model: ${INPUT_FILE}"
echo "Output Base Directory: ${BASE_OUTPUT_DIR}"
echo "Using Xavier Script: ${XAVIER_SCRIPT}"
echo "Python Command: ${PYTHON_CMD}"
echo "Device: ${DEVICE}"
echo "-----------------------------------------------------"

if [ ! -f "$XAVIER_SCRIPT" ]; then
    echo "Error: xavier.py script not found at $XAVIER_SCRIPT"
    echo "Please ensure xavier.py is in the same directory as this script, or update XAVIER_SCRIPT path."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input model file not found at $INPUT_FILE"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"
# Create a top-level plots directory if it doesn't exist
mkdir -p "${BASE_OUTPUT_DIR}/plots"


total_runs=0
successful_runs=0
failed_runs=0

# Loop through quantization methods
for quant_method in "${quant_methods[@]}"; do
    # Loop through FP8 types
    for fp8_type in "${fp8_types[@]}"; do

        if [[ "$quant_method" == "native" ]]; then
            # Native method: loop through rounding and scaling
            for rounding_flag in "${!native_rounding_methods[@]}"; do
                rounding_name="${native_rounding_methods[$rounding_flag]}"

                for scaling_flag in "${!native_scaling_options[@]}"; do
                    scaling_name="${native_scaling_options[$scaling_flag]}"
                    
                    total_runs=$((total_runs + 1))

                    # Construct output filename and plot directory name
                    run_descriptor="${MODEL_BASENAME}_quant-${quant_method}_fp8-${fp8_type}_round-${rounding_name}_scale-${scaling_name}"
                    output_model_file="${BASE_OUTPUT_DIR}/${run_descriptor}.safetensors"
                    plot_dir="${BASE_OUTPUT_DIR}/plots/${run_descriptor}"

                    echo # Blank line for separation
                    echo "=== Run ${total_runs}: Quant=${quant_method}, FP8=${fp8_type}, Rounding=${rounding_name}, Scaling=${scaling_name} ==="
                    echo "Output model will be: ${output_model_file}"
                    echo "Plots will be saved to: ${plot_dir}"

                    mkdir -p "${plot_dir}"

                    # Build the command arguments array
                    cmd_args=()
                    cmd_args+=("$XAVIER_SCRIPT")
                    cmd_args+=("$INPUT_FILE")
                    cmd_args+=("$output_model_file")
                    cmd_args+=("--quant_method" "$quant_method")
                    cmd_args+=("--fp8_type" "$fp8_type")
                    cmd_args+=("--plot")
                    cmd_args+=("--plot_dir" "$plot_dir")
                    cmd_args+=("--device" "$DEVICE")
                    # cmd_args+=("--debug") # Uncomment for verbose xavier.py output

                    if [ -n "$rounding_flag" ]; then
                        cmd_args+=("$rounding_flag")
                    fi
                    if [ -n "$scaling_flag" ]; then
                        cmd_args+=("$scaling_flag")
                    fi
                    
                    echo "Command: $PYTHON_CMD ${cmd_args[*]}"
                    
                    # Execute the command
                    "$PYTHON_CMD" "${cmd_args[@]}"

                    if [ $? -eq 0 ]; then
                        echo "Run ${total_runs} successful for ${run_descriptor}."
                        successful_runs=$((successful_runs + 1))
                    else
                        echo "Error: Run ${total_runs} failed for ${run_descriptor}. Check output above."
                        failed_runs=$((failed_runs + 1))
                    fi
                    echo "======================================================================="
                done
            done
        else
            # TorchAO methods: no separate looping for native rounding/scaling
            total_runs=$((total_runs + 1))

            run_descriptor="${MODEL_BASENAME}_quant-${quant_method}_fp8-${fp8_type}"
            output_model_file="${BASE_OUTPUT_DIR}/${run_descriptor}.safetensors"
            plot_dir="${BASE_OUTPUT_DIR}/plots/${run_descriptor}"

            echo # Blank line for separation
            echo "=== Run ${total_runs}: Quant=${quant_method}, FP8=${fp8_type} ==="
            echo "Output model will be: ${output_model_file}"
            echo "Plots will be saved to: ${plot_dir}"

            mkdir -p "${plot_dir}"

            cmd_args=()
            cmd_args+=("$XAVIER_SCRIPT")
            cmd_args+=("$INPUT_FILE")
            cmd_args+=("$output_model_file")
            cmd_args+=("--quant_method" "$quant_method")
            cmd_args+=("--fp8_type" "$fp8_type")
            cmd_args+=("--plot")
            cmd_args+=("--plot_dir" "$plot_dir")
            cmd_args+=("--device" "$DEVICE")
            # cmd_args+=("--debug")

            echo "Command: $PYTHON_CMD ${cmd_args[*]}"
            
            "$PYTHON_CMD" "${cmd_args[@]}"

            if [ $? -eq 0 ]; then
                echo "Run ${total_runs} successful for ${run_descriptor}."
                successful_runs=$((successful_runs + 1))
            else
                echo "Error: Run ${total_runs} failed for ${run_descriptor}. Check output above."
                failed_runs=$((failed_runs + 1))
            fi
            echo "======================================================================="
        fi
    done
done

echo # Blank line
echo "-----------------------------------------------------"
echo "All quantization variations attempted."
echo "Total runs: ${total_runs}"
echo "Successful runs: ${successful_runs}"
echo "Failed runs: ${failed_runs}"
echo "Outputs are in: ${BASE_OUTPUT_DIR}"
echo "-----------------------------------------------------"

# Make the script executable
chmod +x "$0"
echo "Script ${0} is now executable." 