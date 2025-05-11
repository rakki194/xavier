#!/bin/bash

# Script to generate all possible rounding and scaling variations with xavier.py

# --- Configuration ---
# Path to the input safetensors file
INPUT_FILE="/home/kade/toolkit/diffusion/comfy/models/unet/chroma-unlocked-v29.safetensors"

# Base name for output files (derived from input file, without extension)
MODEL_BASENAME=$(basename "$INPUT_FILE" .safetensors)

# Directory where all outputs (models and plots) will be saved
BASE_OUTPUT_DIR="/home/kade/toolkit/diffusion/comfy/models/unet/" # Modified to save models directly to the specified path

# Path to the xavier.py script
XAVIER_SCRIPT="./xavier.py" # Assumes xavier.py is in the same directory as this script

# Python command
PYTHON_CMD="python3" # Use python3, or change to "python" if preferred

# Device for PyTorch computations
DEVICE="cuda" # Or "cpu" if cuda is not available/preferred
# --- End Configuration ---

# FP8 types
# I don't like e5m2, so I'm not running it.
#fp8_types=("e4m3" "e5m2")
fp8_types=("e4m3")

# Rounding methods:
# Key is the flag to pass to xavier.py ("" for default)
# Value is the name for file/directory naming
declare -A rounding_methods
rounding_methods[""]="default"          # Default (no flag)
rounding_methods["--complex_rounding"]="complex"
rounding_methods["--shifturb"]="shifturb"
rounding_methods["--owlshift"]="owlshift"

# Scaling options:
# Key is the flag to pass to xavier.py ("" for no scaling)
# Value is the name for file/directory naming
declare -A scaling_options
scaling_options[""]="noscale"
scaling_options["--owlscale"]="owlscale"


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

total_runs=0
successful_runs=0
failed_runs=0

# Loop through FP8 types
for fp8_type in "${fp8_types[@]}"; do
    # Loop through rounding methods (using keys of associative array)
    for rounding_flag in "${!rounding_methods[@]}"; do
        rounding_name="${rounding_methods[$rounding_flag]}"

        # Loop through scaling options (using keys of associative array)
        for scaling_flag in "${!scaling_options[@]}"; do
            scaling_name="${scaling_options[$scaling_flag]}"
            
            total_runs=$((total_runs + 1))

            # Construct output filename and plot directory name
            run_descriptor="${MODEL_BASENAME}_fp8-${fp8_type}_round-${rounding_name}_scale-${scaling_name}"
            output_model_file="${BASE_OUTPUT_DIR}/${run_descriptor}.safetensors"
            plot_dir="${BASE_OUTPUT_DIR}/plots/${run_descriptor}"

            echo # Blank line for separation
            echo "=== Run ${total_runs}: FP8=${fp8_type}, Rounding=${rounding_name}, Scaling=${scaling_name} ==="
            echo "Output model will be: ${output_model_file}"
            echo "Plots will be saved to: ${plot_dir}"

            mkdir -p "${plot_dir}"

            # Build the command arguments array
            cmd_args=()
            cmd_args+=("$XAVIER_SCRIPT") # Script name itself
            cmd_args+=("$INPUT_FILE")    # input_file argument
            cmd_args+=("$output_model_file") # output_file argument
            cmd_args+=("--fp8_type" "$fp8_type")
            cmd_args+=("--plot")
            cmd_args+=("--plot_dir" "$plot_dir")
            cmd_args+=("--device" "$DEVICE")
            # cmd_args+=("--debug") # Uncomment for verbose xavier.py output

            # Add rounding flag if not empty (for default, flag is empty string)
            if [ -n "$rounding_flag" ]; then
                cmd_args+=("$rounding_flag")
            fi

            # Add scaling flag if not empty (for noscale, flag is empty string)
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
            echo "======================================================================"
        done
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