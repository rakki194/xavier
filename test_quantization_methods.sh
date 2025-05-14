#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for verbose command printing

INPUT_MODEL="test_linear_16x32.safetensors"
BASE_OUTPUT_DIR="quantized_test_output_linear"
BASE_PLOT_DIR="quantized_test_plots_linear"

# Assuming these scripts are in the same directory or in PATH
XAVIER_PY_SCRIPT="xavier.py"
CREATE_MODEL_SCRIPT="create_test_model.py"
VERIFICATION_SCRIPT="verify_quantization.py"

# Check if dependent scripts exist
if [ ! -f "$XAVIER_PY_SCRIPT" ]; then
    echo "Error: $XAVIER_PY_SCRIPT not found. Please ensure it is in the current directory or PATH." >&2
    exit 1
fi
if [ ! -f "$CREATE_MODEL_SCRIPT" ]; then
    echo "Error: $CREATE_MODEL_SCRIPT not found. Please ensure it is in the current directory or PATH." >&2
    exit 1
fi
if [ ! -f "$VERIFICATION_SCRIPT" ]; then
    echo "Error: $VERIFICATION_SCRIPT not found. Please ensure it is in the current directory or PATH." >&2
    exit 1
fi


echo "Ensuring test model '$INPUT_MODEL' exists..."
python "$CREATE_MODEL_SCRIPT"
if [ ! -f "$INPUT_MODEL" ]; then
    echo "Error: Failed to create '$INPUT_MODEL' using $CREATE_MODEL_SCRIPT." >&2
    exit 1
fi
echo "Test model '$INPUT_MODEL' is ready."

echo "Preparing output directories..."
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_PLOT_DIR"

# Counters for summary
TOTAL_RUNS=0
SUCCESSFUL_VERIFICATIONS=0
FAILED_VERIFICATIONS=0

process_and_verify() {
    echo "DEBUG: process_and_verify CALLED. Arguments: \$1='$1', \$2='$2', \$3='$3', \$4='$4'"
    local name_suffix=$1
    echo "DEBUG: name_suffix set to '$name_suffix'"
    local quant_method_arg=$2
    echo "DEBUG: quant_method_arg set to '$quant_method_arg'"
    local fp8_type_arg=$3
    echo "DEBUG: fp8_type_arg set to '$fp8_type_arg'"
    local native_flags_arg=$4 # Can be empty string
    echo "DEBUG: native_flags_arg set to '$native_flags_arg'"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo "DEBUG: TOTAL_RUNS incremented to $TOTAL_RUNS"

    local output_subdir_name="${name_suffix}_${fp8_type_arg}"
    local full_output_dir="$BASE_OUTPUT_DIR/$output_subdir_name"
    local full_plot_dir="$BASE_PLOT_DIR/$output_subdir_name"
    local output_file="$full_output_dir/quantized_model.safetensors"

    mkdir -p "$full_output_dir"
    mkdir -p "$full_plot_dir"

    echo ""
    echo "======================================================================"
    echo "Test Run $TOTAL_RUNS: Xavier Quantization"
    echo "  Method Description: $name_suffix"
    echo "  Xavier Quant Method: $quant_method_arg"
    echo "  FP8 Type: $fp8_type_arg"
    [[ -n "$native_flags_arg" ]] && echo "  Native Flags: $native_flags_arg"
    echo "  Output File: $output_file"
    echo "  Plot Dir: $full_plot_dir"
    echo "----------------------------------------------------------------------"

    # Construct the command - ensure native_flags_arg is handled correctly if empty
    CMD_ARRAY=(python "$XAVIER_PY_SCRIPT" "$INPUT_MODEL" "$output_file" \
        --quant_method "$quant_method_arg" \
        --fp8_type "$fp8_type_arg" \
        --plot --plot_dir "$full_plot_dir" --debug)
    
    # Add native flags if they are provided
    if [[ -n "$native_flags_arg" ]]; then
        # Split native_flags_arg by space and add to CMD_ARRAY
        read -ra NATIVE_FLAGS_ARRAY <<< "$native_flags_arg"
        CMD_ARRAY+=("${NATIVE_FLAGS_ARRAY[@]}")
    fi

    echo "Executing XAVIER.PY: ${CMD_ARRAY[*]}"
    # Execute the command array to handle spaces in arguments correctly
    "${CMD_ARRAY[@]}"
    XAVIER_EXIT_STATUS=$?
    echo "DEBUG: xavier.py exited with status: $XAVIER_EXIT_STATUS"

    if [ "$XAVIER_EXIT_STATUS" -ne 0 ]; then
        echo "ERROR: xavier.py failed with exit status $XAVIER_EXIT_STATUS. See output above." >&2
        # Script will terminate here due to set -e if XAVIER_EXIT_STATUS is non-zero
        # Adding an explicit exit for clarity in case set -e was somehow bypassed or if we want to stop before verification.
        FAILED_VERIFICATIONS=$((FAILED_VERIFICATIONS + 1))
    fi

    echo "----------------------------------------------------------------------"
    echo "Verifying output: $output_file"
    if [ -f "$output_file" ]; then
        echo "  Output file created." ✅
        echo "Executing VERIFICATION_SCRIPT: python $VERIFICATION_SCRIPT $output_file $fp8_type_arg"
        python "$VERIFICATION_SCRIPT" "$output_file" "$fp8_type_arg"
        VERIFY_EXIT_STATUS=$?
        echo "DEBUG: verify_quantization.py exited with status: $VERIFY_EXIT_STATUS"

        if [ "$VERIFY_EXIT_STATUS" -eq 0 ]; then
            echo "  VERIFICATION SUCCESSFUL for $output_file" 👍
            SUCCESSFUL_VERIFICATIONS=$((SUCCESSFUL_VERIFICATIONS + 1))
        else
            echo "  ERROR: VERIFICATION FAILED for $output_file with status $VERIFY_EXIT_STATUS" ❌
            FAILED_VERIFICATIONS=$((FAILED_VERIFICATIONS + 1))
        fi
    else
        echo "  ERROR: Output file $output_file NOT created." ❌
        FAILED_VERIFICATIONS=$((FAILED_VERIFICATIONS + 1))
    fi
    echo "======================================================================"
    set -x # Ensure tracing is active for the return
    return 0 # Explicitly return 0 to ensure success for set -e
}

FP8_TYPES="e4m3 e5m2"

# --- Native Methods ---
echo ""

echo "DEBUG: Printing first separator (Line 94)"
for _i in $(seq 1 70); do printf "="; done; echo
echo "DEBUG: After first separator"

echo "      STARTING NATIVE QUANTIZATION METHOD TESTS" # Line 95

echo "DEBUG: Printing second separator (Line 96)"
set -x # Enable command tracing
for _i in $(seq 1 70); do printf "="; done; echo
SEPARATOR_EXIT_STATUS=$?
set +x # Disable command tracing
echo "DEBUG: Second separator command exit status: $SEPARATOR_EXIT_STATUS"

if [ "$SEPARATOR_EXIT_STATUS" -ne 0 ]; then
    echo "ERROR: The command to print the second separator failed with exit status $SEPARATOR_EXIT_STATUS." >&2
    echo "Script will likely terminate due to 'set -e'." >&2
    # The script will exit here if SEPARATOR_EXIT_STATUS is non-zero due to set -e
    # If it somehow didn't, force an exit to highlight the problem:
    exit 1
fi
echo "DEBUG: Successfully printed second separator. Continuing..."

echo "DEBUG: About to start the 'native methods' FP8_TYPES loop (Line 110)."
set -x # Enable command tracing for the loop itself
for fp8_type in $FP8_TYPES; do
    echo "DEBUG: INSIDE native methods loop. Current fp8_type: $fp8_type"
    # set +x # Disable command tracing for the content of process_and_verify to reduce noise

    process_and_verify "native_vanilla" "native" "$fp8_type" ""

    process_and_verify "native_comfyscale" "native" "$fp8_type" "--comfyscale"
    process_and_verify "native_complex_rounding" "native" "$fp8_type" "--complex_rounding"
    process_and_verify "native_shifturb" "native" "$fp8_type" "--shifturb"
    process_and_verify "native_owlshift" "native" "$fp8_type" "--owlshift --seed 42" # Added seed for owlshift consistency

    process_and_verify "native_comfyscale_complex" "native" "$fp8_type" "--comfyscale --complex_rounding"
    process_and_verify "native_comfyscale_shifturb" "native" "$fp8_type" "--comfyscale --shifturb"
    process_and_verify "native_comfyscale_owlshift" "native" "$fp8_type" "--comfyscale --owlshift --seed 42"
    # set -x # Re-enable for the next loop iteration check by set -x
done
set +x # Disable command tracing after the loop
echo "DEBUG: FINISHED 'native methods' FP8_TYPES loop."

# --- TorchAO Methods ---
echo ""
for _i in $(seq 1 70); do printf "="; done; echo
echo "      STARTING TORCHAO QUANTIZATION METHOD TESTS"
for _i in $(seq 1 70); do printf "="; done; echo

TORCHAO_METHODS_BASE="fp8_weight_only_aoscale fp8_weight_only_comfyscale fp8_dynamic_act_weight_aoscale fp8_dynamic_act_weight_comfyscale"

for fp8_type in $FP8_TYPES; do
    for torchao_suffix in $TORCHAO_METHODS_BASE; do
        method_name="torchao_${torchao_suffix}"
        # Using the full suffix for description to avoid ambiguity
        desc_name="torchao_${torchao_suffix}"
        process_and_verify "$desc_name" "$method_name" "$fp8_type" ""
    done
done

echo ""
for _i in $(seq 1 70); do printf "="; done; echo
echo "                 ALL QUANTIZATION TESTS COMPLETED"
for _i in $(seq 1 70); do printf "="; done; echo
echo "Summary:"
echo "  Total Test Runs: $TOTAL_RUNS"
echo "  Successful Verifications: $SUCCESSFUL_VERIFICATIONS" 👍
echo "  Failed Verifications: $FAILED_VERIFICATIONS" ❌
echo "Outputs are in: $BASE_OUTPUT_DIR"
echo "Plots are in: $BASE_PLOT_DIR"

if [ "$FAILED_VERIFICATIONS" -gt 0 ]; then
    echo "Some verifications failed. Please check the logs above." >&2
    exit 1
fi

exit 0 