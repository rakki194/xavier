#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Paths to models
CHROMA_MODEL_PATH="/home/kade/toolkit/diffusion/comfy/models/unet/chroma-unlocked-v29.safetensors"
T5_TEXT_ENCODER_PATH="/home/kade/toolkit/diffusion/comfy/models/clip/t5xxl_fp16.safetensors"

# Output paths
CALIBRATION_DATA_OUTPUT_PATH="./chroma_calibration_data.pt"
QUANTIZED_MODEL_OUTPUT_PATH="./chroma-unlocked-v29-svdquant-int4.safetensors"

# Calibration data generation parameters
TOKENIZER_NAME="google/t5-v1_1-xxl" # Or any appropriate T5 tokenizer for your t5xxl model
NUM_CALIBRATION_SAMPLES=128 # Number of text samples for generating embeddings
CALIBRATION_BATCH_SIZE=4    # Batch size for generating embeddings

# Xavier SVDQuant parameters
TARGET_BITS=4

# !!! IMPORTANT: TARGET_MODULE_LIST_JSON !!!
# This list MUST contain the exact names of all torch.nn.Linear modules you want to quantize.
# Generating this list manually for a complex model like Chroma/FluxMod is error-prone.
#
# RECOMMENDATION:
# 1. Auto-generate this list: Modify xavier.py to print all `nn.Linear` module names
#    from your loaded Chroma model (see conversation for an example snippet).
# 2. OR, for a first pass to quantize ALL linear layers, set this to an empty list:
#    TARGET_MODULE_LIST_JSON='[]'
#    The SVDQuant logic in xavier.py will then target every nn.Linear it finds.
#
# The list below is a PARTIALLY CORRECTED and INCOMPLETE example.
# - Names should NOT include ".weight" or ".bias".
# - Some previous entries were likely incorrect (e.g., conv layers mistaken for linear).
TARGET_MODULE_LIST_JSON='[
    # --- Definite UNet Top-Level Linear Layers (from LDM structure) ---
    "model.diffusion_model.time_embed.1",
    "model.diffusion_model.time_embed.3",
    # "model.diffusion_model.label_emb.0", 
    #   ^ Corrected from .1. Assumes adm_in_channels creates label_emb = nn.Sequential(nn.Linear(...)). Verify.

    # --- PROBLEMATIC/LIKELY INCORRECT ENTRIES from previous list (REVIEW AND REMOVE if not nn.Linear) ---
    # "model.diffusion_model.label_emb.1", # If label_emb.0 is correct, this is wrong. If label_emb is nn.Embedding, remove.
    # "model.diffusion_model.out.2", # This is typically a Conv2d in UNet, not Linear. REMOVE.
    #
    # The following "skip_connection" layers are suspect. In standard ResBlocks (LDM),
    # skip_connections are Conv2d or Identity. If these are Linear, they are from custom blocks. VERIFY.
    # "model.diffusion_model.input_blocks.1.0.skip_connection",
    # "model.diffusion_model.input_blocks.2.0.skip_connection",
    # "model.diffusion_model.input_blocks.4.0.skip_connection",
    # "model.diffusion_model.input_blocks.5.0.skip_connection",
    # "model.diffusion_model.input_blocks.7.0.skip_connection",
    # "model.diffusion_model.input_blocks.8.0.skip_connection",
    # "model.diffusion_model.middle_block.2.skip_connection", # middle_block often doesn\'t have indexed skip like this.
    # "model.diffusion_model.output_blocks.0.0.skip_connection",
    # "model.diffusion_model.output_blocks.1.0.skip_connection",
    # "model.diffusion_model.output_blocks.2.0.skip_connection",
    # "model.diffusion_model.output_blocks.3.0.skip_connection",
    # "model.diffusion_model.output_blocks.4.0.skip_connection",
    # "model.diffusion_model.output_blocks.5.0.skip_connection",

    # --- EXAMPLES of how to name layers within BasicTransformerBlocks ---
    # You need to identify ALL such blocks and their linear layers.
    # Let\'s assume model.diffusion_model.input_blocks.0.1 is a SpatialTransformer,
    # and it contains BasicTransformerBlocks in its .transformer_blocks ModuleList.
    # If the first BasicTransformerBlock is at .transformer_blocks.0:
    #
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_q",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_k",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_v",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_out.0",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.ff.net.0.proj", # From GEGLU in FeedForward
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.ff.net.2",    # Second linear in FeedForward
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn2.to_q",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn2.to_k",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn2.to_v",
    # "model.diffusion_model.input_blocks.0.1.transformer_blocks.0.attn2.to_out.0",
    #
    # Repeat for ALL BasicTransformerBlocks in ALL SpatialTransformers throughout
    # input_blocks, middle_block, and output_blocks, using the correct indices.
    # The depth of transformer_blocks comes from unet_config (transformer_depth_encode/decode/middle).

    # --- EXAMPLES for ResBlock embedding projection layers ---
    # If model.diffusion_model.input_blocks.0.0 is a ResBlock:
    # "model.diffusion_model.input_blocks.0.0.emb_layers.1"
    #
    # Repeat for ALL ResBlocks.

    # Add ALL other nn.Linear layer names here, comma-separated, enclosed in double quotes.
    # E.g., "module.path.to.linear1", "another.module.linear2"
]'''

# SVD_RANK_FACTOR, SVD_N_ITER, SVD_N_SAMPLES will use defaults in xavier.py unless specified.
# SVD_ALPHA_MAX_ITER, SVD_ALPHA_LR, SVD_ALPHA_N_SAMPLES also use defaults.

# Python interpreter (ensure it has necessary packages like torch, transformers, safetensors)
PYTHON_CMD="python"

# --- Step 1: Generate Calibration Data (Text Embeddings) ---
echo "Step 1: Generating calibration data (text embeddings)..."
$PYTHON_CMD generate_chroma_calibration_data.py \
    --model_path "$T5_TEXT_ENCODER_PATH" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --output_path "$CALIBRATION_DATA_OUTPUT_PATH" \
    --num_samples $NUM_CALIBRATION_SAMPLES \
    --batch_size $CALIBRATION_BATCH_SIZE \
    --device cuda # Assuming CUDA is available, adjust if not

echo "Calibration data generated at $CALIBRATION_DATA_OUTPUT_PATH"

# --- Step 2: Run SVDQuant using xavier.py ---
echo "Step 2: Running SVDQuant on Chroma model using xavier.py..."
$PYTHON_CMD xavier.py \
    --model_path "$CHROMA_MODEL_PATH" \
    --output_path "$QUANTIZED_MODEL_OUTPUT_PATH" \
    --quant_method svdquant \
    --target_bits $TARGET_BITS \
    --svd_calibration_data_path "$CALIBRATION_DATA_OUTPUT_PATH" \
    --svd_target_module_list_json "$TARGET_MODULE_LIST_JSON" \
    --svd_model_arch "ComfyUI_FluxMod.flux_mod.model.FluxMod" \
    --svd_model_file "third_party/ComfyUI_FluxMod/flux_mod/model.py" 
    # Add other svdquant specific parameters if needed, e.g.:
    # --svd_rank_factor 0.1 \
    # --svd_n_iter 20 \
    # --svd_n_samples 1024 \
    # --svd_alpha_max_iter 100 \
    # --svd_alpha_lr 1e-2 \

echo "SVDQuant complete. Quantized model saved to $QUANTIZED_MODEL_OUTPUT_PATH"

echo "--- All Done ---" 