import torch
from transformers import T5Tokenizer, T5EncoderModel
from safetensors.torch import load_file
import argparse
import os
from tqdm import tqdm


def generate_calibration_data(
    model_path: str,
    tokenizer_name: str,
    output_path: str,
    num_samples: int = 64,
    batch_size: int = 4,
    max_length: int = 77,  # Common max length for text encoders in diffusion
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generates calibration data by passing sample prompts through a T5 text encoder.

    Args:
        model_path: Path to the .safetensors file for the T5 model.
        tokenizer_name: Name or path of the T5 tokenizer (e.g., "google/t5-v1_1-xxl").
        output_path: Path to save the generated calibration data (.pt file).
        num_samples: Total number of sample prompts to generate.
        batch_size: Batch size for processing prompts.
        max_length: Maximum sequence length for the tokenizer.
        device: Device to run the model on ("cuda" or "cpu").
    """
    print(f"Using device: {device}")

    # Try to infer tokenizer if a full model identifier is given, otherwise use specified tokenizer_name
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        print(f"Loaded tokenizer '{tokenizer_name}'")
    except Exception as e:
        print(
            f"Could not load tokenizer '{tokenizer_name}' directly: {e}. Attempting to load from model_path parent directory if applicable."
        )
        try:
            # Assuming the tokenizer might be co-located or identifiable from model_path
            # This is a common pattern if model_path is a Hugging Face model directory
            tokenizer_load_path = (
                os.path.dirname(model_path)
                if os.path.isfile(model_path)
                else model_path
            )
            if (
                not tokenizer_load_path
            ):  # If model_path is just a filename in current dir
                tokenizer_load_path = "."

            # Check if tokenizer_name might be a sub-path of a larger model repo that model_path is part of
            # For example, if model_path is ".../text_encoder/model.safetensors" and tokenizer is at "..."
            # This is a heuristic
            potential_hf_root = os.path.abspath(
                os.path.join(os.path.dirname(model_path), "..")
            )
            try:
                print(
                    f"Attempting to load tokenizer from potential Hugging Face root: {potential_hf_root}"
                )
                tokenizer = T5Tokenizer.from_pretrained(potential_hf_root)
                print(f"Successfully loaded tokenizer from {potential_hf_root}")
            except:
                print(
                    f"Failed to load tokenizer from {potential_hf_root}. Trying '{tokenizer_load_path}' next."
                )
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_load_path)
                print(f"Successfully loaded tokenizer from {tokenizer_load_path}")

        except Exception as e_fallback:
            print(
                f"Failed to load tokenizer from '{tokenizer_load_path}': {e_fallback}"
            )
            print(
                "Ensure the tokenizer_name is correct or the model_path directory contains tokenizer files."
            )
            return

    # Load the T5 Encoder model
    # T5EncoderModel is typically part of a larger T5 model structure.
    # If model_path is a .safetensors file for just the encoder,
    # we might need to load it carefully.
    # For safetensors, we might need to load state_dict and then load it into a T5Config-defined model.
    # However, `from_pretrained` can often handle .safetensors directly if a config is findable.
    try:
        print(f"Loading T5EncoderModel from '{model_path}'...")
        # First, try to load as if it's a full pretrained model directory or ID
        text_encoder = T5EncoderModel.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        print("Loaded T5EncoderModel using from_pretrained directly on model_path.")
    except Exception as e_direct:
        print(f"Failed to load T5EncoderModel directly from '{model_path}': {e_direct}")
        print(
            "Attempting to load state_dict from .safetensors and a standard T5Config."
        )
        try:
            # Fallback: Load config for a T5 XXL model, then load state dict from the .safetensors file
            # This assumes the .safetensors file ONLY contains the encoder weights.
            # We need to know the correct T5Config. Since it's t5xxl_fp16, we use a large T5 config.
            # A common one is "google/t5-v1_1-xxl" for its config, or just T5Config with appropriate params.
            # For simplicity, let's assume the tokenizer_name can also provide the config.
            config_load_name = tokenizer_name  # e.g. "google/t5-v1_1-xxl"
            from transformers import T5Config

            config = T5Config.from_pretrained(config_load_name)

            text_encoder = T5EncoderModel(config).to(torch.float16)
            state_dict = load_file(model_path, device="cpu")  # Load to CPU first

            # Adjust keys if necessary (e.g. if state_dict has a prefix like "text_encoder.")
            # This is a common case if weights are extracted from a larger model.
            # For a standalone T5EncoderModel, keys should typically not have such prefixes.
            # We'll try loading directly, and if it fails, try to strip common prefixes.
            try:
                text_encoder.load_state_dict(state_dict)
            except RuntimeError as e_state_dict:
                print(
                    f"Failed to load state_dict directly: {e_state_dict}. Checking for common prefixes..."
                )
                # Example prefix removal, this might need to be adjusted based on actual key names
                prefixes_to_try = [
                    "encoder.",
                    "model.encoder.",
                ]  # Add other potential prefixes
                cleaned_state_dict = state_dict
                for prefix in prefixes_to_try:
                    if all(key.startswith(prefix) for key in state_dict.keys()):
                        cleaned_state_dict = {
                            k[len(prefix) :]: v for k, v in state_dict.items()
                        }
                        print(
                            f"Attempting to load state_dict after removing prefix '{prefix}'"
                        )
                        try:
                            text_encoder.load_state_dict(cleaned_state_dict)
                            print("Successfully loaded state_dict with prefix removed.")
                            break  # Success
                        except RuntimeError:
                            print(f"Failed with prefix '{prefix}'.")
                            cleaned_state_dict = state_dict  # Reset for next try
                            continue
                else:  # If loop completed without break
                    raise RuntimeError(
                        "Could not load state_dict even after trying to clean prefixes."
                    )

            print(
                f"Loaded T5EncoderModel from state_dict '{model_path}' with config from '{config_load_name}'."
            )

        except Exception as e_fallback_load:
            print(f"Error loading T5EncoderModel with fallback: {e_fallback_load}")
            print(
                "Please ensure model_path is a valid .safetensors file for a T5 Encoder compatible with the tokenizer/config."
            )
            return

    text_encoder = text_encoder.to(device).eval()

    # --- Sample Prompts ---
    # Using a variety of prompts can help capture diverse activations.
    sample_prompts = [
        "A majestic lion basking in the golden hour sun on the African savanna.",
        "A futuristic cityscape with flying vehicles and towering skyscrapers.",
        "A serene forest path with sunlight filtering through the leaves.",
        "An astronaut floating in space, looking at Earth.",
        "A delicious plate of spaghetti carbonara, close-up shot.",
        "A group of friends laughing around a campfire at night.",
        "Abstract painting with vibrant colors and bold strokes.",
        "A cute kitten playing with a ball of yarn.",
        "A historical reenactment of a medieval battle.",
        "A cup of coffee with latte art.",
        "A bustling farmers market with fresh produce.",
        "A quiet library with shelves full of books.",
        "A powerful sports car racing on a track.",
        "A beautiful sunset over the ocean.",
        "A detailed portrait of an elderly person with expressive eyes.",
        "A whimsical illustration of a fantasy creature.",
        "A snowy mountain landscape.",
        "A child blowing bubbles in a park.",
        "A scientist working in a high-tech laboratory.",
        "A still life of fruits and flowers.",
        # Add more diverse prompts if needed
    ]

    # Repeat prompts to reach num_samples if necessary
    prompts_to_process = (sample_prompts * (num_samples // len(sample_prompts) + 1))[
        :num_samples
    ]
    print(f"Processing {len(prompts_to_process)} prompts for calibration data...")

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(prompts_to_process), batch_size), desc="Generating Embeddings"
        ):
            batch_prompts = prompts_to_process[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Get the encoder's last hidden state
            # For T5EncoderModel, the output is BaseModelOutput, where last_hidden_state is the first element.
            outputs = text_encoder(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
            last_hidden_state = outputs.last_hidden_state
            all_embeddings.append(last_hidden_state.cpu())  # Move to CPU before storing

    calibration_data_tensors = torch.cat(all_embeddings, dim=0)
    print(f"Generated calibration data tensor shape: {calibration_data_tensors.shape}")

    # Save the calibration data
    # xavier.py expects a list of tensors, so we'll save it as such,
    # though it's currently a single concatenated tensor.
    # For SVDQuant, typically a list of input activations to the target layers is used.
    # Here we are generating text embeddings. How these map to SVDQuant's calibration needs
    # in xavier.py might need clarification. If xavier.py's SVDQuant calibration
    # expects inputs to specific nn.Linear layers, this approach provides general text features.
    # For now, we save them as a list containing this one tensor.
    # The `load_calibration_data` in xavier.py loads a .pt file and returns its content.
    # If it's a list of tensors, it's used directly.

    # The `get_calibration_dataset_for_svdquant` in `xavier.py` (newly added)
    # actually expects a list of *input activations* to the linear layers.
    # The current script generates text *embeddings*. These are outputs of the text encoder.
    # To get *input activations* for the UNet's linear layers, one would typically run
    # a few forward passes of the full diffusion model (text encoder + UNet) with these embeddings
    # as conditioning, and capture the *inputs* to each nn.Linear layer being quantized.

    # For now, we'll save these text embeddings. The user might intend to use these
    # as inputs to the UNet for a *subsequent* calibration step that captures UNet layer inputs,
    # or perhaps the SVDQuant implementation in xavier.py is adapted to work with these directly
    # for some specific layers (e.g., cross-attention in the UNet).

    # Given `xavier.py`'s `get_calibration_dataset_for_svdquant` and `calibrate_model_svdquant`,
    # it seems it does expect to run the model and capture inputs.
    # So, these generated text embeddings are one part of the input needed for that process.
    # The `calibration_data_path` in `xavier.py`'s SVDQuant section is used to load data
    # which is then fed to the model during calibration.

    # Let's save this as a list containing the single tensor of embeddings.
    torch.save([calibration_data_tensors], output_path)
    print(f"Calibration data saved to {output_path}")
    print(
        f"Structure: List containing one tensor of shape {calibration_data_tensors.shape}"
    )
    print("This data represents text embeddings from the T5 encoder.")
    print(
        "For SVDQuant in xavier.py, these embeddings would typically be used as conditioning"
    )
    print(
        "to the main model (Chroma/FluxMod UNet) during its calibration phase, where actual"
    )
    print("input activations to the target nn.Linear layers are captured.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate calibration data using a T5 text encoder."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the .safetensors file for the T5 model (e.g., /path/to/t5xxl_fp16.safetensors)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="google/t5-v1_1-xxl",  # A common T5 XXL tokenizer
        help="Name or path of the T5 tokenizer (Hugging Face identifier).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="chroma_calibration_data.pt",
        help="Path to save the generated calibration data (.pt file).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,  # Increased default for better calibration
        help="Total number of sample prompts to generate for calibration.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,  # Adjusted for potentially large models
        help="Batch size for processing prompts.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=77,
        help="Maximum sequence length for the tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on ('cuda' or 'cpu').",
    )

    args = parser.parse_args()

    generate_calibration_data(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

# Example usage:
# python generate_chroma_calibration_data.py \
#   --model_path /home/kade/toolkit/diffusion/comfy/models/clip/t5xxl_fp16.safetensors \
#   --tokenizer_name google/t5-v1_1-xxl \
#   --output_path chroma_calibration_data.pt \
#   --num_samples 128 \
#   --batch_size 4 \
#   --device cuda

# Example usage:
# python generate_chroma_calibration_data.py \
#   --model_path /home/kade/toolkit/diffusion/comfy/models/clip/t5xxl_fp16.safetensors \
#   --tokenizer_name google/t5-v1_1-xxl \
#   --output_path chroma_calibration_data.pt \
#   --num_samples 128 \
#   --batch_size 4 \
#   --device cuda
