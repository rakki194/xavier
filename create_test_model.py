import torch
from safetensors.torch import save_file
import os


# Define a simple nn.Module to wrap the Linear layer
class SimpleLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


def create_and_save_linear_model(
    output_path="test_linear_16x32.safetensors", in_features=32, out_features=16
):
    """
    Creates a simple linear model with nested keys (e.g., 'layer.weight')
    and saves its state_dict to a .safetensors file.
    """
    # 1. Define the model using the wrapper
    model = SimpleLinearModel(in_features, out_features)

    # 2. Get the state_dict
    # Keys will now be like 'layer.weight' and 'layer.bias'
    state_dict = model.state_dict()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 3. Save the state_dict to .safetensors
    try:
        save_file(state_dict, output_path)
        print(f"Model state_dict saved to: {output_path}")
        print("Contains tensors:")
        for key, tensor in state_dict.items():
            print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    except Exception as e:
        print(f"Error saving model to {output_path}: {e}")


if __name__ == "__main__":
    filename = "test_linear_16x32.safetensors"
    create_and_save_linear_model(output_path=filename)
