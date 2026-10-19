# Copyright (C) 2026 Xiaomi Corporation.
"""
Convert HuggingFace model weights to PyTorch format.
Reads from Xiaomi-Robotics-0-Pretrain and saves to weights directory.
"""

import os
import sys
import torch
import argparse
import traceback
from pathlib import Path
from transformers import AutoModel, AutoProcessor

from mibot.utils.device import checkpoint_attn_argument, resolve_attn_implementation

def load_and_convert_model(model_path, output_dir, output_filename="pytorch_model.pt"):
    """Load model using transformers and save in PyTorch format."""

    print("=" * 60)
    print("Loading model with AutoModel...")
    print("=" * 60)
    print(f"\nModel path: {model_path}")

    # Load model the same way as server.py does
    print("\nLoading model (using AutoModel)...")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation=checkpoint_attn_argument(resolve_attn_implementation("cpu")), dtype=torch.bfloat16)

    # Move to CPU for saving (model is on GPU by default in server.py)
    print("Moving model to CPU for saving...")
    model = model.cpu()

    # Get state dict
    original_state_dict = model.state_dict()

    # Add "model." prefix to all keys
    state_dict = {}
    for name, tensor in original_state_dict.items():
        new_name = f"model.{name}"
        state_dict[new_name] = tensor

    print(f"\n{'='*60}")
    print(f"Model loaded successfully!")
    print(f"{'='*60}")
    print(f"\nTotal tensors: {len(state_dict)}")

    # Print tensor info
    total_params = 0
    for name, tensor in state_dict.items():
        num_params = tensor.numel()
        total_params += num_params
        print(f"  {name}: {tuple(tensor.shape)} ({tensor.dtype})")

    print(f"Total parameters: {total_params:,}")

    # Save to PyTorch format
    output_path = output_dir / output_filename
    print(f"\nSaving to: {output_path}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "module": state_dict,
    }

    torch.save(state_dict, str(output_path))
    print(f"Successfully saved!")

    # Verify the saved file
    file_size = output_path.stat().st_size / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model weights to PyTorch format.")
    parser.add_argument("--model_path", type=str, default="XiaomiRobotics/Xiaomi-Robotics-0-Pretrain", help="HuggingFace repo ID or local path")
    parser.add_argument("--output_dir", type=str, default="./pretrained_ckpt", help="Output directory for the PyTorch model")
    parser.add_argument("--output_filename", type=str, default="xr0_pretrained.pt", help="Output filename")
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("HuggingFace Model to PyTorch Converter")
    print("=" * 60)
    print(f"\nInput Model: {model_path}")
    print(f"Output directory: {output_dir.absolute()}")

    try:
        output_path = load_and_convert_model(model_path, output_dir, args.output_filename)

        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        print(f"\nSaved PyTorch weights to: {output_path.absolute()}")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
