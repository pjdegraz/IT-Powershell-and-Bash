"""Script to download and convert Nemotron weights from Hugging Face to MLX format."""

import argparse
import os
from pathlib import Path


def download_from_huggingface(model_id: str, output_dir: str):
    """
    Download model from Hugging Face.

    Args:
        model_id: Hugging Face model ID (e.g., nvidia/nemotron-speech-streaming-en-0.6b)
        output_dir: Directory to save the model
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading {model_id} from Hugging Face...")
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Model downloaded to: {model_path}")
        return model_path

    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return None


def convert_pytorch_to_mlx(pytorch_path: str, mlx_path: str):
    """
    Convert PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch checkpoint
        mlx_path: Path to save MLX weights
    """
    try:
        import torch
        import mlx.core as mx
        import numpy as np

        print(f"Loading PyTorch checkpoint from {pytorch_path}...")

        # Load PyTorch checkpoint
        # Note: Nemotron uses NeMo format, which may require special handling
        checkpoint = torch.load(pytorch_path, map_location='cpu')

        print("Converting weights to MLX format...")

        # Convert each parameter
        mlx_weights = {}

        for key, value in checkpoint.items():
            # Convert PyTorch tensor to numpy, then to MLX
            if isinstance(value, torch.Tensor):
                numpy_value = value.numpy()

                # Handle weight transposition for linear layers
                # PyTorch uses [out_features, in_features]
                # MLX uses [out_features, in_features] (same, but verify)
                if 'weight' in key and numpy_value.ndim == 2:
                    # Check if this is a linear layer that needs transposition
                    # Most MLX linear layers follow the same convention as PyTorch
                    pass  # No transpose needed

                mlx_weights[key] = mx.array(numpy_value)
            else:
                # Handle non-tensor values
                mlx_weights[key] = value

        # Save MLX weights
        print(f"Saving MLX weights to {mlx_path}...")
        mx.save(mlx_path, mlx_weights)

        print("Conversion complete!")
        return mlx_path

    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install torch")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


def convert_nemo_to_mlx(nemo_path: str, mlx_path: str):
    """
    Convert NeMo checkpoint to MLX format.

    NeMo models use a different format than standard PyTorch checkpoints.

    Args:
        nemo_path: Path to NeMo .nemo file
        mlx_path: Path to save MLX weights
    """
    try:
        import torch
        import mlx.core as mx
        import tarfile
        import tempfile
        import yaml

        print(f"Extracting NeMo checkpoint from {nemo_path}...")

        # NeMo files are tar archives
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract .nemo file
            with tarfile.open(nemo_path, 'r') as tar:
                tar.extractall(tmpdir)

            # Load model config
            config_path = os.path.join(tmpdir, 'model_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print("Model config loaded")

            # Load weights
            weights_path = os.path.join(tmpdir, 'model_weights.ckpt')
            if not os.path.exists(weights_path):
                weights_path = os.path.join(tmpdir, 'model.ckpt')

            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location='cpu')

                # Extract state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Convert to MLX
                mlx_weights = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        mlx_weights[key] = mx.array(value.numpy())

                # Save
                print(f"Saving MLX weights to {mlx_path}...")
                mx.save(mlx_path, mlx_weights)

                print("Conversion complete!")
                return mlx_path
            else:
                print(f"Error: Could not find weights file in {tmpdir}")
                return None

    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install torch pyyaml")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Nemotron weights to MLX format"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../weights",
        help="Directory to save weights",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, don't convert",
    )
    parser.add_argument(
        "--convert-only",
        type=str,
        help="Convert existing checkpoint (provide path)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.convert_only:
        # Convert existing checkpoint
        checkpoint_path = args.convert_only
        mlx_path = str(output_dir / "nemotron_mlx.npz")

        if checkpoint_path.endswith('.nemo'):
            convert_nemo_to_mlx(checkpoint_path, mlx_path)
        else:
            convert_pytorch_to_mlx(checkpoint_path, mlx_path)

    else:
        # Download from Hugging Face
        download_dir = str(output_dir / "hf_download")
        model_path = download_from_huggingface(args.model_id, download_dir)

        if model_path and not args.download_only:
            # Find checkpoint file
            checkpoint_files = list(Path(download_dir).rglob("*.ckpt"))
            nemo_files = list(Path(download_dir).rglob("*.nemo"))

            if nemo_files:
                mlx_path = str(output_dir / "nemotron_mlx.npz")
                convert_nemo_to_mlx(str(nemo_files[0]), mlx_path)
            elif checkpoint_files:
                mlx_path = str(output_dir / "nemotron_mlx.npz")
                convert_pytorch_to_mlx(str(checkpoint_files[0]), mlx_path)
            else:
                print("Warning: No checkpoint files found.")
                print("You may need to manually convert the weights.")


if __name__ == "__main__":
    print("=" * 70)
    print("Nemotron Speech ASR - Weight Conversion Tool")
    print("=" * 70)
    print()

    main()

    print()
    print("=" * 70)
    print("Next steps:")
    print("1. Verify the converted weights")
    print("2. Update the model configuration to match the checkpoint")
    print("3. Test inference with example audio")
    print("=" * 70)
