"""Script to load Nemotron using NeMo and convert to MLX format."""

import argparse
import os
from pathlib import Path


def load_and_convert_with_nemo(model_name: str, output_dir: str):
    """
    Load Nemotron model using NeMo and convert to MLX format.

    Args:
        model_name: Model name or path (e.g., nvidia/nemotron-speech-streaming-en-0.6b)
        output_dir: Directory to save MLX weights
    """
    try:
        import nemo.collections.asr as nemo_asr
        import mlx.core as mx
        import torch

        print(f"Loading model with NeMo: {model_name}...")
        print("This will download the model if not cached (~2.5GB)")

        # Load the model using NeMo
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name
        )

        print("Model loaded successfully!")
        print(f"Model type: {type(asr_model)}")

        # Extract configuration
        config = asr_model.cfg
        print("\nModel Configuration:")
        print(f"  Encoder: {config.encoder._target_ if hasattr(config, 'encoder') else 'Unknown'}")
        print(f"  Decoder: {config.decoder._target_ if hasattr(config, 'decoder') else 'Unknown'}")
        print(f"  Sample rate: {config.sample_rate if hasattr(config, 'sample_rate') else 16000}")

        # Get model state dict
        state_dict = asr_model.state_dict()

        print(f"\nTotal parameters: {len(state_dict)}")
        print("\nSample parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}: {shape}")

        # Convert to MLX format
        print("\nConverting to MLX format...")
        mlx_weights = {}

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Convert to numpy then MLX
                numpy_value = value.cpu().numpy()
                mlx_weights[key] = mx.array(numpy_value)
            else:
                print(f"Warning: Skipping non-tensor parameter: {key}")

        # Save MLX weights
        output_path = Path(output_dir) / "nemotron_mlx_from_nemo.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving MLX weights to {output_path}...")
        mx.save(str(output_path), mlx_weights)

        # Save configuration
        config_path = Path(output_dir) / "model_config.yaml"
        print(f"Saving configuration to {config_path}...")

        import yaml
        from omegaconf import OmegaConf

        config_dict = OmegaConf.to_container(config, resolve=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Save tokenizer if available
        if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
            tokenizer_path = Path(output_dir) / "tokenizer.json"
            print(f"Saving tokenizer to {tokenizer_path}...")

            tokenizer_info = {
                'vocab_size': asr_model.tokenizer.vocab_size,
                'type': str(type(asr_model.tokenizer)),
            }

            # Try to save tokenizer vocabulary
            if hasattr(asr_model.tokenizer, 'vocab'):
                tokenizer_info['vocab'] = dict(asr_model.tokenizer.vocab)

            import json
            with open(tokenizer_path, 'w') as f:
                json.dump(tokenizer_info, f, indent=2)

        print("\n" + "=" * 70)
        print("Conversion complete!")
        print(f"MLX weights: {output_path}")
        print(f"Configuration: {config_path}")
        print("=" * 70)

        return str(output_path)

    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("\nTo install NeMo:")
        print("  conda create -n nemo python=3.10")
        print("  conda activate nemo")
        print("  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("  pip install nemo_toolkit['all']")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_model_info(model_name: str):
    """
    Extract model information without full conversion.

    Args:
        model_name: Model name or path
    """
    try:
        import nemo.collections.asr as nemo_asr

        print(f"Loading model: {model_name}...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

        # Print model architecture info
        print("\n" + "=" * 70)
        print("Model Architecture Information")
        print("=" * 70)

        print("\nEncoder:")
        print(f"  Type: {type(asr_model.encoder)}")
        if hasattr(asr_model.encoder, 'num_layers'):
            print(f"  Layers: {asr_model.encoder.num_layers}")

        print("\nDecoder:")
        print(f"  Type: {type(asr_model.decoder)}")

        print("\nTokenizer:")
        if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer:
            print(f"  Type: {type(asr_model.tokenizer)}")
            print(f"  Vocab size: {asr_model.tokenizer.vocab_size}")

        # Count parameters
        total_params = sum(p.numel() for p in asr_model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

        # Print sample of state dict keys
        state_dict = asr_model.state_dict()
        print(f"\nState dict keys ({len(state_dict)} total):")
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k][:5]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k][:5]

        print("\n  Encoder keys (sample):")
        for key in encoder_keys:
            print(f"    {key}")

        print("\n  Decoder keys (sample):")
        for key in decoder_keys:
            print(f"    {key}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Load Nemotron with NeMo and convert to MLX"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="Model name or HuggingFace ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../weights",
        help="Directory to save converted weights",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only extract model info, don't convert",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Nemotron Speech ASR - NeMo to MLX Converter")
    print("=" * 70)
    print()

    if args.info_only:
        extract_model_info(args.model_name)
    else:
        load_and_convert_with_nemo(args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
