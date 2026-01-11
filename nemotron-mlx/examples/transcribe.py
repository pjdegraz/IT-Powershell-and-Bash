"""Example script for transcribing audio with Nemotron Speech ASR."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import numpy as np
from typing import Optional
from src.model import NemotronSpeechASR
from src.utils import compute_mel_spectrogram, tokens_to_text


def transcribe_audio(
    model: NemotronSpeechASR,
    audio: np.ndarray,
    sample_rate: int = 16000,
    chunk_size_ms: Optional[int] = None,
    tokenizer=None,
):
    """
    Transcribe audio using the Nemotron model.

    Args:
        model: NemotronSpeechASR model instance
        audio: Audio waveform as numpy array
        sample_rate: Sample rate of the audio
        chunk_size_ms: Optional chunk size for streaming (80, 160, 560, or 1120)
        tokenizer: Optional tokenizer for decoding tokens to text

    Returns:
        Transcription text
    """
    # Compute mel spectrogram
    print("Computing mel spectrogram...")
    mel_features = compute_mel_spectrogram(audio, sample_rate=sample_rate)

    # Add batch dimension
    mel_features = mel_features[None, :, :]  # [1, time, n_mels]

    # Transcribe
    print("Transcribing...")
    if chunk_size_ms is not None:
        # Streaming mode
        print(f"Using streaming mode with {chunk_size_ms}ms chunks")
        chunk_size_frames = int(chunk_size_ms * sample_rate / 1000) // 160
        predictions = model.transcribe(mel_features, chunk_size=chunk_size_frames)
    else:
        # Non-streaming mode
        print("Using non-streaming mode")
        predictions = model.transcribe(mel_features)

    # Get first batch prediction
    tokens = predictions[0]

    # Convert to text
    text = tokens_to_text(tokens, tokenizer)

    return text


def main():
    """Main function demonstrating model usage."""
    print("=" * 60)
    print("Nemotron Speech ASR - MLX Implementation")
    print("=" * 60)

    # Initialize model
    print("\nInitializing model...")
    model = NemotronSpeechASR(
        vocab_size=1024,
        encoder_layers=24,
        encoder_dim=512,
        encoder_heads=8,
    )

    print(f"Model initialized with {sum(p.size for p in model.parameters())} parameters")

    # Example: Create dummy audio for testing
    print("\nCreating test audio (1 second of random noise)...")
    sample_rate = 16000
    duration = 1.0
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Normalize

    # Transcribe with different chunk sizes
    chunk_sizes = [None, 80, 160, 560, 1120]

    for chunk_size in chunk_sizes:
        print("\n" + "-" * 60)
        if chunk_size is None:
            print("Mode: Non-streaming")
        else:
            print(f"Mode: Streaming with {chunk_size}ms chunks")

        try:
            text = transcribe_audio(
                model=model,
                audio=audio,
                sample_rate=sample_rate,
                chunk_size_ms=chunk_size,
            )

            print(f"Result: {text}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("\nTo use with real audio:")
    print("  1. Install librosa: pip install librosa")
    print("  2. Load audio: audio, sr = librosa.load('file.wav', sr=16000)")
    print("  3. Transcribe: text = transcribe_audio(model, audio)")
    print("=" * 60)


if __name__ == "__main__":
    # Note: This is just a demonstration
    # For real usage, you need to:
    # 1. Download the pretrained weights from Hugging Face
    # 2. Convert them to MLX format
    # 3. Load them into the model
    main()
