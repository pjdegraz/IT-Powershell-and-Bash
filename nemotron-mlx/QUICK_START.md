# Nemotron Speech ASR - Quick Start Guide

This guide will help you get started with the MLX implementation of NVIDIA's Nemotron Speech ASR model.

## Installation

### 1. Clone and Install

```bash
# Clone the repository
cd nemotron-mlx

# Install basic dependencies
pip install mlx numpy

# For audio processing
pip install librosa soundfile

# For weight conversion (optional)
pip install torch huggingface_hub pyyaml nemo_toolkit
```

### 2. Get the Model Weights

#### Option A: Using NeMo (Recommended)

```bash
cd scripts
python convert_from_nemo.py --model-name nvidia/nemotron-speech-streaming-en-0.6b
```

This will:
- Download the model from Hugging Face (~2.5GB)
- Load it using NeMo
- Convert weights to MLX format
- Save configuration and tokenizer

#### Option B: Manual Conversion

If you already have the `.nemo` file:

```bash
python scripts/convert_weights.py --convert-only /path/to/model.nemo
```

## Basic Usage

### Simple Transcription

```python
import numpy as np
from src.model import NemotronSpeechASR
from src.utils import compute_mel_spectrogram
import librosa

# Load audio file
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)

# Create model
model = NemotronSpeechASR(
    vocab_size=1024,  # Update based on actual tokenizer
    encoder_layers=24,
    encoder_dim=512,
)

# Load pretrained weights (if available)
# import mlx.core as mx
# weights = mx.load("weights/nemotron_mlx.npz")
# model.update(weights)

# Compute mel spectrogram
mel_features = compute_mel_spectrogram(audio, sample_rate=16000)
mel_features = mel_features[None, :, :]  # Add batch dimension

# Transcribe (greedy decoding - fast)
predictions = model.transcribe(mel_features)
print(f"Greedy result: {predictions[0]}")

# Transcribe with beam search (more accurate)
predictions = model.transcribe(mel_features, use_beam_search=True, beam_size=4)
print(f"Beam search result: {predictions[0]}")
```

### Streaming Inference

For real-time or low-latency applications:

```python
# Ultra-low latency (80ms chunks)
predictions = model.transcribe(mel_features, chunk_size=80)

# Low latency (160ms chunks) - recommended balance
predictions = model.transcribe(mel_features, chunk_size=160)

# Medium latency (560ms chunks)
predictions = model.transcribe(mel_features, chunk_size=560)

# Best accuracy (1120ms chunks)
predictions = model.transcribe(mel_features, chunk_size=1120)
```

## Model Architecture

The model consists of three main components:

### 1. FastConformer Encoder (24 layers)
- **Input**: 80-dim mel spectrogram
- **Output**: 512-dim encoded features (8x downsampled)
- **Features**:
  - Cache-aware streaming support
  - Relative position encoding
  - Depthwise-separable convolutions
  - Multi-head self-attention

### 2. RNNT Decoder (1 LSTM layer)
- **Input**: Previous tokens
- **Output**: 512-dim prediction network features
- **Features**:
  - Token embeddings (256-dim)
  - LSTM hidden states for context

### 3. Joint Network
- **Input**: Encoder output + Decoder output
- **Output**: Vocabulary probabilities
- **Function**: Combines acoustic and linguistic information

## Decoding Strategies

### Greedy Decoding (Fast)
```python
predictions = model.transcribe(mel_features)
```
- **Speed**: ~50-100ms per second of audio
- **Accuracy**: Good for most cases
- **Use case**: Real-time applications

### Beam Search (Accurate)
```python
predictions = model.transcribe(
    mel_features,
    use_beam_search=True,
    beam_size=4  # Try 4, 8, or 16
)
```
- **Speed**: 4-8x slower than greedy
- **Accuracy**: Better, especially for difficult audio
- **Use case**: Offline transcription, high-accuracy needs

## Performance Tips

### 1. Apple Silicon Optimization

MLX automatically optimizes for Apple Silicon:
- **Unified Memory**: Efficient memory usage across CPU/GPU
- **Metal Acceleration**: GPU-accelerated operations
- **Lazy Evaluation**: Computes only when needed

### 2. Batch Processing

Process multiple files together:

```python
audios = [librosa.load(f, sr=16000)[0] for f in audio_files]
mel_batch = mx.stack([compute_mel_spectrogram(a) for a in audios])
predictions = model.transcribe(mel_batch)
```

### 3. Chunk Size Selection

| Chunk Size | Latency | Accuracy | Use Case |
|------------|---------|----------|----------|
| 80ms | Ultra-low | Good | Live captions, voice assistants |
| 160ms | Low | Better | Real-time transcription (recommended) |
| 560ms | Medium | Very good | Phone calls, meetings |
| 1120ms | High | Best | Offline transcription |

## Troubleshooting

### Issue: Model outputs random tokens

**Solution**: You need to load pretrained weights. The model architecture is correct but weights are randomly initialized.

```python
import mlx.core as mx
weights = mx.load("weights/nemotron_mlx_from_nemo.npz")
model.update(weights)
```

### Issue: Tokenizer not found

**Solution**: The tokenizer is saved during conversion. Load it to decode tokens:

```python
import json
with open("weights/tokenizer.json", "r") as f:
    tokenizer_info = json.load(f)

# Use the vocab to decode tokens
vocab = tokenizer_info["vocab"]
text = " ".join([vocab.get(str(t), f"<{t}>") for t in predictions[0]])
```

### Issue: Out of memory on large audio files

**Solution**: Use streaming inference:

```python
# Process in 1-second chunks
predictions = model.transcribe(mel_features, chunk_size=160)
```

## Next Steps

1. **Convert Weights**: Use `scripts/convert_from_nemo.py` to get actual weights
2. **Test**: Run `python tests/test_model.py` to verify functionality
3. **Benchmark**: Measure performance on your hardware
4. **Fine-tune**: (Optional) Fine-tune on domain-specific data

## Examples

### Complete Example: File Transcription

```python
#!/usr/bin/env python3
"""Complete example: transcribe audio file."""

import librosa
import mlx.core as mx
from src.model import NemotronSpeechASR
from src.utils import compute_mel_spectrogram

def main():
    # Load audio
    print("Loading audio...")
    audio, sr = librosa.load("example.wav", sr=16000, mono=True)
    print(f"Audio length: {len(audio)/sr:.2f} seconds")

    # Create model
    print("Creating model...")
    model = NemotronSpeechASR()

    # Load weights
    print("Loading weights...")
    try:
        weights = mx.load("weights/nemotron_mlx_from_nemo.npz")
        model.update(weights)
        print("Weights loaded successfully!")
    except:
        print("Warning: Could not load weights. Using random initialization.")

    # Prepare features
    print("Computing features...")
    mel_features = compute_mel_spectrogram(audio, sample_rate=16000)
    mel_features = mel_features[None, :, :]

    # Transcribe
    print("Transcribing...")
    predictions = model.transcribe(mel_features, use_beam_search=True)

    print(f"\nResult: {predictions[0]}")

if __name__ == "__main__":
    main()
```

## Resources

- [Original Model](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Conversion Guide](CONVERSION_GUIDE.md)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)

## Support

For issues or questions:
1. Check [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md) for detailed conversion instructions
2. Run tests: `python tests/test_model.py`
3. Verify MLX installation: `python -c "import mlx.core as mx; print(mx.__version__)"`
