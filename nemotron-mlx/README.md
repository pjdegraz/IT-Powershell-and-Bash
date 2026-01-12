# Nemotron Speech ASR - MLX Implementation

MLX port of NVIDIA's Nemotron Speech ASR streaming transcription model for Apple Silicon.

## About

This is a conversion of NVIDIA's [Nemotron Speech Streaming model](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) to Apple's MLX framework for optimized inference on Apple Silicon devices.

### Original Model

- **Model**: NVIDIA Nemotron Speech ASR (0.6B parameters)
- **Architecture**: Cache-Aware FastConformer encoder (24 layers) + RNNT decoder
- **Features**:
  - Sub-25ms transcription latency
  - Streaming inference with configurable chunk sizes (80ms, 160ms, 560ms, 1120ms)
  - Word Error Rate (WER): 7.2% - 7.8% on standard benchmarks
  - Native punctuation and capitalization support
  - 16kHz mono audio input

### MLX Port Status

✅ **Core Implementation Complete**
- FastConformer encoder with cache-aware streaming
- RNNT decoder and joint network
- Greedy decoding implementation
- Audio preprocessing utilities
- Streaming inference support

🚧 **In Progress**
- Weight conversion from NeMo format
- Validation against original model
- Optimization for Apple Silicon

## Architecture

The model consists of two main components:

1. **FastConformer Encoder**: 24-layer cache-aware encoder with 8x convolutional downsampling
   - Multi-head self-attention with relative position encoding
   - Depthwise-separable convolution modules
   - Feed-forward networks with SiLU activation
   - Layer normalization and residual connections

2. **RNNT Decoder**: Recurrent Neural Network Transducer for streaming output
   - Token embedding layer
   - LSTM-based prediction network
   - Joint network combining encoder and decoder outputs

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repo-url>
cd nemotron-mlx

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### With Audio Processing

```bash
pip install -e ".[audio]"
```

### For Weight Conversion

```bash
pip install -e ".[conversion]"
```

## Usage

### Basic Transcription

```python
import numpy as np
from src.model import NemotronSpeechASR
from src.utils import compute_mel_spectrogram
import librosa

# Load model
model = NemotronSpeechASR(
    vocab_size=1024,
    encoder_layers=24,
    encoder_dim=512,
)

# Load audio
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)

# Compute mel spectrogram
mel_features = compute_mel_spectrogram(audio, sample_rate=16000)

# Add batch dimension
mel_features = mel_features[None, :, :]

# Transcribe
predictions = model.transcribe(mel_features)
print(f"Transcription: {predictions[0]}")
```

### Streaming Inference

```python
# Transcribe with 160ms chunks for low latency
predictions = model.transcribe(
    mel_features,
    chunk_size=160  # milliseconds
)
```

### Different Latency/Accuracy Trade-offs

```python
# Ultra-low latency (80ms chunks)
predictions = model.transcribe(mel_features, chunk_size=80)

# Low latency (160ms chunks) - recommended
predictions = model.transcribe(mel_features, chunk_size=160)

# Medium latency (560ms chunks)
predictions = model.transcribe(mel_features, chunk_size=560)

# Best accuracy (1120ms chunks)
predictions = model.transcribe(mel_features, chunk_size=1120)
```

## Weight Conversion

To use pretrained weights from Hugging Face:

```bash
cd scripts
python convert_weights.py --model-id nvidia/nemotron-speech-streaming-en-0.6b
```

See [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md) for detailed instructions.

## Testing

Run unit tests:

```bash
python tests/test_model.py
```

## Performance

Expected performance on Apple Silicon (M1/M2/M3):

- **Encoding**: ~50-100ms for 1 second of audio (24 layers)
- **Decoding**: Real-time streaming with sub-100ms latency
- **Memory**: ~2GB for full model

*Note: Actual performance depends on model size, chunk size, and hardware.*

## Project Structure

```
nemotron-mlx/
├── src/
│   ├── model.py           # Main model implementation
│   ├── encoder.py         # FastConformer encoder
│   ├── decoder.py         # RNNT decoder
│   ├── attention.py       # Attention mechanisms
│   └── utils.py           # Utility functions
├── examples/
│   └── transcribe.py      # Example transcription script
├── weights/               # Model weights (download separately)
└── tests/                 # Unit tests
```

## References

- [NVIDIA Nemotron Speech ASR Blog Post](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- [Original Model on Hugging Face](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [MLX Framework](https://github.com/ml-explore/mlx)

## License

This implementation follows the original model's NVIDIA Permissive Open Model License.
