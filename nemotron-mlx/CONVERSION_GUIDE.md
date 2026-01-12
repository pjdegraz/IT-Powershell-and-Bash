# Nemotron Speech ASR - MLX Conversion Guide

This guide explains how to convert NVIDIA's Nemotron Speech ASR model from PyTorch/NeMo format to Apple's MLX framework.

## Overview

The conversion process involves:

1. **Downloading** the original model from Hugging Face
2. **Extracting** the weights from NeMo format
3. **Converting** PyTorch tensors to MLX arrays
4. **Adapting** the model architecture to MLX

## Architecture Mapping

### PyTorch/NeMo → MLX Equivalents

| PyTorch/NeMo | MLX | Notes |
|-------------|-----|-------|
| `torch.nn.Linear` | `mlx.nn.Linear` | Same weight layout |
| `torch.nn.LSTM` | `mlx.nn.LSTM` | Similar API |
| `torch.nn.MultiheadAttention` | Custom attention | Implemented in `attention.py` |
| `torch.nn.Conv1d` | `mlx.nn.Conv1d` | Same parameters |
| `torch.nn.LayerNorm` | `mlx.nn.LayerNorm` | Same behavior |
| `torch.nn.Dropout` | `mlx.nn.Dropout` | Same behavior |

### Model Components

1. **FastConformer Encoder** (`encoder.py`)
   - 24 FastConformer blocks
   - Each block contains:
     - 2x Feed-forward modules (half residual)
     - Multi-head self-attention with relative position encoding
     - Convolution module (depthwise-separable)
     - Layer normalization
   - 8x subsampling via strided convolutions

2. **RNNT Decoder** (`decoder.py`)
   - Token embedding layer
   - LSTM layers (default: 1 layer)
   - Output projection

3. **Joint Network** (`decoder.py`)
   - Combines encoder and decoder outputs
   - Projects to vocabulary space
   - Produces logits for each time-text position

## Weight Conversion Process

### Step 1: Download from Hugging Face

```bash
cd scripts
python convert_weights.py --model-id nvidia/nemotron-speech-streaming-en-0.6b
```

This downloads the model to `../weights/hf_download/`.

### Step 2: Extract NeMo Checkpoint

NeMo checkpoints (`.nemo` files) are tar archives containing:
- `model_config.yaml` - Model configuration
- `model_weights.ckpt` - PyTorch state dict

The conversion script automatically extracts these.

### Step 3: Convert Weights

The script converts each PyTorch tensor:

```python
# PyTorch tensor → NumPy → MLX array
pytorch_tensor = checkpoint['encoder.layers.0.weight']
numpy_array = pytorch_tensor.numpy()
mlx_array = mx.array(numpy_array)
```

### Step 4: Weight Transposition

Some weights may need transposition:

- **Linear layers**: PyTorch and MLX both use `[out_features, in_features]` - no transpose
- **Conv layers**: Check channel ordering (PyTorch: NCHW, MLX: NHWC for Conv2d)
- **Attention**: Verify head dimension ordering

### Step 5: Save in MLX Format

```python
mx.save("nemotron_mlx.npz", mlx_weights)
```

## Key Differences: NeMo vs MLX

### 1. Streaming/Caching

**NeMo**: Uses cache-aware streaming with explicit cache management
**MLX**: We implement similar caching by passing cache tensors through layers

```python
# MLX implementation
encoder_outputs, cache = encoder(features, cache=previous_cache)
```

### 2. Convolution Modules

**NeMo**: Uses depthwise-separable convolutions
**MLX**: Implement using standard Conv1d (groups parameter may differ)

### 3. Relative Position Encoding

**NeMo**: Built into FastConformer attention
**MLX**: Custom implementation in `attention.py`

## Testing the Conversion

### 1. Load Converted Weights

```python
from src.model import NemotronSpeechASR
import mlx.core as mx

# Load weights
weights = mx.load("weights/nemotron_mlx.npz")

# Create model
model = NemotronSpeechASR()

# Update weights
model.update(weights)
```

### 2. Test Inference

```python
import numpy as np
from src.utils import compute_mel_spectrogram

# Load audio (using librosa)
import librosa
audio, sr = librosa.load("test.wav", sr=16000)

# Compute features
mel_features = compute_mel_spectrogram(audio)

# Transcribe
mel_features = mel_features[None, :, :]  # Add batch dim
predictions = model.transcribe(mel_features)
```

### 3. Compare Outputs

Compare MLX outputs with original PyTorch/NeMo outputs:

1. Use same audio input
2. Extract encoder outputs at each layer
3. Compare logits from joint network
4. Verify final predictions match

## Common Issues

### Issue 1: Shape Mismatches

**Problem**: Weight shapes don't match between PyTorch and MLX
**Solution**: Check for needed transpositions, especially in Conv layers

### Issue 2: Missing Keys

**Problem**: Some keys in checkpoint don't exist in MLX model
**Solution**: Update weight mapping in conversion script

### Issue 3: NaN/Inf Values

**Problem**: Model produces NaN or Inf during inference
**Solution**:
- Check numerical stability (add epsilon to divisions)
- Verify activation functions
- Check normalization layers

### Issue 4: Performance Differences

**Problem**: MLX model outputs differ from PyTorch
**Solution**:
- Verify dropout is disabled during inference
- Check batch normalization in eval mode
- Compare intermediate layer outputs

## Optimization Tips

### 1. Use MLX-Specific Optimizations

```python
# Enable automatic mixed precision
mx.set_default_dtype(mx.float16)

# Use efficient operations
# MLX automatically optimizes graph execution
```

### 2. Batch Processing

```python
# Process multiple audio files together
batch_features = mx.stack([compute_mel_spectrogram(a) for a in audios])
batch_predictions = model.transcribe(batch_features)
```

### 3. Streaming Optimization

```python
# Use appropriate chunk sizes for your use case
# Smaller chunks = lower latency, possibly lower accuracy
# Larger chunks = higher latency, better accuracy

chunk_sizes = {
    "ultra_low": 80,    # 80ms - lowest latency
    "low": 160,         # 160ms
    "medium": 560,      # 560ms
    "high": 1120,       # 1120ms - best accuracy
}
```

## Next Steps

1. **Validate accuracy**: Compare WER on benchmark datasets
2. **Optimize performance**: Profile and optimize bottlenecks
3. **Add features**: Implement beam search, language model integration
4. **Fine-tune**: Optionally fine-tune on domain-specific data

## Resources

- [NVIDIA Nemotron Blog Post](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [NeMo Framework Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
- [Original Model](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
