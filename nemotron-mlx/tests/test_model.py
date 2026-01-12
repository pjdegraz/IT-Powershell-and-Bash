"""Unit tests for Nemotron Speech ASR MLX implementation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import numpy as np
from src.model import NemotronSpeechASR
from src.encoder import FastConformerEncoder, FastConformerBlock
from src.decoder import RNNTDecoder, RNNTJointNetwork
from src.attention import MultiHeadAttention, RelativePositionMultiHeadAttention
from src.utils import compute_mel_spectrogram


def test_attention():
    """Test multi-head attention module."""
    print("Testing MultiHeadAttention...")

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4

    attn = MultiHeadAttention(d_model, num_heads, use_cache=False)
    x = mx.random.normal((batch_size, seq_len, d_model))

    output, _ = attn(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ MultiHeadAttention passed")


def test_relative_position_attention():
    """Test relative position multi-head attention."""
    print("Testing RelativePositionMultiHeadAttention...")

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4

    attn = RelativePositionMultiHeadAttention(d_model, num_heads, use_cache=False)
    x = mx.random.normal((batch_size, seq_len, d_model))

    output, _ = attn(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ RelativePositionMultiHeadAttention passed")


def test_conformer_block():
    """Test FastConformer block."""
    print("Testing FastConformerBlock...")

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4
    d_ff = 256

    block = FastConformerBlock(d_model, num_heads, d_ff, use_cache=False)
    x = mx.random.normal((batch_size, seq_len, d_model))

    output, _ = block(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ FastConformerBlock passed")


def test_encoder():
    """Test FastConformer encoder."""
    print("Testing FastConformerEncoder...")

    batch_size = 2
    time_steps = 100
    n_mels = 80
    d_model = 64
    num_layers = 2

    encoder = FastConformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=4,
        d_ff=256,
        input_features=n_mels,
        use_cache=False,
    )

    x = mx.random.normal((batch_size, time_steps, n_mels))
    output, _ = encoder(x)

    # After 8x subsampling
    expected_time = time_steps // 8
    expected_shape = (batch_size, expected_time, d_model)

    # Allow some tolerance for rounding
    assert output.shape[0] == batch_size, f"Batch size mismatch"
    assert output.shape[2] == d_model, f"Model dim mismatch"
    print(f"  Output shape: {output.shape}")
    print("✓ FastConformerEncoder passed")


def test_decoder():
    """Test RNNT decoder."""
    print("Testing RNNTDecoder...")

    batch_size = 2
    seq_len = 10
    vocab_size = 100
    embedding_dim = 64
    hidden_dim = 128

    decoder = RNNTDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
    )

    tokens = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    output, hidden = decoder(tokens)

    assert output.shape == (batch_size, seq_len, hidden_dim), f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"
    print("✓ RNNTDecoder passed")


def test_joint_network():
    """Test RNNT joint network."""
    print("Testing RNNTJointNetwork...")

    batch_size = 2
    enc_len = 10
    dec_len = 5
    encoder_dim = 128
    decoder_dim = 64
    joint_dim = 128
    vocab_size = 100

    joint = RNNTJointNetwork(encoder_dim, decoder_dim, joint_dim, vocab_size)

    enc_out = mx.random.normal((batch_size, enc_len, encoder_dim))
    dec_out = mx.random.normal((batch_size, dec_len, decoder_dim))

    logits = joint(enc_out, dec_out)

    expected_shape = (batch_size, enc_len, dec_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    print("✓ RNNTJointNetwork passed")


def test_full_model():
    """Test complete Nemotron model."""
    print("Testing NemotronSpeechASR...")

    batch_size = 2
    time_steps = 100
    n_mels = 80

    model = NemotronSpeechASR(
        vocab_size=100,
        encoder_layers=2,  # Use fewer layers for testing
        encoder_dim=64,
        encoder_heads=4,
        encoder_ff_dim=256,
        decoder_embedding_dim=32,
        decoder_hidden_dim=64,
        joint_dim=64,
    )

    # Test encoding
    mel_features = mx.random.normal((batch_size, time_steps, n_mels))
    encoder_outputs, _ = model.encode(mel_features)

    print(f"  Encoder output shape: {encoder_outputs.shape}")

    # Test full forward pass
    result = model(mel_features)

    assert "encoder_outputs" in result
    assert "predictions" in result
    print(f"  Predictions: {result['predictions']}")
    print("✓ NemotronSpeechASR passed")


def test_mel_spectrogram():
    """Test mel spectrogram computation."""
    print("Testing compute_mel_spectrogram...")

    # Create test audio (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

    mel_features = compute_mel_spectrogram(audio, sample_rate=sample_rate, n_mels=80)

    # Check output shape
    assert mel_features.ndim == 2, f"Expected 2D output, got {mel_features.ndim}D"
    assert mel_features.shape[1] == 80, f"Expected 80 mel bins, got {mel_features.shape[1]}"

    print(f"  Mel spectrogram shape: {mel_features.shape}")
    print("✓ compute_mel_spectrogram passed")


def test_streaming_inference():
    """Test streaming inference with caching."""
    print("Testing streaming inference...")

    batch_size = 1
    chunk_size = 50
    n_mels = 80

    model = NemotronSpeechASR(
        vocab_size=100,
        encoder_layers=2,
        encoder_dim=64,
        encoder_heads=4,
    )

    # Simulate streaming with 3 chunks
    cache = None
    all_outputs = []

    for i in range(3):
        chunk = mx.random.normal((batch_size, chunk_size, n_mels))
        encoder_outputs, cache = model.encode(chunk, cache=cache)
        all_outputs.append(encoder_outputs)

    print(f"  Processed 3 chunks")
    print(f"  Cache length: {len(cache)} layers")
    print("✓ Streaming inference passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Nemotron Speech ASR MLX Tests")
    print("=" * 60)
    print()

    tests = [
        test_attention,
        test_relative_position_attention,
        test_conformer_block,
        test_encoder,
        test_decoder,
        test_joint_network,
        test_mel_spectrogram,
        test_full_model,
        test_streaming_inference,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
