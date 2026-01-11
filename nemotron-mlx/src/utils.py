"""Utility functions for audio preprocessing and feature extraction."""

import mlx.core as mx
import numpy as np
from typing import Optional, Union


def compute_mel_spectrogram(
    audio: Union[np.ndarray, mx.array],
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
) -> mx.array:
    """
    Compute mel spectrogram from raw audio.

    Args:
        audio: Raw audio waveform [time] or [batch, time]
        sample_rate: Audio sample rate (Hz)
        n_fft: FFT size
        hop_length: Hop length for STFT
        n_mels: Number of mel bins
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)

    Returns:
        Mel spectrogram [time, n_mels] or [batch, time, n_mels]
    """
    # Convert to numpy if needed
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Add batch dimension if needed
    single_sample = audio.ndim == 1
    if single_sample:
        audio = audio[None, :]

    batch_size, audio_length = audio.shape

    # Compute STFT for each sample
    mel_specs = []
    for i in range(batch_size):
        # Simple STFT implementation
        # In production, you'd use a proper audio library like librosa
        spec = _stft(audio[i], n_fft, hop_length)

        # Convert to mel scale
        mel_basis = _mel_filter_bank(n_fft, n_mels, sample_rate, fmin, fmax)
        mel_spec = np.dot(mel_basis, spec)

        # Convert to log scale
        mel_spec = np.log(mel_spec + 1e-8)

        mel_specs.append(mel_spec)

    # Stack batch
    mel_specs = np.stack(mel_specs, axis=0)  # [batch, n_mels, time]

    # Transpose to [batch, time, n_mels]
    mel_specs = mel_specs.transpose(0, 2, 1)

    # Remove batch dimension if single sample
    if single_sample:
        mel_specs = mel_specs[0]

    # Convert to MLX array
    return mx.array(mel_specs)


def _stft(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Simple STFT implementation.

    Args:
        audio: Audio waveform [time]
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Magnitude spectrogram [n_fft//2+1, time]
    """
    # Pad audio
    pad_length = n_fft // 2
    audio = np.pad(audio, (pad_length, pad_length), mode='reflect')

    # Compute number of frames
    n_frames = 1 + (len(audio) - n_fft) // hop_length

    # Hann window
    window = np.hanning(n_fft)

    # Compute STFT
    stft = []
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = audio[start:end] * window

        # FFT
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)
        stft.append(magnitude)

    return np.stack(stft, axis=1)  # [n_fft//2+1, time]


def _mel_filter_bank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: Optional[float],
) -> np.ndarray:
    """
    Create mel filter bank.

    Args:
        n_fft: FFT size
        n_mels: Number of mel bins
        sample_rate: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel filter bank [n_mels, n_fft//2+1]
    """
    if fmax is None:
        fmax = sample_rate / 2

    # Convert Hz to mel
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Create mel scale
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bins
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filter bank
    filter_bank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]

        # Rising slope
        for j in range(left, center):
            filter_bank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            filter_bank[i, j] = (right - j) / (right - center)

    return filter_bank


def normalize_audio(
    audio: Union[np.ndarray, mx.array],
    target_db: float = -20.0,
) -> Union[np.ndarray, mx.array]:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio waveform
        target_db: Target dB level

    Returns:
        Normalized audio
    """
    is_mlx = isinstance(audio, mx.array)

    if is_mlx:
        audio = np.array(audio)

    # Compute RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # Compute current dB
    current_db = 20 * np.log10(rms + 1e-8)

    # Compute gain
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    # Apply gain
    normalized = audio * gain

    # Clip to [-1, 1]
    normalized = np.clip(normalized, -1.0, 1.0)

    if is_mlx:
        normalized = mx.array(normalized)

    return normalized


def load_audio(
    file_path: str,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    NOTE: This is a placeholder. In production, use a proper audio library
    like librosa, soundfile, or torchaudio.

    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate

    Returns:
        Audio waveform at target sample rate
    """
    # This is a placeholder implementation
    # In production, you would use:
    # import librosa
    # audio, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    # return audio

    raise NotImplementedError(
        "Audio loading not implemented. Please use librosa or similar:\n"
        "  import librosa\n"
        "  audio, sr = librosa.load(file_path, sr=16000, mono=True)\n"
        "  mel_features = compute_mel_spectrogram(audio)\n"
    )


def tokens_to_text(
    tokens: list,
    tokenizer: Optional[object] = None,
) -> str:
    """
    Convert token IDs to text.

    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer object (e.g., from HuggingFace)

    Returns:
        Decoded text string
    """
    if tokenizer is None:
        # If no tokenizer provided, just return token IDs as string
        return " ".join(map(str, tokens))

    # Use tokenizer to decode
    return tokenizer.decode(tokens)


def chunk_audio(
    mel_features: mx.array,
    chunk_size_ms: int = 80,
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> list:
    """
    Split mel spectrogram into chunks for streaming inference.

    Args:
        mel_features: Mel spectrogram [time, n_mels]
        chunk_size_ms: Chunk size in milliseconds
        sample_rate: Audio sample rate
        hop_length: STFT hop length

    Returns:
        List of mel spectrogram chunks
    """
    # Convert chunk size from ms to frames
    chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
    chunk_size_frames = chunk_size_samples // hop_length

    # Split into chunks
    total_frames = mel_features.shape[0]
    chunks = []

    for start in range(0, total_frames, chunk_size_frames):
        end = min(start + chunk_size_frames, total_frames)
        chunk = mel_features[start:end, :]
        chunks.append(chunk)

    return chunks
