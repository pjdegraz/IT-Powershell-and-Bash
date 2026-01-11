"""FastConformer encoder implementation for MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple
from .attention import RelativePositionMultiHeadAttention


class ConvolutionModule(nn.Module):
    """Convolution module used in Conformer blocks."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        activation: str = "swish",
    ):
        """
        Args:
            d_model: Model dimension
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
        )

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            # Note: MLX doesn't have groups parameter, we'll implement depthwise differently
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm(d_model)

        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        residual = x
        x = self.layer_norm(x)

        # Transpose for Conv1d: [batch, d_model, seq_len]
        x = x.transpose(0, 2, 1)

        # Pointwise conv 1 with GLU activation
        x = self.pointwise_conv1(x)
        x, gate = mx.split(x, 2, axis=1)
        x = x * mx.sigmoid(gate)  # GLU activation

        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise conv 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Transpose back: [batch, seq_len, d_model]
        x = x.transpose(0, 2, 1)

        return residual + x


class FeedForwardModule(nn.Module):
    """Feed-forward module used in Conformer blocks."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "swish",
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x


class FastConformerBlock(nn.Module):
    """FastConformer block with cache-aware attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_cache: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            use_cache: Whether to use caching for streaming
        """
        super().__init__()

        self.use_cache = use_cache

        # First feed-forward module (1/2 scale)
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)

        # Self-attention with relative position encoding
        self.self_attn = RelativePositionMultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            use_cache=use_cache,
        )
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)

        # Second feed-forward module (1/2 scale)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask
            cache: Cached (key, value) tensors

        Returns:
            output: Block output [batch, seq_len, d_model]
            new_cache: Updated cache
        """
        # First feed-forward (half residual)
        x = x + 0.5 * (self.ff1(x) - x)

        # Self-attention with caching
        residual = x
        x = self.attn_layer_norm(x)
        attn_out, new_cache = self.self_attn(x, mask, cache)
        x = residual + self.attn_dropout(attn_out)

        # Convolution module
        x = self.conv(x)

        # Second feed-forward (half residual)
        x = x + 0.5 * (self.ff2(x) - x)

        # Final layer norm
        x = self.final_layer_norm(x)

        return x, new_cache


class SubsamplingModule(nn.Module):
    """8x depthwise-separable convolutional downsampling."""

    def __init__(self, d_model: int, input_channels: int = 1):
        """
        Args:
            d_model: Model dimension
            input_channels: Number of input channels (1 for mono audio)
        """
        super().__init__()

        # Two layers of 2x2 strided convolution for 4x downsampling
        # Then 2x via additional stride
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=d_model // 2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=d_model // 2,
            out_channels=d_model,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
        )

        self.activation = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input audio features [batch, time, freq]

        Returns:
            Downsampled features [batch, time/8, d_model]
        """
        # Add channel dimension: [batch, 1, time, freq]
        x = x[:, None, :, :]

        # Apply convolutions
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        # Flatten spatial dimensions: [batch, time/8, d_model]
        batch, channels, time, freq = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(batch, time, channels * freq)

        return x


class FastConformerEncoder(nn.Module):
    """Cache-aware FastConformer encoder with 24 layers."""

    def __init__(
        self,
        num_layers: int = 24,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        kernel_size: int = 31,
        dropout: float = 0.1,
        input_features: int = 80,  # Mel spectrogram features
        use_cache: bool = True,
    ):
        """
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            input_features: Number of input features (mel bins)
            use_cache: Whether to use caching for streaming
        """
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.use_cache = use_cache

        # 8x subsampling
        self.subsampling = SubsamplingModule(d_model, input_channels=1)

        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)

        # Positional encoding
        self.pos_dropout = nn.Dropout(dropout)

        # FastConformer blocks
        self.layers = [
            FastConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout,
                use_cache=use_cache,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Args:
            x: Input mel spectrogram [batch, time, freq]
            mask: Attention mask
            cache: List of cached (key, value) tensors for each layer

        Returns:
            output: Encoder output [batch, time/8, d_model]
            new_cache: Updated cache for all layers
        """
        # Apply 8x subsampling
        x = self.subsampling(x)

        # Input projection
        x = self.input_proj(x)
        x = self.pos_dropout(x)

        # Initialize cache if needed
        if self.use_cache and cache is None:
            cache = [None] * self.num_layers

        # Apply all FastConformer blocks
        new_cache = [] if self.use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(x, mask, layer_cache)

            if self.use_cache:
                new_cache.append(layer_new_cache)

        return x, new_cache
