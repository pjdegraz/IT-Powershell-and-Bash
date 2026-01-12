"""Attention mechanisms for FastConformer encoder."""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional caching for streaming."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_cache: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_cache: Whether to use caching for streaming inference
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_cache = use_cache

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len]
            cache: Cached (key, value) tensors for streaming

        Returns:
            output: Attention output [batch, seq_len, d_model]
            new_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Handle caching for streaming inference
        new_cache = None
        if self.use_cache:
            if cache is not None:
                cached_k, cached_v = cache
                k = mx.concatenate([cached_k, k], axis=2)
                v = mx.concatenate([cached_v, v], axis=2)
            new_cache = (k, v)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output, new_cache


class RelativePositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding (used in Conformer)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_pos_encoding: int = 5000,
        use_cache: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_pos_encoding: Maximum position for relative encoding
            use_cache: Whether to use caching for streaming inference
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_cache = use_cache
        self.max_pos_encoding = max_pos_encoding

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable biases for relative position
        self.pos_bias_u = mx.zeros((self.num_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((self.num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)

    def _compute_relative_position_encoding(self, length: int) -> mx.array:
        """Compute relative position encoding."""
        position = mx.arange(length, dtype=mx.float32)
        div_term = mx.exp(
            mx.arange(0, self.head_dim, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.head_dim)
        )

        pos_enc = mx.zeros((length, self.head_dim))
        pos_enc[:, 0::2] = mx.sin(position[:, None] * div_term)
        pos_enc[:, 1::2] = mx.cos(position[:, None] * div_term)

        return pos_enc

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
            cache: Cached (key, value) tensors for streaming

        Returns:
            output: Attention output [batch, seq_len, d_model]
            new_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Handle caching for streaming inference
        new_cache = None
        if self.use_cache:
            if cache is not None:
                cached_k, cached_v = cache
                k = mx.concatenate([cached_k, k], axis=2)
                v = mx.concatenate([cached_v, v], axis=2)
            new_cache = (k, v)

        # Compute relative position encoding
        total_len = k.shape[2]
        pos_enc = self._compute_relative_position_encoding(total_len)
        pos_enc = self.pos_proj(pos_enc)
        pos_enc = pos_enc.reshape(total_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Compute attention with relative position bias
        # Content-based addressing
        q_with_u = q + self.pos_bias_u[None, :, None, :]
        scores_content = (q_with_u @ k.transpose(0, 1, 3, 2)) / self.scale

        # Position-based addressing
        q_with_v = q + self.pos_bias_v[None, :, None, :]
        scores_pos = (q_with_v @ pos_enc.transpose(0, 2, 1)) / self.scale

        # Combine scores
        scores = scores_content + scores_pos[:, :, -seq_len:, :]

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output, new_cache
