"""Main Nemotron Speech ASR model implementation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Dict
from .encoder import FastConformerEncoder
from .decoder import RNNTDecoder, RNNTJointNetwork, GreedyRNNTDecoder, BeamSearchRNNTDecoder


class NemotronSpeechASR(nn.Module):
    """
    Complete Nemotron Speech ASR model with FastConformer encoder and RNNT decoder.

    This model supports streaming inference with configurable chunk sizes.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        encoder_layers: int = 24,
        encoder_dim: int = 512,
        encoder_heads: int = 8,
        encoder_ff_dim: int = 2048,
        decoder_embedding_dim: int = 256,
        decoder_hidden_dim: int = 512,
        decoder_layers: int = 1,
        joint_dim: int = 512,
        dropout: float = 0.1,
        blank_index: int = 0,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ):
        """
        Args:
            vocab_size: Size of the output vocabulary
            encoder_layers: Number of encoder layers
            encoder_dim: Encoder model dimension
            encoder_heads: Number of attention heads in encoder
            encoder_ff_dim: Feed-forward dimension in encoder
            decoder_embedding_dim: Decoder embedding dimension
            decoder_hidden_dim: Decoder hidden dimension
            decoder_layers: Number of decoder LSTM layers
            joint_dim: Joint network hidden dimension
            dropout: Dropout probability
            blank_index: Index of the blank token
            sample_rate: Audio sample rate (Hz)
            n_mels: Number of mel spectrogram bins
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.blank_index = blank_index
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Encoder
        self.encoder = FastConformerEncoder(
            num_layers=encoder_layers,
            d_model=encoder_dim,
            num_heads=encoder_heads,
            d_ff=encoder_ff_dim,
            dropout=dropout,
            input_features=n_mels,
            use_cache=True,
        )

        # Decoder
        self.decoder = RNNTDecoder(
            vocab_size=vocab_size,
            embedding_dim=decoder_embedding_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )

        # Joint network
        self.joint_network = RNNTJointNetwork(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_hidden_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )

        # Greedy decoder for inference
        self.greedy_decoder = GreedyRNNTDecoder(
            decoder=self.decoder,
            joint_network=self.joint_network,
            blank_index=blank_index,
        )

        # Beam search decoder for better accuracy
        self.beam_search_decoder = BeamSearchRNNTDecoder(
            decoder=self.decoder,
            joint_network=self.joint_network,
            blank_index=blank_index,
            beam_size=4,
        )

    def encode(
        self,
        mel_features: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Encode audio features.

        Args:
            mel_features: Mel spectrogram [batch, time, n_mels]
            mask: Attention mask
            cache: Encoder cache for streaming

        Returns:
            encoder_outputs: Encoded features [batch, time/8, encoder_dim]
            new_cache: Updated cache
        """
        return self.encoder(mel_features, mask, cache)

    def decode(
        self,
        tokens: mx.array,
        hidden_states: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """
        Decode tokens.

        Args:
            tokens: Input tokens [batch, seq_len]
            hidden_states: Decoder hidden states

        Returns:
            decoder_outputs: Decoder outputs [batch, seq_len, decoder_dim]
            new_hidden_states: Updated hidden states
        """
        return self.decoder(tokens, hidden_states)

    def joint(
        self,
        encoder_outputs: mx.array,
        decoder_outputs: mx.array,
    ) -> mx.array:
        """
        Compute joint network logits.

        Args:
            encoder_outputs: Encoder outputs [batch, enc_len, encoder_dim]
            decoder_outputs: Decoder outputs [batch, dec_len, decoder_dim]

        Returns:
            logits: Joint network logits [batch, enc_len, dec_len, vocab_size]
        """
        return self.joint_network(encoder_outputs, decoder_outputs)

    def __call__(
        self,
        mel_features: mx.array,
        target_tokens: Optional[mx.array] = None,
        encoder_cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        decoder_hidden: Optional[list] = None,
    ) -> Dict[str, mx.array]:
        """
        Forward pass.

        Args:
            mel_features: Mel spectrogram [batch, time, n_mels]
            target_tokens: Target tokens for training [batch, target_len]
            encoder_cache: Encoder cache for streaming
            decoder_hidden: Decoder hidden states

        Returns:
            Dictionary containing:
                - encoder_outputs: Encoded features
                - logits: Joint network logits (if target_tokens provided)
                - predictions: Greedy predictions (if no target_tokens)
        """
        # Encode
        encoder_outputs, new_encoder_cache = self.encode(
            mel_features, cache=encoder_cache
        )

        result = {
            "encoder_outputs": encoder_outputs,
            "encoder_cache": new_encoder_cache,
        }

        # Training mode: compute logits
        if target_tokens is not None:
            decoder_outputs, new_decoder_hidden = self.decode(
                target_tokens, decoder_hidden
            )
            logits = self.joint(encoder_outputs, decoder_outputs)
            result["logits"] = logits
            result["decoder_hidden"] = new_decoder_hidden

        # Inference mode: greedy decoding
        else:
            predictions = self.greedy_decoder(encoder_outputs)
            result["predictions"] = predictions

        return result

    def transcribe(
        self,
        mel_features: mx.array,
        chunk_size: Optional[int] = None,
        use_beam_search: bool = False,
        beam_size: int = 4,
    ) -> List[List[int]]:
        """
        Transcribe audio using greedy or beam search decoding.

        Args:
            mel_features: Mel spectrogram [batch, time, n_mels]
            chunk_size: Optional chunk size for streaming (in frames)
            use_beam_search: Whether to use beam search (slower but more accurate)
            beam_size: Beam size for beam search (only if use_beam_search=True)

        Returns:
            List of token sequences (one per batch item)
        """
        if chunk_size is None:
            # Non-streaming inference
            encoder_outputs, _ = self.encode(mel_features)

            if use_beam_search:
                # Update beam size if needed
                if beam_size != self.beam_search_decoder.beam_size:
                    self.beam_search_decoder.beam_size = beam_size
                predictions = self.beam_search_decoder(encoder_outputs)
            else:
                predictions = self.greedy_decoder(encoder_outputs)

            return predictions
        else:
            # Streaming inference (uses greedy only for now)
            return self._transcribe_streaming(mel_features, chunk_size)

    def _transcribe_streaming(
        self,
        mel_features: mx.array,
        chunk_size: int,
    ) -> List[List[int]]:
        """
        Streaming transcription with caching.

        Args:
            mel_features: Mel spectrogram [batch, time, n_mels]
            chunk_size: Chunk size in frames

        Returns:
            List of token sequences
        """
        batch_size, total_frames, _ = mel_features.shape
        predictions = [[] for _ in range(batch_size)]

        encoder_cache = None
        decoder_hidden = None

        # Process in chunks
        for start_idx in range(0, total_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, total_frames)
            chunk = mel_features[:, start_idx:end_idx, :]

            # Encode chunk
            encoder_outputs, encoder_cache = self.encode(
                chunk, cache=encoder_cache
            )

            # Decode chunk
            chunk_predictions = self.greedy_decoder(encoder_outputs)

            # Accumulate predictions
            for b in range(batch_size):
                predictions[b].extend(chunk_predictions[b])

        return predictions

    @staticmethod
    def from_pretrained(model_path: str) -> "NemotronSpeechASR":
        """
        Load a pretrained model from disk.

        Args:
            model_path: Path to model weights

        Returns:
            Loaded model instance
        """
        # Load weights
        weights = mx.load(model_path)

        # Extract model configuration from weights if available
        # For now, use default configuration
        model = NemotronSpeechASR()

        # Load state dict
        model.update(weights)

        return model

    def save(self, save_path: str):
        """
        Save model weights to disk.

        Args:
            save_path: Path to save weights
        """
        weights = self.parameters()
        mx.save(save_path, weights)
