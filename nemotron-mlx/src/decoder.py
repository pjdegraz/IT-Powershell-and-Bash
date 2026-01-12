"""RNNT decoder implementation for MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple


class RNNTDecoder(nn.Module):
    """
    Recurrent Neural Network Transducer (RNNT) decoder.

    The RNNT decoder predicts the next token given the current audio encoding
    and the previous predicted tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of the output vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            input_size = embedding_dim if i == 0 else hidden_dim
            self.lstm_layers.append(nn.LSTM(input_size, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(
        self,
        tokens: mx.array,
        hidden_states: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """
        Args:
            tokens: Previous tokens [batch, seq_len]
            hidden_states: Previous hidden states for each LSTM layer

        Returns:
            output: Decoder output [batch, seq_len, hidden_dim]
            new_hidden_states: Updated hidden states
        """
        # Embed tokens
        x = self.embedding(tokens)
        x = self.dropout(x)

        # Initialize hidden states if needed
        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        # Apply LSTM layers
        new_hidden_states = []
        for i, lstm in enumerate(self.lstm_layers):
            x, h = lstm(x, hidden_states[i])
            x = self.dropout(x)
            new_hidden_states.append(h)

        # Project output
        output = self.output_proj(x)

        return output, new_hidden_states


class RNNTJointNetwork(nn.Module):
    """
    Joint network for RNNT that combines encoder and decoder outputs.

    The joint network predicts the probability of each token given both
    the audio encoding and the language model state.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joint_dim: int,
        vocab_size: int,
        activation: str = "tanh",
    ):
        """
        Args:
            encoder_dim: Encoder output dimension
            decoder_dim: Decoder output dimension
            joint_dim: Joint network hidden dimension
            vocab_size: Size of output vocabulary
            activation: Activation function
        """
        super().__init__()

        self.encoder_proj = nn.Linear(encoder_dim, joint_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, joint_dim, bias=False)

        # Activation
        if activation == "tanh":
            self.activation = mx.tanh
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = lambda x: x

        # Output projection to vocabulary
        self.output_proj = nn.Linear(joint_dim, vocab_size)

    def __call__(
        self,
        encoder_outputs: mx.array,
        decoder_outputs: mx.array,
    ) -> mx.array:
        """
        Args:
            encoder_outputs: Encoder outputs [batch, enc_len, encoder_dim]
            decoder_outputs: Decoder outputs [batch, dec_len, decoder_dim]

        Returns:
            Joint network logits [batch, enc_len, dec_len, vocab_size]
        """
        # Project encoder and decoder outputs
        enc = self.encoder_proj(encoder_outputs)  # [batch, enc_len, joint_dim]
        dec = self.decoder_proj(decoder_outputs)  # [batch, dec_len, joint_dim]

        # Add dimensions for broadcasting
        enc = enc[:, :, None, :]  # [batch, enc_len, 1, joint_dim]
        dec = dec[:, None, :, :]  # [batch, 1, dec_len, joint_dim]

        # Combine encoder and decoder representations
        joint = enc + dec  # [batch, enc_len, dec_len, joint_dim]

        # Apply activation
        if callable(self.activation):
            joint = self.activation(joint)
        else:
            joint = mx.tanh(joint)

        # Project to vocabulary
        logits = self.output_proj(joint)  # [batch, enc_len, dec_len, vocab_size]

        return logits


class GreedyRNNTDecoder:
    """Greedy decoder for RNNT inference."""

    def __init__(
        self,
        decoder: RNNTDecoder,
        joint_network: RNNTJointNetwork,
        blank_index: int = 0,
        max_symbols_per_step: int = 10,
    ):
        """
        Args:
            decoder: RNNT decoder module
            joint_network: RNNT joint network
            blank_index: Index of the blank token
            max_symbols_per_step: Maximum number of symbols to emit per time step
        """
        self.decoder = decoder
        self.joint_network = joint_network
        self.blank_index = blank_index
        self.max_symbols_per_step = max_symbols_per_step

    def __call__(self, encoder_outputs: mx.array) -> list:
        """
        Greedy decoding of encoder outputs.

        Args:
            encoder_outputs: Encoder outputs [batch, time, encoder_dim]

        Returns:
            List of decoded token sequences (one per batch item)
        """
        batch_size, time_steps, _ = encoder_outputs.shape

        # Initialize decoder state
        predictions = [[] for _ in range(batch_size)]
        hidden_states = None

        # Current tokens (start with blank)
        current_tokens = mx.ones((batch_size, 1), dtype=mx.int32) * self.blank_index

        # Decode each time step
        for t in range(time_steps):
            enc_t = encoder_outputs[:, t : t + 1, :]  # [batch, 1, encoder_dim]

            # Emit symbols until blank is predicted
            symbols_count = 0
            while symbols_count < self.max_symbols_per_step:
                # Get decoder output
                dec_out, hidden_states = self.decoder(current_tokens, hidden_states)

                # Get joint network logits
                logits = self.joint_network(enc_t, dec_out)  # [batch, 1, 1, vocab_size]
                logits = logits[:, 0, 0, :]  # [batch, vocab_size]

                # Get predictions
                predicted_tokens = mx.argmax(logits, axis=-1)  # [batch]

                # Check for blank
                is_blank = predicted_tokens == self.blank_index

                # Add non-blank predictions
                for b in range(batch_size):
                    if not is_blank[b]:
                        predictions[b].append(int(predicted_tokens[b]))

                # Update current tokens
                current_tokens = predicted_tokens[:, None]  # [batch, 1]

                # Break if all predicted blank
                if mx.all(is_blank):
                    break

                symbols_count += 1

        return predictions


class BeamSearchRNNTDecoder:
    """Beam search decoder for RNNT inference with improved accuracy."""

    def __init__(
        self,
        decoder: RNNTDecoder,
        joint_network: RNNTJointNetwork,
        blank_index: int = 0,
        beam_size: int = 4,
        max_symbols_per_step: int = 10,
    ):
        """
        Args:
            decoder: RNNT decoder module
            joint_network: RNNT joint network
            blank_index: Index of the blank token
            beam_size: Number of beams to keep
            max_symbols_per_step: Maximum number of symbols to emit per time step
        """
        self.decoder = decoder
        self.joint_network = joint_network
        self.blank_index = blank_index
        self.beam_size = beam_size
        self.max_symbols_per_step = max_symbols_per_step

    def __call__(self, encoder_outputs: mx.array) -> list:
        """
        Beam search decoding of encoder outputs.

        Args:
            encoder_outputs: Encoder outputs [batch, time, encoder_dim]

        Returns:
            List of decoded token sequences (one per batch item)
        """
        batch_size, time_steps, _ = encoder_outputs.shape

        # For simplicity, process one batch item at a time
        results = []
        for b in range(batch_size):
            enc_b = encoder_outputs[b : b + 1]  # [1, time, encoder_dim]
            result = self._beam_search_single(enc_b)
            results.append(result)

        return results

    def _beam_search_single(self, encoder_outputs: mx.array) -> list:
        """
        Beam search for a single sequence.

        Args:
            encoder_outputs: Encoder outputs [1, time, encoder_dim]

        Returns:
            List of decoded tokens
        """
        time_steps = encoder_outputs.shape[1]

        # Initialize beams: (tokens, score, hidden_states)
        beams = [
            {
                "tokens": [],
                "score": 0.0,
                "hidden_states": None,
            }
        ]

        # Process each time step
        for t in range(time_steps):
            enc_t = encoder_outputs[:, t : t + 1, :]  # [1, 1, encoder_dim]

            new_beams = []

            # Expand each beam
            for beam in beams:
                tokens = beam["tokens"]
                score = beam["score"]
                hidden_states = beam["hidden_states"]

                # Try emitting symbols
                current_tokens = tokens.copy()
                current_score = score
                current_hidden = hidden_states

                symbols_emitted = 0

                while symbols_emitted < self.max_symbols_per_step:
                    # Get current token (or blank if empty)
                    if len(current_tokens) == 0:
                        token_input = mx.ones((1, 1), dtype=mx.int32) * self.blank_index
                    else:
                        token_input = mx.array([[current_tokens[-1]]], dtype=mx.int32)

                    # Get decoder output
                    dec_out, new_hidden = self.decoder(token_input, current_hidden)

                    # Get joint network logits
                    logits = self.joint_network(enc_t, dec_out)  # [1, 1, 1, vocab_size]
                    logits = logits[0, 0, 0, :]  # [vocab_size]

                    # Get log probabilities
                    log_probs = mx.log_softmax(logits, axis=-1)

                    # Get top-k candidates
                    top_k = min(self.beam_size, self.decoder.vocab_size)
                    top_scores, top_indices = mx.topk(log_probs, k=top_k)

                    # Check if blank is most likely
                    blank_score = float(log_probs[self.blank_index])

                    # Always consider blank (move to next time step)
                    new_beams.append(
                        {
                            "tokens": current_tokens,
                            "score": current_score + blank_score,
                            "hidden_states": current_hidden,
                        }
                    )

                    # Consider emitting non-blank tokens
                    for i in range(top_k):
                        token_id = int(top_indices[i])
                        token_score = float(top_scores[i])

                        if token_id != self.blank_index:
                            new_tokens = current_tokens + [token_id]
                            new_beam_score = current_score + token_score

                            new_beams.append(
                                {
                                    "tokens": new_tokens,
                                    "score": new_beam_score,
                                    "hidden_states": new_hidden,
                                }
                            )

                    # If blank is most likely, stop emitting for this beam
                    if blank_score > float(top_scores[0]):
                        break

                    # Continue with best non-blank token
                    best_non_blank_idx = 0
                    for i in range(top_k):
                        if int(top_indices[i]) != self.blank_index:
                            best_non_blank_idx = i
                            break

                    best_token = int(top_indices[best_non_blank_idx])
                    best_score = float(top_scores[best_non_blank_idx])

                    if best_token == self.blank_index:
                        break

                    current_tokens = current_tokens + [best_token]
                    current_score = current_score + best_score
                    current_hidden = new_hidden
                    symbols_emitted += 1

            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[
                : self.beam_size
            ]

        # Return best beam
        if beams:
            return beams[0]["tokens"]
        else:
            return []
