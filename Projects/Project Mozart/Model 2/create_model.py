import torch
from torch import nn
import math


# -------------------------------------------------------
# Sinusoidal Positional Encoding
# -------------------------------------------------------
def sinusoidal_positional_encoding(max_len: int, embed_dim: int, device=None):
    """
    Create sinusoidal positional encodings.
    Shape: (1, max_len, embed_dim)
    """
    pe = torch.zeros(max_len, embed_dim, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device).float() *
        -(math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, embed_dim)


# -------------------------------------------------------
# MusicTransformer2 Model (with improvements)
# -------------------------------------------------------
class MusicTransformer2(nn.Module):
    """
    Transformer model for autoregressive music token prediction.
    Improvements:
        - Causal masking (no peeking ahead)
        - Sinusoidal positional encoding
        - Embedding scaling (GPT-style)
        - LayerNorm before output
    """

    def __init__(self,
                vocab_size: int,
                embed_dim: int = 512,
                n_heads: int = 8,
                n_layers: int = 8,
                ff_dim: int = 2048,
                dropout: float = 0.1,
                max_seq_len: int = 2048):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # ---- Embeddings ----
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_scale = embed_dim ** 0.5  # GPT-style scaling

        # ---- Positional Encoding ----
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(max_seq_len, embed_dim),
            persistent=False
        )

        # ---- Transformer ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- LayerNorm & Output ----
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def _generate_causal_mask(self, seq_len: int, device):
        """
        Creates an upper-triangular mask to prevent attention to future positions.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) of token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        device = x.device

        # ---- Embedding + Position ----
        x = self.embed(x) * self.embed_scale
        pos_enc = self.positional_encoding[:, :seq_len, :].to(device)
        x = x + pos_enc

        # ---- Causal Mask ----
        mask = self._generate_causal_mask(seq_len, device)

        # ---- Transformer ----
        out = self.transformer(x, mask=mask)

        # ---- LayerNorm & Projection ----
        out = self.norm(out)
        logits = self.fc(out)
        return logits


# -------------------------------------------------------
# MusicTransformer Model
# -------------------------------------------------------
class MusicTransformer(nn.Module):
    """
    A simple Transformer model for music token prediction.
    Takes a sequence of token IDs (from MIDI tokenization)
    and predicts the next token in the sequence.
    """

    def __init__(self, 
                vocab_size: int,     # number of unique tokens (notes, events, etc.)
                embed_dim: int = 256, 
                n_heads: int = 4, 
                n_layers: int = 4,
                ff_dim: int = 512,   # feedforward dimension inside Transformer
                dropout: float = 0.1,
                max_seq_len: int = 1024):
        """
        Args:
            vocab_size: total number of tokens in vocabulary
            embed_dim: embedding dimension
            n_heads: number of attention heads
            n_layers: number of transformer encoder layers
            ff_dim: feedforward layer hidden size
            dropout: dropout rate
            max_seq_len: maximum sequence length (for positional encoding)
        """
        super().__init__()

        # ---- Embeddings ----
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Learnable positional encodings (not sinusoidal)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True   # keeps shape as (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- Output Projection ----
        self.fc = nn.Linear(embed_dim, vocab_size)

    # -------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) of token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embed(x)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Pass through Transformer
        out = self.transformer(x)  # (batch, seq_len, embed_dim)

        # Project to vocabulary size
        return self.fc(out)  # (batch, seq_len, vocab_size)
