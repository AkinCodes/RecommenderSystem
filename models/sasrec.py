"""SASRec — Self-Attentive Sequential Recommendation.

A minimal, faithful implementation of the SASRec model (Kang & McAuley, 2018).
Given a chronological sequence of a user's liked items, the model predicts the
next item by attending over the past with a *causal* (left-to-right) mask.

Index convention (IMPORTANT — this is what the training script relies on):
  SASRec item id = (item2idx value) + 1, and 0 is reserved for PADDING.
  So the embedding table has ``num_items + 1`` rows and ``padding_idx=0`` keeps
  the PAD row's embedding pinned at zero (and receives no gradient).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAD_IDX = 0  # reserved row 0 in the item table; all short sequences are left-padded with it


class SASRec(nn.Module):
    """Self-attentive next-item recommender over a user's liked-item sequence."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # +1 row for PAD (index 0). padding_idx keeps that row's vector at 0
        # and excludes it from gradient updates, so padding never adds signal.
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=PAD_IDX)
        # Learned absolute position embeddings, one per slot in the window.
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Two identical self-attention blocks. Each block is:
        #   x -> LayerNorm -> MultiheadAttention (causal) -> residual add
        #     -> LayerNorm -> 2-layer FFN            -> residual add
        self.attn_layernorms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_blocks)]
        )
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embedding_dim, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_blocks)
            ]
        )
        self.ffn_layernorms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_blocks)]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_blocks)
            ]
        )
        self.last_layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of item sequences.

        Args:
            seq: LongTensor of shape (B, L) of SASRec item ids (0 = PAD).
        Returns:
            FloatTensor (B, L, embedding_dim): the hidden state at every position.
            The state at position t summarises items 0..t (never t+1..L-1),
            because of the causal mask built below.
        """
        _, seq_len = seq.shape
        pad_mask = seq == PAD_IDX                      # (B, L) True where padded

        # Position embeddings are RIGHT-ALIGNED: because sequences are left-padded,
        # the LAST column is always the most recent item, so we index positions
        # from the right — most recent item = position 0, the one before it = 1,
        # etc. This makes a user's summary invariant to how many PAD tokens (or
        # how long a window) precede the real history. (With naive left-to-right
        # positions, extra padding would shift every real item onto a different
        # position vector and change the encoding.) Padded columns get large
        # position indices but are zeroed out below, so they never contribute.
        col = torch.arange(seq_len, device=seq.device).unsqueeze(0)  # (1, L)
        positions = (seq_len - 1 - col).clamp(max=self.max_len - 1)   # (1, L) 0 at last column
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # A padded position holds no real item, so zero its embedding so it can
        # never contribute a value when other positions attend to it.
        x = x * (~pad_mask).unsqueeze(-1).float()

        # --- CAUSAL MASK (the crux of SASRec — expect to whiteboard this) ------
        # attn_mask[i, j] == True means "query position i is FORBIDDEN from
        # attending to key position j". We forbid every j > i, i.e. the strict
        # upper triangle. Result: position t sees only positions 0..t (its own
        # past and itself), never the future. That is what makes next-item
        # prediction honest — position t predicts item t+1 without ever having
        # peeked at it. Same mask reused for both blocks.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq.device),
            diagonal=1,
        )

        for ln_attn, attn, ln_ffn, ffn in zip(
            self.attn_layernorms, self.attn_layers, self.ffn_layernorms, self.ffns
        ):
            # Pre-norm self-attention with a residual connection.
            q = ln_attn(x)
            attn_out, _ = attn(
                q, q, q,
                attn_mask=causal_mask,             # future is blocked (causal)
                key_padding_mask=pad_mask,         # padded keys are blocked
                need_weights=False,
            )
            x = x + attn_out
            # Pre-norm position-wise FFN with a residual connection.
            x = x + ffn(ln_ffn(x))
            # Re-zero padded rows so padding stays inert across blocks.
            x = x * (~pad_mask).unsqueeze(-1).float()

        return self.last_layernorm(x)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Return the (B, embedding_dim) hidden state at each user's LAST real item.

        Sequences are LEFT-padded, so the last column (index L-1) is always the
        most recent real item — its hidden state is the user's history summary
        used to score candidate next-items via a dot product with item_emb.
        """
        hidden = self.forward(seq)                 # (B, L, D)
        return hidden[:, -1, :]                     # last position = most recent item

    def item_vectors(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for candidate items (SASRec ids, i.e. item2idx+1)."""
        return self.item_emb(item_ids)
