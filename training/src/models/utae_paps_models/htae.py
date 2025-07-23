import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .positional_encoding import PositionalEncoder

class HTAE2d(nn.Module):
    """
    Heavy-weight Temporal Attention Encoder (HTAE2d) for image time series.
    Adapted from LTAE by Garnot et al.
    This module computes full self-attention over the temporal dimension using a learnable
    aggregation (CLS) token. It accepts an input tensor of shape (B, T, d, H, W) and returns:
      - Aggregated features: (B, d_out, H, W)
      - (Optionally) attention masks: (n_head, B, T, H, W)
    The interface is designed to be a drop-in replacement for the lightweight LTAE2d.
    """
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,  # kept for interface consistency though not used explicitly here
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=True,
        positional_encoding=True,
    ):
        super(HTAE2d, self).__init__()
        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head

        # Project input to d_model if needed.
        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, kernel_size=1)
        else:
            self.d_model = in_channels
            self.inconv = None

        # Ensure the MLP starts with d_model.
        assert mlp[0] == self.d_model, "The first element of mlp must equal d_model"

        # Optional positional encoding.
        if positional_encoding:
            # PositionalEncoder should output a tensor with shape (..., d_model)
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        # Full multi-head attention module.
        # Note: 'average_attn_weights' is removed for compatibility with older PyTorch versions.
        self.attn_full = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )

        # Normalization layers.
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        # MLP layers.
        layers = []
        for i in range(len(mlp) - 1):
            layers.extend([
                nn.Linear(mlp[i], mlp[i + 1]),
                nn.BatchNorm1d(mlp[i + 1]),
                nn.ReLU(inplace=True)
            ])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # Learnable CLS token to aggregate information across time.
        self.cls_token = nn.Parameter(torch.randn(1, self.d_model))

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, d, H, W).
            batch_positions (Tensor): (Optional) Tensor of shape (B, T) providing temporal positions.
            pad_mask (Tensor): (Optional) Boolean mask of shape (B, T) indicating padded tokens.
            return_comp (bool): Unused in this implementation; kept for interface compatibility.
        Returns:
            out (Tensor): Aggregated features of shape (B, mlp[-1], H, W).
            (optionally) cls_attn (Tensor): Attention masks of shape (n_head, B, T, H, W)
        """
        B, T, d, H, W = x.shape

        # Process pad_mask: expand spatially and flatten so that each spatial location gets its own mask.
        if pad_mask is not None:
            # Expand to (B, T, H, W) then flatten spatial dims → (B*H*W, T)
            pad_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand(B, T, H, W)
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(B * H * W, T)

        # Reshape x from (B, T, d, H, W) to (B*H*W, T, d)
        x_seq = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, d)

        # Apply group normalization along the channel dimension.
        x_seq = self.in_norm(x_seq.permute(0, 2, 1)).permute(0, 2, 1)

        # Optionally project to d_model.
        if self.inconv is not None:
            # inconv expects shape (B*H*W, in_channels, T).
            x_seq = self.inconv(x_seq.permute(0, 2, 1)).permute(0, 2, 1)
        # Now x_seq has shape (B*H*W, T, d_model).

        # Optionally add positional encoding (only if batch_positions is provided).
        if self.positional_encoder is not None and batch_positions is not None:
            # Expand batch_positions from (B, T) → (B, T, H, W) then flatten to (B*H*W, T).
            bp = batch_positions.unsqueeze(-1).unsqueeze(-1).expand(B, T, H, W)
            bp = bp.permute(0, 2, 3, 1).contiguous().view(B * H * W, T)
            pos_enc = self.positional_encoder(bp)  # Expected shape: (B*H*W, T, d_model)
            x_seq = x_seq + pos_enc

        # Prepend the CLS token to each sequence.
        cls_tokens = self.cls_token.expand(B * H * W, -1).unsqueeze(1)  # (B*H*W, 1, d_model)
        x_seq = torch.cat([cls_tokens, x_seq], dim=1)  # (B*H*W, T+1, d_model)

        # Extend pad_mask to account for the CLS token (which is never masked).
        if pad_mask is not None:
            cls_pad = torch.zeros(B * H * W, 1, dtype=pad_mask.dtype, device=pad_mask.device)
            pad_mask = torch.cat([cls_pad, pad_mask], dim=1)  # (B*H*W, T+1)

        # Run full multi-head attention.
        # Input shape: (B*H*W, T+1, d_model).
        attn_output, attn_weights = self.attn_full(
            x_seq, x_seq, x_seq, key_padding_mask=pad_mask
        )
        # attn_output: (B*H*W, T+1, d_model)
        # attn_weights: (B*H*W, T+1, T+1) --> averaged over heads

        # Replicate the averaged attention weights along a new head dimension.
        attn_weights = attn_weights.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        # Now attn_weights has shape: (B*H*W, n_head, T+1, T+1)

        # Use the output corresponding to the CLS token as the aggregated feature.
        agg_out = attn_output[:, 0, :]  # (B*H*W, d_model)
        agg_out = agg_out.view(B, H, W, self.d_model).permute(0, 3, 1, 2)  # (B, d_model, H, W)

        # Process the aggregated feature through the MLP.
        mlp_in = agg_out.permute(0, 2, 3, 1).contiguous().view(B * H * W, self.d_model)
        mlp_out = self.dropout(self.mlp(mlp_in))
        mlp_out = mlp_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        if self.out_norm is not None:
            mlp_out = self.out_norm(mlp_out)

        # Extract the attention masks from the CLS token.
        # We take the attention weights from the CLS token (first token) to all time tokens (excluding the CLS token).
        cls_attn = attn_weights[:, :, 0, 1:]  # (B*H*W, n_head, T)
        # Reshape to (B, H, W, n_head, T) then permute to (n_head, B, T, H, W)
        cls_attn = cls_attn.view(B, H, W, self.n_head, T).permute(3, 0, 4, 1, 2)

        if self.return_att:
            return mlp_out, cls_attn
        else:
            return mlp_out
