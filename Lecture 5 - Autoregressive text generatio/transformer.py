import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding:
    @classmethod
    def get_positional_encoding(cls, seq_len, d_model):
        """
        Get positional encoding for a sequence.

        Args:
            seq_len (int): The length of the sequence.
            d_model (int): The dimension of the model.

        Returns:
            Tensor: Positional encoding tensor of shape (seq_len, d_model).
        """
        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(d_model // 2).unsqueeze(0)
        angle = pos / 10000 ** (2 * i / d_model)
        positional_encoding = torch.zeros(seq_len, d_model)
        positional_encoding[:, 0::2] = torch.sin(angle)
        positional_encoding[:, 1::2] = torch.cos(angle)
        return positional_encoding


class MultiheadAttention(nn.Module):
    """
    Multihead Attention mechanism.

    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of attention heads.
        dropout (float, optional): Dropout probability. Default: 0.0.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, is_causal=False):
        """
        Forward pass for the multihead attention mechanism.

        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
            key_padding_mask (Tensor, optional): Mask to avoid attention on padding tokens. Default: None.
            is_causal (bool, optional): If True, applies causal attention mask. Default: False.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = query.size()

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

        # Linear projections for queries, keys, and values
        query = self.q_proj(query)  # (batch_size, seq_len, embed_dim)
        key = self.k_proj(key)      # (batch_size, seq_len, embed_dim)
        value = self.v_proj(value)  # (batch_size, seq_len, embed_dim)

        # Reshape the inputs to separate the heads dimension
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query=query,  # (batch_size, num_heads, seq_len, head_dim)
            key=key,      # (batch_size, num_heads, seq_len, head_dim)
            value=value,  # (batch_size, num_heads, seq_len, head_dim)
            attn_mask=key_padding_mask,  # (batch_size, seq_len) if key_padding_mask is provided
            dropout_p=(self.dropout or 0.0),
            is_causal=is_causal,
        )

        # Concatenate the heads back together
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final output projection
        return self.out_proj(attn_output)

class CausalTransformerLayer(nn.Module):
    """
    Transformer layer with causal attention.

    Args:
        d_model (int): The dimension of the model.
        nhead (int): The number of attention heads.
        dropout (float): Dropout probability.
        feedforward_dim (int): The dimension of the feedforward network.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dropout,
        feedforward_dim,
    ):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        """
        Forward pass for the transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask (Tensor, optional): Mask to avoid attention on padding tokens. Default: None.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention
        x_norm = self.norm1(x)  # Apply LayerNorm first
        attn_output = self.attn(x_norm, x_norm, x_norm, attention_mask, is_causal=True)  # (batch_size, seq_len, d_model)
        x = x + self.dropout(attn_output)  # Residual connection after the operation

        # Feed-forward
        x_norm = self.norm2(x)  # Apply LayerNorm first
        ff_output = self.ff(x_norm)  # (batch_size, seq_len, d_model)
        x = x + self.dropout(ff_output)  # Residual connection after the operation

        return x

class Transformer(nn.Module):
    """
    Transformer model with multiple layers.

    Args:
        d_model (int): The dimension of the model.
        nhead (int): The number of attention heads.
        dim_feedforward (int): The dimension of the feedforward network.
        dropout (float): Dropout probability.
        num_layers (int): The number of transformer layers.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, nhead, dropout, dim_feedforward)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, attention_mask):
        """
        Forward pass for the transformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask (Tensor): Mask to avoid attention on padding tokens.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # x: (batch_size, seq_len, d_model)
        # attention_mask: (batch_size, seq_len)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
