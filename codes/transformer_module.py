import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(x, pad_id):
    r"""Mask out subsequent positions."""
    # `lm_mask` shape: (1, S, S)
    lm_mask = torch.ones(1, x.size(1), x.size(1)).triu(1).bool()
    # `mask` shape: (B, 1, S)
    mask = (x == pad_id).unsqueeze(1)
    # `mask | lm_mask` shape: (B, S, S)
    return mask.to(x.device) | lm_mask.to(x.device)


def n_gram_mask(mask, n=2):
    r"""Mask out subsequent positions."""
    # `n_gram_mask` shape: (1, S, S)
    n_gram_mask = torch.ones(mask.size(-1), mask.size(-1)).tril(-n).unsqueeze(0).bool()
    # `mask | n_gram_mask` shape: (B, S, S)
    return mask | n_gram_mask.to(mask.device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        r"""
        args:
        d_model: Dimension of input vector.
        dropout: Droupout rate.
        max_len: Max length of input sequence. Default is 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq: (B, S, d_model)
        return self.dropout(input_seq + self.pe[:, :input_seq.size(1)])


class PositionalEncodingLearned(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        r"""
        args:
        d_model: Dimension of input vector.
        dropout: Droupout rate.
        max_len: Max length of input sequence. Default is 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        position = torch.arange(0, max_len)
        self.register_buffer('position', position.unsqueeze(0))

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq: (B, S, d_model)
        return self.dropout(input_seq + self.pe(self.position[:, :input_seq.size(1)]))


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.Dropout(p=dropout),
        )

        self.init_param(d_model=d_model, d_ff=d_ff)

    def init_param(self, d_ff: int, d_model: int):
        in_norm_val = 1 / math.sqrt(d_model)
        out_norm_val = 1 / math.sqrt(d_ff)

        nn.init.uniform_(self.layers[0].weight, -in_norm_val, in_norm_val)
        nn.init.uniform_(self.layers[2].weight, -out_norm_val, out_norm_val)
        print(f'FeedForward init_param finish!')

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.norm_factor = math.sqrt(self.d_k)

        # Projection linear layers of q, k, v respectively.
        self.q_range = [0, d_model]
        self.k_range = [self.q_range[1], self.q_range[1] + d_model]
        self.v_range = [self.k_range[1], self.k_range[1] + d_model]
        self.qkv_linear = nn.Linear(in_features=d_model, out_features=3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.init_param()

    def init_param(self):
        norm_val = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.qkv_linear.weight, -norm_val, norm_val)
        nn.init.uniform_(self.qkv_linear.bias, -norm_val, norm_val)
        nn.init.uniform_(self.out_linear.weight, -norm_val, norm_val)
        nn.init.uniform_(self.out_linear.bias, -norm_val, norm_val)
        print(f'MultiHeadAttention init_param finish!')

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        batch_size = x.size(0)
        seq_len = x.size(1)

        qkv = self.qkv_linear(x)
        # (B, n_heads, S, d_k)
        q = qkv[:, :, self.q_range[0]:self.q_range[1]].reshape(batch_size, seq_len, self.n_heads,
                                                               self.d_k).transpose(1, 2)
        # (B, n_heads, d_k, S)
        k = qkv[:, :, self.k_range[0]:self.k_range[1]].reshape(batch_size, seq_len, self.d_k,
                                                               self.n_heads).transpose(1, 3)
        # (B, n_heads, S, d_k)
        v = qkv[:, :, self.v_range[0]:self.v_range[1]].reshape(batch_size, seq_len, self.n_heads,
                                                               self.d_k).transpose(1, 2)

        # (B, n_heads, S, S)
        sim = q @ k / self.norm_factor
        sim.masked_fill_(mask.unsqueeze(1), -1e9)
        # (B, n_heads, S, S)
        attn = F.softmax(sim, dim=3)
        # (B, n_heads, S, d_k) -> (B, S, n_heads, d_k) -> (B, S, d_model)
        ws_x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Return shape: (B, S, `d_model`)
        return self.dropout(self.out_linear(ws_x))


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
        super().__init__()

        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        r""" MultiHeadAttention => AddNorm => FeedForward => AddNorm."""
        # Apply residual connection with layer normalization.
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.feedforward(x))
        # Return shape: (B, S, `d_model`)
        return x


class Decoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.decoder_layers:
            x = layer(x, mask)

        return x
