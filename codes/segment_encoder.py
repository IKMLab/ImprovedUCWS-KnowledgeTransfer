from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel

from codes.model_output import SegmentOutput
from codes.transformer_module import (subsequent_mask,
    PositionalEncodingLearned, PositionalEncoding, Decoder)
from codes.util import init_module


class SegEmbedding(nn.Module):
    r"""Define the embedding module for the segment model."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        vocab_size: int,
        init_embedding: Dict,
        is_pos: bool,
        max_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model, padding_idx=pad_id)
        init_module(self.embedding)

        if is_pos:
            self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
            # self.positional_encoding = PositionalEncodingLearned(d_model=d_model, dropout=dropout, max_len=max_len)
            # self.positional_encoding.apply(init_module)

        if init_embedding:
            embed = init_embedding['embedding']
            pos_embed = init_embedding['position']
            assert embed.shape[0] == vocab_size
            assert embed.shape[1] == d_model
            self.embedding = nn.Embedding.from_pretrained(nn.Parameter(torch.from_numpy(embed).float()))
            if pos_embed is not None:
                self.positional_encoding.pe = nn.Embedding.from_pretrained(nn.Parameter(torch.from_numpy(pos_embed[:max_len, :]).float()))

        self.embedding2vocab = nn.Linear(d_model, vocab_size)
        self.embedding2vocab.weight = self.embedding.weight

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        return self.embedding2vocab(hidden)

    def forward(self, x: Tensor):
        # `embeds` shape: (B, S, d_model)
        embeds = self.embedding(x)
        if hasattr(self, 'positional_encoding'):
            embeds = self.positional_encoding(embeds)

        return embeds

def SegmentEncoder(
        d_model: int,
        d_ff: int,
        dropout: float,
        embedding: SegEmbedding,
        n_layers: int,
        n_heads: int,
        pad_id: int,
        vocab_size: int = None,
        hug_name: str = None,
        **kwargs,
):

    if n_heads:
        return SegmentTransformerEnocder(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            n_layers=n_layers,
            n_heads=n_heads,
            pad_id=pad_id,
            embedding=embedding,
            **kwargs,
        )
    elif hug_name:
        return SegmentBERTEnocder(
            hug_name=hug_name,
            pad_id=pad_id,
            vocab_size=vocab_size,
            max_seg_len=None,
            **kwargs,
        )

    return SegmentLSTMEnocder(
        d_model=d_model,
        dropout=dropout,
        n_layers=n_layers,
        embedding=embedding,
        **kwargs,
    )

class SegmentTransformerEnocder(nn.Module):
    r"""Transformer-based Enocder."""

    def __init__(
        self,
        embedding: SegEmbedding,
        d_model: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        pad_id: int,
        **kwargs
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = embedding

        # Transformer-based encoder.
        self.encoder = Decoder(d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

        self.encoder.apply(init_module)

    def forward(self, x: Tensor, **kwargs):

        # `embeds` shape: (B, S, d_model)
        embeds = self.embedding(x)

        lm_mask = subsequent_mask(x, self.pad_id)

        # `hidden_states` shape: (B, S, d_model)
        # hidden_states = self.encoder(embeds, mask=lm_mask.to(x.device), src_key_padding_mask=src_key_padding_mask)
        hidden_states = self.encoder(embeds, mask=lm_mask.to(x.device))

        logits = self.embedding.emb2vocab(hidden_states)

        return SegmentOutput(
            logits=logits[:, :-1, :],
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def get_weight(self):
        return self.encoder.decoder_layers[-1].feedforward.layers[2].weight[0]

class SegmentLSTMEnocder(nn.Module):
    r"""LSTM-based Enocder."""

    def __init__(
        self,
        embedding: SegEmbedding,
        d_model: int,
        dropout: float,
        n_layers: int,
        **kwargs
    ):
        super().__init__()

        self.embedding = embedding

        # LSTM-based encoder.
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )

        # `h_init_state` shpae: (n_layers * num_directions, B, d_model)
        # `c_init_state` shpae: (n_layers * num_directions, B, d_model)
        self.h_init_state = nn.Parameter(torch.zeros(n_layers, 1, d_model))
        self.c_init_state = nn.Parameter(torch.zeros(n_layers, 1, d_model))

        self.encoder_input_dropout = nn.Dropout(kwargs['encoder_input_dropout_rate'])

        self.encoder.apply(init_module)

    def forward(self, x: Tensor, **kwargs):
        self.encoder.flatten_parameters()

        # `embeds` shape: (B, S, d_model)
        embeds = self.encoder_input_dropout(self.embedding(x))

        # Make LSTM init states (h, c).
        # `h` shape: (n_layers * num_directions, B, d_model)
        # `c` shape: (n_layers * num_directions, B, d_model)
        h = self.h_init_state.expand(-1, x.size(0), -1).contiguous()
        c = self.c_init_state.expand(-1, x.size(0), -1).contiguous()

        # `hidden_states` shape: (B, S, d_model)
        hidden_states, _ = self.encoder(embeds, (h, c))

        logits = self.embedding.emb2vocab(hidden_states)

        return SegmentOutput(
            logits=logits[:, :-1, :],
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        return self.embedding.emb2vocab(hidden)

    def get_weight(self):
        return self.encoder.all_weights[-1][1][0]

class SegmentBERTEnocder(nn.Module):
    r"""BERT-based Encoder."""

    def __init__(
        self,
        pad_id: int,
        hug_name: str = 'bert-base-chinese',
        vocab_size: int = 21131,
        max_seg_len: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id

        self.encoder = BertModel.from_pretrained(hug_name)
        self.encoder.resize_token_embeddings(vocab_size)

        self.embedding = self.encoder.get_input_embeddings()

        self.max_seg_len = max_seg_len
        self.vocab_size = vocab_size

    def forward(self, x: Tensor, **kwargs):
        embeds = self.embedding(x)
        # attn_mask = (x != self.pad_id).bool().to(x.device)
        attn_mask = self.generate_mask(x, self.max_seg_len)
        output = self.encoder(inputs_embeds=embeds, attention_mask=attn_mask)

        hidden_states = output.last_hidden_state

        return SegmentOutput(
            logits=None,
            embeds=embeds,
            decoder_hidden=hidden_states,
        )

    def generate_mask(self, x: Tensor, max_seg_len: int = None):
        attn_mask = (x != self.pad_id).bool().to(x.device)
        if max_seg_len:
            seq_len = x.size(1)
            # Make 3-dimension mask.
            seg_mask = (torch.ones((seq_len, seq_len))) == 1
            for i in range(seq_len):
                for j in range(1, min(max_seg_len + 1, seq_len - i)):
                    seg_mask[i, i + j] = False
            # `seg_mask` shape: (1, S, S)
            seg_mask = seg_mask[None, :, :].bool()
            attn_mask = attn_mask[:, None, :,] & seg_mask.to(x.device)
        return attn_mask

    def emb2vocab(self, hidden: Tensor):
        r"""Map embedding vectors to vocabulary."""
        return hidden @ self.embedding.weight.transpose(0, 1)
