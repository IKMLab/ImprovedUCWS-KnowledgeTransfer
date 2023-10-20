r"""SLM Model."""
import json
import six
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from codes.segment_encoder import SegEmbedding, SegmentEncoder
from codes.segment_decoder import SegmentDecoder


class SLMConfig:
    r"""Configuration for `SegmentalLM`."""

    def __init__(
        self,
        embedding_path: str = None,
        vocab_file: str = None,
        vocab_size: str = None,
        embedding_size: int = 256,
        hidden_size: int = 256,
        max_segment_length: int = 4,
        encoder_layer_number: int = 1,
        decoder_layer_number: int = 1,
        encoder_input_dropout_rate: float = 0.0,
        decoder_input_dropout_rate: float = 0.0,
        encoder_dropout_rate: float = 0.0,
        decoder_dropout_rate: float = 0.0,
        punc_id: int = 2,
        num_id: int = 3,
        eos_id: int = 5,
        eng_id: int = 7,
        **kwargs
    ):
        r"""Constructs SLMConfig."""
        self.embedding_path = embedding_path
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_segment_length = max_segment_length
        self.encoder_layer_number = encoder_layer_number
        self.decoder_layer_number = decoder_layer_number
        self.encoder_input_dropout_rate = encoder_input_dropout_rate
        self.decoder_input_dropout_rate = decoder_input_dropout_rate
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.eos_id = eos_id
        self.punc_id = punc_id
        self.eng_id = eng_id
        self.num_id = num_id

    @classmethod
    def from_dict(cls, json_object):
        r"""Constructs a `SLMConfig` from a Python dictionary of parameters."""
        config = SLMConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        r"""Constructs a `SLMConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        r"""Serializes this instance to a Python dictionary."""
        output = self.__dict__
        return output

    def to_json_string(self):
        r"""Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_json_file(self, json_file):
        r"""Serializes this instance to a JSON file."""
        with open(json_file, "w") as writer:
            writer.write(self.to_json_string())


class SegmentalLM(nn.Module):
    r"""Segmental Language Model.
    Estimate the likelihoods of small text fragments, progressing one step at a
    time until the end of the given input sentence. With the likelihoods of all
    possible word units, utilize dynamic programming to identify the best
    segmentations.
    """

    def __init__(
        self,
        config: SLMConfig,
        init_embedding = None,
        hug_name: str = None,
        n_heads: int = None,
        dec_n_heads: int = None,
        max_len: int = 32,
        pad_id: int = 0,
        **kwargs,
    ):
        super(SegmentalLM, self).__init__()

        self.config = config

        # Whether use pre-trained embedding (word-vector) or not.
        if init_embedding is not None:
            assert np.shape(init_embedding)[0] == config.vocab_size
            assert np.shape(init_embedding)[1] == config.embedding_size
            shared_embedding = nn.Parameter(torch.from_numpy(init_embedding).float())
        else:
            shared_embedding = torch.zeros(config.vocab_size, config.embedding_size)
            nn.init.uniform_(shared_embedding, a=-1.0, b=1.0)


        embedding = SegEmbedding(
            d_model=config.embedding_size,
            dropout=0.1,
            vocab_size=config.vocab_size,
            init_embedding={
                'embedding': shared_embedding.detach().numpy(),
                'position': None,
            },
            is_pos=False if n_heads is None else True,
            max_len=max_len,
            pad_id=pad_id,
        )

        self.context_encoder = SegmentEncoder(
            d_model=config.embedding_size,
            d_ff=config.hidden_size,
            dropout=config.encoder_dropout_rate,
            embedding=embedding,
            n_layers=config.encoder_layer_number,
            n_heads=n_heads,
            vocab_size=config.vocab_size,
            pad_id=pad_id,
            encoder_input_dropout_rate=config.encoder_input_dropout_rate,
            hug_name=hug_name,
            config=config, # kwargs
            max_len=max_len, # kwargs
        )

        self.segment_decoder = SegmentDecoder(
            d_model=config.hidden_size,
            dec_n_layers=config.decoder_layer_number,
            dec_n_heads=dec_n_heads,
            dropout=config.decoder_dropout_rate,
        )

        self.decoder_input_dropout = nn.Dropout(p=config.decoder_input_dropout_rate)

    def forward(
        self,
        x: Tensor,
        lengths: List[int],
        segments: List[int] = None,
        mode: str = 'unsupervised',
        supervised_weight: float = 0.1
    ):
        if mode == 'supervised' and segments is None:
            raise ValueError('Supervised mode needs segmented text.')

        # input format: (seq_len, batch_size)
        x = x.transpose(0, 1).contiguous()

        # transformed format: (seq_len, batch_size)
        max_length, batch_size = x.size()

        loginf = 1000000.0

        max_length = max(lengths)

        lm_output = self.context_encoder(x.transpose(0, 1).contiguous())
        inputs = lm_output.embeds.transpose(0, 1)
        encoder_output = lm_output.decoder_hidden.transpose(0, 1)

        is_single = -loginf * ((x == self.config.punc_id) | (x == self.config.eng_id) | (x == self.config.num_id)).type_as(x)

        if mode == 'supervised':
          is_single = torch.zeros_like(is_single)

        neg_inf_vector = torch.full_like(inputs[0, : ,0], -loginf)

        logpy = neg_inf_vector.repeat(max_length - 1, self.config.max_segment_length, 1)
        logpy[0][0] = 0

        # Make context encoder and segment decoder have different learning rate.
        encoder_output = encoder_output * 0.5

        seg_dec_hiddens = self.segment_decoder.gen_start_symbol_hidden(encoder_output)

        for j_start in range(1, max_length - 1):
            j_end = j_start + min(self.config.max_segment_length, (max_length-1) - j_start)

            decoder_output = self.segment_decoder(
                seg_start_hidden=seg_dec_hiddens[j_start-1, :, :].unsqueeze(0),
                seg_embeds=self.decoder_input_dropout(inputs[j_start:j_end, :, :]),
            )
            decoder_output = self.context_encoder.emb2vocab(decoder_output)
            decoder_logpy = F.log_softmax(decoder_output, dim=2)

            decoder_target = x[j_start:j_end, :]

            target_logpy = decoder_logpy[:-1, :, :].gather(dim=2, index=decoder_target.unsqueeze(-1)).squeeze(-1)

            tmp_logpy = torch.zeros_like(target_logpy[0])

            # j is a temporary j_end.
            for j in range(j_start, j_end):
                tmp_logpy = tmp_logpy + target_logpy[j - j_start, :]
                if j > j_start:
                    tmp_logpy = tmp_logpy + is_single[j, :]
                if j == j_start + 1:
                    tmp_logpy = tmp_logpy + is_single[j_start, :]
                logpy[j_start][j - j_start] = tmp_logpy + decoder_logpy[j - j_start + 1, :, self.config.eos_id]

        if mode == 'unsupervised' or mode == 'supervised':

            # Total log probability.
            # Log probability for generate [bos] at beginning is 0.
            alpha = neg_inf_vector.repeat(max_length - 1, 1)
            alpha[0] = 0

            for j_end in range(1, max_length - 1):
                logprobs = []
                for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                    logprobs.append(alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1])
                alpha[j_end] =  torch.logsumexp(torch.stack(logprobs), dim=0)

            NLL_loss = 0.0
            total_length = 0

            index = (torch.LongTensor(lengths) - 2).view(1, -1)

            NLL_loss = - torch.gather(input=alpha, dim=0, index=index.to(x.device))

            assert NLL_loss.view(-1).size(0) == batch_size

            total_length += sum(lengths) - 2 * batch_size

            normalized_NLL_loss = NLL_loss.sum() / float(total_length)

            if mode == 'supervised':

                supervised_NLL_loss = 0.0
                total_length = 0

                # Ex:
                # Texts: 今天  天氣  非常好
                # segments[i] : 2 2 3
                for i in range(batch_size):
                    j_start = 1
                    for j_length in segments[i]:
                        if j_length <= self.config.max_segment_length:
                            supervised_NLL_loss = supervised_NLL_loss - logpy[j_start][j_length - 1][i]
                            total_length += j_length
                        j_start += j_length

                normalized_supervised_NLL_loss = supervised_NLL_loss / float(total_length)

                normalized_NLL_loss = normalized_supervised_NLL_loss * supervised_weight + normalized_NLL_loss

                return normalized_NLL_loss, normalized_supervised_NLL_loss

            return normalized_NLL_loss

        elif mode == 'decode':
            ret = []

            # sentence_length = true_length + 2 (bos, eos)
            # alpha: List(-Inf) = 紀錄下當前的分割選擇機率
            # path: List(-Inf) = 紀錄下路徑
            for i in range(batch_size):
                alpha = [-loginf] * (lengths[i] - 1)
                path = [-1] * (lengths[i] - 1)
                alpha[0] = 0.0
                for j_end in range(1, lengths[i] - 1):
                    for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                        logprob = alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1][i].item()
                        if logprob > alpha[j_end]:
                            alpha[j_end] = logprob
                            path[j_end] = j_start

                j_end = lengths[i] - 2
                segment_lengths = []
                while j_end > 0:
                    prev_j = path[j_end]
                    segment_lengths.append(j_end - prev_j)
                    j_end = prev_j

                segment_lengths = segment_lengths[::-1]

                ret.append(segment_lengths)

            return ret

        else:
            raise ValueError('Mode %s not supported' % mode)
