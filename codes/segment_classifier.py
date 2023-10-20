r"""The classifier for sequecne tagging.
Tagging schema:
    binary: Word boundary or not.
    B/M/E/S: Begin/Middle/End/Single.
    Spin-WS: Predict whether two characters are connected or not.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from codes.segment_encoder import SegEmbedding, SegmentEncoder
from codes.util import init_module


class ClassifierModel(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        init_embedding,
        d_model: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        pad_id: int,
        encoder: SegmentEncoder = None,
        **kwargs,
    ) -> None:
        super().__init__()
        r"""Classifier model's encoder.(LSTM-based, pre-trained-based)"""

        if encoder is not None:
            self.encoder = encoder
        else:
            shard_embedding = nn.Parameter(torch.from_numpy(init_embedding).float())
            self.embedding = SegEmbedding(
                d_model=embedding_size,
                dropout=0.1,
                vocab_size=vocab_size,
                init_embedding={
                    'embedding': shard_embedding.detach().numpy(),
                    'position': None,
                },
                is_pos=False if n_heads is None else True,
                max_len=32,
                pad_id=pad_id,
            )
            self.encoder = SegmentEncoder(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                embedding=self.embedding,
                n_layers=n_layers,
                n_heads=n_heads,
                pad_id=pad_id,
                max_len=None,
                vocab_size=None,
                encoder_input_dropout_rate=dropout,
            )

            self.encoder.apply(init_module)

        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )


        self.pooler.apply(init_module)

    def forward(self, x: Tensor, attention_mask: Tensor):
        output = self.encoder(x)
        encoder_outputs = output.decoder_hidden # * 0.5

        pool_output = self.pooler(encoder_outputs)

        # `pool_output` shape: (B, S, d_model)
        return (pool_output, )


class SegmentClassifier(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        init_embedding,
        d_model: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        model_type: str,
        pad_id: int,
        tk_cls,
        encoder: SegmentEncoder = None,
        num_labels: int = 2,
        label_smoothing: float = 0.0,
        is_blank: bool = False,
        is_c_nc: bool = False,
        is_mlm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Tagging schema from SpIn-WS (Tong, Yu, et al. ACL, 2022.)
        self.is_c_nc = is_c_nc
        self.is_mlm = is_mlm

        self.num_labels = num_labels
        self.pad_id = tk_cls.pad_id
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

        self.dropout = nn.Dropout(dropout)

        logging.info(f'=== Shared encoder or not: {encoder=} ===')

        if 'lstm_encoder' not in model_type:
            if is_blank:
                # Load the empty BERT model.
                logging.info('=== Load empty BERT model. ===')
                config = AutoConfig.from_pretrained(model_type)
                self.model = AutoModel.from_config(config=config)
            else:
                logging.info('=== Load pre-trained BERT model. ===')
                self.model = AutoModel.from_pretrained(model_type)

            self.model.resize_token_embeddings(len(tk_cls))
            # d_model = 768
            assert d_model == 768
        else:
            # If enocder is not None, means that share encoder.
            self.model = ClassifierModel(
                embedding_size=embedding_size,
                vocab_size=vocab_size,
                init_embedding=init_embedding,
                encoder=encoder,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                n_layers=n_layers,
                n_heads=n_heads,
                pad_id=pad_id,
            )

        # For token's tag prediction.
        # Whether use SpIn-WS schema or not. (d_model*2)
        self.classifier = nn.Linear(
            in_features=d_model*2 if self.is_c_nc else d_model,
            out_features=num_labels
        )
        init_module(self.classifier)

        if self.is_mlm:
            config = BertConfig.from_pretrained(model_type)
            config.vocab_size = len(tk_cls)
            self.lm_classifier = BertOnlyMLMHead(config)


    def forward(self, x: Tensor, labels: Tensor = None):
        attention_mask = (x != self.pad_id).bool()
        model_output = self.model(x, attention_mask=attention_mask.to(x.device))

        # `sequence_output` shape: (B, S, d_model)
        sequence_output = model_output[0]

        if self.is_c_nc:
            # SpIn-WS method from (Tong, Yu, et al, ACL, 2022).

            # `first` shape: (B, S-1, d_model) 0 ~ n-1
            # `second` shape: (B, S-1, d_model) 1 ~ n
            # `rest` shape: (B, 1, d_model*2)
            first = sequence_output[:, :-1, :]
            second = sequence_output[:, 1:, :]
            rest = sequence_output[:, -1, :][:, None, :].repeat(1, 1, 2)

            # `sequence_output` shape: (B, S-1, d_model*2)
            # `sequence_output` shape: (B, S, d_model*2)
            sequence_output = torch.cat([first, second], dim=-1)
            sequence_output = torch.cat([sequence_output, rest], dim=1)

        # `logits` shape: (B, S, num_labels)
        logits = self.classifier(self.dropout(sequence_output))

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss

        return F.softmax(logits, -1)

    def generate_segments(
        self,
        x: Tensor,
        lengths: Tensor,
        return_confidence: bool = False,
        is_bi: bool = False,
    ):
        r"""Generate the segments for segment model or inference."""

        # bos and eos.
        lengths = torch.tensor(lengths) - 2

        # `logits` shape: (B, S, num_labels)
        logits = self(x)

        # `probs` shape: (B, S)
        # `labels` shape: (B, S)
        probs, labels = logits.max(dim=-1)

        if self.num_labels == 4:
            # In BMES tagging schema.
            # Replace Single tag with word boundary.
            labels[labels == 3] = 1
            labels[ (labels !=1) & (labels!=-100)] = 0

        confidence = 0
        batch_segments = []
        # Find the end-of-word boundary.
        for seq_len, line, prob in zip(lengths, labels, probs):
            line = line[1:seq_len+1]

            cur_conf = prob[1:seq_len+1].mean() if line.nelement != 0 else 0
            if self.is_c_nc:
                # 今 天  天 氣  好
                #   0  1  0  1
                cur_conf = prob[1:seq_len-1].mean() if line.nelement != 0 else 0

            segment = []
            seg_len = 1
            if is_bi:
                for i in range(seq_len):
                    if i == seq_len - 1 or line[i+1] == 0:
                        segment.append(seg_len)
                        seg_len = 1
                        continue
                    seg_len += 1
            else:
                for i in range(seq_len):
                    if i == seq_len - 1 or line[i] == 1:
                        segment.append(seg_len)
                        seg_len = 1
                        continue
                    seg_len += 1

            assert sum(segment) == seq_len

            confidence += cur_conf
            batch_segments.append(segment)

        if return_confidence == True:
            return batch_segments, confidence / len(batch_segments)

        return batch_segments

    def generate_label(
        self,
        input_ids_batch: Tensor,
        segments_batch: Tensor,
        tk,
        is_bmes: bool = False,
        is_bi: bool = False,
        **kwargs,
    ):
        r"""Generate pseudo-labels for the classifier using the results of the
        segment model.

        There are four types of tagging schemas.
            1. Word boundary schema (default): 1 stand for delimeter.
                - ex: 今天/的/天氣/非常好 - > 01/1/01/001
            2. BI schema: { 0: Begin , 1: Inside }
                - ex: 今天/的/天氣/非常好 - > 01/0/01/011
            3. BMES schema: { 0: Begin , 1: End, 2: Middle, 3: Single }
                - ex: 今天/的/天氣/非常好 - > 01/3/01/021
            4. SpIn-WS schema: Predict the two character is connect or not.
        """

        # SpIn_WS_tag_2id: { Concat: 0, Split: 1 }
        n_nc_dict = {1: [1], 2: [0, 1], 3: [0, 0, 1], 4: [0, 0 ,0 ,1]}

        # bmes_tag2id: { B:0 , E:1 , M:2 , S:3 }
        bmes_dict = {0: [],  1: [3] , 2: [0, 1] , 3: [0, 2, 1], 4: [0, 2, 2, 1]}


        batch_labels_cls = []
        for input_ids, segment in zip(input_ids_batch, segments_batch):
            e_idx = (input_ids == tk.eos_id).nonzero(as_tuple=True)[0]
            label = torch.zeros(input_ids.size(0))
            label[e_idx+1:].fill_(-100)

            if self.is_c_nc:
                tmp = []
                for s in segment:
                    if s == 0:
                        continue
                    if s not in n_nc_dict:
                        tmp.extend([0]*(s-1) + [1])
                        continue
                    tmp.extend(n_nc_dict[s])

                label[1: e_idx] = torch.tensor(tmp)
                label[e_idx-1:].fill_(-100)
                label[0].fill_(-100)
            elif is_bmes:
                tmp = []
                for s in segment:
                    if s not in bmes_dict:
                        tmp.extend([0] + [2 for i in range(s-2)] + [1])
                        continue
                    tmp.extend(bmes_dict[s])
                label[1: e_idx] = torch.tensor(tmp)
            elif is_bi:
                label[:e_idx+1].fill_(1)
                idx = [1] + [1 + sum(segment[:i])-1+1 for i in range(1, len(segment))]
                label[idx] = 0
            else:
                idx = [sum(segment[:i])-1+1 for i in range(1, len(segment)+1)]
                label[idx] = 1

            batch_labels_cls.append(label)

        return torch.stack(batch_labels_cls, 0).long()

    def mlm_forward(self, x: Tensor, ratio: float = .15):
        r"""Get the MLM loss."""

        # Randomly mask the input ids.
        masked_matrix = torch.randn(x.size(0), x.size(1))
        masked_matrix = masked_matrix < ratio

        labels = torch.zeros_like(x).fill_(-100)
        labels[masked_matrix] = x[masked_matrix]

        # x[masked_matrix] = 103
        masked_input_ids = torch.clone(x)
        masked_input_ids.masked_fill_(masked_matrix.to(x.device), 103)

        attention_mask = (x != self.pad_id).bool()
        model_output = self.model(masked_input_ids, attention_mask=attention_mask.to(x.device))

        # `sequence_output` shape: (B, S, d_model)
        sequence_output = model_output[0]

        mlm_logits = self.lm_classifier(sequence_output)

        mlm_loss = self.loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

        return mlm_loss
