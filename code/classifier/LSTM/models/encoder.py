# encoding=utf-8

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import logging
from torch import nn, Tensor
import torch.nn.functional as F

from common import PADDING
from models.base import LSTM, get_sent_masks
from vocab import VocabEntry


################################################################################
# Base Layers
################################################################################
class LastEncodeLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def output_size(self):
        pass


class BaseEmbeddingLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.nl_embed_layer = None


################################################################################
# Encoder Layers
################################################################################
class EmbeddingLayer(BaseEmbeddingLayer):
    def __init__(self, embed_size: int, code_vocab: VocabEntry, action_vocab: VocabEntry, nl_vocab: VocabEntry,
                 is_mix_vocab: bool):
        super().__init__()
        self.code_vocab, self.action_vocab, self.nl_vocab = code_vocab, action_vocab, nl_vocab
        # init_embeddings
        self.nl_embed_layer = nn.Embedding(len(nl_vocab), embed_size, padding_idx=nl_vocab[PADDING])
        self.action_embed_layer = nn.Embedding(len(action_vocab), embed_size, padding_idx=action_vocab[PADDING])
        self.mix_vocab = is_mix_vocab
        if self.mix_vocab:
            logging.info("Code and nl share embeddings")
            self.code_embed_layer = self.nl_embed_layer
        else:
            self.code_embed_layer = nn.Embedding(len(code_vocab), embed_size, padding_idx=code_vocab[PADDING])

    def init_pretrain_embeddings(self, freeze: bool):
        self.nl_embed_layer.weight.data.copy_(torch.from_numpy(self.nl_vocab.embeddings))
        self.nl_embed_layer.weight.requires_grad = not freeze
        if not self.mix_vocab:
            self.code_embed_layer.weight.data.copy_(torch.from_numpy(self.code_vocab.embeddings))
            self.code_embed_layer.weight.requires_grad = not freeze

    def forward(self, old_token_tensor: Tensor, new_token_tensor: Tensor, action_tensor: Tensor, nl_tensor: Tensor, focal_src_tensor: Tensor, focal_tgt_tensor: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.code_embed_layer(old_token_tensor), self.code_embed_layer(new_token_tensor), \
               self.action_embed_layer(action_tensor), self.nl_embed_layer(nl_tensor),\
                self.code_embed_layer(focal_src_tensor), self.code_embed_layer(focal_tgt_tensor)
        

    @property
    def device(self):
        return self.code_embed_layer.weight.device


class RNNLayer(LastEncodeLayer):
    def __init__(self, embed_size, hidden_size, num_layers, dropout, bidirectional=True, batch_first=False):
        super(RNNLayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.rnn_layer = LSTM(self.embed_size, self.hidden_size, self.num_layers, bidirectional=bidirectional,
                              batch_first=batch_first, dropout=dropout)

    @property
    def output_size(self):
        if self.bidirectional:
            return self.hidden_size * 2
        return self.hidden_size

    def forward(self, embeddings: torch.Tensor, lens: List[int], enforce_sorted: bool = False):
        """
        :param embeddings: (src_sent_len, batch_size, embed_size)
        :param lens:
        :return: (batch_size, src_sent_len, self.output_size),
                (num_layers*num_directions, batch, hidden_size),
                 (num_layers*num_directions, batch, hidden_size)
        """
        encodings, (last_state, last_cell) = self.rnn_layer(embeddings, lens, enforce_sorted=enforce_sorted)
        if not self.batch_first:
            encodings = encodings.permute(1, 0, 2)
        last_state = torch.cat([s.squeeze(0) for s in last_state.split(1, dim=0)], dim=-1)
        last_cell = torch.cat([c.squeeze(0) for c in last_cell.split(1, dim=0)], dim=-1)
        return encodings, last_state, last_cell


class EditNLEncodeLayer(LastEncodeLayer):
    """
    A layer for encoding edit and nl embeddings
    """
    def __init__(self, embed_size, edit_vec_size, nl_hidden_size, dropout):
        super().__init__()
        self.edit_rnn_layer = RNNLayer(embed_size * 3, edit_vec_size // 2, 1, dropout)
        self.nl_rnn_layer = RNNLayer(embed_size, nl_hidden_size, 1, dropout)

    @property
    def output_size(self):
        return self.edit_rnn_layer.output_size + self.nl_rnn_layer.output_size

    @property
    def edit_output_size(self):
        return self.edit_rnn_layer.output_size

    @property
    def nl_output_size(self):
        return self.nl_rnn_layer.output_size

    def forward(self, old_token_ems: Tensor, new_token_ems: Tensor, action_ems: Tensor, code_lens: List[int],
                src_ems: Tensor, src_lens: List[int], focal_src_ems: Tensor, focal_tgt_ems: Tensor, focal_src_lens: List[int], focal_tgt_lens:List[int]):
        """
        :param old_token_ems: (sent_len, batch_size, embed_size)
        :param new_token_ems: (sent_len, batch_size, embed_size)
        :param action_ems: (sent_len, batch_size, embed_size)
        :param code_lens: code sent lens
        :param src_lens: source nl lens
        :return:
            edit_encodings: (batch_size, sent_len, edit_vec_size)
            edit_last_state: (batch_size, edit_vec_size)
            edit_last_cell: (batch_size, edit_vec_size)
            src_encodings: (batch_size, sent_len, nl_hidden_size * 2)
            src_last_state: (batch_size, nl_hidden_size * 2)
            src_last_cell: (batch_size, nl_hidden_size * 2)
        """
        edit_ems = torch.cat([old_token_ems, new_token_ems, action_ems], dim=-1)
        edit_encodings, edit_last_state, edit_last_cell = self.edit_rnn_layer(edit_ems, code_lens)
        src_encodings, src_last_state, src_last_cell = self.nl_rnn_layer(src_ems, src_lens)


        focal_src_encodings, focal_src_last_state, focal_src_last_cell = self.nl_rnn_layer(focal_src_ems, focal_src_lens)
        focal_tgt_encodings, focal_tgt_last_state, focal_tgt_last_cell = self.nl_rnn_layer(focal_tgt_ems, focal_tgt_lens)

        return edit_encodings, edit_last_state, edit_last_cell, src_encodings, src_last_state, src_last_cell, \
                focal_src_encodings, focal_src_last_state, focal_src_last_cell,focal_tgt_encodings, focal_tgt_last_state, focal_tgt_last_cell

class CoAttnLayer(nn.Module):
    def __init__(self, edit_encoding_size, src_encoding_size):
        super().__init__()
        self.edit_encoding_size = edit_encoding_size
        self.src_encoding_size = src_encoding_size
        self.edit_src_linear = nn.Linear(edit_encoding_size, src_encoding_size, bias=False)

    @property
    def edit_output_size(self):
        return self.src_encoding_size

    @property
    def nl_output_size(self):
        return self.edit_encoding_size

    def forward(self, edit_encodings: Tensor, src_encodings: Tensor, edit_sent_masks: Tensor, src_sent_masks: Tensor) \
            -> Tuple[Tensor, Tensor]:
        """
        :param edit_encodings: (batch_size, edit_len, edit_encoding_size)
        :param src_encodings: (batch_size, src_len, src_encoding_size)
        :param edit_sent_masks: (batch_size, edit_max_len), **1 for padding**
        :param src_sent_masks: (batch_size, src_max_len), **1 for padding**
        :return: edit_ctx_encodings, src_ctx_encodings
        """
        # similar to dot_prod_attention
        sim_matrix = self.edit_src_linear(edit_encodings).bmm(src_encodings.permute(0, 2, 1))
        # should not mask on the same sim_matrix
        # since softmax on a all-inf column will produce nan
        edit_sim_matrix = sim_matrix.masked_fill(src_sent_masks.unsqueeze(1).bool(), -float('inf'))
        src_sim_matrix = sim_matrix.masked_fill(edit_sent_masks.unsqueeze(-1).bool(), -float('inf'))
        edit_weights = F.softmax(edit_sim_matrix, dim=-1)
        src_weights = F.softmax(src_sim_matrix, dim=1)
        # NOTE: even padding will have ctx_encoding, but such encodings are ignored when calculating attentions in
        #       decoder
        # (batch_size, edit_len, src_encoding_size)
        edit_ctx_encodings = edit_weights.bmm(src_encodings)
        # (batch_size, src_len, edit_encoding_size)
        src_ctx_encodings = src_weights.permute(0, 2, 1).bmm(edit_encodings)
        return edit_ctx_encodings, src_ctx_encodings


class ModelingLayer(LastEncodeLayer):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.rnn_layer = RNNLayer(input_size, hidden_size, 1, bidirectional=True, batch_first=True, dropout=dropout)
        self._output_size = hidden_size * 2

    @property
    def output_size(self):
        return self._output_size

    def forward(self, input_tensor: Tensor, sent_lens: List[int]):
        """
        :param input_tensor: (batch_size, seq_len, input_size)
        :param sent_lens: List[int]
        :return:
            encodings: (batch_size, seq_len, hidden_size * num_directions)
            last_state: (batch_size, hidden_size * num_layer * num_directions)
            last_cell: (batch_size, hidden_size * num_layer * num_directions)
        """
        # input is sorted by nl input len, hence enforce_sorted should be False
        return self.rnn_layer(input_tensor, sent_lens, enforce_sorted=False)


################################################################################
# Base Encoders
################################################################################
class BaseEncoder(LastEncodeLayer, ABC):
    @property
    @abstractmethod
    def device(self):
        return

    @abstractmethod
    def init_pretrain_embeddings(self, freeze: bool):
        pass


class BaseEditNLEncoder(BaseEncoder, ABC):
    def __init__(self,
                 embed_size: int,
                 edit_vec_size: int,
                 enc_hidden_size: int,
                 dropout: float,
                 mix_vocab: bool):
        super().__init__()
        self.embed_size = embed_size
        self.edit_vec_size = edit_vec_size
        self.enc_hidden_size = enc_hidden_size
        self.dropout = dropout
        self.mix_vocab = mix_vocab
        self.embed_layer = None

    @property
    @abstractmethod
    def edit_output_size(self):
        pass

    @property
    @abstractmethod
    def nl_output_size(self):
        pass

    @property
    def output_size(self):
        return self.edit_output_size + self.nl_output_size

    @property
    def device(self):
        return self.embed_layer.device

    @property
    def nl_embed_layer(self):
        return self.embed_layer.nl_embed_layer

    def init_pretrain_embeddings(self, freeze: bool):
        self.embed_layer.init_pretrain_embeddings(freeze)


################################################################################
# Encoders
################################################################################
class EditNLEncoder(BaseEditNLEncoder):
    def __init__(self, code_vocab: VocabEntry, action_vocab: VocabEntry, nl_vocab: VocabEntry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_layer = EmbeddingLayer(self.embed_size, code_vocab, action_vocab, nl_vocab, self.mix_vocab)
        # code edit encoder
        # (code_len, batch_size) -> [(code_len, batch_size, edit_vec_size), (batch_size, edit_vec_size)]
        self.context_layer = EditNLEncodeLayer(self.embed_size, self.edit_vec_size, self.enc_hidden_size, self.dropout)

    @property
    def edit_output_size(self):
        return self.context_layer.edit_output_size

    @property
    def nl_output_size(self):
        return self.context_layer.nl_output_size

    def forward(self, code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens):
        """
        :param code_tensor_a: code seq before updating
        :param code_tensor_b: code seq after updating
        :param action_tensor: action seq
        :param code_lens: edit length
        :param src_tensor: source comment seq
        :param src_lens: source comment length
        :return: encoder output
        """
        # encodings: (batch_size, sent_len, hidden_size * direction * #layer)
        code_a_ems, code_b_ems, action_ems, nl_ems = self.embed_layer(code_tensor_a, code_tensor_b, action_tensor,
                                                                      src_tensor)
        edit_encodings, edit_last_state, edit_last_cell, src_encodings, src_last_state, src_last_cell = \
            self.context_layer(code_a_ems, code_b_ems, action_ems, code_lens, nl_ems, src_lens)
        src_sent_masks = get_sent_masks(src_encodings.size(1), src_lens, self.device)
        edit_sent_masks = get_sent_masks(edit_encodings.size(1), code_lens, self.device)
        return edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
               src_last_cell, src_sent_masks


class CoAttnEncoder(BaseEditNLEncoder):
    def __init__(self,
                 code_vocab: VocabEntry,
                 action_vocab: VocabEntry,
                 nl_vocab: VocabEntry,
                 embed_size: int,
                 edit_vec_size: int,
                 enc_hidden_size: int,
                 dropout: float,
                 mix_vocab: bool):
        super().__init__(embed_size, edit_vec_size, enc_hidden_size, dropout, mix_vocab)
        self.embed_layer = EmbeddingLayer(self.embed_size, code_vocab, action_vocab, nl_vocab, self.mix_vocab)
        self.context_layer = EditNLEncodeLayer(self.embed_size, self.edit_vec_size, self.enc_hidden_size, self.dropout)
        # for co-attn
        self.co_attn_layer = CoAttnLayer(self.context_layer.edit_output_size,
                                         self.context_layer.nl_output_size)
        self.code_edit_modeling_layer = ModelingLayer(self.context_layer.output_size, self.enc_hidden_size,
                                                      self.dropout)
        self.nl_modeling_layer = ModelingLayer(self.context_layer.output_size, self.enc_hidden_size, self.dropout)

    @property
    def edit_output_size(self):
        return self.code_edit_modeling_layer.output_size

    @property
    def nl_output_size(self):
        return self.nl_modeling_layer.output_size
    # ADD
    def forward(self, code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens, focal_src_tensor, focal_tgt_tensor, focal_src_lens, focal_tgt_lens, *args):
        code_a_ems, code_b_ems, action_ems, src_ems, focal_src_ems, focal_tgt_ems\
                                             = self.embed_layer(code_tensor_a, code_tensor_b, action_tensor,src_tensor, focal_src_tensor, focal_tgt_tensor)

        c_edit_encodings, c_edit_last_state, c_edit_last_cell, c_src_encodings, c_src_last_state, c_src_last_cell, \
            focal_src_encodings, focal_src_last_state, focal_src_last_cell, focal_tgt_encodings, focal_tgt_last_state, focal_tgt_last_cell = \
                self.context_layer(code_a_ems, code_b_ems, action_ems, code_lens, src_ems, src_lens, focal_src_ems, focal_tgt_ems, focal_src_lens, focal_tgt_lens)
        

        edit_sent_masks = get_sent_masks(c_edit_encodings.size(1), code_lens, self.device)
        src_sent_masks = get_sent_masks(c_src_encodings.size(1), src_lens, self.device)
        focal_src_sent_masks = get_sent_masks(focal_src_encodings.size(1), focal_src_lens, self.device)
        focal_tgt_sent_masks = get_sent_masks(focal_tgt_encodings.size(1), focal_tgt_lens, self.device)

        # for co-attention
        edit_ctx_encodings, src_ctx_encodings = self.co_attn_layer(c_edit_encodings, c_src_encodings, edit_sent_masks,
                                                                   src_sent_masks)
        
        edit_modeling_input = torch.cat([c_edit_encodings, edit_ctx_encodings], dim=-1)
        edit_encodings, edit_last_state, edit_last_cell = self.code_edit_modeling_layer(edit_modeling_input, code_lens)

        src_modeling_input = torch.cat([c_src_encodings, src_ctx_encodings], dim=-1)
        src_encodings, src_last_state, src_last_cell = self.nl_modeling_layer(src_modeling_input, src_lens)

        return edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
                src_last_cell, src_sent_masks, focal_src_encodings, focal_src_last_state, focal_src_last_cell, focal_src_sent_masks,\
                    focal_tgt_encodings, focal_tgt_last_state, focal_tgt_last_cell, focal_tgt_sent_masks

