# encoding=utf-8

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, Tuple

from torch import DoubleTensor, nn, Tensor
from common import *
import torch.nn.functional as F
from dataset import Example, Batch
from models.attention import BothAttention
from models.beam import Hypothesis
from models.base import LSTMCell, Linear
from vocab import VocabEntry, BaseVocabEntry


class AbstractEditor(nn.Module, ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def _init_loop(self, **forward_args) -> Tuple:
        """
        :param forward_args:
        :return: static_input
        """
        pass

    @abstractmethod
    def _init_step(self, **forward_args) -> Tuple:
        """
        :param forward_args:
        :return: state_tm1
        """
        pass

    @abstractmethod
    def step(self, y_tm1_embed: Tensor, static_input: Any, state_tm1: Tuple) -> Tuple[Tuple, Tuple[Tensor]]:
        """
        :param y_tm1_embed:
        :param static_input:
        :param state_tm1:
        :return: state_tm1, out_vec
                 out_vec may contain all information to calculate words_log_prob
        """
        pass

    @abstractmethod
    def get_decode_vocab(self, example: Example) -> BaseVocabEntry:
        return self.dec_vocab

    @abstractmethod
    def init_pretrain_embeddings(self, freeze: bool):
        pass

    @abstractmethod
    def prepare_tgt_out_tensor(self, batch: Batch, nl_vocab: VocabEntry, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def prepare_forward_kwargs(self, encoder_output: dict, batch: Batch, dec_init_state: Any, **kwargs):
        pass


class BaseEditor(AbstractEditor, ABC):
    def __init__(self, dec_vocab: VocabEntry, enc_embed_layer: nn.Module, edit_out_size: int, src_out_size: int,
                 embed_size: int, dec_hidden_size: int, teacher_forcing: float, share_embed: bool, dropout: float):
        super().__init__()
        self.dec_vocab = dec_vocab
        self.edit_out_size = edit_out_size
        self.src_out_size = src_out_size
        self.embed_size = embed_size
        self.hidden_size = dec_hidden_size
        self.teacher_forcing_ratio = teacher_forcing
        self.share_embed = share_embed
        self.dropout = dropout

        if self.share_embed:
            logging.info("Encoder and decoder share embeddings")
            self.embed_layer = enc_embed_layer
        else:
            self.embed_layer = nn.Embedding(len(self.dec_vocab), self.embed_size, padding_idx=self.dec_vocab[PADDING])
        self.rnn_cell = None
        self.readout = None

    def init_pretrain_embeddings(self, freeze: bool):
        if not self.share_embed:
            self.embed_layer.weight.data.copy_(torch.from_numpy(self.dec_vocab.embeddings))
            self.embed_layer.weight.requires_grad = not freeze

    @property
    def device(self):
        return self.embed_layer.weight.device

    def prepare_tgt_out_tensor(self, batch: Batch, nl_vocab: VocabEntry, **kwargs) -> Tensor:
        return batch.get_tgt_out_tensor(nl_vocab, self.device)

    def prepare_forward_kwargs(self, encoder_output: dict, batch: Batch, dec_init_state: Any, **kwargs):
        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
        src_last_cell, src_sent_masks = encoder_output
        return {
            'edit_last_state': edit_last_state,
            'dec_init_state': dec_init_state
        }

    def prepare_prob_input(self, out_vecs: Union[Tuple[Tensor], List[Tuple[Tensor]]], **forward_kwargs) -> Tuple:
        """
        :param out_vecs: the out_vec of a step or the whole loop
        :return: inputs required by cal_words_log_prob
        """
        if isinstance(out_vecs, Tuple):
            return tuple([out_vecs[0]])
        elif isinstance(out_vecs, List):
            return tuple([torch.stack([out_v[0] for out_v in out_vecs])])
        else:
            raise Exception("Unexpected type of out_vecs: {}".format(type(out_vecs)))

    def cal_words_log_prob(self, att_ves: Tensor, *args) -> DoubleTensor:
        # (tgt_sent_len - 1, batch_size, tgt_vocab_size) or (batch_size, tgt_vocab_size)
        tgt_vocab_scores = self.readout(att_ves)
        words_log_prob = F.log_softmax(tgt_vocab_scores, dim=-1).double()
        return words_log_prob

    def forward(self, tgt_in_tensor: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        :param tgt_in_tensor: input tensor
        :param kwargs: other arguments
        :return: words_log_prob: (tgt_len - 1, batch_size, vocab_size)
        """
        static_input = self._init_loop(**kwargs)
        state_tm1 = self._init_step(**kwargs)

        teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if teacher_forcing:
            out_vecs = []
            tgt_in_embeddings = self.embed_layer(tgt_in_tensor)
            # start from y_0=`<s>`, iterate until y_{T-1}
            for y_tm1_embed in tgt_in_embeddings.split(split_size=1, dim=0):
                # (batch_size, embed_size)
                y_tm1_embed = y_tm1_embed.squeeze(0)
                # out_vec may contain tensors related to attentions
                state_tm1, out_vec = self.step(y_tm1_embed, static_input, state_tm1)
                out_vecs.append(out_vec)
            # (tgt_in_sent_len - 1, batch_size, hidden_size)
            prob_input = self.prepare_prob_input(out_vecs, **kwargs)
            words_log_prob = self.cal_words_log_prob(*prob_input)
            ys = words_log_prob.max(dim=-1)[1]
        else:
            words_log_prob = []
            ys = []
            y_t = tgt_in_tensor[0]
            for di in range(tgt_in_tensor.size(0)):
                out_of_vocab = (y_t >= len(self.dec_vocab))
                y_tm1 = y_t.masked_fill(out_of_vocab, self.dec_vocab.unk_id)

                # (batch_size, embed_size)
                y_tm1_embed = self.embed_layer(y_tm1)
                # out_vec may contain tensors related to attentions
                state_tm1, out_vec = self.step(y_tm1_embed, static_input, state_tm1)

                prob_input = self.prepare_prob_input(out_vec, **kwargs)
                # (batch_size, vocab_size)
                log_prob_t = self.cal_words_log_prob(*prob_input)
                words_log_prob.append(log_prob_t)
                y_t = log_prob_t.max(dim=1)[1]
                ys.append(y_t)
            words_log_prob = torch.stack(words_log_prob, dim=0)
            ys = torch.stack(ys, dim=0)

        return words_log_prob, ys

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int, BeamClass, **kwargs) -> List[Hypothesis]:
        """
        NOTE: the batch size must be 1
        """
        dec_vocab = self.get_decode_vocab(example)
        static_input = self._init_loop(**kwargs)
        state_tm1 = self._init_step(**kwargs)

        beam = BeamClass(dec_vocab, self.device, beam_size, example.src_tokens)

        cur_step = 0
        while (not beam.is_finished) and cur_step < max_dec_step:
            cur_step += 1
            y_tm1 = beam.next_y_tm1()
            y_tm1_embed = self.embed_layer(y_tm1)
            cur_static_input = beam.expand_static_input(static_input)

            state_tm1, out_vec = self.step(y_tm1_embed, cur_static_input, state_tm1)
            prob_input = self.prepare_prob_input(out_vec, **kwargs)
            words_log_prob = self.cal_words_log_prob(*prob_input)

            state_tm1 = beam.step(words_log_prob, state_tm1)

        return beam.get_final_hypos()


class BothAttnSeqEditor(BaseEditor):
    def __init__(self, dec_vocab: VocabEntry, enc_embed_layer: nn.Module, edit_out_size: int, src_out_size: int,
                 embed_size: int, dec_hidden_size: int, teacher_forcing: float, share_embed: bool, dropout: float,
                 input_feed: bool, attn_func: Callable):
        super(BothAttnSeqEditor, self).__init__(dec_vocab, enc_embed_layer, edit_out_size, src_out_size, embed_size,
                                                dec_hidden_size, teacher_forcing, share_embed, dropout)

        self.input_feed = input_feed
        self.attn_func = attn_func

        self.attention = BothAttention(self.attn_func, self.src_out_size, self.edit_out_size, self.hidden_size)

        if self.input_feed:
            input_size = self.embed_size + self.hidden_size
        else:
            input_size = self.embed_size
        self.rnn_cell = LSTMCell(input_size, self.hidden_size, dropout=self.dropout)
        self.att_src_linear = nn.Linear(self.src_out_size, self.hidden_size, bias=False)
        self.att_edit_linear = nn.Linear(self.edit_out_size, self.hidden_size, bias=False)
        self.readout = Linear(self.hidden_size, len(self.dec_vocab), dropout=self.dropout, bias=False)

    def _init_loop(self, **forward_args):
        src_encodings_att_linear = self.att_src_linear(forward_args['src_encodings'])
        edit_encodings_att_linear = self.att_edit_linear(forward_args['edit_encodings'])
        static_input = (forward_args['src_encodings'], src_encodings_att_linear, forward_args['edit_encodings'],
                        edit_encodings_att_linear, forward_args['src_sent_masks'], forward_args['edit_sent_masks'])
        return static_input

    def _init_step(self, **forward_args):
        h_tm1 = forward_args['dec_init_state']
        # (batch_size, hidden_size)
        att_tm1 = torch.zeros(forward_args['src_encodings'].size(0), self.hidden_size, device=self.device)
        state_tm1 = (h_tm1, att_tm1)
        return state_tm1

    def step(self, y_tm1_embed: Tensor, static_input: Any, state_tm1: Tuple) -> Tuple[Tuple, Tuple]:
        src_encodings, src_encodings_att_linear, edit_encodings, edit_encodings_att_linear, \
        src_sent_masks, edit_sent_masks = static_input[:6]
        h_tm1, att_tm1 = state_tm1
        if self.input_feed:
            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
        else:
            x = torch.cat([y_tm1_embed], dim=-1)
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.rnn_cell(x, h_tm1)
        # ctx_t: src_encoding_state

        att_t, src_alpha_t, edit_alpha_t = self.attention(
            h_t, src_encodings, src_encodings_att_linear, edit_encodings, edit_encodings_att_linear,
            src_sent_masks, edit_sent_masks)

        state_tm1 = ((h_t, cell_t), att_t)
        out_vec = att_t
        return state_tm1, (out_vec, src_alpha_t, edit_alpha_t)

    def get_decode_vocab(self, example: Example) -> BaseVocabEntry:
        return self.dec_vocab

    def prepare_forward_kwargs(self, encoder_output: dict, batch: Batch, dec_init_state: Any, **kwargs):
        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
        src_last_cell, src_sent_masks = encoder_output
        return {
            'src_encodings': src_encodings,
            'edit_encodings': edit_encodings,
            'src_sent_masks': src_sent_masks,
            'edit_sent_masks': edit_sent_masks,
            'dec_init_state': dec_init_state
        }


class SrcPtrBASeqEditor(BothAttnSeqEditor):
    def __init__(self, dec_vocab: VocabEntry, enc_embed_layer: nn.Module, edit_out_size: int, src_out_size: int,
                 embed_size: int, dec_hidden_size: int, teacher_forcing: float, share_embed: bool, dropout: float,
                 *args, **kwargs):
        super(SrcPtrBASeqEditor, self).__init__(dec_vocab, enc_embed_layer, edit_out_size, src_out_size, embed_size,
                                                dec_hidden_size, teacher_forcing, share_embed, dropout, *args, **kwargs)
        self.p_gen_linear = nn.Linear(self.hidden_size, 1)

    def prepare_tgt_out_tensor(self, batch: Batch, nl_vocab: VocabEntry, **kwargs) -> Tensor:
        return batch.get_src_ext_tgt_out_tensor(nl_vocab, self.device)

    def prepare_prob_input(self, out_vecs: Union[Tuple[Tensor], List[Tuple[Tensor]]], **forward_kwargs) -> Tuple:
        """
        :param out_vecs: the out_vec of a step or the whole loop: (out_vec, src_alpha_t, edit_alpha_t)
        :param forward_kwargs: kwargs of forward function
        :return: (att_vecs, src_att_weights, src_ext_tensor, src_zeros)
        """
        static_input = [forward_kwargs['src_ext_tensor'], forward_kwargs['max_ext_size']]
        if isinstance(out_vecs, Tuple):
            return tuple(list(out_vecs[:2]) + static_input)
        elif isinstance(out_vecs, List):
            att_vec = torch.stack([out_v[0] for out_v in out_vecs], dim=0)
            att_weights = torch.stack([out_v[1] for out_v in out_vecs], dim=0)
            return tuple([att_vec, att_weights] + static_input)
        else:
            raise Exception("Unexpected type of out_vecs: {}".format(type(out_vecs)))

    def cal_gen_words_prob(self, att_ves, max_ext_size) -> Tensor:
        tgt_vocab_scores = self.readout(att_ves)
        gen_words_prob = F.softmax(tgt_vocab_scores, dim=-1)
        if max_ext_size != 0:
            ext_zero_size = [*gen_words_prob.size()[:-1], max_ext_size]
            ext_zeros = torch.zeros(ext_zero_size, dtype=FLOAT_TYPE, device=self.device)
            # (batch_size, vocab_size + max_ext_size)
            gen_words_prob = torch.cat([gen_words_prob, ext_zeros], dim=-1)
        return gen_words_prob

    def cal_words_log_prob(self, att_ves: Tensor, *args) -> DoubleTensor:
        """
        :param att_ves: (tgt_sent_len - 1, batch_size, hidden_size) or (batch_size, hidden_size)
        :param src_att_weights: (tgt_sent_len - 1, batch_size, src_sent_len) or (batch_size, src_sent_len), already masked
        :param src_ext_tensor: (src_sent_len, batch_size)
        :param max_ext_size: int
        :return: words_log_prob
        """
        src_att_weights, src_ext_tensor, max_ext_size = args[0:3]
        src_ext_tensor = src_ext_tensor.transpose(1, 0)
        p_gen = torch.sigmoid(self.p_gen_linear(att_ves))
        gen_words_prob = self.cal_gen_words_prob(att_ves, max_ext_size)
        words_prob = p_gen * gen_words_prob

        # copy prob
        copy_words_prob = (1 - p_gen) * src_att_weights
        words_prob = words_prob.double()
        copy_words_prob = copy_words_prob.double()
        # final prob
        words_prob = words_prob.scatter_add_(-1, src_ext_tensor.expand_as(copy_words_prob), copy_words_prob)
        # avoid nan!
        words_log_prob = torch.log(torch.clamp_min(words_prob, 1e-12))
        return words_log_prob

    def get_decode_vocab(self, example: Example) -> BaseVocabEntry:
        return example.get_src_ext_vocab()

    def prepare_forward_kwargs(self, encoder_output: dict, batch: Batch, dec_init_state: Any, **kwargs):
        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
        src_last_cell, src_sent_masks = encoder_output
        nl_vocab = kwargs['nl_vocab']
        return {
            'src_encodings': src_encodings,
            'edit_encodings': edit_encodings,
            'src_sent_masks': src_sent_masks,
            'edit_sent_masks': edit_sent_masks,
            'dec_init_state': dec_init_state,
            'src_ext_tensor': batch.get_src_ext_tensor(nl_vocab, self.device),
            'max_ext_size': batch.get_max_src_ext_size()
        }


class BothPtrBASeqEditor(SrcPtrBASeqEditor):
    def __init__(self, dec_vocab: VocabEntry, enc_embed_layer: nn.Module, edit_out_size: int, src_out_size: int,
                 embed_size: int, dec_hidden_size: int, teacher_forcing: float, share_embed: bool, dropout: float,
                 *args, **kwargs):
        super(BothPtrBASeqEditor, self).__init__(dec_vocab, enc_embed_layer, edit_out_size, src_out_size, embed_size,
                                                 dec_hidden_size, teacher_forcing, share_embed, dropout, *args,
                                                 **kwargs)
        self.p_copy_src_linear = nn.Linear(self.hidden_size, 1)

    def prepare_tgt_out_tensor(self, batch: Batch, nl_vocab: VocabEntry, **kwargs) -> Tensor:
        return batch.get_both_ext_tgt_out_tensor(nl_vocab, self.device)

    def prepare_prob_input(self, out_vecs: Union[Tuple[Tensor], List[Tuple[Tensor]]], **forward_kwargs) -> Tuple:
        """
        :param out_vecs: the out_vec of a step or the whole loop: (out_vec, src_alpha_t, edit_alpha_t)
        :param forward_kwargs: kwargs of forward function
        :return: (att_vecs, src_att_weights, src_ext_tensor, src_zeros)
        """
        static_input = [forward_kwargs['src_ext_tensor'], forward_kwargs['code_ext_tensor'],
                        forward_kwargs['max_ext_size']]
        if isinstance(out_vecs, Tuple):
            return tuple(list(out_vecs[:3]) + static_input)
        elif isinstance(out_vecs, List):
            # att_vec
            att_vec = torch.stack([out_v[0] for out_v in out_vecs], dim=0)
            # src_alpha_t
            src_att_weights = torch.stack([out_v[1] for out_v in out_vecs], dim=0)
            # edit_alpha_t
            code_att_weights = torch.stack([out_v[2] for out_v in out_vecs], dim=0)
            return tuple([att_vec, src_att_weights, code_att_weights] + static_input)
        else:
            raise Exception("Unexpected type of out_vecs: {}".format(type(out_vecs)))

    def cal_words_log_prob(self, att_ves: Tensor, *args) -> DoubleTensor:
        """
        :param att_ves: (tgt_sent_len - 1, batch_size, hidden_size) or (batch_size, hidden_size)
        :param src_att_weights: (tgt_sent_len - 1, batch_size, src_sent_len) or (batch_size, src_sent_len), already masked
        :param src_ext_tensor: (src_sent_len, batch_size)
        :param max_ext_size: int
        :return: words_log_prob
        """
        src_att_weights, code_att_weights, src_ext_tensor, code_ext_tensor, max_ext_size = args[0:5]
        src_ext_tensor = src_ext_tensor.transpose(1, 0)
        code_ext_tensor = code_ext_tensor.transpose(1, 0)
        p_gen = torch.sigmoid(self.p_gen_linear(att_ves))
        gen_words_prob = self.cal_gen_words_prob(att_ves, max_ext_size)
        words_prob = p_gen * gen_words_prob

        # src copy prob
        p_copy_src = torch.sigmoid(self.p_copy_src_linear(att_ves))
        src_copy_words_prob = (1 - p_gen) * p_copy_src * src_att_weights

        # for reproducability
        words_prob = words_prob.double()
        src_copy_words_prob = src_copy_words_prob.double()
        words_prob = words_prob.scatter_add_(-1, src_ext_tensor.expand_as(src_copy_words_prob), src_copy_words_prob)

        # code copy prob
        code_copy_words_prob = (1 - p_gen) * (1 - p_copy_src) * code_att_weights
        # for reproducability
        code_copy_words_prob = code_copy_words_prob.double()
        words_prob = words_prob.scatter_add_(-1, code_ext_tensor.expand_as(code_copy_words_prob), code_copy_words_prob)

        # avoid nan!
        words_log_prob = torch.log(torch.clamp_min(words_prob, 1e-12))
        return words_log_prob

    def get_decode_vocab(self, example: Example) -> BaseVocabEntry:
        return example.get_both_ext_vocab()

    def prepare_forward_kwargs(self, encoder_output: Tuple, batch: Batch, dec_init_state: Any, **kwargs):
        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
        src_last_cell, src_sent_masks = encoder_output
        nl_vocab = kwargs['nl_vocab']
        src_ext_tensor, code_ext_tensor = batch.get_both_ext_tensor(nl_vocab, self.device)
        return {
            'src_encodings': src_encodings,
            'edit_encodings': edit_encodings,
            'src_sent_masks': src_sent_masks,
            'edit_sent_masks': edit_sent_masks,
            'dec_init_state': dec_init_state,
            'src_ext_tensor': src_ext_tensor,
            'code_ext_tensor': code_ext_tensor,
            'max_ext_size': batch.get_max_both_ext_size()
        }
