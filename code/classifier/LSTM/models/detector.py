# encoding=utf-8
import logging
import math
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Callable, Union

import torch

from torch import DoubleTensor
from torch import nn
from torch.nn import functional as F

from common import get_attr_by_name
from dataset import Batch
from models.base import BaseModel, Linear
from models.encoder import CoAttnEncoder
from models.utils import dot_prod_attention, DetectorLossFactory, cross_entropy, focal_loss
from vocab import Vocab, MixVocab


class BaseClassifier(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def prepare_classifier_input(cls, edit_encodings, edit_last_state, edit_sent_masks, src_encodings, src_last_state,
                                 src_sent_masks):
        pass


class Classifier(BaseClassifier):
    def __init__(self, in_features, hidden_size, class_num, dropout_rate, final_bias: Union[float, None] = None):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_size, bias=False)
        self.readout = Linear(hidden_size, class_num, dropout_rate, bias=True)
        # for focal loss
        if final_bias is not None:
            logging.info("Initialize the final bias to {}".format(final_bias))
            self.readout.init_bias(final_bias)

    def forward(self, input):
        """
        :param input: (*, feature_size)
        :return: probs, (*, class_num)
        """
        features = torch.tanh(self.linear(input))
        logits = self.readout(features)
        probs = F.softmax(logits, dim=-1)
        return probs

    @classmethod
    def prepare_classifier_input(cls, edit_encodings, edit_last_state, edit_sent_masks, src_encodings, src_last_state,
                                 src_sent_masks):
        """
        We concatenate the last states as the feature vector
        """
        features = torch.cat([edit_last_state, src_last_state], dim=-1)
        return (features,)


class AttnClassifier(BaseClassifier):
    def __init__(self, src_in_features: int, edit_in_features: int, hidden_size: int, class_num: int,
                 dropout_rate: float, att_func: Callable, final_bias: Union[float, None] = None):
        super().__init__()
        self.src_query = nn.Parameter(torch.Tensor(1, src_in_features))
        self.edit_query = nn.Parameter(torch.Tensor(1, edit_in_features))
        self.att_func = att_func
        self.output_layer = Classifier(src_in_features + edit_in_features, hidden_size, class_num, dropout_rate,
                                       final_bias=final_bias)

    def forward(self, src_encodings, edit_encodings, src_sent_masks, edit_sent_masks):
        batch_size = src_encodings.size(0)
        # use encodings to directly multiply with query
        # (batch_size, in_features)
        cur_src_query = self.src_query.repeat(batch_size, 1)
        cur_edit_query = self.edit_query.repeat(batch_size, 1)
        src_ctx, src_alpha = self.att_func(cur_src_query, src_encodings, src_encodings, src_sent_masks)
        edit_ctx, edit_alpha = self.att_func(cur_edit_query, edit_encodings, edit_encodings, edit_sent_masks)
        features = torch.cat([src_ctx, edit_ctx], dim=-1)
        return self.output_layer(features)

    @classmethod
    def prepare_classifier_input(cls, edit_encodings, edit_last_state, edit_sent_masks, src_encodings, src_last_state,
                                 src_sent_masks):
        return src_encodings, edit_encodings, src_sent_masks, edit_sent_masks


class BaseDetector(BaseModel, ABC):
    EMBED_SIZE = 300
    EDIT_VEC_SIZE = 512
    ENC_HIDDEN_SIZE = 256
    USE_PRE_EMBED = True
    FREEZE_PRE_EMBED = True
    DROPOUT = 0
    CLF_LOSS_FUNC = cross_entropy
    ALPHA = None
    GAMMA = 2.0
    PI = None

    def __init__(self,
                 vocab: Union[Vocab, MixVocab],
                 embed_size: int = EMBED_SIZE,
                 edit_vec_size: int = EDIT_VEC_SIZE,
                 enc_hidden_size: int = ENC_HIDDEN_SIZE,
                 use_pre_embed: bool = USE_PRE_EMBED,
                 freeze_pre_embed: bool = FREEZE_PRE_EMBED,
                 dropout: float = DROPOUT,
                 clf_loss_func: Union[Callable, str] = CLF_LOSS_FUNC,
                 alpha: Union[None, float] = ALPHA,
                 gamma: float = GAMMA,
                 pi: float = PI,
                 *args,
                 **kwargs):
        super().__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.edit_vec_size = edit_vec_size
        self.enc_hidden_size = enc_hidden_size
        self.dropout = dropout
        self.mix_vocab = isinstance(self.vocab, MixVocab)
        self.use_pre_embed = use_pre_embed
        self.freeze_pre_embed = freeze_pre_embed
        loss_func = get_attr_by_name(clf_loss_func) if isinstance(clf_loss_func, str) else clf_loss_func
        self.loss_func = DetectorLossFactory.get_instance(loss_func, alpha, gamma)
        self.final_bias = None
        if self.loss_func == focal_loss and pi is not None:
            self.final_bias = -math.log((1-pi)/pi)


class Detector(BaseDetector):
    @classmethod
    def prepare_model_params(cls, args: Namespace) -> Namespace:
        mix_vocab = getattr(args, "mix_vocab", cls.MIX_VOCAB)
        if mix_vocab:
            logging.info("Using mix vocab")
        vocab_class = MixVocab if mix_vocab else Vocab
        args.vocab = vocab_class.load(args.vocab_file)
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = locals().copy()
        self.log_args(**params)

        self.code_vocab, self.nl_vocab, self.action_vocab = self.vocab.get_subvocabs()

        self.encoder = CoAttnEncoder(self.code_vocab, self.action_vocab, self.nl_vocab, self.embed_size,
                                     self.edit_vec_size, self.enc_hidden_size, self.dropout, self.mix_vocab)

        self.classifier = Classifier(self.encoder.output_size, self.enc_hidden_size, 2, self.dropout,
                                     final_bias=self.final_bias)

    def init_pretrain_embeddings(self, freeze: bool):
        self.encoder.init_pretrain_embeddings(freeze)

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def _forward(self, batch: Batch) -> DoubleTensor:

        input_tensor = batch.construct_encoder_input(self.code_vocab, self.action_vocab, self.nl_vocab, self.device)

        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
            src_last_cell, src_sent_masks, focal_src_encodings, focal_src_last_state, focal_src_last_cell, focal_src_sent_masks, \
                focal_tgt_encodings, focal_tgt_last_state, focal_tgt_last_cell, focal_tgt_sent_masks = self.encoder(*input_tensor)
        src_encodings = torch.cat([focal_src_encodings, focal_tgt_encodings, src_encodings], dim = 1)
        src_sent_masks = torch.cat([focal_src_sent_masks, focal_tgt_sent_masks, src_sent_masks], dim = 1)
        src_last_state = torch.cat([focal_src_last_state, focal_tgt_last_state, src_last_state], dim = 1)
        
        cls_input = self.classifier.prepare_classifier_input(edit_encodings, edit_last_state, edit_sent_masks,
                                                             src_encodings, src_last_state, src_sent_masks)
        probs = self.classifier(*cls_input).double()
        return probs

    def cal_losses(self, probs: torch.DoubleTensor, label_tensor: torch.LongTensor) -> torch.DoubleTensor:
        # [batch_size, ]
        masks = torch.ones_like(label_tensor).double()
        example_losses = self.loss_func(probs, label_tensor, masks)
        return example_losses

    def forward(self, batch: Batch):
        label_tensor = batch.get_label_tensor(self.device)
        probs = self._forward(batch)
        return self.cal_losses(probs, label_tensor)

    def predict(self, batch: Batch):
        # [batch_size, 2]
        probs = self._forward(batch)
        return probs


class AttnDetector(Detector):
    ATTN_FUNC = dot_prod_attention

    def __init__(self,
                 attn_func: Union[Callable, str] = ATTN_FUNC,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        attn_func = get_attr_by_name(attn_func) if isinstance(attn_func, str) else attn_func
        self.classifier = AttnClassifier(self.encoder.nl_output_size, self.encoder.edit_output_size,
                                         self.enc_hidden_size, 2, self.dropout, attn_func,
                                         final_bias=self.final_bias)
