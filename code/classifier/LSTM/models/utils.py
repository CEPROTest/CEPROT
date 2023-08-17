import logging
from functools import partial
from typing import Tuple, Union, Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from common import DeterOPWrapper


def dot_prod_attention(h_t: Tensor, src_encodings: Tensor, src_encoding_att_linear: Tensor,
                       mask: Tensor = None) -> Tuple[Tensor, Tensor]:
    """
    :param h_t: (batch_size, hidden_state)
    :param src_encodings: (batch_size, src_sent_len, src_output_size)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_state)
    :param mask: (batch_size, src_sent_len), paddings are marked as 1
    :return:
        ctx_vec: (batch_size, src_output_size)
        softmaxed_att_weight: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    with DeterOPWrapper():
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

    if mask is not None:
        att_weight.data.masked_fill_(mask.bool(), -float('inf'))

    softmaxed_att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    with DeterOPWrapper():
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encodings).squeeze(1)

    return ctx_vec, softmaxed_att_weight


def negative_log_likelihood(log_probs: torch.FloatTensor, gold_ids: torch.LongTensor, masks: torch.FloatTensor) \
        -> Tensor:
    """
    :param log_probs: (tgt_src_len - 1, batch_size, tgt_vocab_size) or (batch_size, tgt_vocab_size), log_softmax
    :param gold_ids: (tgt_src_len - 1, batch_size) or (batch_size, )
    :param masks: same size as gold_ids, a matrix to mask target words, 1.0 for non-pad
                       NOTE: this mask is different from dot-production mask
    :return: losses: (tgt_src_len - 1, batch_size) or (batch_size, )
    """
    # (tgt_src_len-1, batch_size) or (batch_size, )
    gold_word_log_probs = torch.gather(log_probs, index=gold_ids.unsqueeze(-1), dim=-1).squeeze(-1) * masks
    losses = -gold_word_log_probs
    return losses


#######################################
# loss function for detectors
#######################################


class DetectorLossFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_instance(raw_loss_func: Callable, alpha: Union[None, float] = None, gamma: float = 2.0):
        """
        :return: loss_func, Callable, takes probs, gold_ids and masks as input and returns losses
        """
        if raw_loss_func == focal_loss:
            logging.info("Using focal loss, the alpha is {}, the gamma is {}".format(alpha, gamma))
            loss_func = partial(raw_loss_func, alpha=alpha, gamma=gamma)
        elif raw_loss_func == balanced_cross_entropy:
            assert isinstance(alpha, float), "For balanced cross entropy, alpha must be a float"
            logging.info("Using balanced cross entropy, the alpha is {}".format(alpha))
            loss_func = partial(raw_loss_func, alpha=alpha)
        else:
            loss_func = raw_loss_func
        return loss_func


def cross_entropy(probs: torch.FloatTensor, gold_ids: torch.LongTensor, masks: torch.FloatTensor) -> Tensor:
    """
    :param probs: (tgt_src_len - 1, batch_size, tgt_vocab_size) or (batch_size, tgt_vocab_size), raw probabilities
    :param gold_ids:  (tgt_src_len - 1, batch_size) or (batch_size, )
    :param masks:  same size as gold_ids, a matrix to mask target words, 1.0 for non-pad
                       note: this mask is different from dot-production mask
    :return: losses: (tgt_src_len - 1, batch_size) or (batch_size, )
    """
    log_probs = torch.log(torch.clamp_min(probs, 1e-12))
    losses = negative_log_likelihood(log_probs, gold_ids, masks)
    return losses


def _construct_alpha_tensor(alpha: float, gold_ids: torch.LongTensor, device: torch.device, dtype: torch.dtype):
    """
    :param alpha: float, the weight of the class 1
    :param gold_ids: (batch_size, ) the gold indexes
    :param device: torch.device
    :return: (batch_size, )
    """
    alpha_tensor = torch.tensor([1 - alpha, alpha], dtype=dtype).to(device)
    # (batch_size, )
    alphas = alpha_tensor.gather(0, gold_ids)
    return alphas


def balanced_cross_entropy(probs: torch.FloatTensor, gold_ids: torch.LongTensor, masks: torch.FloatTensor,
                           alpha: float) -> Tensor:
    """
    :param probs: (tgt_src_len - 1, batch_size, tgt_vocab_size) or (batch_size, tgt_vocab_size), raw probabilities
    :param gold_ids:  (tgt_src_len - 1, batch_size) or (batch_size, )
    :param masks:  same size as gold_ids, a matrix to mask target words, 1.0 for non-pad
                       note: this mask is different from dot-production mask
    :param alpha: the weight of the rare class
    :return: losses: (tgt_src_len - 1, batch_size) or (batch_size, )
    """
    alpha_tensor = _construct_alpha_tensor(alpha, gold_ids, probs.device, probs.dtype)
    ce_losses = cross_entropy(probs, gold_ids, masks)
    losses = alpha_tensor * ce_losses
    return losses


def focal_loss(probs: torch.FloatTensor, gold_ids: torch.LongTensor, masks: torch.FloatTensor,
               alpha: Union[None, float] = 0.25, gamma: float = 2.0) -> Tensor:
    """
    :param probs: (batch_size, class_num), raw probabilities
    :param gold_ids: (batch_size, )
    :param masks: (batch_size, )
    :param alpha: default: 0.25 according to the original paper
    :param gamma: default: 2.0 according to the original paper
    :return: focal losses: (batch_size, )
    """
    gold_word_probs = torch.gather(probs, index=gold_ids.unsqueeze(-1), dim=-1).squeeze(-1) * masks
    gold_word_log_probs = torch.log(torch.clamp_min(gold_word_probs, 1e-12))
    if alpha is not None:
        alphas = _construct_alpha_tensor(alpha, gold_ids, probs.device, probs.dtype)
        gold_word_log_probs = alphas * gold_word_log_probs
    losses = -((1 - gold_word_probs) ** gamma) * gold_word_log_probs
    return losses
