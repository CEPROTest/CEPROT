# encoding=utf-8

"""
Commen Updater Model
"""
from argparse import Namespace

from models import base
from models.beam import Beam
from vocab import MixVocab, Vocab
from .decoder import *
from .encoder import CoAttnEncoder
from .utils import negative_log_likelihood, dot_prod_attention


class BaseUpdater(base.BaseModel, ABC):
    EMBED_SIZE = 300
    EDIT_VEC_SIZE = 512
    ENC_HIDDEN_SIZE = 256
    DEC_HIDDEN_SIZE = 512
    INPUT_FEED = True
    SHARE_EMBED = True
    MIX_VOCAB = True
    USE_PRE_EMBED = True
    FREEZE_PRE_EMBED = True
    DROPOUT = 0
    TEACHER_FORCING = 1.0
    GEN_LOSS_TYPE = "sent_level"
    GEN_LOSS_FUNC = negative_log_likelihood
    ATTN_FUNC = dot_prod_attention

    def __init__(self, vocab: Union[Vocab, MixVocab],
                 embed_size: int = EMBED_SIZE,
                 edit_vec_size: int = EDIT_VEC_SIZE,
                 enc_hidden_size: int = ENC_HIDDEN_SIZE,
                 dec_hidden_size: int = DEC_HIDDEN_SIZE,
                 input_feed: bool = INPUT_FEED,
                 share_embed: bool = SHARE_EMBED,
                 mix_vocab: bool = MIX_VOCAB,
                 use_pre_embed: bool = USE_PRE_EMBED,
                 freeze_pre_embed: bool = FREEZE_PRE_EMBED,
                 dropout: float = DROPOUT,
                 teacher_forcing: float = TEACHER_FORCING,
                 gen_loss_func: Union[Callable, str] = GEN_LOSS_FUNC,
                 gen_loss_type: str = GEN_LOSS_TYPE,
                 attn_func: Union[Callable, str] = ATTN_FUNC,
                 ocd_model_path: str = None,
                 *args,
                 **kwargs):
        super(BaseUpdater, self).__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.edit_vec_size = edit_vec_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.input_feed = input_feed
        self.share_embed = share_embed
        self.mix_vocab = isinstance(vocab, MixVocab)
        self.use_pre_embed = use_pre_embed
        self.freeze_pre_embed = freeze_pre_embed
        self.dropout = dropout
        self.teacher_forcing = teacher_forcing
        assert gen_loss_type in ['sent_level', 'word_level']
        self.loss_type = gen_loss_type
        logging.info("gen_loss_type: {}".format(self.loss_type))
        self.loss_func = get_attr_by_name(gen_loss_func) if isinstance(gen_loss_func, str) else gen_loss_func
        self.attn_func = get_attr_by_name(attn_func) if isinstance(attn_func, str) else attn_func
        self.ocd_model_path = ocd_model_path

        self.code_vocab, self.nl_vocab, self.action_vocab = self.vocab.get_subvocabs()
        self.tgt_action_vocab = None

        self.encoder = None
        self.dec_state_init = None
        self.decoder = None

    def init_pretrain_embeddings(self, freeze: bool):
        self.encoder.init_pretrain_embeddings(freeze)
        self.decoder.init_pretrain_embeddings(freeze)

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def _prepare_dec_init_state(self, last_cells: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        :param last_cells: List[(batch_size, edit_vec_size)]
        :return:
            (batch_size, hidden_size)
            (batch_size, hidden_size)
        """
        # (batch_size, hidden_size)
        dec_init_cell = self.dec_state_init(torch.cat(last_cells, dim=-1))
        dec_init_state = torch.tanh(dec_init_cell)
        return dec_init_state, dec_init_cell

    def prepare_decoder_params(self):
        return self.nl_vocab, self.encoder.nl_embed_layer, self.encoder.edit_output_size, \
               self.encoder.nl_output_size, self.embed_size, self.dec_hidden_size, self.teacher_forcing,\
               self.share_embed, self.dropout, self.input_feed, self.attn_func

    def prepare_decoder_kwargs(self, encoder_output: dict, batch: Batch) -> dict:
        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
        src_last_cell, src_sent_masks = encoder_output
        dec_init_state = self._prepare_dec_init_state([src_last_cell, edit_last_cell])
        return self.decoder.prepare_forward_kwargs(encoder_output, batch, dec_init_state)

    def cal_sent_losses(self, target_tensor, words_log_prob: DoubleTensor) -> DoubleTensor:
        """
        :param target_tensor: (*tgt_len* - 1, batch_size)
        :param words_log_prob: logits
        :return: sent-level losses
        """
        # double for reproducability
        words_mask = (target_tensor != self.nl_vocab[PADDING]).double()
        # (tgt_sent_len - 1, batch_size)
        word_losses = self.loss_func(words_log_prob, target_tensor, words_mask)
        example_losses = word_losses.sum(dim=0)
        return example_losses

    def cal_word_losses(self, target_tensor, words_log_prob: DoubleTensor) -> DoubleTensor:
        """
        :param target_tensor: (*tgt_len* - 1, batch_size)
        :param words_log_prob: logits
        :return: word-level losses
        """
        # double for reproducability
        words_mask = (target_tensor != self.nl_vocab[PADDING]).double()
        # (tgt_sent_len - 1, batch_size)
        word_losses = self.loss_func(words_log_prob, target_tensor, words_mask)
        example_losses = word_losses.sum(dim=0) / words_mask.sum(dim=0)
        return example_losses

    def forward(self, batch: Batch) -> Tensor:
        input_tensor = batch.construct_encoder_input(self.code_vocab, self.action_vocab, self.nl_vocab, self.device)
        tgt_in_tensor = batch.get_tgt_in_tensor(self.nl_vocab, self.device)
        tgt_out_tensor = self.decoder.prepare_tgt_out_tensor(batch, self.nl_vocab,
                                                             tgt_action_vocab=self.tgt_action_vocab)
        encoder_output = self.encoder(*input_tensor)

        decoder_kwargs = self.prepare_decoder_kwargs(encoder_output, batch)
        # omit the last word of tgt, which is </s>
        # (tgt_sent_len - 1, batch_size, hidden_size)
        words_log_prob, ys = self.decoder(tgt_in_tensor, **decoder_kwargs)

        if self.loss_type == 'sent_level':
            example_losses = self.cal_sent_losses(tgt_out_tensor, words_log_prob)
        else:
            example_losses = self.cal_word_losses(tgt_out_tensor, words_log_prob)
        return example_losses

    def beam_search(self, example: Example, beam_size: int, max_dec_step: int,
                    BeamClass=Beam) -> List[Hypothesis]:
        batch = Batch([example])
        input_tensor = batch.construct_encoder_input(self.code_vocab, self.action_vocab, self.nl_vocab, self.device)
        encoder_output = self.encoder(*input_tensor)
        decoder_kwargs = self.prepare_decoder_kwargs(encoder_output, batch)
        hypos = self.decoder.beam_search(example, beam_size, max_dec_step, BeamClass, **decoder_kwargs)
        return hypos


class BaseBothPtrUpdater(BaseUpdater, ABC):
    @classmethod
    def prepare_model_params(cls, args: Namespace):
        mix_vocab = getattr(args, "mix_vocab", cls.MIX_VOCAB)
        if mix_vocab:
            logging.info("Using mix vocab")
        vocab_class = MixVocab if mix_vocab else Vocab
        args.vocab = vocab_class.load(args.vocab_file)
        return args

    def prepare_decoder_kwargs(self, encoder_output: dict, batch: Batch) -> dict:
        edit_encodings, edit_last_state, edit_last_cell, edit_sent_masks, src_encodings, src_last_state, \
        src_last_cell, src_sent_masks = encoder_output
        dec_init_state = self._prepare_dec_init_state([src_last_cell, edit_last_cell])
        return self.decoder.prepare_forward_kwargs(encoder_output, batch, dec_init_state, nl_vocab=self.nl_vocab)


class BaseBPBAUpdater(BaseBothPtrUpdater, ABC):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)


class CoAttnBPBAUpdater(BaseBPBAUpdater):
    """
    Use attention mechanism in Encoder
    """
    def __init__(self, *args, **kwargs):
        super(CoAttnBPBAUpdater, self).__init__(*args, **kwargs)
        params = locals().copy()
        self.log_args(**params)

        self.encoder = CoAttnEncoder(self.code_vocab, self.action_vocab, self.nl_vocab, self.embed_size,
                                     self.edit_vec_size, self.enc_hidden_size, self.dropout, self.mix_vocab)
        self.dec_state_init = nn.Linear(self.encoder.output_size, self.dec_hidden_size)
        self.decoder = BothPtrBASeqEditor(*self.prepare_decoder_params())
