
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Union, Optional, Callable

import dataset
from config import load_yaml_config, dump_config_to_yaml
from common import *
from dataset import Batch, load_dataset_from_file
from models.base import BaseModel
from models.model import build_model_from_args
from models.utils import cross_entropy
from vocab import build_vocab_from_args
import os

torch.set_default_dtype(torch.float32)


class AbstractReporter(ABC):
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def report(self, *args, **kwargs):
        pass

    @abstractmethod
    def report_valid(self, *args, **kwargs):
        pass

    @abstractmethod
    def report_cum(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_report_stat(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_cum_stat(self, *args, **kwargs):
        pass


class BaseReporter(AbstractReporter):
    def __init__(self, metric_name, logger: logging.Logger = logging.root):
        self.logger = logger
        self._report_loss = 0
        self._cum_loss = 0
        self._report_examples = 0
        self._cum_examples = 0
        self._train_begin_time = self._begin_time = None
        self.metric_name = metric_name

    def initialize(self):
        self._train_begin_time = self._begin_time = time.time()

    @property
    def report_loss(self):
        return self._report_loss

    @property
    def cum_loss(self):
        return self._cum_loss

    @property
    def avg_loss(self):
        return self._report_loss / self._report_examples

    @property
    def avg_cum_loss(self):
        return self._cum_loss / self._cum_examples

    @property
    def cum_examples(self):
        return self._cum_examples

    def get_train_time(self):
        return time.time() - self._train_begin_time

    def get_spend_time(self):
        return time.time() - self._begin_time

    def update(self, batch_loss: float, batch_size: int, *args, **kwargs):
        self._report_loss += batch_loss
        self._cum_loss += batch_loss
        self._report_examples += batch_size
        self._cum_examples += batch_size

    def reset_report_stat(self):
        self._report_loss = 0
        self._report_examples = 0
        self._train_begin_time = time.time()

    def reset_cum_stat(self):
        self._cum_loss = 0
        self._cum_examples = 0

    def report(self, epoch, iter):
        train_time = time.time() - self._train_begin_time
        spend_time = time.time() - self._begin_time
        self.logger.info('epoch %d, iter %d, avg. %s %.6f cum. examples %d, speed %.2f examples/sec, time elapsed %.2f sec'
                     % (epoch, iter, self.metric_name, self.avg_loss, self._cum_examples,
                        self._report_examples / train_time, spend_time))

    def report_valid(self, iter, metric_value):
        self.logger.info('validation: iter %d, dev. %s %f' % (iter, self.metric_name, metric_value))

    def report_cum(self, epoch, iter):
        self.logger.info('epoch %d, iter %d, cum. %s %.6f, cum. examples %d' % (epoch, iter, self.metric_name,
                                                                            self.avg_cum_loss, self._cum_examples))


class PPLReporter(AbstractReporter):
    def __init__(self, logger: logging.Logger = logging.root):
        self.logger = logger
        self.base = BaseReporter("ppl", None)
        self._report_tgt_words = 0
        self._cum_tgt_words = 0

    def initialize(self):
        self.base.initialize()

    @property
    def avg_ppl(self):
        return np.exp(self.base.report_loss / self._report_tgt_words)

    @property
    def avg_cum_ppl(self):
        return np.exp(self.base.cum_loss / self._cum_tgt_words)

    def update(self, batch_loss: float, batch_size: int, tgt_words_num: int, *args, **kwargs):
        self.base.update(batch_loss, batch_size)
        # self._report_tgt_words += tgt_words_num
        # self._cum_tgt_words += tgt_words_num

    def reset_report_stat(self):
        self.base.reset_report_stat()
        self._report_tgt_words = 0

    def reset_cum_stat(self):
        self.base.reset_cum_stat()
        self._cum_tgt_words = 0

    def report(self, epoch, iter):
        self.logger.info('epoch %d, iter %d, avg. loss %.6f, avg. %s %.6f '
                     'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec'
                     % (epoch, iter, self.base.avg_loss, self.base.metric_name, self.avg_ppl,
                        self.base.cum_examples, self._report_tgt_words / self.base.get_train_time(),
                        self.base.get_spend_time()))

    def report_cum(self, epoch, iter):
        self.logger.info('epoch %d, iter %d, cum. loss %.6f, cum. %s %.6f cum. examples %d'
                     % (epoch, iter, self.base.avg_cum_loss, self.base.metric_name, self.avg_cum_ppl,
                        self.base.cum_examples))

    def report_valid(self, iter, metric_value):
        self.logger.info('validation: iter %d, dev. %s %f' % (iter, self.base.metric_name, metric_value))


class Procedure(ABC):
    def __init__(self,
                 model: BaseModel,
                 gpu_id: Optional[int] = None,
                 cuda: bool = True,
                 **kwargs):
        """
        :param gpu_id: `int`, optional
            for back compatible
        :param cuda: `bool`
            use cuda or not
        :param seed: `int`
            random seed
        :param kwargs:
        """
        self._gpu_count = 1
        if gpu_id is not None:
            self._device = torch.device("cuda:{}".format(gpu_id))
        elif cuda:
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")
        self._args = Namespace(**kwargs)
        self._model = model

    def _set_device(self):
        self._model.to(self._device)

    @abstractmethod
    def _init_model(self):
        pass


class Trainer(Procedure):
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    LR = 0.001
    LR_DECAY = 0.5
    CLIP_GRAD = 5.0
    UNIFORM_INIT = 0.1
    LOG_EVERY = 100
    VALID_NITER = 500
    PATIENCE = 5
    MAX_TRIAL_NUM = 5
    MAX_EPOCH = 50
    VALID_METRIC = "cross_entropy"

    def __init__(self,
                 model: BaseModel,
                 model_args: dict,
                 train_data: str,
                 dev_data: str,
                 save_to: str,
                 example_class: Callable = dataset.Example,
                 train_dataset_class: Union[Callable, str] = None,
                 pos_train_data: str = None,
                 neg_train_data: str = None,
                 load: bool = False,
                 reload: bool = False,
                 train_batch_size: int = TRAIN_BATCH_SIZE,
                 valid_batch_size: int = VALID_BATCH_SIZE,
                 lr: float = LR,
                 lr_decay: float = LR_DECAY,
                 clip_grad: float = CLIP_GRAD,
                 uniform_init: float = UNIFORM_INIT,
                 log_every: int = LOG_EVERY,
                 valid_niter: int = VALID_NITER,
                 patience: int = PATIENCE,
                 max_trial_num: int = MAX_TRIAL_NUM,
                 max_epoch: int = MAX_EPOCH,
                 valid_metric: str = VALID_METRIC,
                 **kwargs):
        """
        :param model_class:
            model class
        :param train_data:
            train set file path
        :param dev_data:
            valid set file path
        :param save_to:
            model save path
        :param example_class: `Callable`
            Example class
        :param load:
            Load trained model
        :param reload:
            restore from an interrupted train
        :param train_batch_size:
        :param valid_batch_size:
        :param lr:
            learning rate
        :param lr_decay:
        :param clip_grad:
        :param uniform_init:
        :param log_every:
        :param valid_niter: 
        :param patience:
        :param max_trial_num:
        :param max_epoch:
        :param kwargs:
        """
        super(Trainer, self).__init__(model=model, **kwargs)
        self._model_args = model_args
        # set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.initialized = False
        self.logger.debug("Create Trainer with args: {}".format(self._args))

        self._model = model
        self._model_save_path = save_to
        self._train_data = train_data
        self._dev_data = dev_data
        self._example_class = example_class
        self._train_dataset_class = train_dataset_class
        self._pos_train_data = pos_train_data
        self._neg_train_data = neg_train_data
        if type(train_dataset_class) == str:
            self._train_dataset_class = get_attr_by_name(train_dataset_class)
        else:
            self._train_dataset_class = train_dataset_class
        logging.info("Using dataset class {} for training".format(self._train_dataset_class))

        self._load = load
        self._reload = reload
        self._train_batch_size = train_batch_size * self._gpu_count
        self._valid_batch_size = valid_batch_size * self._gpu_count
        self._lr = lr
        self._lr_decay = lr_decay
        self._clip_grad = clip_grad
        self._uniform_init = uniform_init
        self._log_every = log_every // self._gpu_count
        self._valid_niter = valid_niter // self._gpu_count
        self._patience = patience
        self._max_trial_num = max_trial_num
        self._max_epoch = max_epoch

        self._epoch = 0
        self._train_iter = 0
        self._cur_patience = 0
        self._cur_trial = 0
        self._hist_valid_scores = []
        self.loss_reporter = PPLReporter(self.logger)
        self.valid_metric = valid_metric
        logging.info("Valid metric is {}".format(self.valid_metric))

    @property
    def max_epoch(self):
        return self._max_epoch

    @classmethod
    def get_optim_save_path(cls, model_save_path: str):
        return model_save_path + ".optim"

    @classmethod
    def get_state_save_path(cls, model_save_path: str):
        return model_save_path + ".state"

    def _uniform_init_model_params(self):
        uniform_init = self._uniform_init
        if np.abs(uniform_init) > 0.:
            self.logger.info('uniformly initialize parameters [-{}, +{}]'.format(uniform_init, uniform_init))
            for name, p in self._model.named_parameters():
                p.data.uniform_(-uniform_init, uniform_init)

    def _init_optimizer(self):
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

    def train_a_batch(self, batch: Batch) -> float:
        self._optimizer.zero_grad()
        # (batch_size,)
        example_losses = self._model(batch)
        batch_loss = example_losses.sum()
        loss = batch_loss / len(batch)
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
        self._optimizer.step()
        return batch_loss.item()

    def _save_states(self, state_save_path: str):
        keys_to_store = ['_epoch', '_train_iter', '_cur_patience', '_cur_trial', '_hist_valid_scores']
        states_to_store = {}
        for key in keys_to_store:
            states_to_store[key] = getattr(self, key)
        torch.save(states_to_store, state_save_path)

    def _load_states(self, state_save_path: str):
        self.logger.info('restore trainer states')
        states = torch.load(state_save_path)
        for key, value in states.items():
            # restore all states
            setattr(self, key, value)

    def save_model(self, model_save_path: Optional[str] = None):
        model_save_path = model_save_path or self._model_save_path
        self.logger.info('save currently the best model to [%s]' % model_save_path)
        self._model.save(model_save_path, self._model_args)

        # also save the optimizers' state
        optim_save_path = self.get_optim_save_path(model_save_path)
        torch.save(self._optimizer.state_dict(), optim_save_path)

        state_save_path = self.get_state_save_path(model_save_path)
        self._save_states(state_save_path)

    def remove_model(self, model_save_path: Optional[str] = None):
        model_save_path = model_save_path or self._model_save_path
        os.remove(model_save_path)
        optim_save_path = self.get_optim_save_path(model_save_path)
        os.remove(optim_save_path)
        state_save_path = self.get_state_save_path(model_save_path)
        os.remove(state_save_path)

    def load_model(self, load_state: bool = False, model_save_path: Optional[str] = None):
        model_save_path = model_save_path or self._model_save_path
        optim_save_path = self.get_optim_save_path(model_save_path)
        state_save_path = self.get_state_save_path(model_save_path)
        if load_state:
            # we don't load the states of trainer if restoring from validation
            self._load_states(state_save_path)
        self.logger.info('load previously best model')
        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        self._model.load_state_dict(params['state_dict'])
        self._set_device()

        self.logger.info('restore parameters of the optimizers')
        self._optimizer.load_state_dict(torch.load(optim_save_path))

    def decay_lr(self, load: bool = True):
        # decay lr, and restore from previously best checkpoint
        lr = self._optimizer.param_groups[0]['lr'] * self._lr_decay
        self.logger.info('decay learning rate to %f' % lr)
        if load:
            self.load_model()

        # set new lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _validate(self, dev_set):
        was_training = self._model.training
        self._model.eval()

        cum_loss = 0
        cum_tgt_words = 0
        with torch.no_grad():
            for batch in dev_set.train_batch_iter(self._valid_batch_size, shuffle=False):
                batch_loss = self._model(batch).sum()
                cum_loss += batch_loss.item()
                cum_tgt_words += batch.tgt_words_num
            dev_ppl = np.exp(cum_loss / cum_tgt_words)
        valid_metric = -dev_ppl

        if was_training:
            self._model.train()

        return valid_metric

    def validate(self, train_iter, dev_set):
        self.logger.info('begin validation ...')

        valid_metric = self._validate(dev_set)
        self.loss_reporter.report_valid(train_iter, valid_metric)

        is_better = len(self._hist_valid_scores) == 0 or valid_metric > max(self._hist_valid_scores)
        self._hist_valid_scores.append(valid_metric)

        return is_better

    def _init_model(self):
        self._init_optimizer()

        assert not (self._load and self._reload)

        if self._load:
            self.load_model(load_state=False)
        elif self._reload:
            self.load_model(load_state=True)
        else:
            self._uniform_init_model_params()

            if self._model.use_pre_embed:
                freeze = self._model.freeze_pre_embed
                self.logger.info("initialize word embeddings with pretrained embeddings")
                self._model.init_pretrain_embeddings(freeze)

        self.logger.info("use device: {}".format(self._device))
        self._set_device()
        self._model.train()

    def load_dataset(self):
        self.logger.info("Load example using {}".format(self._example_class))
        example_class = self._example_class
        if self._train_dataset_class and self._train_dataset_class == dataset.BalancedSampleDataset:
            pos_train_set = dataset.Dataset.create_from_file(self._pos_train_data, example_class)
            neg_train_set = dataset.LargeDataset.create_from_file(self._neg_train_data, example_class)
            train_set = dataset.BalancedSampleDataset(neg_train_set, pos_train_set)
        else:
            train_set = load_dataset_from_file(self._train_data, example_class)
        dev_set = load_dataset_from_file(self._dev_data, example_class)
        return train_set, dev_set

    def sanity_test(self, batch_count=10):
        if self.initialized:
            self.logger.error("The trainer has already been initialized")
            return
        self.initialized = True
        train_set, dev_set = self.load_dataset()
        self._init_model()

        for idx, batch in enumerate(train_set.train_batch_iter(batch_size=self._train_batch_size, shuffle=True)):
            if idx >= batch_count:
                break
            batch_loss_val = self.train_a_batch(batch)
            self.logger.info("Sanity train: {}".format(batch_loss_val))
            # break
        self.validate(1, dev_set)
        model_save_path = self._model_save_path + ".test"
        self.save_model(model_save_path)
        self.load_model(True, model_save_path)
        self.logger.info("Remove test model")
        self.remove_model(model_save_path)

    def train(self):
        if self.initialized:
            self.logger.error("The trainer has already been initialized")
            return
        self.initialized = True
        # load_dataset
        train_set, dev_set = self.load_dataset()
        self._init_model()

        self.loss_reporter.initialize()
        # self.logger.info("Start training")
        # self.logger.info("This is my output for debugging, train_set class!")
        # self.logger.info(f"{type(train_set)}")
        # train_set:Example
        # DatasetClass:Dataset
        # exampleclass:Example
        while True:
            self._epoch += 1
            for batch in train_set.train_batch_iter(batch_size=self._train_batch_size, shuffle=True):
                self._train_iter += 1
                batch_loss_val = self.train_a_batch(batch)
                self.loss_reporter.update(batch_loss_val, len(batch), batch.tgt_words_num)

                if self._train_iter % self._log_every == 0:
                    self.loss_reporter.report(self._epoch, self._train_iter)
                    self.loss_reporter.reset_report_stat()

                if self._train_iter % self._valid_niter == 0:
                    self.loss_reporter.report_cum(self._epoch, self._train_iter)
                    self.loss_reporter.reset_cum_stat()

                    is_better = self.validate(self._train_iter, dev_set)
                    if is_better:
                        self._cur_patience = 0
                        self.save_model()
                    else:
                        self._cur_patience += 1
                        self.logger.info('hit patience {}'.format(self._cur_patience))

                        if self._cur_patience == self._patience:
                            self._cur_trial += 1
                            self.logger.info('hit #{} trial'.format(self._cur_trial))
                            if self._cur_trial >= self._max_trial_num:
                                self.logger.info('early stop!')
                                return

                            self.decay_lr()

                            # reset patience
                            self._cur_patience = 0

            if self._epoch == self._max_epoch:
                self.logger.info('reached maximum number of epochs')
                is_better = self.validate(self._train_iter, dev_set)
                if is_better:
                    self.logger.info('save the final model')
                    self.save_model()
                return


class ClassifierTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_reporter = BaseReporter("cross_entropy", self.logger)

    def cal_cross_entropy(self, dev_set):
        cum_loss = 0
        example_num = 0
        with torch.no_grad():
            for batch in dev_set.train_batch_iter(self._valid_batch_size, shuffle=False):
                label_tensor = batch.get_label_tensor(self._model.device)
                probs = self._model.predict(batch)
                masks = torch.ones_like(label_tensor).double()
                example_losses = cross_entropy(probs, label_tensor, masks)
                batch_loss = example_losses.sum()
                cum_loss += batch_loss.item()
                example_num += len(batch)
        valid_perf = -(cum_loss / example_num)

        return valid_perf

    def _validate(self, dev_set):
        was_training = self._model.training
        self._model.eval()

        valid_perf = self.cal_cross_entropy(dev_set)

        if was_training:
            self._model.train()

        return valid_perf


def build_trainer_raw(
    configs: dict,
    log_dir: str,
    load: bool = False,
    reload: bool = False,
    log: bool = True,
    gpu_id: Optional[int] = None,
    check_conf: bool = True,
    **kwargs
):
    """
    This is the factory method for building trainer

    :param configs: `dict`
        The configurations read from a config file
    :param log_dir: `str`
        The directory to store all files of this training.
    :param load: `bool`
        Load trained model and re-train from the beginning
    :param reload: `bool`
        Reload the model or train from scratch.
    :param log: `bool`
        Whether to log
    :param configs: `dict`
        The parameters specified by the configuration file.
    :param gpu_id: `int`, optional
        For back compatible
    :param check_conf: `bool`, (default = None)
        Check configuration mismatch. Only when reload=True, check_conf=False can make sense.
    :return: `Trainer`
        The trainer which can be used to conduct train
    """
    def _get_conf_file():
        return os.path.join(log_dir, "train.yml")

    train_config = configs['train']
    train_config.update(kwargs)
    train_config['save_to'] = os.path.join(log_dir, train_config['save_to'])

    config_file = _get_conf_file()
    # create log dir
    # load = True: we re-train from a model in different dir
    # reload = True: we re-train from a model in this dir
    # so only when reload == True, the log dir can exist
    # if the log dir exist and check_conf, we check whether the two configs are equal
    # otherwise, we simple create the log_dir or pass by
    if os.path.exists(log_dir):
        if not reload:
            logging.error("Dir {} has existed! Recheck it!".format(log_dir))
            exit(1)
        elif check_conf:
            old_configs = load_yaml_config(config_file)
            if old_configs != train_config:
                logging.error("Configs are not consistent with old configs!")
                exit(1)
        # if reload and not check_conf, we just pass by
    else:
        os.makedirs(log_dir)
    # dump the configs
    dump_config_to_yaml(train_config, config_file)

    # log
    log_file = os.path.join(log_dir, "train.log")
    if log:
        setup_logger(logging.root, log_file, logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    seed = train_config.pop('seed', 0)
    logging.info("Random Seed is set to {}".format(seed))
    set_reproducibility(seed)

    vocab = build_vocab_from_args(**configs['vocab'])
    # store model_args for re-build model
    model_args = dict(vocab=vocab, **configs['model'])
    model = build_model_from_args(**model_args)
    model_type = train_config.pop('model_type')

    if model_type == "classifier":
        trainer_class = ClassifierTrainer
    else:
        trainer_class = Trainer

    logging.info(f"Create {trainer_class} using parameters:")
    logging.info(f"{train_config}")
    trainer = trainer_class(
        model=model,
        model_args=model_args,
        load=load,
        reload=reload,
        gpu_id=gpu_id,
        **train_config
    )
    return trainer


def train_from_args(
    configs: dict,
    log_dir: str,
    load: bool = False,
    reload: bool = False,
    log: bool = True,
    gpu_id: int = None,
    **kwargs
):
    logging.info(f"This is my output for debug!")
    logging.info(f"{configs}")
    trainer = build_trainer_raw(configs, log_dir, load, reload, log, gpu_id, **kwargs)
    return trainer.train()


def train_from_config(
    config_file: str,
    log_dir: str,
    load: bool = False,
    reload: bool = False,
    log: bool = True,
    gpu_id: int = None,
    **kwargs
):
    
    configs = load_yaml_config(config_file)
    print("This is my output for debug!")
    logging.info(f"This is my output for debug!")
    logging.info(f"{configs}")
    
    return train_from_args(configs, log_dir, load, reload, log, gpu_id, **kwargs)


def sanity_test_from_args(
    configs: dict,
    log_dir: str,
    load: bool = False,
    reload: bool = False,
    log: bool = True,
    gpu_id: int = None,
    **kwargs
):
    trainer = build_trainer_raw(configs, log_dir, load, reload, log, gpu_id, **kwargs)
    return trainer.sanity_test()


def sanity_test_from_config(
    config_file: str,
    log_dir: str,
    load: bool = False,
    reload: bool = False,
    log: bool = True,
    gpu_id: int = None,
    **kwargs
):
    configs = load_yaml_config(config_file)
    return sanity_test_from_args(configs, log_dir, load, reload, log, gpu_id, **kwargs)


def main():
    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True,
                        help="dir for storing all data of this training")
    parser.add_argument('--config', type=str, required=True,
                        help="config file")
    parser.add_argument('--load', action="store_true",
                        help="load trained model and re-train from the beginning")
    parser.add_argument('--reload', action="store_true",
                        help="restore the model")
    parser.add_argument('--no-log', action="store_true",
                        help="do not log")
    parser.add_argument('--gpu-id', type=int, default=None,
                        help="for back compatible")
    parser.add_argument('--sanity-test', action="store_true",
                        help="run sanity test instead of full train")

    args = parser.parse_args()
    if args.sanity_test:
        return sanity_test_from_config(args.config, args.log_dir, args.load, args.reload, not args.no_log, args.gpu_id)
    else:
        return train_from_config(args.config, args.log_dir, args.load, args.reload, not args.no_log, args.gpu_id)


if __name__ == '__main__':
    main()
