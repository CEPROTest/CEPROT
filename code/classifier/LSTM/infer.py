# encoding=utf-8
import os

from config import load_yaml_config
from models import beam
from argparse import ArgumentParser
from typing import Callable, Union, Tuple, Optional

import torch
import json
import logging
from tqdm import tqdm
from dataset import Dataset, Example, LargeDataset, load_dataset_from_file
from common import setup_logger, set_reproducibility
from models.model import load_model_from_args
from train import Procedure, List


class Infer(Procedure):
    BEAM_SIZE = 5
    MAX_DEC_STEP = 100
    BEAM_CLASS = beam.Beam

    def __init__(self,
                 pbar: bool = True,
                 beam_size: int = BEAM_SIZE,
                 max_dec_step: int = MAX_DEC_STEP,
                 beam_class: Callable = BEAM_CLASS,
                 **kwargs):
        super(Infer, self).__init__(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.beam_size = beam_size
        self.max_dec_step = max_dec_step
        self.beam_class = beam_class
        self.pbar = pbar
        self._init_model()

    def _init_model(self):
        self._set_device()

    def beam_search(self, data_set):
        self.logger.debug("Using beam class: {}".format(self.beam_class))
        BeamClass = self.beam_class
        was_training = self._model.training
        self._model.eval()

        hypos = []
        with torch.no_grad():
            for example in tqdm(data_set, disable=(not self.pbar)):
                example_hypos = self._model.beam_search(example, self.beam_size, self.max_dec_step, BeamClass)
                hypos.append(example_hypos)

        if was_training:
            self._model.train()
        return hypos

    def infer_raw(self) -> List:
        test_set_file = self._args.test_set_file
        test_set = load_dataset_from_file(test_set_file)
        hypos = self.beam_search(test_set)
        return hypos

    def infer(self) -> List:
        hypos = self.infer_raw()
        with open(self._args.output_file, 'w') as f:
            json.dump(hypos, f)
        return hypos

    def infer_one(self, code_change_seq: List[List[str]], src_desc_tokens: List[str],
                  ExampleClass: Callable = Example):
        example = ExampleClass.create_partial_example({
            'code_change_seq': code_change_seq,
            'src_desc_tokens': src_desc_tokens,
        })
        test_set = Dataset([example])
        hypos = self.beam_search(test_set)
        dst_desc_tokens = hypos[0][0][0]
        with open(self._args.output_file, 'w') as f:
            json.dump(hypos, f)
        return dst_desc_tokens


class ClassifierInfer(Infer):
    BATCH_SIZE = 32

    @classmethod
    def add_trainer_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = super().add_trainer_specific_args(parent_parser)
        parser.add_argument('--batch-size', type=int, default=32,
                            help="Infer batch size, only useful for classifier")
        return parser

    def __init__(self, pbar: bool = True, batch_size: int = BATCH_SIZE, **kwargs):
        super().__init__(pbar=pbar, **kwargs)
        self.batch_size = batch_size

    def predict(self, data_set: Union[Dataset, LargeDataset]):
        was_training = self._model.training
        self._model.eval()

        batch_preds = []
        with torch.no_grad():
            for batch in tqdm(data_set.infer_batch_iter(self.batch_size), disable=(not self.pbar)):
                batch_preds.append(self._model.predict(batch))

        if was_training:
            self._model.train()

        probs = torch.cat(batch_preds, dim=0).to('cpu')
        assert probs.size(-1) == 2
        return probs

    def infer_raw(self) -> torch.Tensor:
        test_set_file = self._args.test_set_file
        test_set = load_dataset_from_file(test_set_file)
        probs = self.predict(test_set)
        return probs

    def infer(self) -> torch.Tensor:
        probs = self.infer_raw()
        torch.save(probs, self._args.output_file)
        return probs


class CompositeInfer(ClassifierInfer):
    def __init__(self, pbar: bool = True, **kwargs):
        super().__init__(pbar=pbar, **kwargs)

    @classmethod
    def get_probs_file(cls, prefix):
        return prefix + ".probs"

    @classmethod
    def get_hypos_file(cls, prefix):
        return prefix + ".hypos"

    def _infer(self, data_set: [Dataset, LargeDataset]) -> Tuple[torch.Tensor, List]:
        self.logger.info("Using beam class: {}".format(self.beam_class))
        BeamClass = self.beam_class
        was_training = self._model.training
        self._model.eval()

        hypos = []
        probs = []
        with torch.no_grad():
            for batch in tqdm(data_set.infer_batch_iter(self.batch_size), disable=(not self.pbar)):
                cur_probs, cur_hypos = self._model.infer(batch, self.beam_size, self.max_dec_step, BeamClass)
                probs.append(cur_probs)
                hypos.extend(cur_hypos)

        probs = torch.cat(probs, dim=0).to('cpu')
        assert probs.size(-1) == 2
        assert len(hypos) == len(probs)

        if was_training:
            self._model.train()

        return probs, hypos

    def infer_raw(self):
        test_set_file = self._args.test_set_file
        test_set = load_dataset_from_file(test_set_file)
        probs, hypos = self._infer(test_set)
        return probs, hypos

    def infer(self) -> Tuple[torch.Tensor, List]:
        probs, hypos = self.infer_raw()
        torch.save(probs, self.get_probs_file(self._args.output_file))
        with open(self.get_hypos_file(self._args.output_file), 'w') as f:
            json.dump(hypos, f)
        return probs, hypos


def build_inferer(configs: dict,
                  log_dir: str,
                  log: bool = True,
                  gpu_id: Optional[int] = None,
                  **kwargs) -> \
        Union[Infer, ClassifierInfer, CompositeInfer]:
    log_file = os.path.join(log_dir, "infer.log")
    if log:
        setup_logger(logging.root, log_file, logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    seed = kwargs.pop('seed', configs['infer'].get('seed', 0))
    logging.info("Random seed is set to {}".format(seed))
    set_reproducibility(seed)

    model_class = kwargs.pop('model_class', configs['model'].pop('model_class'))
    infer_configs = configs['infer']
    infer_configs.update(kwargs)
    model_path = infer_configs.pop('model_path')
    model_path = os.path.join(log_dir, model_path)
    model = load_model_from_args(model_class, model_path)
    model_type = infer_configs.pop('model_type')

    if model_type == "classifier":
        infer_class = ClassifierInfer
    elif model_type == "composite":
        infer_class = CompositeInfer
    else:
        infer_class = Infer

    infer_configs['output_file'] = os.path.join(log_dir, infer_configs['output_file'])

    logging.info("Infer using parameters: {}".format(infer_configs))

    infer_instance = infer_class(pbar=log, model=model, gpu_id=gpu_id, **infer_configs)
    return infer_instance


def build_inferer_from_config(config_file: str,
                              log_dir: str,
                              log: bool = True,
                              **kwargs):
    configs = load_yaml_config(config_file)
    infer_instance = build_inferer(configs, log_dir, log, **kwargs)
    return infer_instance


def infer_from_args(
    configs: dict,
    log_dir: str,
    log: bool = True,
    gpu_id: Optional[int] = None
):
    inferer = build_inferer(configs, log_dir, log, gpu_id)
    return inferer.infer()


def infer_from_config(
    config_file: str,
    log_dir: str,
    log: bool = True,
    gpu_id: Optional[int] = None
):
    configs = load_yaml_config(config_file)
    return infer_from_args(configs, log_dir, log, gpu_id)


def main():
    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True,
                        help="dir for storing log and model")
    parser.add_argument('--config', type=str, required=True,
                        help="config file")
    args = parser.parse_args()
    return infer_from_config(args.config, args.log_dir, log=True)


if __name__ == '__main__':
    main()
