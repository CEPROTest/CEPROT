# encoding=utf-8
import os

from common import get_attr_by_name
from config import load_yaml_config
from models.base import BaseModel
from vocab import BaseVocab


def build_model_from_args(
    vocab: BaseVocab,
    model_class: str,
    **kwargs
) -> BaseModel:
    ModelClass = get_attr_by_name(model_class)
    model = ModelClass(vocab=vocab, **kwargs)
    return model


def build_model_from_config(
    config_file: str,
    vocab: BaseVocab
):
    model_args = load_yaml_config(config_file)['model']
    return build_model_from_args(vocab=vocab, **model_args)


def load_model_from_args(
    model_class: str,
    model_path: str,
) -> BaseModel:
    ModelClass = get_attr_by_name(model_class)
    model = ModelClass.load(model_path)
    return model


def load_model_from_config(
    config_file: str,
    log_dir: str,
    model_class: str = None
):
    all_configs = load_yaml_config(config_file)
    configs = all_configs['infer']
    model_class = all_configs['model']['model_class'] if model_class is None else model_class
    model_path = configs['model_path']
    model_path = os.path.join(log_dir, model_path)
    return load_model_from_args(model_class, model_path)
