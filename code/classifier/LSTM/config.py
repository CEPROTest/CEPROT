# encoding=utf-8
import os

import yaml
from common import get_attr_by_name


def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def parse_class(loader, node):
    return get_attr_by_name(str(node.value))


def load_yaml_config(path: str) -> dict:
    yaml.Loader.add_constructor('!join', join)
    yaml.Loader.add_constructor('!callable', parse_class)
    file = open(path, 'r')
    kwargs = yaml.load(file, Loader=yaml.Loader)
    file.close()

    if 'includes' in kwargs:
        base_kwargs = {}
        includes = kwargs.pop('includes')
        for ext_file in includes:
            if not os.path.exists(ext_file):
                ext_file = os.path.join(os.path.dirname(path), ext_file)
                assert os.path.exists(ext_file)
            ext_kwargs = load_yaml_config(ext_file)
            base_kwargs.update(ext_kwargs)
        base_kwargs.update(kwargs)
        kwargs = base_kwargs

    return kwargs


def dump_config_to_yaml(configs: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(configs, f)
