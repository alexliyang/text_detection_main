import yaml
from easydict import EasyDict as edict


def load_config():
    config_path = 'configure.yml'
    with open(config_path, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    return yaml_cfg
