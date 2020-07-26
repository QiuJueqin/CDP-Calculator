import os.path as op
from types import SimpleNamespace

import json
import numpy as np


def load_config(cfg_file):
    """ Load configurations from a json and store them in a namespace """
    assert op.isfile(cfg_file), 'configuration file {} not found.'.format(cfg_file)

    config = SimpleNamespace()
    with open(cfg_file) as fp:
        cfg = json.load(fp)
    for key, value in cfg.items():
        setattr(config, key, value)

    return config


def rgb2luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
