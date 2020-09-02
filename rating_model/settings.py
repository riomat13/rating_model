#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import inspect


def get_logger(name: str, level: str = None) -> logging.RootLogger:
    if level is not None:
        level = level.upper()

    if level not in (None, 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
        raise ValueError('Invalid log level')

    if level is None:
        level = Config.LOG_LEVEL
    else:
        level = getattr(logging, level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    fh = logging.FileHandler(f'{Config.LOG_DIR}/{name}.log')
    fh.setLevel(level)

    sh = logging.StreamHandler()
    sh.setLevel(level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


class _Prop:

    def __init__(self,
                 prop: str = None,
                 key: str = None,
                 bool_key: str = None,
                 bool_val: [object, object] = None):
        """Set property.

        If `bool_key` is set, set property as `prop = instance.bool_key ? bool_key[0] : bool_key[1]`
        """

        self.prop = prop
        self.key = key
        self.bool_key = bool_key
        self.bool_val = bool_val

    def __get__(self, inst, inst_type):
        if inst is None:
            return self

        if self.prop is None:
            if self.key is not None:
                self.prop = getattr(inst, self.key)
            elif self.bool_key is not None:
                self.prop = self.bool_val[0] if getattr(inst, self.bool_key) else self.bool_val[1]
        return self.prop

    def __set__(self, inst, value, forced=False):
        if not forced:
            raise AttributeError('Accessing Read-only property')
        
        self.prop = value


class _DirPathProp:
    """Set path for Config property."""

    def __init__(self, subpath, /, env_val=None):
        self.subpath = subpath
        if env_val is not None:
            self.prop = os.getenv(env_val)
        else:
            self.prop = None

    def __get__(self, inst, inst_type):
        if self.prop is None:
            self.prop = os.path.join(inst.ROOT_DIR, self.subpath)
        return self.prop

    def __set__(self, *args):
        raise AttributeError('Accessing Read-only property')


class _Config:  # pragma: no cover
    """Configuration for this app.
    If need to set `DEBUG` to True, use `get_config(debug=True)` instead.
    """
    ROOT_DIR = _Prop(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_DIR = _DirPathProp('data', env_val='ML_MODEL_DATA_DIR')

    LOG_DIR = _DirPathProp('logs')
    LOG_LEVEL = _Prop('INFO')

    MODEL_DIR = _DirPathProp('data/models')
    MODEL_CHECKPOINT = _DirPathProp('data/models/checkpoints')

    RANDOM_SEED = _Prop(None)

    MODEL_CACHE_DIR = _Prop(os.getenv('TRANSFORMER_MODEL_CACHE_DIR',
                                      os.path.expanduser('~/.cache/transformer')))

    ML_MODEL = {
        'EPOCH': 10,
        'ITER_PER_EPOCH': 100,
    }

    def print_all(self):
        """Print all properties."""
        for key in dir(self):
            if key.startswith('_') or \
                    inspect.isfunction((prop := getattr(self, key))) or \
                    inspect.ismethod(prop):
                continue

            print(f'  {key:<20}: {prop}')

    @classmethod
    def from_config(cls, config: object) -> None:
        """Update configuration by class.
        Note that this can not override initial properties.
        """
        for key in dir(config):
            if key.startswith('_'):
                continue

            setattr(cls, key, getattr(config, key))

    @classmethod
    def from_dict(cls, config: dict) -> None:
        """Update configuration by dict-like objct."""
        if not isinstance(config, dict):
            raise TypeError('`config` must be dict-like object')

        for key, val in config.items():
            setattr(cls, key, _Prop(val))


Config = _Config()

for dir_type in ('DATA_DIR', 'LOG_DIR', 'MODEL_DIR'):
    dirpath = getattr(Config, dir_type)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)


def set_random_seed(seed: int):
    global Config

    _Config.RANDOM_SEED.__set__(Config, seed, forced=True)
