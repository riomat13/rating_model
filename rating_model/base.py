#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rating_model._base import _BaseMeta


class BaseDatasetMeta(_BaseMeta):
    """Metaclass for Dataset class."""
    _fields = ['generate', 'generate_all', 'reset']


class Dataset(metaclass=BaseDatasetMeta):  # pragma: no cover
    """Abstract class for Dataset."""

    def generate(self):
        pass

    def generate_all(self):
        pass

    def reset(self):
        pass


class BaseTrainerMeta(_BaseMeta):
    """Metaclass for Trainer class."""
    _fields = ['train']


class BaseTrainer(metaclass=BaseTrainerMeta):  # pragma: no cover
    """Abstract class for Trainer."""

    def train(self,
              train_data: Dataset,
              epochs: int,
              iter_per_epoch: int,
              val_data: Dataset = None,
              verbose_step: int = 100):
        pass
