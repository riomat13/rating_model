#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from functools import partial

import tensorflow as tf

from rating_model.settings import Config
from rating_model.base import Dataset


class ReviewDataset(Dataset):
    """Extract data from database."""

    __feature = {
        'input_ids': tf.io.FixedLenFeature([], tf.string),
        'token_type_ids': tf.io.FixedLenFeature([], tf.string),
        'attention_mask': tf.io.FixedLenFeature([], tf.string),
        'rating': tf.io.FixedLenFeature([], tf.float32)
    }

    def __init__(self,
                 filepaths=None,
                 cls='train',
                 batch_size=32,
                 shuffle=True,
                 repeat=False,
                 path=None):
        """Take files for `tf.data.Dataset`.

        If you want `shuffle` to run deterministically,
        set random seed by `settings.set_random_seed` function.

        Args:
            filepaths: list of str
                list of path to tfrecord files
                if this is provided, `cls` and `path` will not be used.
            cls: str (default: "train")
                type of data eigther "train" or "test"
                "train" will be shuffled and repeated
            batch_size: int (default: 32)
            shuffle: bool (default: True)
                shuffle data, this is only for cls='train'
                `rating_model.settings.Config.RANDOM_SEED` value will be used
                for random seed
            repeat: bool (default: False)
                repeat data, this is only for cls='train'
            path: str or None
                path to directory containing data to be used.
                if not set, use from config

        Raises:
            ValueError: if cls is neigther "train" nor "test"
        """
        if filepaths is None and cls not in ('train', 'test'):  # pragma: no cover
            raise ValueError('`cls` must be chosen from ("train", "test")')

        self.repeat = False

        if filepaths is None:
            self.path = path or Config.DATA_DIR

            filepaths = list(
                f.path for f in os.scandir(f'{self.path}/{cls}')
                if f.is_file() and os.path.splitext(f.name)[-1] == '.tfrecord'
            )

        self._data = tf.data.TFRecordDataset(filepaths) \
            .map(self._decode) \
            .batch(batch_size, drop_remainder=True)

        if cls == 'train':
            if shuffle:
                self._data = self._data \
                    .shuffle(1024,
                             seed=Config.RANDOM_SEED,
                             reshuffle_each_iteration=True)

            if repeat:
                self._data = self._data.repeat()
                self.repeat = True

        self._iter_data = iter(self._data)

    _decoder = partial(tf.io.decode_raw, out_type=tf.int32)

    @tf.autograph.experimental.do_not_convert
    def _decode(self, proto):
        data = tf.io.parse_single_example(proto, self.__feature)

        out_data = {
            'token': {
                'input_ids': self._decoder(data['input_ids']),
                'token_type_ids': self._decoder(data['token_type_ids']),
                'attention_mask': self._decoder(data['attention_mask'])
            },
            'rating': data['rating']
        }
        return out_data

    def generate(self):
        """Generate data per batch."""
        try:
            data = next(self._iter_data)
        except StopIteration:
            return None

        return data

    def generate_all(self):
        """Return generator of dataset.
        This can be run when `repeat` is set to False.
        """
        if self.repeat:
            raise RuntimeError('can not call if dataset is set to repeated')

        def generator():
            while (data := self.generate()) is not None:
                yield data
            self.reset()

        return generator()

    def reset(self):
        """Reset data generator.
        This can be run when `repeat` is set to False.
        """
        if self.repeat:
            raise RuntimeError('can not call if dataset is set to repeated')

        self._iter_data = iter(self._data)
