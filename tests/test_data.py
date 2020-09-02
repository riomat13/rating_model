#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from rating_model.data import ReviewDataset
from rating_model.settings import Config, set_random_seed

# total number of data in tfrecord files
TOTAL_DATA_SIZE = 16
BATCH_SIZE = 4
NUM_FILES = 3

# no use
del tf.test.TestCase.test_session


class ReviewDatasetTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        set_random_seed(1234)

    @classmethod
    def tearDownClass(cls):
        set_random_seed(None)

    def test_generate_data_repeatedly(self):
        # each test file contains 16 data
        dataset = ReviewDataset(
            cls='train',
            batch_size=BATCH_SIZE,
            repeat=True,
            path=os.path.join(Config.ROOT_DIR, 'tests/data')
        )

        # all generated data must be fixed size
        # if there are remainders, they will be ignored
        for _ in range(10):
            items = dataset.generate()
            self.assertEqual(len(items['token']['input_ids']), BATCH_SIZE)
            self.assertEqual(len(items['rating']), BATCH_SIZE)

    def test_prohibit_call_generate_all_when_data_is_repeated(self):
        dataset = ReviewDataset(
            cls='train',
            batch_size=BATCH_SIZE,
            shuffle=True,
            repeat=True,
            path=os.path.join(Config.ROOT_DIR, 'tests/data')
        )

        with self.assertRaises(RuntimeError):
            for items in dataset.generate_all():
                pass

    def test_prohibit_call_reset_when_data_is_repeated(self):
        dataset = ReviewDataset(
            cls='train',
            batch_size=BATCH_SIZE,
            shuffle=True,
            repeat=True,
            path=os.path.join(Config.ROOT_DIR, 'tests/data')
        )

        with self.assertRaises(RuntimeError):
            dataset.reset()

    def test_generate_all_data_in_files(self):
        dataset = ReviewDataset(
            cls='train',
            batch_size=BATCH_SIZE,
            shuffle=True,
            repeat=True,
            path=os.path.join(Config.ROOT_DIR, 'tests/data')
        )

        appeared = set()

        for _ in range(10):
            items = dataset.generate()
            for token in items['token']['input_ids']:
                appeared.add(token.ref())

        self.assertEqual(len(appeared), 10 * BATCH_SIZE)

    def test_generate_deterministically_with_seed_from_config(self):
        dataset = ReviewDataset(
            cls='test',
            batch_size=BATCH_SIZE,
            path=os.path.join(Config.ROOT_DIR, 'tests/data')
        )

        data1 = list(items['token']['input_ids'] for items in dataset.generate_all())

        dataset = ReviewDataset(
            cls='test',
            batch_size=BATCH_SIZE,
            path=os.path.join(Config.ROOT_DIR, 'tests/data')
        )

        data2 = list(items['token']['input_ids'] for items in dataset.generate_all())

        for t1, t2 in zip(data1, data2):
            self.assertAllEqual(t1, t2)

    def test_not_repeat_in_test(self):
        dataset = ReviewDataset(
            cls='test',
            batch_size=BATCH_SIZE,
            path=os.path.join(Config.ROOT_DIR, 'tests/data'),
            shuffle=True
        )

        count = 0

        for items in dataset.generate_all():
            count += 1

        # number of called must be number of total data
        # divided by batch size without remainder
        self.assertEqual(count, TOTAL_DATA_SIZE // BATCH_SIZE)
