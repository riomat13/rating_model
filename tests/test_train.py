#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import unittest
from unittest.mock import patch, mock_open

patch('rating_model.train.get_logger')

import io
import json
import random

from faker import Faker

from rating_model.train import ModelState


class ModelStateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fake = Faker()

    @patch('builtins.open')
    def test_save_with_properties_as_json(self, mock_open):
        stream = io.StringIO()

        target_keys = ['name', 'loss']
        info = ModelState(name=self.fake.word(), loss=random.random())

        mock_open.return_value.__enter__.return_value.write.side_effect = stream.write
        info.save()

        mock_open.assert_called_once()
        stream.seek(0)

        data = json.loads(stream.read())

        for key in target_keys:
            self.assertEqual(data.get(key), getattr(info, key))

    def test_load_from_json_file(self):
        target_data = {
            'name': self.fake.word(),
            'loss': random.random()
        }

        mo = mock_open(read_data=json.dumps(target_data))

        with patch('builtins.open', mo):
            info = ModelState.from_jsonfile('file')

        for key, val in target_data.items():
            self.assertEqual(getattr(info, key), val)

    def test_load_invalid_fail_and_raise_key_error(self):
        target_data = {
            'test': self.fake.word(),
            'loss': random.random()
        }

        mo = mock_open(read_data=json.dumps(target_data))

        with patch('builtins.open', mo):
            with self.assertRaises(KeyError):
                ModelState.from_jsonfile('file')


if __name__ == '__main__':
    unittest.main()
