#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import inspect

from faker import Faker

from rating_model.settings import Config


fake = Faker()


class ConfigTest(unittest.TestCase):

    def test_assing_variables_to_config_from_dict_and_check_immutablity(self):

        func_or_methods = set((
            'print_all', 'get_config', 'from_config', 'from_dict'
        ))

        # take all variables(not callable) and change them
        keys = []
        for key in dir(Config):
            if key.startswith('_') or \
                    inspect.isfunction((prop := getattr(Config, key))) or \
                    inspect.ismethod(prop):
                continue
            keys.append(key)

        config = {key: fake.word() for key in keys}

        Config.from_dict(config)

        for key, val in config.items():
            self.assertEqual(getattr(Config, key), val)

            with self.assertRaises(AttributeError):
                setattr(Config, key, fake.word())


if __name__ == '__main__':
    unittest.main()
