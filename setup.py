#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='rating_model',
    version='0.0.1',
    description='Rating prediction model',
    packages=find_packages(exclude=['tests']),
    install_requires=[],
    setup_requires=['setuptools', 'wheel', 'pipenv'],
    python_requires='>=3.8.5',
)
