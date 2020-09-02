#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta
import inspect


class _BaseMeta(ABCMeta):
    _fields = []

    def __init__(self, name, bases, cls_dict):
        super().__init__(name, bases, cls_dict)
        sup = super(self, self)

        for field in self._fields:
            fn = cls_dict.get(field)

            if fn is None:
                raise TypeError(f'Invalid class definition: "{field}" is not defined')

            sup_fn = getattr(sup, field, None)

            if sup_fn is not None:
                fn_sig = inspect.signature(fn)
                sup_fn_sig = inspect.signature(sup_fn)

                if fn_sig != sup_fn_sig:
                    raise TypeError(f'Invalid signature in "{field}" method - expected: {sup_fn_sig}, actual: {fn_sig}')
