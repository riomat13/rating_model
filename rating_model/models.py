#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from rating_model.transformers import (
    AlbertConfig,
    AlbertTokenizer,
    TFAlbertMainLayer
)

from rating_model.settings import Config
from .data import ReviewDataset


def _get_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    def fn():
        """Default tokenizer to be used."""
        nonlocal tokenizer
        return tokenizer

    return fn

get_tokenizer = _get_tokenizer()


class AlbertRatingModel(tf.keras.Model):  # pragma: no cover 

    def __init__(self, config, *args, **kwargs):
        super(AlbertRatingModel, self).__init__(*args, **kwargs)

        self.albert = TFAlbertMainLayer(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name='output'
        )

    def shape_list(self, x):
        static = x.shape.as_list()
        dynamic = tf.shape(x)
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def call(self, inputs: dict, training=True):
        input_ids = inputs.get("input_ids")  # <-
        attention_mask = inputs.get("attention_mask")  # <-
        token_type_ids = inputs.get("token_type_ids")  # <-
        position_ids = inputs.get("position_ids")
        head_mask = inputs.get("head_mask")
        inputs_embeds = inputs.get("inputs_embeds")
        output_attentions = inputs.get("output_attentions")
        output_hidden_states = inputs.get("output_hidden_states")

        assert len(inputs) <= 8, "Too many inputs."

        seq_length = self.shape_list(input_ids)[1]

        flat_input_ids = (tf.reshape(input_ids, (-1, seq_length))
            if input_ids is not None else None)
        flat_attention_mask = (tf.reshape(attention_mask, (-1, seq_length))
            if attention_mask is not None else None)
        flat_token_type_ids = (tf.reshape(token_type_ids, (-1, seq_length))
            if token_type_ids is not None else None)
        flat_position_ids = (tf.reshape(position_ids, (-1, seq_length))
            if position_ids is not None else None)

        flat_inputs = [
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        ]

        outputs = self.albert(flat_inputs, training=training)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=training)
        outputs = self.out(pooled_output)

        return outputs
