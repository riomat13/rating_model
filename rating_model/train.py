#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import Optional

import tensorflow as tf

from rating_model.transformers import AlbertConfig

from rating_model.base import BaseTrainer, Dataset
from rating_model.models import AlbertRatingModel, get_tokenizer
from rating_model.settings import Config, get_logger

logger = get_logger(__name__)


@dataclass
class ModelState:
    # TODO: add information to save
    name: str
    loss: float = float('inf')
    learning_rate: float = 3e-4

    def save(self):
        data = json.dumps({
            'name': self.name,
            'loss': self.loss,
            'learning_rate': self.learning_rate
        })

        with open(os.path.join(Config.MODEL_DIR, 'model-info.json'), 'w') as f:
            f.write(data)

    def set_loss(self, loss):
        self.loss = loss
        self.save()
        logger.info('Saved model info')

    def update_learning_rate(self, ratio: float):
        self.learning_rate *= ratio

    def is_updated_loss(self, loss) -> bool:
        if self.loss > loss:
            print(f'Updated: Loss {self.loss} -> {loss}')
            self.set_loss(loss)
            return True
        return False

    @classmethod
    def from_jsonfile(cls, filepath):

        with open(filepath, 'r') as f:
            data = json.load(f)

        try:
            obj = cls(name=data['name'], loss=data['loss'])
        except KeyError as e:
            logger.error('Provided file does not have necessary items')
            raise

        return obj


class Trainer(BaseTrainer):

    def __init__(self, batch_size: int, from_pretrained: bool = False):
        self.model = None
        self.model_dirpath = os.path.join(Config.MODEL_DIR, 'best')

        if from_pretrained:
            try:
                self.model = tf.keras.models.load_model(self.model_dirpath)
                self.state = ModelState.from_jsonfile(
                    os.path.join(Config.MODEL_DIR, 'model-info.json')
                )
            except (OSError, FileNotFoundError):
                logger.warning('Model data not found. Use default value.')

        if self.model is None:
            # TODO: move configuration to settings
            config = AlbertConfig(
                vocab_size=30000,
                embedding_size=128,
                hidden_size=512,
                num_hidden_layers=1,
                num_attention_heads=8,
                intermediate_size=1024,
                inner_group_num=1,
                hidden_act='gelu',
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
                max_position_embeddings=512,
                type_vocab_size=2,
                initilizer_range=0.02,
                layer_norm_eps=1e-12,
                classifier_dropout_prob=0.1
            )
            self.model = AlbertRatingModel(config)
            self.state = ModelState(name='albert-model')

        self.batch_size = batch_size

        self.opt_func = tf.keras.optimizers.Adam
        self.optimizer = self.opt_func(learning_rate=self.state.learning_rate)

        self.tokenizer = get_tokenizer()

    @tf.function
    def _train(self, inputs, labels):
        """training step."""

        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            loss = tf.keras.losses.mean_squared_error(labels, outputs)

        trainables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainables)
        self.optimizer.apply_gradients(zip(gradients, trainables))

        return loss, outputs

    def train(self,
              train_data: Dataset,
              epochs: int,
              iter_per_epoch: int,
              val_data: Dataset = None,
              verbose_step: int = 100):
        """Main training process.

        Args:
            epochs: int
                total number of epoch to train
            iter_per_epoch: int
                number of iterations to train per epoch
            verbose_step: int (default: 100)
                print each `verbose_step` step in each epoch
        """
        print_format = '{header}: Loss={loss:.4f} - Time={time:2f}s'

        best_loss = float('inf')

        dttm = datetime.now().strftime('%Y%m%d%H00')
        curr_model_dirpath = os.path.join(Config.MODEL_DIR, dttm)

        no_updated_cnt = 0

        for epoch in range(1, epochs + 1):
            print(f'Epoch: {epoch:>3d}')
            start_epoch = time.perf_counter()

            total_loss = 0
            size = 0

            for it in range(1, iter_per_epoch + 1):
                st = time.perf_counter()

                batch = train_data.generate()

                loss, outputs = self._train(batch['token'], batch['rating'])
                total_loss += tf.reduce_sum(loss)
                size += outputs.shape[0]
                
                end = time.perf_counter()

                if it % verbose_step == 0:
                    curr_loss = total_loss / size
                    print(print_format.format(header=f'Iteration: {it}', loss=curr_loss, time=end-st))
            
            end_epoch = time.perf_counter()
            if not size:
                logger.warning('Training dataset size is too small. Skip the training')
                return

            print(print_format.format(
                header='Train(epoch)',
                loss=total_loss/size,
                time=end_epoch-start_epoch))

            # check validation score and save the model if get best score
            if val_data is not None:
                st_val = time.perf_counter()

                count = 0
                val_loss = 0

                for data in val_data.generate_all():
                    count += data['rating'].shape[0]
                    val_out = self.model(data['token'], training=False)
                    val_loss += tf.reduce_sum(tf.keras.losses.mean_squared_error(data['rating'], val_out))

                val_loss_average = float(val_loss / count)
                
                # TODO: design learning_rate decay pattern
                if self.state.is_updated_loss(val_loss_average):
                    # global best including previous runs
                    self.model.save(self.model_dirpath)
                    self.model.save(curr_model_dirpath)
                    best_loss = val_loss_average
                    no_updated_cnt = 0
                elif best_loss < val_loss_average:
                    # best score in current run
                    tf.saved_model.save(self.model, curr_model_dirpath)
                    best_loss = val_loss_average
                    no_updated_cnt = 0
                elif self.state.learning_rate > 1e-7:
                    no_updated_cnt += 1
                    if no_updated_cnt == 3:
                        no_updated_cnt = 0
                        # update optimizer with lower learning rate
                        self.state.update_learning_rate(0.7)
                        self.optimizer = self.opt_func(learning_rate=self.state.learning_rate)

                end_val = time.perf_counter()

                print(print_format.format(header='Validation', loss=float(val_loss_average), time=end_val-st_val))

            end_epoch = time.perf_counter()
            print(f'Time elapsed(epoch): {end_epoch-start_epoch:.2f} sec.')


def main():  # pragma: no cover
    trainer = Trainer(batch_size=16)
    train_dataset = ReviewDataset(
        cls='train',
        path=Config.DATA_DIR,
        batch_size=16,
        shiffle=True,
        repeat=False,
    )

    val_dataset = ReviewDataset(
        cls='test',
        path=Config.DATA_DIR,
        batch_size=16,
    )

    trainer.train(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=Config.ML_MODEL['EPOCH'],
        iter_per_epoch=Config.ML_MODEL['ITER_PER_EPOCH'],
    )
