import os
import tempfile
from unittest import TestCase

import numpy as np
from tensorflow import keras

from keras_transformer_xl import build_transformer_xl, get_custom_objects


class TestTransformerXL(TestCase):

    def test_build(self):
        model = build_transformer_xl(
            units=6,
            embed_dim=16,
            hidden_dim=12,
            num_token=13,
            num_block=3,
            num_head=2,
            batch_size=3,
            memory_len=15,
            target_len=5,
            dropout=0.1,
            attention_dropout=0.1,
            cutoffs=[3],
            div_val=2,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_transformer_xl_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
        try:
            current_path = os.path.dirname(os.path.abspath(__file__))
            visual_path = os.path.join(current_path, 'test_build.jpg')
            keras.utils.vis_utils.plot_model(model, visual_path)
        except Exception as e:
            pass

    def test_fit_batch_changes(self):
        model = build_transformer_xl(
            units=4,
            embed_dim=4,
            hidden_dim=4,
            num_token=2,
            num_block=1,
            num_head=1,
            batch_size=2,
            memory_len=0,
            target_len=5,
        )
        model.compile('adam', 'mse')
        model.summary()
        model.train_on_batch([np.ones((2, 5)), np.zeros((2, 1))], np.zeros((2, 5, 2)))
        model.train_on_batch([np.ones((1, 5)), np.zeros((1, 1))], np.zeros((1, 5, 2)))
        model.train_on_batch([np.ones((2, 5)), np.zeros((2, 1))], np.zeros((2, 5, 2)))
        model.train_on_batch([np.ones((1, 5)), np.zeros((1, 1))], np.zeros((1, 5, 2)))
