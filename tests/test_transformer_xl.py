import os
from unittest import TestCase
from keras_transformer_xl.backend import keras
from keras_transformer_xl import build_transformer_xl


class TestTransformerXL(TestCase):

    def test_build(self):
        model = build_transformer_xl(
            units=16,
            embed_dim=16,
            hidden_dim=64,
            num_token=13,
            num_block=3,
            num_head=4,
            dropout=0.1,
            attention_dropout=0.1,
            cutoffs=[3],
            div_val=2,
        )
        model.summary()
        try:
            current_path = os.path.dirname(os.path.abspath(__file__))
            visual_path = os.path.join(current_path, 'test_build.jpg')
            keras.utils.vis_utils.plot_model(model, visual_path)
        except Exception as e:
            pass
