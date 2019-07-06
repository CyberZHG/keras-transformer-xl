from unittest import TestCase
import numpy as np
from keras_transformer_xl.backend import keras
from keras_transformer_xl import MemorySequence, build_transformer_xl


class DummySequence(keras.utils.Sequence):

    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return np.ones((3, 5 * (index + 1))), np.ones((3, 5 * (index + 1), 3))


class TestSequence(TestCase):

    def test_dummy(self):
        model = build_transformer_xl(
            units=4,
            embed_dim=4,
            hidden_dim=4,
            num_token=3,
            num_block=3,
            num_head=2,
            batch_size=3,
            memory_len=20,
            target_len=10,
        )
        seq = MemorySequence(
            units=4,
            model=model,
            sequence=DummySequence(),
            target_len=10,
        )
        for i in range(len(seq)):
            self.assertEqual((3, 10), seq[0][0][0].shape)
            self.assertEqual((3, 10, 3), seq[0][1].shape)

        self.assertEqual([0, 0, 0], seq[0][0][1].tolist())

        self.assertEqual([0, 0, 0], seq[1][0][1].tolist())

        self.assertEqual([0, 0, 0], seq[2][0][1].tolist())
        self.assertEqual([10, 10, 10], seq[3][0][1].tolist())

        self.assertEqual([0, 0, 0], seq[4][0][1].tolist())
        self.assertEqual([10, 10, 10], seq[5][0][1].tolist())

        self.assertEqual([0, 0, 0], seq[6][0][1].tolist())
        self.assertEqual([10, 10, 10], seq[7][0][1].tolist())
        self.assertEqual([20, 20, 20], seq[8][0][1].tolist())

        self.assertEqual([0, 0, 0], seq[9][0][1].tolist())

        model.predict_generator(seq)
