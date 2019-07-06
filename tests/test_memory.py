from unittest import TestCase
import numpy as np
from keras_transformer_xl.backend import keras
from keras_transformer_xl.backend import backend as K
from keras_transformer_xl import Memory


class TestMemory(TestCase):

    def test_sample(self):
        input_data = keras.layers.Input(shape=(3, 3))
        input_length = keras.layers.Input(shape=())
        memory_layer = Memory(3, 5, 3)
        output = memory_layer([input_data, input_length])
        func = K.function(
            inputs=[input_data, input_length],
            outputs=[output],
            updates=memory_layer.updates,
        )

        data_1 = np.array([
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
        ])
        length = np.array([0, 0])
        output = func([data_1, length])[0]
        self.assertEqual(data_1.tolist(), output.tolist())

        data_2 = np.array([
            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
        ])
        length = np.array([3, 3])
        output = func([data_2, length])[0]
        self.assertEqual(np.concatenate([data_1, data_2], axis=1).tolist(), output.tolist())

        data_3 = np.array([
            [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
            [[45, 46, 47], [48, 49, 50], [51, 52, 53]],
        ])
        length = np.array([6, 6])
        output = func([data_3, length])[0]
        self.assertEqual(np.concatenate([data_1, data_2, data_3], axis=1)[:, -8:, :].tolist(), output.tolist())
