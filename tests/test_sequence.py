from unittest import TestCase
import numpy as np
from keras_transformer_xl.backend import keras, TF_KERAS
from keras_transformer_xl import MemorySequence, build_transformer_xl, fit_generator, predict_generator


class DummySequence(keras.utils.Sequence):

    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return np.ones((3, 5 * (index + 1))), np.ones((3, 5 * (index + 1), 3))


class TestSequence(TestCase):

    def test_dummy(self):
        if TF_KERAS:
            return
        model = build_transformer_xl(
            units=4,
            embed_dim=4,
            hidden_dim=4,
            num_token=3,
            num_block=3,
            num_head=2,
        )
        seq = MemorySequence(
            units=4,
            model=model,
            sequence=DummySequence(),
            target_len=10,
            memory_len=20,
        )
        first_batch = seq[2]
        outputs = model.predict_on_batch(first_batch[0])
        seq.update_memories(outputs)
        second_batch = seq[3]
        outputs = model.predict_on_batch(second_batch[0])
        seq.update_memories(outputs)
        seq.update_memories(outputs)
        seq.update_memories(outputs)
        seq.update_memories(outputs)
        second_batch = seq[3]
        for i in range(1, 4):
            self.assertEqual((3, 20, 4), second_batch[0][i].shape)
        model.compile(optimizer='adam', loss={'Softmax': 'mse'})
        fit_generator(model, seq, epochs=2, validation_data=seq)
        predict_generator(model, seq, verbose=True)
