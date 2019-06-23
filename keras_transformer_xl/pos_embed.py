from .backend import keras
from .backend import backend as K

__all__ = ['PositionalEmbedding']


class PositionalEmbedding(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        output_dim: int >= 0. Dimension of the embedding. Should be even.

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self, output_dim, clamp_len=None, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.output_dim = output_dim
        self.clamp_len = clamp_len

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, **kwargs):
        if self.clamp_len is not None:
            inputs = K.clip(inputs, min_value=0, max_value=self.clamp_len)
        inputs = K.expand_dims(inputs, axis=-1)
        output_dim = K.cast(self.output_dim, K.floatx())
        ranges = K.expand_dims(K.arange(0.0, self.output_dim, 2.0), axis=0) / output_dim
        inverse = 1.0 / K.pow(10000.0, ranges)
        positions = inputs * inverse
        return K.concatenate([K.sin(positions), K.cos(positions)], axis=-1)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'clamp_len': self.clamp_len,
        }
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
