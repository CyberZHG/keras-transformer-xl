import tensorflow as tf
from .backend import keras, activations, initializers, regularizers, constraints, TF_KERAS
from .backend import backend as K

__all__ = ['RelativePartialMultiHeadSelfAttention']


class RelativePartialMultiHeadSelfAttention(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        units: int >= 0. Dimensions of all tensors.
        num_head: int >= 0. Number of heads. Should divide units.
        use_bias: Boolean. Whether to use bias term.
        attention_dropout: 0.0 < float < 1.0. Dropout rate for attention weights.

    # Input shape
        First 3D tensor with shape: `(batch_size, sequence_length, units)`.
        Second 3D tensor with shape: `(batch_size, previous_sequence_length + sequence_length, units)`.
        Third 3D tensor with shape: `(batch_size, previous_sequence_length, units)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self,
                 units,
                 num_head,
                 activation=None,
                 use_bias=False,
                 attention_dropout=0.0,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(RelativePartialMultiHeadSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.num_head = num_head
        self.units_head = units // num_head
        self.activation = activation
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel_q, self.kernel_kv, self.kernel_o, self.kernel_r = (None,) * 4
        self.bias_q, self.bias_kv, self.bias_o, self.bias_r = (None,) * 4
        self.att_drop_layer = None

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None

    def build(self, input_shape):
        self.kernel_q = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_q',
        )
        if self.use_bias:
            self.bias_q = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_q',
            )

        self.kernel_kv = self.add_weight(
            shape=(self.units, self.units * 2),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_kv',
        )
        if self.use_bias:
            self.bias_kv = self.add_weight(
                shape=(self.units * 2,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_kv',
            )

        self.kernel_o = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_o',
        )
        if self.use_bias:
            self.bias_o = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_o',
            )

        self.kernel_r = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_r',
        )
        if self.use_bias:
            self.bias_r = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_r',
            )
        if 0.0 < self.attention_dropout < 1.0:
            self.att_drop_layer = keras.layers.Dropout(self.attention_dropout)
        super(RelativePartialMultiHeadSelfAttention, self).build(input_shape)

    def _reshape_to_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size, seq_len, self.num_head, self.units_head))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * self.num_head, seq_len, self.units_head))

    def _reshape_from_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // self.num_head, self.num_head, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // self.num_head, seq_len, feature_dim * self.num_head))

    def _reshape_mask(self, mask):
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, self.num_head, 1])
        return K.reshape(mask, (-1, seq_len))

    @staticmethod
    def _relative_shift(x):
        batch_size, q_len, k_len = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2]
        x = tf.pad(x, [[0, 0], [0, 0], [1, 0]])               # (batch * n_head, seq_len, prev_len + seq_len + 1)
        x = K.reshape(x, (batch_size, k_len + 1, q_len))      # (batch * n_head, prev_len + seq_len + 1, seq_len)
        x = x[:, 1:, :]                                       # (batch * n_head, prev_len + seq_len, seq_len)
        return K.reshape(x, (batch_size, q_len, k_len))       # (batch * n_head, seq_len, prev_len + seq_len)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, mask=None, training=None):
        inputs, relatives, memories, bias_context, bias_relative = inputs
        full = K.concatenate([memories, inputs], axis=1)      # (batch, prev_len + seq_len, units)
        w_q = K.dot(inputs, self.kernel_q)                    # (batch, seq_len, units)
        w_kv = K.dot(full, self.kernel_kv)                    # (batch, prev_len + seq_len, units * 2)
        w_r = K.dot(relatives, self.kernel_r)                 # (batch, prev_len + seq_len, units)
        if self.use_bias:
            w_q = K.bias_add(w_q, self.bias_q)
            w_kv = K.bias_add(w_kv, self.bias_kv)
            w_r = K.bias_add(w_r, self.bias_r)
        if self.activation is not None:
            w_q = self.activation(w_q)
            w_kv = self.activation(w_kv)
            w_r = self.activation(w_r)

        w_k = w_kv[:, :, :self.units]                         # (batch, prev_len + seq_len, units)
        w_v = w_kv[:, :, self.units:]                         # (batch, prev_len + seq_len, units)

        w_qc = K.bias_add(w_q, bias_context)
        w_qc = self._reshape_to_batches(w_qc)                 # (batch * n_head, seq_len, units_head)
        w_k = self._reshape_to_batches(w_k)                   # (batch * n_head, prev_len + seq_len, units_head)
        a_context = K.batch_dot(w_qc, w_k, axes=2)            # (batch * n_head, seq_len, prev_len + seq_len)

        w_qr = K.bias_add(w_q, bias_relative)
        w_qr = self._reshape_to_batches(w_qr)                 # (batch * n_head, seq_len, units_head)
        w_r = self._reshape_to_batches(w_r)                   # (batch * n_head, prev_len + seq_len, units_head)
        a_relative = K.batch_dot(w_qr, w_r, axes=2)           # (batch * n_head, seq_len, prev_len + seq_len)
        a_relative = self._relative_shift(a_relative)         # (batch * n_head, seq_len, prev_len + seq_len)

        att = (a_context + a_relative) / K.sqrt(K.constant(self.units_head, dtype=K.floatx()))
        exp = K.exp(att - K.max(att, axis=-1, keepdims=True))

        q_len, k_len = K.shape(w_q)[1], K.shape(w_k)[1]
        indices = K.expand_dims(K.arange(0, k_len), axis=0)
        upper = K.expand_dims(K.arange(k_len - q_len, k_len), axis=-1)
        exp *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None and mask[0] is not None:
            mask = K.cast(mask[0], K.floatx())
            mask = K.concatenate([K.ones_like(memories[:, :, 0]), mask], axis=1)
            exp *= K.expand_dims(self._reshape_mask(mask), axis=1)

        att = exp / K.sum(exp, axis=-1, keepdims=True)
        if self.att_drop_layer is not None:
            att = self.att_drop_layer(att, training=training)
        w_v = self._reshape_to_batches(w_v)                   # (batch * n_head, prev_len + seq_len, units_head)
        w_o = K.batch_dot(att, w_v)                           # (batch * n_head, seq_len, units_head)

        w_o = self._reshape_from_batches(w_o)                 # (batch, seq_len, units)
        w_o = K.dot(w_o, self.kernel_o)                       # (batch, seq_len, units)
        if self.use_bias:
            w_o = K.bias_add(w_o, self.bias_o)
        if self.activation is not None:
            w_o = self.activation(w_o)

        if TF_KERAS:
            # Add shape information to tensor when using `tf.keras`
            input_shape = K.int_shape(inputs)
            if input_shape[1] is not None:
                w_o = K.reshape(w_o, (-1,) + input_shape[1:])
        return w_o

    def get_config(self):
        config = {
            'units': self.units,
            'num_head': self.num_head,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(RelativePartialMultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
