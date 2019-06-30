import numpy as np
from .backend import keras
from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward
from .scale import Scale
from .pos_embed import PositionalEmbedding
from .rel_bias import RelativeBias
from .rel_multi_head import RelativePartialMultiHeadSelfAttention
from .identity import Identity


__all__ = [
    'get_custom_objects',
    'set_custom_objects',
    'build_transformer_xl',
]


def get_custom_objects():
    return {
        'AdaptiveEmbedding': AdaptiveEmbedding,
        'AdaptiveSoftmax': AdaptiveSoftmax,
        'Scale': Scale,
        'LayerNormalization': LayerNormalization,
        'FeedForward': FeedForward,
        'PositionalEmbedding': PositionalEmbedding,
        'RelativeBias': RelativeBias,
        'RelativePartialMultiHeadSelfAttention': RelativePartialMultiHeadSelfAttention,
        'Identity': Identity,
    }


def set_custom_objects():
    for name, layer in get_custom_objects().items():
        keras.utils.get_custom_objects()[name] = layer


def build_transformer_xl(units,
                         embed_dim,
                         hidden_dim,
                         num_token,
                         num_block,
                         num_head,
                         dropout=0.0,
                         attention_dropout=0.0,
                         cutoffs=None,
                         div_val=1,
                         force_projection=None,
                         bind_embeddings=True,
                         bind_projections=True,
                         target_len=None,
                         memory_len=None,
                         clamp_len=None,
                         share_biases=True):
    """Build transformer-XL model.

    :param units: Units inside the transformer.
    :param embed_dim: Dimension of embeddings.
    :param hidden_dim: Dimension inside position-wise feed-forward layer.
    :param num_token: Number of distinct input tokens.
    :param num_block: Number of basic encoder blocks.
    :param num_head: Number of heads for attention.
    :param dropout: General dropout rate.
    :param attention_dropout: Dropout rate inside attention layer.
    :param cutoffs: Cutoffs of adaptive embedding.
    :param div_val: Scale factor of adaptive embedding.
    :param force_projection: Add projection when the dimensions are equal.
    :param bind_embeddings: Whether to bind embeddings to adaptive softmax.
    :param bind_projections: Whether to bind projections to adaptive softmax.
    :param target_len: The length of prediction block.
    :param memory_len: The maximum length of memories.
    :param clamp_len: The maximum value of relative position.
    :param share_biases: Whether to use the same biases for all layers.
    :return: The built model.
    """
    token_input = keras.layers.Input(shape=(target_len,), name='Input-Token')
    memories = []
    for i in range(num_block):
        memory_input = keras.layers.Input(shape=(None, units), name='Input-Memory-{}'.format(i + 1))
        memories.append(memory_input)
    inputs = [token_input] + memories

    results = AdaptiveEmbedding(
        input_dim=num_token,
        output_dim=units,
        embed_dim=embed_dim,
        cutoffs=cutoffs,
        div_val=div_val,
        mask_zero=True,
        force_projection=force_projection,
        return_embeddings=True,
        return_projections=True,
        name='Embed-Token',
    )(token_input)
    token_embed, embedding_weights = results[0], results[1:]
    token_embed = Scale(scale=np.sqrt(units), name='Embed-Token-Scaled')(token_embed)

    position_embed = PositionalEmbedding(
        output_dim=units,
        clamp_len=clamp_len,
        name='Embed-Position',
    )([token_input, memories[0]])

    if 0.0 < dropout < 1.0:
        token_embed = keras.layers.Dropout(rate=dropout, name='Embed-Token-Dropped')(token_embed)
        position_embed = keras.layers.Dropout(rate=dropout, name='Embed-Position-Dropped')(position_embed)

    if share_biases:
        context_biases, relative_biases = RelativeBias(units=units, name='Biases')(token_input)
    else:
        context_biases, relative_biases = [], []
        for i in range(num_block):
            context_bias, relative_bias = RelativeBias(units=units, name='Biases-{}'.format(i + 1))(token_input)
            context_biases.append(context_bias)
            relative_biases.append(relative_bias)

    outputs = [token_embed]
    for i in range(num_block):
        block_input, block_output = outputs[-1], outputs[-1]
        if share_biases:
            context_bias, relative_bias = context_biases, relative_biases
        else:
            context_bias, relative_bias = context_biases[i], relative_biases[i]
        block_output = RelativePartialMultiHeadSelfAttention(
            units=units,
            num_head=num_head,
            use_bias=False,
            attention_dropout=attention_dropout,
            name='Attention-{}'.format(i + 1),
        )([block_output, position_embed, memories[i], context_bias, relative_bias])
        block_output = keras.layers.Add(name='Attention-Res-{}'.format(i + 1))([block_input, block_output])
        if 0.0 < dropout < 1.0:
            block_output = keras.layers.Dropout(rate=dropout, name='Attention-Dropped-{}'.format(i + 1))(block_output)
        block_output = LayerNormalization(name='Attention-Norm-{}'.format(i + 1))(block_output)

        block_input = block_output
        block_output = FeedForward(
            units=hidden_dim,
            dropout_rate=dropout,
            name='FeedForward-{}'.format(i + 1),
        )(block_output)
        block_output = keras.layers.Add(name='FeedForward-Res-{}'.format(i + 1))([block_input, block_output])
        if 0.0 < dropout < 1.0:
            block_output = keras.layers.Dropout(rate=dropout, name='FeedForward-Dropped-{}'.format(i + 1))(block_output)
        block_output = LayerNormalization(name='FeedForward-Norm-{}'.format(i + 1))(block_output)

        outputs.append(block_output)

    softmax = AdaptiveSoftmax(
        input_dim=units,
        output_dim=num_token,
        embed_dim=embed_dim,
        cutoffs=cutoffs,
        div_val=div_val,
        force_projection=force_projection,
        bind_embeddings=bind_embeddings,
        bind_projections=bind_projections,
        name='Softmax',
    )(outputs[-1:] + embedding_weights)
    outputs = outputs[:-1]
    for i, output in enumerate(outputs):
        outputs[i] = Identity(name='Output-Memory-{}'.format(i + 1))(output)
    outputs = [softmax] + outputs

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model
