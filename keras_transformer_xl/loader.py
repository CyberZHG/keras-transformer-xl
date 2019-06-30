import json
import numpy as np
import tensorflow as tf
from .backend import backend as K
from .transformer_xl import build_transformer_xl


__all__ = [
    'build_model_from_config',
    'load_model_weights_from_checkpoint',
    'load_trained_model_from_checkpoint',
]


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def build_model_from_config(config_path):
    """Build the model from config file.

    :param config_path: The path to the JSON configuration file.
    :return: model and config
    """
    if isinstance(config_path, dict):
        config = config_path
    else:
        with open(config_path, 'r') as reader:
            config = json.loads(reader.read())
    model = build_transformer_xl(
        units=config['d_model'],
        embed_dim=config['d_embed'],
        hidden_dim=config['d_inner'],
        num_token=config['vocab_size'],
        num_block=config['n_layer'],
        num_head=config['n_head'],
        dropout=config.get('dropout', 0.0),
        attention_dropout=config.get('dropatt', 0.0),
        cutoffs=config.get('cutoffs', None),
        div_val=config.get('div_val', 1),
        force_projection=config.get('proj_same_dim', None),
        bind_embeddings=True,
        bind_projections=config.get('share_proj', True),
        target_len=config['tgt_len'],
        memory_len=config['mem_len'],
        clamp_len=config.get('clamp_len', None),
        share_biases=not config.get('untie_r', False),
    )
    return model, config


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    """
    loader = checkpoint_loader(checkpoint_file)

    if config.get('div_val', 1) == 1:
        model.get_layer(name='Embed-Token').set_weights([
            loader('transformer/adaptive_embed/lookup_table'),
        ])
    else:
        embed_layer = model.get_layer(name='Embed-Token')
        weights = []
        for i in range(len(embed_layer.cutoffs) - 1):
            weights.append(loader('transformer/adaptive_embed/cutoff_{}/lookup_table'.format(i)))
            if K.int_shape(embed_layer.weights[i * 2 + 1]) != ():
                weights.append(loader('transformer/adaptive_embed/cutoff_{}/proj_W'.format(i)))
            else:
                weights.append(np.zeros(()))
        embed_layer.set_weights(weights)

    r_w_bias = loader('transformer/r_w_bias')
    r_r_bias = loader('transformer/r_r_bias')
    if config.get('untie_r', False):
        for i in range(config['n_layer']):
            model.get_layer(name='Biases-{}'.format(i + 1)).set_weights([
                r_w_bias[i].flatten(),
                r_r_bias[i].flatten(),
            ])
    else:
        model.get_layer(name='Biases').set_weights([
            r_w_bias.flatten(),
            r_r_bias.flatten(),
        ])
    for i in range(config['n_layer']):
        qkv_kernel = loader('transformer/layer_{}/rel_attn/qkv/kernel'.format(i))
        model.get_layer(name='Attention-{}'.format(i + 1)).set_weights([
            qkv_kernel[:, :config['d_model']],
            qkv_kernel[:, config['d_model']:],
            loader('transformer/layer_{}/rel_attn/o/kernel'.format(i)),
            loader('transformer/layer_{}/rel_attn/r/kernel'.format(i)),
        ])
        model.get_layer(name='Attention-Norm-{}'.format(i + 1)).set_weights([
            loader('transformer/layer_{}/rel_attn/LayerNorm/gamma'.format(i)),
            loader('transformer/layer_{}/rel_attn/LayerNorm/beta'.format(i)),
        ])
        model.get_layer(name='FeedForward-{}'.format(i + 1)).set_weights([
            loader('transformer/layer_{}/ff/layer_1/kernel'.format(i)),
            loader('transformer/layer_{}/ff/layer_1/bias'.format(i)),
            loader('transformer/layer_{}/ff/layer_2/kernel'.format(i)),
            loader('transformer/layer_{}/ff/layer_2/bias'.format(i)),
        ])
        model.get_layer(name='FeedForward-Norm-{}'.format(i + 1)).set_weights([
            loader('transformer/layer_{}/ff/LayerNorm/gamma'.format(i)),
            loader('transformer/layer_{}/ff/LayerNorm/beta'.format(i)),
        ])

    if config.get('div_val', 1) == 1:
        model.get_layer(name='Softmax').set_weights([
            loader('transformer/adaptive_softmax/bias'),
        ])
    else:
        softmax_layer = model.get_layer(name='Softmax')
        weights = [
            loader('transformer/adaptive_softmax/cutoff_0/cluster_W').transpose(),
            loader('transformer/adaptive_softmax/cutoff_0/cluster_b'),
        ]
        for i in range(len(softmax_layer.cutoffs) - 1):
            if not softmax_layer.bind_projections[i] and softmax_layer.projections[i] is not None:
                weights.append(loader('transformer/adaptive_softmax/cutoff_{}/proj'.format(i)))
            weights.append(loader('transformer/adaptive_softmax/cutoff_{}/b'.format(i)))
        softmax_layer.set_weights(weights)


def load_trained_model_from_checkpoint(config_path,
                                       checkpoint_path):
    """Load trained official model from checkpoint.

    :param config_path: The path to the JSON configuration file.
    :param checkpoint_path: The path to the checkpoint files, should end with '.ckpt'.
    :return: model
    """
    model, config = build_model_from_config(
        config_path,
    )
    load_model_weights_from_checkpoint(model, config, checkpoint_path)
    return model
