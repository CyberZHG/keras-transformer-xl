import json
import codecs
import numpy as np
import tensorflow as tf
from .backend import keras
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
        dropout_attention=config.get('dropatt', 0.0),
        cutoffs=config.get('cutoffs', None),
        div_val=config.get('div_val', 1),
        bind_embeddings=True,
        bind_projections=config.get('tie_projs', True),
        fixed_input_len=config.get('fixed_input_len', False),
        target_len=config['tgt_len'],
        memory_len=config['mem_len'],
        clamp_len=config.get('clamp_len', None),
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

    model.get_layer(name='Embed-Token').set_weights([
        loader('transformer/adaptive_embed/lookup_table'),
    ])
    model.get_layer(name='Biases').set_weights([
        loader('transformer/r_w_bias').flatten(),
        loader('transformer/r_r_bias').flatten(),
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
    model.get_layer(name='Softmax').set_weights([
        loader('transformer/adaptive_softmax/bias'),
    ])


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
