# Keras Transformer-XL

[![Version](https://img.shields.io/pypi/v/keras-transformer-xl.svg)](https://pypi.org/project/keras-transformer-xl/)
![License](https://img.shields.io/pypi/l/keras-transformer-xl.svg)

\[[中文](https://github.com/CyberZHG/keras-transformer-xl/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-transformer-xl/blob/master/README.md)\]

Unofficial implementation of [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf).

## Install

```bash
pip install keras-transformer-xl
```

## Usage

### Load Pretrained Weights

Several configuration files can be found at [the info directory](./keras_transformer_xl/info).

```python
import os
from keras_transformer_xl import load_trained_model_from_checkpoint

checkpoint_path = 'foo/bar/sota/enwiki8'
model = load_trained_model_from_checkpoint(
    config_path=os.path.join(checkpoint_path, 'config.json'),
    checkpoint_path=os.path.join(checkpoint_path, 'model.ckpt')
)
model.summary()
```

### About IO

The generated model has two inputs, and the second input is the lengths of memories.

You can use `MemorySequence` wrapper for training and prediction:

```python
from tensorflow import keras
import numpy as np
from keras_transformer_xl import MemorySequence, build_transformer_xl


class DummySequence(keras.utils.Sequence):

    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return np.ones((3, 5 * (index + 1))), np.ones((3, 5 * (index + 1), 3))


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
    model=model,
    sequence=DummySequence(),
    target_len=10,
)

model.predict(model, seq, verbose=True)
```
