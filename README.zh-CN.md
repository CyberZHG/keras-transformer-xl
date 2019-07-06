# Keras Transformer-XL

[![Travis](https://travis-ci.org/CyberZHG/keras-transformer-xl.svg)](https://travis-ci.org/CyberZHG/keras-transformer-xl)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-transformer-xl/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-transformer-xl)
[![Version](https://img.shields.io/pypi/v/keras-transformer-xl.svg)](https://pypi.org/project/keras-transformer-xl/)
![Downloads](https://img.shields.io/pypi/dm/keras-transformer-xl.svg)
![License](https://img.shields.io/pypi/l/keras-transformer-xl.svg)

<!--![](https://img.shields.io/badge/keras-tensorflow-blue.svg)-->
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0_beta-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-transformer-xl/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-transformer-xl/blob/master/README.md)\]

[Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)的非官方实现.


## 安装

```bash
pip install keras-transformer-xl
```

## 使用

### 注意

在环境变量里加入`TF_KERAS=1`启用`tensorflow.python.keras`。

### 加载预训练权重

在[信息文件夹](./keras_transformer_xl/info)中有一些预训练模型的配置信息，可以根据实际情况进行选择：

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

### 关于输入输出

模型有两个输入，第一个是词的下标，第二个是已经记忆的长度，第二个输入要求所有数值相同。`MemorySequence`对输入进行的切割并加入了记忆长度的输入：

```python
import keras
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
)
seq = MemorySequence(
    units=4,
    model=model,
    sequence=DummySequence(),
    target_len=10,
)

model.predict(model, seq, verbose=True)
```
