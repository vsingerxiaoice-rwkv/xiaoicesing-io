# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## Configuration Schemas

This document explains the meaning and usages of all editable keys in a configuration file.

### Common configurations

#### base_config

Path(s) of other config files that the current config is based on and will override.

##### used by

all

##### type

str, List[str]

##### default

_none_

##### Constraints

_none_

### Neural networks

#### hidden_size

Dimension of hidden layers of FastSpeech2, token and variance embeddings, and diffusion condition.

##### used by

acoustic model

##### type

int

##### default

_256_

##### Constraints

__none__

#### residual_channels

TBD

#### residual_layers

TBD

#### diff_decoder_type

Denoiser type of the DDPM.

##### used by

acoustic model

##### type

str

##### default

_wavenet_

##### Constraints

choose from [ _wavenet_ ]

#### diff_loss_type

Loss type of the DDPM.

##### used by

acoustic model

##### type

str

##### default

_l2_

##### Constraints

choose from [ _l1_, _l2_ ]

### Dataset information and preprocessing

#### raw_data_dir

Path(s) to the raw data including wave files, transcriptions, etc.

##### used by

all

##### type

str, List[str]

##### default

_none_

##### Constraints

_none_

### Training, validation and inference

#### task_cls

TBD

#### lr

Initial learning rate of the scheduler.

##### used by

all

##### type

float

##### default

_0.0004_

##### Constraints

_none_

#### max_batch_frames

TBD

#### max_batch_size

TBD

