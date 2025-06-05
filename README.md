# creme-rl

`creme-rl` is an experimental reinforcement learning toolkit focused on the optimisation of genomic sequences. It contains an AlphaZero‑style agent built with PyTorch and a collection of helper utilities for working with sequence data.

## Features

- **SeqGame environment** (`cremerl.env.SeqGame`)
  - models the sequential editing of DNA sequences using a sliding window.
- **AlphaDNA agent** (`cremerl.alphadna`)
  - implements Monte Carlo Tree Search (MCTS) and policy/value networks.
  - includes a parallel variant for faster rollouts.
- **Model zoo** (`cremerl.model_zoo`)
  - simple convolutional architectures used for policy/value prediction.
- **Surrogate models** (`cremerl.surrogate_model_zoo`)
  - PyTorch Lightning wrappers (e.g. DeepSTARR) for supervising the agent.
- **Utility helpers** (`cremerl.utils` and `cremerl.shuffle`)
  - sequence shuffling, batch generation and scoring functions.

Example notebooks demonstrating different training strategies are provided in
`notebooks/` along with a few pretrained checkpoints.

## Installation

`creme-rl` requires Python 3.8 or later. Install the minimal dependencies and the
package itself in editable mode:

```bash
pip install numpy torch pytorch-lightning
pip install -e .
```

The `data/` directory contains a `download.sh` script that retrieves the
DeepSTARR dataset and other helper files used in the notebooks.

## Basic usage

Below is a minimal example for creating an environment and agent.

```python
import numpy as np
from cremerl.env import SeqGame
from cremerl.model_zoo import CNN_v0
from cremerl.alphadna import AlphaDNA
import torch.optim as optim

sequence = np.zeros((4, 250), dtype=np.float32)  # one‑hot encoded DNA
model = CNN_v0(action_dim=50)
optimizer = optim.Adam(model.parameters())

env = SeqGame(sequence, model_func=model, num_trials=10)
args = {
    "batch_size": 16,
    "num_iterations": 1,
    "num_selfPlay_iterations": 4,
    "num_epochs": 1,
    "rlop_patience": 5,
    "rlop_factor": 0.5,
}
agent = AlphaDNA(model, optimizer, env, args)
```

Refer to the notebooks for more detailed training pipelines and evaluation
procedures.

## Testing

Run the unit tests with `pytest`:

```bash
pip install numpy
pytest
```

Only a handful of lightweight tests are included (`tests/` and
`notebooks/shuffle_test.py`).

## License

This repository does not currently include an explicit license file. Treat it as
research code and use at your own discretion.
