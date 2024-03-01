import keras
import pytest
import torch
import numpy as np
from keras import ops
from k3_addons.metrics.audio.stoi import (
    short_time_objective_intelligibility as stoi_keras,
)
from torchmetrics.functional.audio.stoi import (
    short_time_objective_intelligibility as stoi_torch,
)


@pytest.mark.parametrize(
    "input_shape, fs, extended",
    [
        ((8000,), 16000, False),
        ((8000,), 16000, True),
        ((8000,), 16000, False),
        ((8000,), 16000, True),
    ],
)
def test_stoi(input_shape, fs, extended):
    inputs = keras.random.uniform(input_shape)
    target = keras.random.uniform(input_shape)
    stoi_keras_val = stoi_keras(inputs, target, fs, extended)
    inputs = torch.tensor(ops.convert_to_numpy(inputs))
    target = torch.tensor(ops.convert_to_numpy(target))
    stoi_torch_val = stoi_torch(inputs, target, fs, extended).numpy()

    assert np.allclose(stoi_keras_val, stoi_torch_val, atol=1e-4)
