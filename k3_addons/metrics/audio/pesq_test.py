import keras
import pytest
import torch
import numpy as np
from keras import ops
from k3_addons.metrics.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq_keras,
)
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq_torch,
)


@pytest.mark.parametrize(
    "input_shape, fs, mode",
    [
        ((8000,), 16000, "wb"),
        ((8000,), 16000, "nb"),
    ],
)
def test_stoi(input_shape, fs, mode):
    inputs = keras.random.uniform(input_shape)
    target = keras.random.uniform(input_shape)
    stoi_keras_val = pesq_keras(inputs, target, fs, mode)
    inputs = torch.tensor(ops.convert_to_numpy(inputs))
    target = torch.tensor(ops.convert_to_numpy(target))
    stoi_torch_val = pesq_torch(inputs, target, fs, mode).numpy()

    assert np.allclose(stoi_keras_val, stoi_torch_val, atol=1e-4)
