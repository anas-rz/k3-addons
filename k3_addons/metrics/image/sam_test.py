import pytest
import keras
from keras import ops
import numpy as np
import torch

from k3_addons.metrics.image.sam import (
    spectral_angle_mapper as spectral_angle_mapper_keras,
)
from torchmetrics.functional.image.sam import (
    spectral_angle_mapper as spectral_angle_mapper_torch,
)


# parametrize the test
@pytest.mark.parametrize(
    "input_shape, reduction, data_format",
    [
        ((4, 3, 32, 32), "sum", "channels_first"),
        ((4, 3, 32, 32), "elementwise_mean", "channels_first"),
        ((4, 32, 32, 3), "none", "channels_first"),
        ((4, 32, 32, 3), "sum", "channels_last"),
        ((4, 32, 32, 3), "elementwise_mean", "channels_last"),
        ((4, 32, 32, 3), "none", "channels_last"),
    ],
)
def test_total_variation(input_shape, reduction, data_format):
    inputs = keras.random.uniform(input_shape)
    labels = keras.random.uniform(input_shape)
    tv_keras = spectral_angle_mapper_keras(
        inputs, labels, data_format=data_format, reduction=reduction
    )
    if data_format == "channels_last":
        inputs = ops.transpose(inputs, (0, 3, 1, 2))
        labels = ops.transpose(labels, (0, 3, 1, 2))
    inputs = torch.tensor(ops.convert_to_numpy(inputs))
    labels = torch.tensor(ops.convert_to_numpy(labels))
    tv_torch = spectral_angle_mapper_torch(inputs, labels, reduction=reduction).numpy()

    assert np.allclose(tv_keras, tv_torch, atol=1e-4)
