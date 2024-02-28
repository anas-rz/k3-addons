import pytest
import keras
from keras import ops
import numpy as np
import torch

from k3_addons.metrics.image.tv import total_variation as total_variation_keras
from torchmetrics.functional.image.tv import total_variation as total_variation_torch
from torchmetrics.image.tv import TotalVariation as TotalVariationTorch

# parametrize the test
@pytest.mark.parametrize(
    "input_shape, reduction, data_format",
    [
        ((4, 3, 32, 32), "sum", "channels_first"),
        ((4, 32, 32, 3), "mean", "channels_last"),
    ],
)
def test_total_variation(input_shape, reduction, data_format):
    inputs = keras.random.uniform(input_shape)
    tv_keras = total_variation_keras(inputs, data_format=data_format, reduction=reduction)
    if data_format == "channels_last":
        inputs = ops.transpose(inputs, (0, 3, 1, 2))
    inputs = torch.tensor(ops.convert_to_numpy(inputs))
    tv_torch = total_variation_torch(inputs, reduction=reduction).numpy()

    assert np.allclose(tv_keras, tv_torch, atol=1e-5)
