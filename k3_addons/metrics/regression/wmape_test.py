import pytest
import keras
from keras import ops
import numpy as np
import torch

from k3_addons.metrics.regression.wmape import (
    weighted_mean_absolute_percentage_error as weighted_mean_absolute_percentage_error_keras,
)
from torchmetrics.functional.regression.wmape import (
    weighted_mean_absolute_percentage_error as weighted_mean_absolute_percentage_error_torch,
)


# parametrize the test
@pytest.mark.parametrize(
    "input_shape",
    [
        ((32,)),
        ((32,)),
        ((40,)),
        ((51,)),
    ],
)
def test_total_variation(input_shape):
    inputs = keras.random.uniform(input_shape)
    target = keras.random.uniform(input_shape)
    tv_keras = weighted_mean_absolute_percentage_error_keras(inputs, target)
    inputs = torch.tensor(ops.convert_to_numpy(inputs))
    target = torch.tensor(ops.convert_to_numpy(target))
    tv_torch = weighted_mean_absolute_percentage_error_torch(inputs, target).numpy()

    assert np.allclose(tv_keras, tv_torch, atol=1e-4)
