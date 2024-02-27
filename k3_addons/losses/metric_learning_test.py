import keras
from keras import ops
import numpy as np
from k3_addons.losses.metric_learning import pairwise_distance


def test_zero_distance():
    equal_embeddings = ops.convert_to_tensor([[1.0, 0.5], [1.0, 0.5]])

    distances = pairwise_distance(equal_embeddings, squared=False)
    np.allclose(ops.convert_to_numpy(ops.sum(distances)), 0, 1e-6, 1e-6)


def test_positive_distances():
    embeddings = 1.0 + 2e-7 * keras.random.uniform([64, 6], dtype="float32")
    distances = pairwise_distance(embeddings, squared=False)
    assert np.all(ops.convert_to_numpy(distances) >= 0)


def test_correct_distance():
    k_embeddings = ops.convert_to_tensor([[0.5, 0.5], [1.0, 1.0]])

    expected_distance = np.array([[0, np.sqrt(2) / 2], [np.sqrt(2) / 2, 0]])

    distances = pairwise_distance(k_embeddings, squared=False)
    np.allclose(ops.convert_to_numpy(expected_distance), distances, 1e-6, 1e-6)


def test_correct_distance_squared():
    tf_embeddings = ops.convert_to_tensor([[0.5, 0.5], [1.0, 1.0]])

    expected_distance = np.array([[0, 0.5], [0.5, 0]])

    distances = pairwise_distance(tf_embeddings, squared=True)
    np.testing.assert_allclose(
        ops.convert_to_numpy(expected_distance), distances, 1e-6, 1e-6
    )
