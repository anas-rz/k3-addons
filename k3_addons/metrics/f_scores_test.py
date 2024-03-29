# ported from tensorflow_addons

import pytest

from k3_addons.metrics.f_scores import FBetaScore, F1Score
from k3_addons.metrics.utils import _get_model
from keras import ops, backend

import numpy as np


def test_config_fbeta():
    fbeta_obj = FBetaScore(num_classes=3, beta=0.5, threshold=0.3, average=None)
    assert fbeta_obj.beta == 0.5
    assert fbeta_obj.average is None
    assert fbeta_obj.threshold == 0.3
    assert fbeta_obj.num_classes == 3
    assert backend.standardize_dtype(fbeta_obj.dtype) == "float32"
    # Check save and restore config
    fbeta_obj2 = FBetaScore.from_config(fbeta_obj.get_config())
    assert fbeta_obj2.beta == 0.5
    assert fbeta_obj2.average is None
    assert fbeta_obj2.threshold == 0.3
    assert fbeta_obj2.num_classes == 3
    assert backend.standardize_dtype(fbeta_obj.dtype) == "float32"


def _test_tf(avg, beta, act, pred, sample_weights, threshold):
    act = ops.convert_to_tensor(act, "float32")
    pred = ops.convert_to_tensor(pred, "float32")

    fbeta = FBetaScore(3, avg, beta, threshold)
    fbeta.update_state(act, pred, sample_weights)
    return ops.convert_to_numpy(fbeta.result())


def _test_fbeta_score(actuals, preds, sample_weights, avg, beta_val, result, threshold):
    actuals = ops.convert_to_tensor(actuals, "float32")
    preds = ops.convert_to_tensor(preds, "float32")
    if sample_weights is not None:
        sample_weights = ops.convert_to_tensor(sample_weights, "float32")

    tf_score = _test_tf(avg, beta_val, actuals, preds, sample_weights, threshold)
    np.testing.assert_allclose(tf_score, result, atol=1e-7, rtol=1e-6)


def test_fbeta_perfect_score():
    preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
    actuals = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]

    for avg_val in ["micro", "macro", "weighted"]:
        for beta in [0.5, 1.0, 2.0]:
            _test_fbeta_score(actuals, preds, None, avg_val, beta, 1.0, 0.66)


def test_fbeta_worst_score():
    preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
    actuals = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

    for avg_val in ["micro", "macro", "weighted"]:
        for beta in [0.5, 1.0, 2.0]:
            _test_fbeta_score(actuals, preds, None, avg_val, beta, 0.0, 0.66)


@pytest.mark.parametrize(
    "avg_val, beta, result",
    [
        (None, 0.5, [0.71428573, 0.5, 0.833334]),
        (None, 1.0, [0.8, 0.5, 0.6666667]),
        (None, 2.0, [0.9090904, 0.5, 0.555556]),
        ("micro", 0.5, 0.6666667),
        ("micro", 1.0, 0.6666667),
        ("micro", 2.0, 0.6666667),
        ("macro", 0.5, 0.6825397),
        ("macro", 1.0, 0.6555555),
        ("macro", 2.0, 0.6548822),
        ("weighted", 0.5, 0.6825397),
        ("weighted", 1.0, 0.6555555),
        ("weighted", 2.0, 0.6548822),
    ],
)
def test_fbeta_random_score(avg_val, beta, result):
    preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
    actuals = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
    _test_fbeta_score(actuals, preds, None, avg_val, beta, result, 0.66)


@pytest.mark.parametrize(
    "avg_val, beta, result",
    [
        (None, 0.5, [0.9090904, 0.555556, 1.0]),
        (None, 1.0, [0.8, 0.6666667, 1.0]),
        (None, 2.0, [0.71428573, 0.833334, 1.0]),
        ("micro", 0.5, 0.833334),
        ("micro", 1.0, 0.833334),
        ("micro", 2.0, 0.833334),
        ("macro", 0.5, 0.821549),
        ("macro", 1.0, 0.822222),
        ("macro", 2.0, 0.849206),
        ("weighted", 0.5, 0.880471),
        ("weighted", 1.0, 0.844445),
        ("weighted", 2.0, 0.829365),
    ],
)
def test_fbeta_random_score_none(avg_val, beta, result):
    preds = [
        [0.9, 0.1, 0],
        [0.2, 0.6, 0.2],
        [0, 0, 1],
        [0.4, 0.3, 0.3],
        [0, 0.9, 0.1],
        [0, 0, 1],
    ]
    actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
    _test_fbeta_score(actuals, preds, None, avg_val, beta, result, None)


@pytest.mark.parametrize(
    "avg_val, beta, sample_weights, result",
    [
        (None, 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.909091, 0.555556, 1.0]),
        (None, 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (None, 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.9375, 0.714286, 1.0]),
        (None, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.8, 0.666667, 1.0]),
        (None, 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (None, 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.857143, 0.8, 1.0]),
        (None, 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.714286, 0.833333, 1.0]),
        (None, 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        (None, 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.789474, 0.909091, 1.0]),
        ("micro", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
        ("micro", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("micro", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
        ("micro", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
        ("micro", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("micro", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
        ("micro", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
        ("micro", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("micro", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
        ("macro", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.821549),
        ("macro", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
        ("macro", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.883929),
        ("macro", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.822222),
        ("macro", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
        ("macro", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.885714),
        ("macro", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.849206),
        ("macro", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
        ("macro", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.899522),
        ("weighted", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.880471),
        ("weighted", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("weighted", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.917857),
        ("weighted", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.844444),
        ("weighted", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("weighted", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.902857),
        ("weighted", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.829365),
        ("weighted", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
        ("weighted", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.897608),
    ],
)
def test_fbeta_weighted_random_score_none(avg_val, beta, sample_weights, result):
    preds = [
        [0.9, 0.1, 0],
        [0.2, 0.6, 0.2],
        [0, 0, 1],
        [0.4, 0.3, 0.3],
        [0, 0.9, 0.1],
        [0, 0, 1],
    ]
    actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
    _test_fbeta_score(actuals, preds, sample_weights, avg_val, beta, result, None)


def test_keras_model():
    if backend.backend() == "jax":
        pytest.skip("JAX does not support F1Score with model.fit() yet.")
    fbeta = FBetaScore(5, "micro", 1.0)
    _get_model(fbeta, 5)


def test_eq():
    f1 = F1Score(3)
    fbeta = FBetaScore(3, beta=1.0)

    preds = [
        [0.9, 0.1, 0],
        [0.2, 0.6, 0.2],
        [0, 0, 1],
        [0.4, 0.3, 0.3],
        [0, 0.9, 0.1],
        [0, 0, 1],
    ]
    actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
    preds = ops.convert_to_tensor(preds, "float32")
    actuals = ops.convert_to_tensor(actuals, "float32")
    fbeta.update_state(actuals, preds)
    f1.update_state(actuals, preds)
    np.testing.assert_allclose(
        ops.convert_to_numpy(fbeta.result()), ops.convert_to_numpy(f1.result())
    )


def test_sample_eq():
    f1 = F1Score(3)
    f1_weighted = F1Score(3)

    preds = ops.convert_to_tensor(
        [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
    )
    actuals = ops.convert_to_tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
    )
    sample_weights = ops.convert_to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    f1.update_state(actuals, preds)
    f1_weighted(actuals, preds, sample_weights)
    np.testing.assert_allclose(
        ops.convert_to_numpy(f1.result()), ops.convert_to_numpy(f1_weighted.result())
    )


def test_keras_model_f1():
    if backend.backend() == "jax":
        pytest.skip("JAX does not support F1Score with model.fit() yet.")
    f1 = F1Score(5)
    _get_model(f1, 5)


def test_config_f1():
    f1 = F1Score(3)
    config = f1.get_config()
    assert "beta" not in config
