import numpy as np
from keras import layers, ops, backend, Sequential

from k3_addons.metrics.hamming import HammingLoss, hamming_distance


def test_config():
    hl_obj = HammingLoss(mode="multilabel", threshold=0.8)
    assert hl_obj.name == "hamming_loss"
    assert backend.standardize_dtype(hl_obj.dtype) == "float32"


def check_results(obj, value):
    np.testing.assert_allclose(
        ops.convert_to_numpy(value), ops.convert_to_numpy(obj.result()), atol=1e-5
    )


def test_mc_4_classes():
    actuals = ops.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype="float32",
    )
    predictions = ops.convert_to_tensor(
        [
            [0.85, 0.12, 0.03, 0],
            [0, 0, 1, 0],
            [0.10, 0.045, 0.045, 0.81],
            [1, 0, 0, 0],
            [0.80, 0.10, 0.10, 0],
            [1, 0, 0, 0],
            [0.05, 0, 0.90, 0.05],
        ],
        dtype="float32",
    )
    # Initialize
    hl_obj = HammingLoss("multiclass", threshold=0.8)
    hl_obj.update_state(actuals, predictions)
    # Check results
    check_results(hl_obj, 0.2857143)


def test_mc_5_classes():
    actuals = ops.convert_to_tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype="float32",
    )

    predictions = ops.convert_to_tensor(
        [
            [0.85, 0, 0.15, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0.05, 0.90, 0.04, 0, 0.01],
            [0.10, 0, 0.81, 0.09, 0],
            [0.10, 0.045, 0, 0.81, 0.045],
            [1, 0, 0, 0, 0],
            [0, 0.85, 0, 0, 0.15],
        ],
        dtype="float32",
    )
    # Initialize
    hl_obj = HammingLoss("multiclass", threshold=0.8)
    hl_obj.update_state(actuals, predictions)
    # Check results
    check_results(hl_obj, 0.25)


def test_ml_4_classes():
    actuals = ops.convert_to_tensor(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]], dtype="float32"
    )
    predictions = ops.convert_to_tensor(
        [[0.97, 0.56, 0.83, 0.77], [0.34, 0.95, 0.7, 0.89], [0.95, 0.45, 0.23, 0.56]],
        dtype="float32",
    )
    # Initialize
    hl_obj = HammingLoss("multilabel", threshold=0.8)
    hl_obj.update_state(actuals, predictions)
    # Check results
    check_results(hl_obj, 0.16666667)


def test_ml_5_classes():
    actuals = ops.convert_to_tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
        ],
        dtype="float32",
    )
    predictions = ops.convert_to_tensor(
        [
            [1, 0.75, 0.2, 0.55, 0],
            [0.65, 0.22, 0.97, 0.88, 0],
            [0, 1, 0, 1, 0],
            [0, 0.85, 0.9, 0.34, 0.5],
            [0.4, 0.65, 0.87, 0, 0.12],
            [0.66, 0.55, 1, 0.98, 0],
            [0.95, 0.34, 0.67, 0.65, 0.10],
            [0.45, 0.97, 0.89, 0.67, 0.46],
        ],
        dtype="float32",
    )
    # Initialize
    hl_obj = HammingLoss("multilabel", threshold=0.7)
    hl_obj.update_state(actuals, predictions)
    # Check results
    check_results(hl_obj, 0.075)


def hamming_distance_test():
    actuals = ops.convert_to_tensor([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype="int32")
    predictions = ops.convert_to_tensor([1, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype="int32")
    test_result = hamming_distance(actuals, predictions)
    np.testing.assert_allclose(0.3, test_result, atol=1e-5)


# Keras model check
def test_keras_model():
    model = Sequential()
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    h1 = HammingLoss(mode="multiclass")
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=[h1])
    data = np.random.random((100, 10))
    labels = np.random.random((100, 3))
    model.fit(data, labels, epochs=1, batch_size=32, verbose=0)
