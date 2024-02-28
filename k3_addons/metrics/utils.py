import keras
from keras import ops
import numpy as np


class MeanMetricWrapper(keras.metrics.Mean):
    def __init__(
        self,
        fn,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, self._dtype)
        y_pred = ops.cast(y_pred, self._dtype)
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {k: v for k, v in self._fn_kwargs.items()}
        base_config = super().get_config()
        return {**base_config, **config}


def _get_model(metric, num_output):
    # Test API comptibility with tf.keras Model
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(num_output, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["acc", metric]
    )

    data = ops.convert_to_tensor(np.random.random((10, 3)))
    labels = ops.convert_to_tensor(np.random.random((10, num_output)))
    model.fit(data, labels, epochs=1, batch_size=5, verbose=0)


def sample_weight_shape_match(v, sample_weight):
    if sample_weight is None:
        return ops.ones_like(v)
    if np.size(sample_weight) == 1:
        return ops.full(ops.shape(v), sample_weight)
    return ops.convert_to_tensor(sample_weight)
