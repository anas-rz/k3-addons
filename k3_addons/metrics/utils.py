import keras
import numpy as np


def _get_model(metric, num_output):
    # Test API comptibility with tf.keras Model
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(num_output, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["acc", metric]
    )

    data = np.random.random((10, 3))
    labels = np.random.random((10, num_output))
    model.fit(data, labels, epochs=1, batch_size=5, verbose=0)
