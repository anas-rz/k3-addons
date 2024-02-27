from keras import layers, Sequential
from k3_addons.layers.pooling.adaptive_pooling import AdaptiveAveragePool2D
from k3_addons.api_export import k3_export


@k3_export(path="k3_addons.layers.ParNetAttention")
class ParNetAttention(layers.Layer):
    def __init__(self, activation="selu", **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.sse = Sequential(
            [
                AdaptiveAveragePool2D(1),
                layers.Conv2D(input_dim, kernel_size=1, activation="sigmoid"),
            ]
        )

        self.conv1x1 = Sequential(
            [layers.Conv2D(input_dim, kernel_size=1), layers.BatchNormalization()]
        )
        self.conv3x3 = Sequential(
            [
                layers.Conv2D(input_dim, kernel_size=3, padding="same"),
                layers.BatchNormalization(),
            ]
        )
        self.activation = layers.Activation(self.activation)

    def call(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        out = self.activation(x1 + x2 + x3)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
            }
        )
        return config
