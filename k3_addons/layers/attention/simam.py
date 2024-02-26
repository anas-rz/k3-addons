from keras import layers, ops
from k3_addons.api_export import k3_export

@k3_export(path="k3_addons.layers.SimAM")
class SimAM(layers.Layer):
    def __init__(self, e_lambda=1e-4, activation="sigmoid"):
        super().__init__()
        self.e_lambda = e_lambda
        self.activaton = layers.Activation(activation)

    def call(self, x):
        b, h, w, c = ops.shape(x)
        n = w * h - 1
        x_mean = ops.mean(x, axis=1, keepdims=True)
        x_mean = ops.mean(x_mean, axis=2, keepdims=True)
        x_minus_mu_square = ops.square(x - x_mean)
        denom = ops.sum(x_minus_mu_square, axis=1, keepdims=True)
        denom = ops.sum(denom, axis=2, keepdims=True) / n
        weights = x_minus_mu_square / (4 * (denom + self.e_lambda)) + 0.5
        return x * self.activaton(weights)
