from keras import ops
from k3_addons.api_export import k3_export

@k3_export("k3_addons.activations.snake")
def snake(x, frequency=1):
    x = ops.convert_to_tensor(x)
    frequency = ops.cast(frequency, x.dtype)

    return x + (1 - ops.cos(2 * frequency * x)) / (2 * frequency)
