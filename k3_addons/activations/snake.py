from keras import ops


def snake(x, frequency=1):
    x = ops.convert_to_tensor(x)
    frequency = ops.cast(frequency, x.dtype)

    return x + (1 - ops.cos(2 * frequency * x)) / (2 * frequency)
