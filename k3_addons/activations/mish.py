from keras import ops


def mish(x):
    x = ops.convert_to_tensor(x)
    return x * ops.tanh(ops.softplus(x))
