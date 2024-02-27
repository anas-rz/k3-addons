from keras import ops


def tanhshrink(x):
    x = ops.convert_to_tensor(x)
    return x - ops.tanh(x)
