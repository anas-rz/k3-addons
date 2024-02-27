from keras import ops
from k3_addons.api_export import k3_export


@k3_export("k3_addons.activations.mish")
def mish(x):
    x = ops.convert_to_tensor(x)
    return x * ops.tanh(ops.softplus(x))
