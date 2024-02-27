from keras import ops
from k3_addons.api_export import k3_export


@k3_export("k3_addons.activations.hardshrink")
def hardshrink(x, lower=-0.5, upper=0.5):
    if lower > upper:
        raise ValueError(
            "The value of lower is {} and should"
            " not be higher than the value "
            "variable upper, which is {} .".format(lower, upper)
        )
    x = ops.convert_to_tensor(x)
    mask_lower = x < lower
    mask_upper = upper < x
    mask = ops.logical_or(mask_lower, mask_upper)
    mask = ops.cast(mask, x.dtype)
    return x * mask
