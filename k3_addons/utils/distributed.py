from keras import ops


def reduce(x, reduction):
    if reduction == "elementwise_mean":
        return ops.mean(x)
    if reduction == "none" or reduction is None:
        return x
    if reduction == "sum":
        return ops.sum(x)
    raise ValueError("Reduction parameter unknown.")
