from keras import ops


def _check_same_shape(preds, target):
    """Check that predictions and target have the same shape, else raise error."""
    if ops.shape(preds) != ops.shape(target):
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )
