from keras import ops, backend, metrics
from k3_addons.api_export import k3_export


def _total_variation_update(img, data_format=None):
    """Compute total variation statistics on current batch."""
    if len(ops.shape(img)) != 4:
        raise RuntimeError(
            f"Expected input `img` to be an 4D tensor, but got {img.shape}"
        )
    data_format = data_format or backend.image_data_format()
    # Calculate differences along spatial dimensions
    if data_format == "channels_first":
        diff1 = img[:, :, 1:, :] - img[:, :, :-1, :]
        diff2 = img[:, :, :, 1:] - img[:, :, :, :-1]
    elif data_format == "channels_last":
        diff1 = img[:, 1:, :, :] - img[:, :-1, :, :]
        diff2 = img[:, :, 1::] - img[:, :, :-1, :]
    else:
        raise ValueError(f"Invalid value for image_data_format: {data_format}")

    # Compute absolute values and L1 norm (sum of absolute values)
    res1 = ops.sum(ops.abs(diff1), axis=[1, 2, 3])
    res2 = ops.sum(ops.abs(diff2), axis=[1, 2, 3])

    score = res1 + res2
    return score, ops.shape(img)[0]


def _total_variation_compute(score, num_elements, reduction):
    if reduction == "mean":
        return ops.sum(score) / num_elements
    if reduction == "sum":
        return ops.sum(score)
    if reduction is None or reduction == "none":
        return score
    raise ValueError(
        "Expected argument `reduction` to either be 'sum', 'mean', 'none' or None"
    )


@k3_export(
    [
        "k3_addons.metrics.total_variation",
        "k3_addons.metrics.functional.total_variation",
        "k3_addons.metrics.image.total_variation",
    ]
)
def total_variation(img, reduction="sum", data_format=None):
    score, num_elements = _total_variation_update(img, data_format=data_format)
    return _total_variation_compute(score, num_elements, reduction)
