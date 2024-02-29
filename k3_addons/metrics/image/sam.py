from keras import backend, ops

from k3_addons.utils.checks import _check_same_shape
from k3_addons.utils.distributed import reduce
from k3_addons.api_export import k3_export

get_channel_axis = lambda data_format: 1 if data_format == "channels_first" else -1


def _sam_update(preds, target, data_format=None):
    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got preds: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(ops.shape(preds)) != 4:
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW or BxHxWxC shape."
            f" Got preds: {ops.shape(preds)} and target: {ops.shape(target)}."
        )
    channel_axis = get_channel_axis(data_format)
    if (ops.shape(preds)[channel_axis] <= 1) or (ops.shape(target)[channel_axis] <= 1):
        raise ValueError(
            "Expected channel dimension of `preds` and `target` to be larger than 1."
            f" Got preds: {preds.shape[channel_axis]} and target: {target.shape[channel_axis]}."
        )
    return preds, target


def _sam_compute(
    preds,
    target,
    reduction="elementwise_mean",
    data_format=None,
):
    if data_format is None:
        data_format = backend.image_data_format()
    channel_axis = get_channel_axis(data_format)
    dot_product = ops.sum((preds * target), axis=channel_axis)
    preds_norm = ops.norm(preds, axis=channel_axis)
    target_norm = ops.norm(target, axis=channel_axis)
    denom = preds_norm * target_norm
    sam_score = ops.clip(dot_product / denom, -1, 1)
    sam_score = ops.arccos(sam_score)
    return reduce(sam_score, reduction)


@k3_export(
    [
        "k3_addons.metrics.spectral_angle_mapper",
        "k3_addons.metrics.functional.spectral_angle_mapper",
        "k3_addons.metrics.image.spectral_angle_mapper",
    ]
)
def spectral_angle_mapper(preds, target, reduction, data_format=None):
    preds, target = _sam_update(preds, target, data_format=data_format)
    return _sam_compute(preds, target, reduction=reduction, data_format=data_format)
