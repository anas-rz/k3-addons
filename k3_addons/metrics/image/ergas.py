from keras import ops
from k3_addons.utils.checks import _check_same_shape
from k3_addons.utils.distributed import reduce
from k3_addons.metrics.image.utils import default_data_format
from k3_addons.api_export import k3_export


def _ergas_update(preds, target):
    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got preds: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(ops.shape(preds)) != 4:
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW shape."
            f" Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _ergas_compute(
    preds,
    target,
    ratio=4,
    reduction="elementwise_mean",
    data_format=None,
):
    data_format = default_data_format(data_format)
    if data_format == "channels_first":
        b, c, h, w = ops.shape(preds)
        preds = ops.reshape(preds, (b, c, h * w))
        target = ops.reshape(target, (b, c, h * w))
        spatial_axis = 2
    else:
        b, h, w, c = ops.shape(preds)
        preds = ops.reshape(preds, (b, h * w, c))
        target = ops.reshape(target, (b, h * w, c))
        spatial_axis = 1

    diff = preds - target
    sum_squared_error = ops.sum(diff * diff, axis=spatial_axis)  #
    rmse_per_band = ops.sqrt(sum_squared_error / (h * w))
    mean_target = ops.mean(target, axis=spatial_axis)

    ergas_score = (
        100 * ratio * ops.sqrt(ops.sum((rmse_per_band / mean_target) ** 2, axis=1) / c)
    )
    return reduce(ergas_score, reduction)


@k3_export(
    [
        "k3_addons.metrics.ergas",
        "k3_addons.metrics.functional.ergas",
        "k3_addons.metrics.image.ergas",
    ]
)
def error_relative_global_dimensionless_synthesis(
    preds,
    target,
    ratio=4,
    reduction="elementwise_mean",
    data_format=None,
):
    preds, target = _ergas_update(preds, target)
    return _ergas_compute(preds, target, ratio, reduction, data_format=data_format)
