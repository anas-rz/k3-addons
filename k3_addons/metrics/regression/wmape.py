from keras import ops
from k3_addons.utils.checks import _check_same_shape
from k3_addons.api_export import k3_export


def _weighted_mean_absolute_percentage_error_update(
    preds,
    target,
):
    _check_same_shape(preds, target)

    sum_abs_error = preds - target
    sum_abs_error = ops.sum(ops.abs(sum_abs_error))

    sum_scale = ops.sum(ops.abs(target))

    return sum_abs_error, sum_scale


def _weighted_mean_absolute_percentage_error_compute(
    sum_abs_error,
    sum_scale,
    epsilon=1.17e-06,
):
    return sum_abs_error / ops.clip(sum_scale, x_min=epsilon, x_max=float("inf"))


@k3_export(["k3_addons.metrics.wmape", "k3_addons.metrics.functional.wmape"])
def weighted_mean_absolute_percentage_error(preds, target):
    sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(
        preds, target
    )
    return _weighted_mean_absolute_percentage_error_compute(sum_abs_error, sum_scale)
