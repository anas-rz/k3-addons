from keras import ops, backend
from k3_addons.utils.keras_utils import LossFunctionWrapper
from k3_addons.api_export import k3_export


@k3_export("k3_addons.losses.GIoULoss")
class GIoULoss(LossFunctionWrapper):
    def __init__(
        self,
        mode="giou",
        reduction=None,
        name="giou_loss",
    ):
        super().__init__(giou_loss, name=name, reduction=reduction, mode=mode)


def giou_loss(y_true, y_pred, mode="giou"):
    if mode not in ["giou", "iou"]:
        raise ValueError("Value of mode should be 'iou' or 'giou'")
    y_pred = ops.convert_to_tensor(y_pred)
    if not backend.is_float_dtype(y_pred.dtype):
        y_pred = ops.cast(y_pred, "float32")
    y_true = ops.cast(y_true, y_pred.dtype)
    giou = ops.squeeze(_calculate_giou(y_pred, y_true, mode))
    return 1 - giou


def _calculate_giou(b1, b2, mode="giou"):
    zero = ops.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = ops.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = ops.unstack(b2, 4, axis=-1)
    b1_width = ops.maximum(zero, b1_xmax - b1_xmin)
    b1_height = ops.maximum(zero, b1_ymax - b1_ymin)
    b2_width = ops.maximum(zero, b2_xmax - b2_xmin)
    b2_height = ops.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = ops.maximum(b1_ymin, b2_ymin)
    intersect_xmin = ops.maximum(b1_xmin, b2_xmin)
    intersect_ymax = ops.minimum(b1_ymax, b2_ymax)
    intersect_xmax = ops.minimum(b1_xmax, b2_xmax)
    intersect_width = ops.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = ops.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = ops.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = ops.minimum(b1_ymin, b2_ymin)
    enclose_xmin = ops.minimum(b1_xmin, b2_xmin)
    enclose_ymax = ops.maximum(b1_ymax, b2_ymax)
    enclose_xmax = ops.maximum(b1_xmax, b2_xmax)
    enclose_width = ops.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = ops.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - ops.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou
