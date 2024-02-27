from keras.losses import Loss
from keras import ops, backend
from k3_addons.api_export import k3_export


@k3_export("k3_addons.losses.WeightedKappaLoss")
class WeightedKappaLoss(Loss):
    def __init__(
        self,
        num_classes,
        weightage="quadratic",
        name="cohen_kappa_loss",
        epsilon=1e-6,
        reduction=None,
    ):
        super().__init__(name=name, reduction=reduction)

        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.epsilon = epsilon or backend.epsilon()
        label_vec = ops.arange(num_classes, dtype=backend.floatx())
        self.row_label_vec = ops.reshape(label_vec, [1, num_classes])
        self.col_label_vec = ops.reshape(label_vec, [num_classes, 1])
        col_mat = ops.tile(self.col_label_vec, [1, num_classes])
        row_mat = ops.tile(self.row_label_vec, [num_classes, 1])
        if weightage == "linear":
            self.weight_mat = ops.abs(col_mat - row_mat)
        else:
            self.weight_mat = (col_mat - row_mat) ** 2

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, dtype=self.col_label_vec.dtype)
        y_pred = ops.cast(y_pred, dtype=self.weight_mat.dtype)
        batch_size = ops.shape(y_true)[0]
        cat_labels = ops.matmul(y_true, self.col_label_vec)
        cat_label_mat = ops.tile(cat_labels, [1, self.num_classes])
        row_label_mat = ops.tile(self.row_label_vec, [batch_size, 1])
        if self.weightage == "linear":
            weight = ops.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = ops.sum(weight * y_pred)
        label_dist = ops.sum(y_true, axis=0, keepdims=True)
        pred_dist = ops.sum(y_pred, axis=0, keepdims=True)
        w_pred_dist = ops.matmul(self.weight_mat, ops.transpose(pred_dist))
        denominator = ops.sum(ops.matmul(label_dist, w_pred_dist))
        denominator /= ops.cast(batch_size, dtype=denominator.dtype)
        loss = ops.divide_no_nan(numerator, denominator)
        return ops.log(loss + self.epsilon)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}
