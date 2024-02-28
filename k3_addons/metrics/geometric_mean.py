from keras.metrics import Metric
from keras import ops
from k3_addons.metrics.utils import sample_weight_shape_match


class GeometricMean(Metric):
    def __init__(self, name="geometric_mean", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(
            shape=(), initializer="zeros", dtype=dtype, name="total"
        )
        self.count = self.add_weight(
            shape=(), initializer="zeros", dtype=dtype, name="count"
        )

    def update_state(self, values, sample_weight=None) -> None:
        values = ops.cast(values, dtype=self.dtype)
        sample_weight = sample_weight_shape_match(values, sample_weight)
        sample_weight = ops.cast(sample_weight, dtype=self.dtype)

        self.count.assign_add(ops.sum(sample_weight))
        if not ops.isinf(self.total):
            log_v = ops.log(values)
            log_v = ops.multiply(sample_weight, log_v)
            log_v = ops.sum(log_v)
            self.total.assign_add(log_v)

    def result(self):
        if ops.isinf(self.total):
            return ops.convert_to_tensor(0, dtype=self.dtype)
        ret = ops.exp(self.total / self.count)
        return ops.cast(ret, dtype=self.dtype)

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)
