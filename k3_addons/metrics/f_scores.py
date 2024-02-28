from keras import ops, metrics


class FBetaScore(metrics.Metric):
    def __init__(
        self,
        num_classes,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, 'micro', 'macro', 'weighted']"
            )

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError("The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                shape=self.init_shape,
                initializer="zeros",
                dtype=self.dtype,
                name=name,
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = ops.max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = ops.logical_and(y_pred >= threshold, ops.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = ops.cast(y_true, self.dtype)
        y_pred = ops.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = ops.multiply(val, ops.expand_dims(sample_weight, 1))
            return ops.sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))

    def result(self):
        precision = ops.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = ops.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (ops.square(self.beta) * precision) + recall
        mean = ops.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + ops.square(self.beta))

        if self.average == "weighted":
            weights = ops.divide_no_nan(
                self.weights_intermediate, ops.sum(self.weights_intermediate)
            )
            f1_score = ops.sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = ops.mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = ops.zeros(self.init_shape, dtype=self.dtype)
        for v in self.variables:
            v.assign(reset_value)
