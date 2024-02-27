from keras.losses import Loss


class LossFunctionWrapper(Loss):
    def __init__(self, fn, reduction=None, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        return self.fn(y_true, y_pred, **self._fn_kwargs)
