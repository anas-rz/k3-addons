from keras import ops, layers, backend
from keras.src.ops.operation_utils import compute_pooling_output_shape

from k3_addons.utils.misc import standardize_pool_size, get_input_dim
from k3_addons.api_export import k3_export


class BaseAdaptivePool(layers.Layer):
    def __init__(
        self,
        output_size,
        pool_dimensions,
        data_format=None,
        padding="valid",
        pool_mode="average",
        **kwargs,
    ):
        super(BaseAdaptivePool, self).__init__(**kwargs)
        assert pool_mode in [
            "average",
            "max",
        ], "Adaptive Pooling mode must be either average or max"
        self.pool_mode = pool_mode
        self.pool_dimensions = pool_dimensions
        self.output_size = standardize_pool_size(pool_dimensions, output_size)
        if data_format is None:
            self.data_format = backend.image_data_format()
        else:
            self.data_format = data_format
        assert self.data_format in [
            "channels_last",
            "channels_first",
        ], "Data format must be either channels_last or channels_first"
        self.input_dim = get_input_dim(self.data_format, pool_dimensions)
        self.padding = padding

    def build(self, input_shape):
        input_size = input_shape[self.input_dim]
        if self.pool_dimensions == 1:
            self.strides = input_size // self.output_size
            self.kernel_size = input_size - (self.output_size - 1) * self.strides
        elif self.pool_dimensions == 2:
            self.strides = (
                input_size[0] // self.output_size[0],
                input_size[1] // self.output_size[1],
            )
            self.kernel_size = (
                input_size[0] - (self.output_size[0] - 1) * self.strides[0],
                input_size[1] - (self.output_size[1] - 1) * self.strides[1],
            )
        elif self.pool_dimensions == 3:
            self.strides = (
                input_size[0] // self.output_size[0],
                input_size[1] // self.output_size[1],
                input_size[2] // self.output_size[2],
            )
            self.kernel_size = (
                input_size[0] - (self.output_size[0] - 1) * self.strides[0],
                input_size[1] - (self.output_size[1] - 1) * self.strides[1],
                (self.output_size[2] - 1) * self.strides[2],
            )

    def call(self, inputs):
        if self.pool_mode == "max":
            return ops.max_pool(
                inputs,
                pool_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        elif self.pool_mode == "average":
            return ops.average_pool(
                inputs,
                pool_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        else:
            raise ValueError(
                "`pool_mode` must be either 'max' or 'average'. Received: "
                f"{self.pool_mode}."
            )

    def compute_output_shape(self, input_shape):
        return compute_pooling_output_shape(
            input_shape,
            self.kernel_size,
            self.strides,
            self.padding,
            self.data_format,
        )


@k3_export(path="k3_addons.layers.AdaptiveMaxPool1D")
class AdaptiveMaxPool1D(BaseAdaptivePool):
    """
    Adaptive Pooling like torch.nn.AdaptiveMaxPool1d
    """

    def __init__(self, output_size, data_format=None, padding="valid", **kwargs):
        super(AdaptiveMaxPool1D, self).__init__(
            output_size,
            pool_dimensions=1,
            pool_mode="max",
            data_format=data_format,
            padding=padding,
            **kwargs,
        )


@k3_export(path="k3_addons.layers.AdaptiveAveragePool1D")
class AdaptiveAveragePool1D(BaseAdaptivePool):
    """Adaptive Pooling like torch.nn.AdaptiveAvgPool1d"""

    def __init__(self, output_size, data_format=None, padding="valid", **kwargs):
        super(AdaptiveAveragePool1D, self).__init__(
            output_size,
            pool_dimensions=1,
            pool_mode="average",
            data_format=data_format,
            padding=padding,
            **kwargs,
        )


@k3_export(path="k3_addons.layers.AdaptiveMaxPool2D")
class AdaptiveMaxPool2D(BaseAdaptivePool):
    """Adaptive Pooling like torch.nn.AdaptiveMaxPool2d"""

    def __init__(self, output_size, data_format=None, padding="valid", **kwargs):
        super(AdaptiveMaxPool2D, self).__init__(
            output_size,
            pool_dimensions=2,
            pool_mode="max",
            data_format=data_format,
            padding=padding,
            **kwargs,
        )


@k3_export(path="k3_addons.layers.AdaptiveAveragePool2D")
class AdaptiveAveragePool2D(BaseAdaptivePool):
    """Adaptive Pooling like torch.nn.AdaptiveAvgPool2d"""

    def __init__(self, output_size, data_format=None, padding="valid", **kwargs):
        super(AdaptiveAveragePool2D, self).__init__(
            output_size,
            pool_dimensions=2,
            pool_mode="average",
            data_format=data_format,
            padding=padding,
            **kwargs,
        )
