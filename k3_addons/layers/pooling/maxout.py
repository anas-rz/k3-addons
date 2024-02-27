from keras import layers, ops
from k3_addons.api_export import k3_export


@k3_export("k3_addons.layers.Maxout")
class Maxout(layers.Layer):
    def __init__(self, num_units: int, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs):
        shape = list(ops.shape(inputs))
        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = ops.shape(inputs)[i]

        num_channels = shape[self.axis]
        if num_channels % self.num_units:
            raise ValueError(
                "number of features({}) is not "
                "a multiple of num_units({})".format(num_channels, self.num_units)
            )

        if self.axis < 0:
            axis = self.axis + len(shape)
        else:
            axis = self.axis
        assert axis >= 0, "Find invalid axis: {}".format(self.axis)

        expand_shape = shape[:]
        expand_shape[axis] = self.num_units
        k = num_channels // self.num_units
        expand_shape.insert(axis, k)

        outputs = ops.max(ops.reshape(inputs, expand_shape), axis, keepdims=False)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_units": self.num_units,
                "axis": self.axis,
            }
        )
        return config
