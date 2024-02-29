from keras import ops, backend


def _gaussian(kernel_size, sigma, dtype):
    dist = ops.arange(
        start=(1 - kernel_size) / 2, stop=(1 + kernel_size) / 2, step=1, dtype=dtype
    )
    gauss = ops.exp(-ops.square(dist / sigma) / 2)
    gauss = gauss / ops.sum(gauss)
    return ops.expand_dims(gauss, axis=0)  # (1, kernel_size)


def _gaussian_kernel_2d(channel, kernel_size, sigma, dtype, data_format=None):
    if data_format is None:
        data_format = backend.image_data_format()
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype)
    kernel = ops.matmul(
        ops.transpose(gaussian_kernel_x), gaussian_kernel_y
    )  # (kernel_size, 1) * (1, kernel_size)
    if data_format == "channels_first":
        kernel = ops.expand_dims(kernel, axis=0)
        kernel = ops.expand_dims(kernel, axis=0)
        return ops.broadcast_to(kernel, (channel, 1, kernel_size[0], kernel_size[1]))
    elif data_format == "channels_last":
        kernel = ops.expand_dims(kernel, axis=0)
        kernel = ops.expand_dims(kernel, axis=-1)
        return ops.broadcast_to(kernel, (1, kernel_size[0], kernel_size[1], channel))
    else:
        raise ValueError(
            "Expected argument `data_format` to be 'channels_first' or 'channels_last'"
        )


def default_data_format(data_format):
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "Expected `data_format` to be one of 'channels_first', 'channels_last'."
            f" Got {data_format}."
        )
    return data_format
