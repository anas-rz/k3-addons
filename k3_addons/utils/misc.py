from functools import partial


def n_tuple(inputs, n):
    if isinstance(inputs, int):
        return tuple([inputs] * n)
    elif isinstance(inputs, (list, tuple)):
        assert len(inputs) == n, f"Pool size must have length {n}"
        return tuple(inputs)
    else:
        raise ValueError(
            f"inputs must be an integer or a {tuple, list} found: {inputs}"
        )


two_tuple = partial(n_tuple, n=2)
three_tuple = partial(n_tuple, n=3)


def standardize_pool_size(dims, output_size, data_format=None):
    if dims == 1:
        return output_size
    elif dims == 2:
        return two_tuple(output_size)
    elif dims == 3:
        return three_tuple(output_size)
    else:
        raise ValueError(
            f"Invalid Output Size received, Expected Int or Tuple of size {dims}"
        )


def get_input_dim(data_format, dims):
    if data_format == "channels_last":
        if dims == 1:
            return 1
        elif dims == 2:
            return slice(1, 3)
        elif dims == 3:
            return slice(1, 4)
        else:
            raise ValueError(
                f"Invalid Pool Dimensions received, Expected Int or Tuple of size {dims}"
            )
    if data_format == "channels_first":
        if dims == 1:
            return 2
        elif dims == 2:
            return slice(2, 4)
        elif dims == 3:
            return slice(2, 4)
        else:
            raise ValueError(
                f"Invalid Pool Dimensions received, Expected Int or Tuple of size {dims}"
            )
