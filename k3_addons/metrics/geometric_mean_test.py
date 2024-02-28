import numpy as np
import pytest
from keras import ops, backend

from k3_addons.metrics.geometric_mean import GeometricMean


def get_test_data():
    return [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ([0, 0, 0, 0, 0, 0, 0, 1, 2, 6], 0),
        ([0.2, 0.5, 0.3, 0.6, 0.1, 0.7], 0.32864603),
        ([8, 4, 1, 7, 2, 11, 9, 22, 52], 7.1804023),
        ([8.2, 9.7, 9.1, 2.7, 1.1, 2.0], 4.0324492),
        ([0.6666666, 0.215213, 0.15167], 0.27918512),
    ]


def assert_result(expected, result):
    np.testing.assert_allclose(expected, result, atol=1e-5)


def check_result(obj, expected_result, expected_count):
    result = ops.convert_to_numpy(obj.result())
    count = ops.convert_to_numpy(obj.count)
    assert_result(expected_result, result)
    np.testing.assert_equal(expected_count, count)


def test_config_gmean():
    def _check_config(obj, name):
        assert obj.name == name
        assert backend.standardize_dtype(obj.dtype) == "float32"
        assert len(obj.variables) == 2

    name = "my_gmean"
    obj1 = GeometricMean(name=name)
    _check_config(obj1, name)
    obj2 = GeometricMean.from_config(obj1.get_config())
    _check_config(obj2, name)


def test_init_states_gmean():
    obj = GeometricMean()
    assert obj.total.numpy() == 0.0
    assert obj.count.numpy() == 0.0
    assert backend.standardize_dtype(obj.total.dtype) == "float32"
    assert backend.standardize_dtype(obj.count.dtype) == "float32"


@pytest.mark.parametrize("values, expected", get_test_data())
def test_scalar_update_state_gmean(values, expected):
    obj = GeometricMean()
    values = ops.convert_to_tensor(values, "float32")
    for v in values:
        obj.update_state(v)
    check_result(obj, expected, len(values))


@pytest.mark.parametrize("values, expected", get_test_data())
def test_vector_update_state_gmean(values, expected):
    obj = GeometricMean()
    values = ops.convert_to_tensor(values, "float32")
    obj.update_state(values)
    check_result(obj, expected, len(values))


@pytest.mark.parametrize("values, expected", get_test_data())
def test_call_gmean(values, expected):
    obj = GeometricMean()
    result = obj(ops.convert_to_tensor(values, "float32"))
    count = ops.convert_to_numpy(obj.count)
    assert_result(expected, result)
    np.testing.assert_equal(len(values), count)


def test_reset_state():
    obj = GeometricMean()
    obj.update_state([1, 2, 3, 4, 5])
    obj.reset_state()
    assert ops.convert_to_numpy(obj.total) == 0.0
    assert ops.convert_to_numpy(obj.count) == 0.0


@pytest.mark.parametrize(
    "values, sample_weight, expected",
    [
        ([1, 2, 3, 4, 5], 1, 2.6051712),
        ([2.1, 4.6, 7.1], [1, 2, 3], 5.014777),
        ([9.6, 1.8, 8.2], [0.2, 0.5, 0.3], 3.9649222),
    ],
)
def test_sample_weight_gmean(values, sample_weight, expected):
    obj = GeometricMean()
    obj.update_state(values, sample_weight=sample_weight)
    assert_result(expected, ops.convert_to_numpy(obj.result()))
