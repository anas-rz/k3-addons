from keras import ops
from k3_addons.utils.imports import PYSOTI_AVAILABLE

from k3_addons.utils.checks import _check_same_shape


def short_time_objective_intelligibility(preds, target, fs, extended=False):
    if not PYSOTI_AVAILABLE:
        raise ModuleNotFoundError(
            "ShortTimeObjectiveIntelligibility metric requires that `pystoi` is installed."
            " You can install it using `pip install pystoi`."
        )
    from pystoi import stoi as stoi_backend

    _check_same_shape(preds, target)

    if len(preds.shape) == 1:
        stoi_val_np = stoi_backend(
            ops.convert_to_numpy(target), ops.convert_to_numpy(preds), fs, extended
        )
        stoi_val = ops.convert_to_tensor(stoi_val_np)
    else:
        preds_np = ops.convert_to_numpy(ops.reshape(preds, (-1, preds.shape[-1])))
        target_np = ops.convert_to_numpy(ops.reshape(target, (-1, preds.shape[-1])))
        stoi_val_np = ops.empty(shape=(preds_np.shape[0]))
        for b in range(ops.shape(preds_np)[0]):
            stoi_val_np[b] = stoi_backend(target_np[b, :], preds_np[b, :], fs, extended)
        stoi_val = ops.convert_to_tensor(stoi_val_np)
        stoi_val = ops.reshape(stoi_val, (preds.shape[:-1]))
    return stoi_val
