import numpy as np
import torch
from keras import ops
from k3_addons.utils.checks import _check_same_shape
from k3_addons.utils.imports import MULTIPROCESSING_AVAILABLE, PESQ_AVAILABLE
from k3_addons.api_export import k3_export


@k3_export(
    [
        "k3_addons.metrics.pesq",
        "k3_addons.metrics.functional.pesq",
        "k3_addons.metrics.audio.pesq",
    ]
)
def perceptual_evaluation_speech_quality(
    preds,
    target,
    fs,
    mode,
    n_processes=1,
):
    if not PESQ_AVAILABLE:
        raise ModuleNotFoundError(
            "PESQ metric requires that pesq is installed."
            " Install it using  `pip install pesq`."
        )
    import pesq as pesq_backend

    if fs not in (8000, 16000):
        raise ValueError(
            f"Expected argument `fs` to either be 8000 or 16000 but got {fs}"
        )
    if mode not in ("wb", "nb"):
        raise ValueError(
            f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}"
        )
    _check_same_shape(preds, target)

    if len(ops.shape(preds)) == 1:
        pesq_val_np = pesq_backend.pesq(
            fs, ops.convert_to_numpy(target), ops.convert_to_numpy(preds), mode
        )
        pesq_val = torch.tensor(pesq_val_np)
    else:
        preds_np = ops.convert_to_numpy(ops.reshape(preds, (-1, preds.shape[-1])))
        target_np = ops.convert_to_numpy(ops.reshape(target, (-1, preds.shape[-1])))

        if MULTIPROCESSING_AVAILABLE and n_processes != 1:
            pesq_val_np = pesq_backend.pesq_batch(
                fs, target_np, preds_np, mode, n_processor=n_processes
            )
            pesq_val_np = np.array(pesq_val_np)
        else:
            pesq_val_np = np.empty(shape=(preds_np.shape[0]))
            for b in range(preds_np.shape[0]):
                pesq_val_np[b] = pesq_backend.pesq(
                    fs, target_np[b, :], preds_np[b, :], mode
                )
        pesq_val = ops.convert_to_tensor(pesq_val_np)
        pesq_val = ops.reshape(pesq_val, (preds.shape[:-1]))

    return pesq_val
