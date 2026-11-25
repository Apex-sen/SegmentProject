import hashlib
import json

from torch import nn


def use_current_batch_stats(model):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.train()
            layer.track_running_stats = False
            layer.running_mean = None
            layer.running_var = None


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        item = tensor.detach().cpu().contiguous()
        digest.update(repr((name, str(item.dtype), tuple(item.shape))).encode("utf-8"))
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer):
    state = optimizer.state_dict()
    digest = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True).encode("utf-8"))
    for index, entry in sorted(state["state"].items()):
        digest.update(tensor_digest((f"{index}.{name}", tensor)
                                    for name, tensor in sorted(entry.items())).encode("utf-8"))
    return digest.hexdigest()
