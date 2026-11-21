import hashlib
import json
from inspect import signature

import torch
from torch import nn


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        item = tensor.detach().cpu().contiguous()
        digest.update(repr((name, str(item.dtype), tuple(item.shape))).encode("utf-8"))
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


class SourceOnly:
    adapted_parameter_names = frozenset()

    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def __call__(self, image, valid):
        return self.model(image)

    def state_summary(self):
        return {}

    def verify_run(self, start, end, sample_count):
        if start != end:
            raise RuntimeError("Source Only changed model parameters or buffers.")


def entropy_loss(logits, valid=None):
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
    if valid is not None:
        entropy = entropy[valid]
    if entropy.numel() == 0:
        raise ValueError("TENT requires at least one valid pixel per update.")
    return entropy.mean()


class TentContinual:
    def __init__(self, model, learning_rate, momentum):
        self.model = model.eval().requires_grad_(False)
        params = []
        self.adapted_parameter_names = set()
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            for parameter_name, parameter in module.named_parameters(recurse=False):
                parameter.requires_grad_(True)
                params.append(parameter)
                self.adapted_parameter_names.add(".".join(filter(None, (name, parameter_name))))
        if not params:
            raise ValueError("TENT requires affine BatchNorm2d parameters.")
        self.optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
        self.steps = 0

    @torch.enable_grad()
    def __call__(self, image, valid):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(image)
        loss = entropy_loss(logits, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite test entropy during TENT adaptation.")
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        return logits.detach()

    def state_summary(self):
        state = self.optimizer.state_dict()
        digest = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True).encode("utf-8"))
        for index, entry in sorted(state["state"].items()):
            digest.update(tensor_digest((f"{index}.{name}", tensor)
                                        for name, tensor in sorted(entry.items())).encode("utf-8"))
        return {"optimizer_sha256": digest.hexdigest(), "update_steps": self.steps}

    def verify_run(self, start, end, sample_count):
        if end["update_steps"] - start["update_steps"] != sample_count:
            raise RuntimeError("Expected one TENT update for each target image.")


METHOD_REGISTRY = {
    "source_only": SourceOnly,
    "tent_continual": TentContinual,
}


def resolve_methods(config, selection=None):
    names = config["experiment"]["methods"] if selection in (None, "both") else [selection]
    methods = {}
    for name in names:
        if name not in METHOD_REGISTRY:
            registered = ", ".join(METHOD_REGISTRY)
            raise ValueError(f"Unknown method '{name}'. Registered methods: {registered}")
        settings = config["methods"].get(name, {})
        if not isinstance(settings, dict):
            raise ValueError(f"methods.{name} must be a TOML table.")
        try:
            signature(METHOD_REGISTRY[name]).bind(None, **settings)
        except TypeError as exc:
            raise ValueError(f"Invalid [methods.{name}] configuration: {exc}") from exc
        methods[name] = settings
    return methods
