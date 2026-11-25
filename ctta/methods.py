from copy import deepcopy
from inspect import signature

import torch
from torch import nn

from .cotta import CoTTA
from .method_state import optimizer_digest, tensor_digest, use_current_batch_stats
from .model import STRUCTURES
from .sar import SAR
from .vptta import VPTTA


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


class EmptyPredictionBNRetry(SourceOnly):
    """Run frozen, one-image inference and retry if a foreground structure is empty."""

    def __init__(self, model, task):
        super().__init__(model.requires_grad_(False))
        self.retry_model = deepcopy(self.model)
        use_current_batch_stats(self.retry_model)
        # The optic cup is part of the optic disc, so checking the cup covers both.
        self.foreground_label, = STRUCTURES[task]["oc" if task == "fundus" else "polyp"]
        self.retry_images = 0

    @torch.inference_mode()
    def __call__(self, image, valid):
        logits = self.model(image)
        if ((logits.argmax(dim=1) == self.foreground_label) & valid).any().item():
            return logits
        self.retry_images += 1
        return self.retry_model(image)

    def state_summary(self):
        return {"retry_model_sha256": tensor_digest(self.retry_model.state_dict().items()),
                "retry_images": self.retry_images}

    def verify_run(self, start, end, sample_count):
        if any(start[key] != end[key] for key in ("model_sha256", "retry_model_sha256")):
            raise RuntimeError("Empty-prediction retry changed frozen parameters or buffers.")
        if not 0 <= end["retry_images"] - start["retry_images"] <= sample_count:
            raise RuntimeError("Expected at most one BN retry per image.")


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
        return {"optimizer_sha256": optimizer_digest(self.optimizer), "update_steps": self.steps}

    def verify_run(self, start, end, sample_count):
        if end["update_steps"] - start["update_steps"] != sample_count:
            raise RuntimeError("Expected one TENT update for each target image.")


METHOD_REGISTRY = {
    "source_only": SourceOnly,
    "empty_prediction_bn_retry": EmptyPredictionBNRetry,
    "tent_continual": TentContinual,
    "cotta": CoTTA,
    "sar": SAR,
    "vptta": VPTTA,
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
        if name == "cotta":
            settings = {"seed": config["seed"], **settings}
        if name == "empty_prediction_bn_retry":
            settings = {**settings, "task": config.get("task", "fundus")}
        try:
            signature(METHOD_REGISTRY[name]).bind(None, **settings)
        except TypeError as exc:
            raise ValueError(f"Invalid [methods.{name}] configuration: {exc}") from exc
        methods[name] = settings
    return methods
