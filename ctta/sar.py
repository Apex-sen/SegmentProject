import math

import torch
from torch import nn

from .method_state import optimizer_digest


class SAR:
    """Pixel-wise port of mr-eggplant/SAR's sar.py and sam.py."""

    def __init__(self, model, learning_rate=0.000015625, momentum=0.9, rho=0.05,
                 margin_fraction=0.4, reset_entropy=0.2):
        if not 0 < margin_fraction <= 1 or not math.isfinite(rho) or rho < 0:
            raise ValueError("SAR requires 0 < margin_fraction <= 1 and a finite rho >= 0.")
        if not math.isfinite(reset_entropy) or reset_entropy < 0:
            raise ValueError("SAR requires a finite reset_entropy >= 0.")
        self.model = model.train().requires_grad_(False)
        names, params = [], []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            if "layer4" in name or not isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias"):
                    names.append(f"{name}.{param_name}" if name else param_name)
                    params.append(param.requires_grad_(True))
        if not params:
            raise ValueError("SAR requires trainable normalization affine parameters.")
        self.adapted_parameter_names = frozenset(names)
        self.params = params
        self.source_params = [param.detach().clone() for param in params]
        self.optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
        self.rho = rho
        self.margin_fraction = margin_fraction
        self.reset_entropy = reset_entropy
        self.ema = None
        self.steps = self.skipped = self.resets = self.samples = 0

    @staticmethod
    def pixel_entropy(logits, valid):
        pixels = logits.movedim(1, -1)[valid]
        return -(pixels.softmax(1) * pixels.log_softmax(1)).sum(1)

    @torch.enable_grad()
    def __call__(self, image, valid):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(image)
        margin = self.margin_fraction * math.log(logits.shape[1])
        entropy = self.pixel_entropy(logits, valid)
        reliable = entropy < margin
        self.samples += 1
        if not reliable.any():
            self.skipped += 1
            return logits.detach()

        entropy[reliable].mean().backward()
        # SAM's first step ascends along the joint L2-normalized gradient.
        with torch.no_grad():
            perturbed = [(param, param.clone()) for param in self.params if param.grad is not None]
            grad_norm = torch.stack([param.grad.norm(p=2) for param, _ in perturbed]).norm(p=2)
            scale = self.rho / (grad_norm + 1e-12)
            for param, _ in perturbed:
                param.add_(param.grad * scale)

        self.optimizer.zero_grad(set_to_none=True)
        try:
            second_entropy = self.pixel_entropy(self.model(image), valid)[reliable]
            second_reliable = second_entropy < margin
            if not second_reliable.any():
                self.skipped += 1
                return logits.detach()
            loss = second_entropy[second_reliable].mean()
            loss.backward()
        finally:
            with torch.no_grad():
                for param, before in perturbed:
                    param.copy_(before)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        value = loss.item()
        self.ema = value if self.ema is None else 0.9 * self.ema + (1 - 0.9) * value
        self.steps += 1
        if self.ema < self.reset_entropy:
            # Upstream resets the weights and SAM's transient state. Its separate
            # base SGD momentum survives the reset, as does the newly computed EMA.
            with torch.no_grad():
                for param, source in zip(self.params, self.source_params):
                    param.copy_(source)
            self.resets += 1
        return logits.detach()

    def state_summary(self):
        return {"optimizer_sha256": optimizer_digest(self.optimizer), "entropy_ema": self.ema,
                "samples": self.samples, "update_steps": self.steps,
                "skipped_steps": self.skipped, "recovery_resets": self.resets}

    def verify_run(self, start, end, sample_count):
        updates = end["update_steps"] - start["update_steps"]
        skipped = end["skipped_steps"] - start["skipped_steps"]
        resets = end["recovery_resets"] - start["recovery_resets"]
        if (end["samples"] - start["samples"] != sample_count or updates + skipped != sample_count
                or not 0 <= resets <= updates):
            raise RuntimeError("SAR expected one update or skip per target image.")
