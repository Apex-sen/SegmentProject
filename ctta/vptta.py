import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .method_state import optimizer_digest, tensor_digest


class WarmupBatchNorm(nn.BatchNorm2d):
    """VPTTA OPTIC AdaBN with the local checkpoint's affine and source statistics."""

    def __init__(self, source):
        super().__init__(source.num_features, eps=source.eps, device=source.weight.device,
                         dtype=source.weight.dtype)
        self.load_state_dict(source.state_dict())
        self.target_weight = 1.0
        self.alignment_loss = None

    def forward(self, x):
        weight = self.target_weight
        target = x.detach()
        source_mean = self.running_mean[None, :, None, None]
        source_var = self.running_var[None, :, None, None]
        mean = weight * target.mean((0, 2, 3), keepdim=True) + (1 - weight) * source_mean
        var = weight * target.var((0, 2, 3), keepdim=True) + (1 - weight) * source_var
        if torch.is_grad_enabled():
            self.alignment_loss = ((mean - x.mean((2, 3), keepdim=True)).abs().mean()
                                   + (var.sqrt() - x.std((2, 3), keepdim=True)).abs().mean())
        normalized = (x - mean) / (var + self.eps).sqrt()
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class VPTTA:
    """Port of Chen-Ziyang/VPTTA's prompt, AdaBN and memory loop."""

    adapted_parameter_names = frozenset()

    def __init__(self, model, learning_rate=0.05, beta1=0.9, beta2=0.99,
                 prompt_alpha=0.01, warmup=5, memory_size=40, neighbors=16, iterations=1,
                 encoder_only=False):
        if not (0 < prompt_alpha <= 1 and math.isfinite(warmup) and warmup > 0
                and type(memory_size) is int and type(neighbors) is int and 1 <= neighbors <= memory_size
                and type(iterations) is int and iterations >= 1):
            raise ValueError("VPTTA requires 0 < prompt_alpha <= 1, warmup > 0, "
                             "1 <= neighbors <= memory_size, and at least one iteration.")
        self.model = model
        self.norms = []
        encoder_blocks = {"stem", "layer1", "layer2", "layer3", "layer4"}
        for name, norm in list(model.named_modules()):
            if isinstance(norm, nn.BatchNorm2d) and (not encoder_only or name.split(".", 1)[0] in encoder_blocks):
                if not name:
                    raise ValueError("VPTTA cannot replace the root module.")
                replacement = WarmupBatchNorm(norm)
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, replacement)
                self.norms.append(replacement)
        if not self.norms:
            raise ValueError("VPTTA requires BatchNorm2d layers with source statistics.")
        self.model.eval().requires_grad_(False)
        self.prompt_alpha = prompt_alpha
        self.warmup = warmup
        self.memory_size = memory_size
        self.neighbors = neighbors
        self.iterations = iterations
        self.memory = {}
        self.prompt = self.optimizer = None
        self.adam_options = dict(lr=learning_rate, betas=(beta1, beta2))
        self.samples = self.steps = self.memory_hits = 0

    def initial_prompt(self, key):
        if len(self.memory) < self.neighbors:
            return torch.ones_like(self.prompt)

        keys = list(self.memory)
        keys_array = np.frombuffer(b"".join(keys), dtype=np.float32).reshape(len(keys), key.size)

        def cosine_similarity(candidates):
            denominator = np.linalg.norm(candidates, axis=1) * np.linalg.norm(key)
            return np.divide(np.dot(candidates, key), denominator, out=np.zeros_like(denominator),
                             where=denominator != 0)

        similarity = cosine_similarity(keys_array)
        indices = np.argpartition(similarity, -self.neighbors)[-self.neighbors:]
        weights = np.exp(cosine_similarity(keys_array[indices]) / 0.2)
        weights /= weights.sum()
        result = self.memory[keys[indices[0]]] * weights[0]
        for index, weight in zip(indices[1:], weights[1:]):
            result += self.memory[keys[index]] * weight
        self.memory_hits += 1
        return torch.from_numpy(result).unsqueeze(0)

    @torch.enable_grad()
    def __call__(self, image, valid):
        height, width = image.shape[-2:]
        if image.shape[:2] != (1, 3) or height != width:
            raise ValueError("VPTTA expects one square RGB input at a time.")
        size = max(1, int(height * self.prompt_alpha))
        if self.prompt is None:
            self.prompt = nn.Parameter(image.new_ones((1, 3, size, size)))
            self.optimizer = torch.optim.Adam([self.prompt], **self.adam_options)
        if size != self.prompt.shape[-1]:
            raise ValueError("VPTTA requires a fixed prompt size throughout the test stream.")

        start = (height - size) // 2
        end = height - start - size
        spectrum = torch.fft.fft2(image.detach(), dim=(-2, -1))
        amplitude = torch.fft.fftshift(spectrum.abs())
        phase = spectrum.angle()
        cos, sin = phase.cos(), phase.sin()
        key = amplitude[:, :, start:start + size, start:start + size].reshape(-1).cpu().numpy()

        def prompted_image():
            multiplier = F.pad(self.prompt, (start, end, start, end), value=1.0)
            adjusted = torch.fft.ifftshift(amplitude * multiplier)
            return torch.fft.ifft2(torch.complex(cos * adjusted, sin * adjusted), dim=(-2, -1)).real

        with torch.no_grad():
            self.prompt.copy_(self.initial_prompt(key))
        target_weight = 1 / (math.sqrt(self.samples + 1) / self.warmup + 1)
        for norm in self.norms:
            norm.target_weight = target_weight

        for _ in range(self.iterations):
            self.optimizer.zero_grad(set_to_none=True)
            self.model(prompted_image())
            loss = sum(norm.alignment_loss for norm in self.norms) / len(self.norms)
            for norm in self.norms:
                norm.alignment_loss = None
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite BN alignment loss in VPTTA.")
            loss.backward()
            self.optimizer.step()
            self.steps += 1

        with torch.no_grad():
            logits = self.model(prompted_image())
        # Keep the public OPTIC implementation's strict > FIFO check before insertion.
        if len(self.memory) > self.memory_size:
            self.memory.pop(next(iter(self.memory)))
        self.memory[key.tobytes()] = self.prompt[0].detach().cpu().numpy().copy()
        self.samples += 1
        return logits

    def state_summary(self):
        return {"samples": self.samples, "update_steps": self.steps, "memory_entries": len(self.memory),
                "memory_hits": self.memory_hits,
                "memory_sha256": tensor_digest((key.hex(), torch.from_numpy(value)) for key, value in self.memory.items()),
                "prompt_sha256": tensor_digest([("prompt", self.prompt)]) if self.prompt is not None else None,
                "optimizer_sha256": optimizer_digest(self.optimizer) if self.optimizer is not None else None}

    def verify_run(self, start, end, sample_count):
        if start["model_sha256"] != end["model_sha256"]:
            raise RuntimeError("VPTTA changed the source model parameters or BN buffers.")
        if (end["samples"] - start["samples"] != sample_count
                or end["update_steps"] - start["update_steps"] != self.iterations * sample_count):
            raise RuntimeError("VPTTA did not complete the configured prompt updates.")
