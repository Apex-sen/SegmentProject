import math
from copy import deepcopy

import torch
from torch.nn import functional as F

from .method_state import optimizer_digest, tensor_digest, use_current_batch_stats


class CoTTA:
    """Port of qinenergy/cotta's online update with the paper's segmentation views."""

    def __init__(self, model, learning_rate=0.0000075, teacher_decay=0.999, restore_probability=0.01,
                 confidence_threshold=0.69, scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), seed=42,
                 bn_stats="source"):
        if not (0 <= teacher_decay <= 1 and 0 <= restore_probability <= 1 and 0 <= confidence_threshold <= 1):
            raise ValueError("CoTTA probabilities and EMA decay must be between zero and one.")
        if not scales or not all(math.isfinite(scale) and scale > 0 for scale in scales):
            raise ValueError("CoTTA requires positive, finite augmentation scales.")
        if bn_stats not in ("source", "current_batch"):
            raise ValueError("CoTTA bn_stats must be source or current_batch.")
        self.model = model.eval().requires_grad_(True)
        self.anchor = deepcopy(model).requires_grad_(False)
        self.teacher = deepcopy(model).requires_grad_(False)
        if bn_stats == "current_batch":
            for network in (self.model, self.teacher, self.anchor):
                use_current_batch_stats(network)
        self.adapted_parameter_names = frozenset(name for name, _ in model.named_parameters())
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.teacher_decay = teacher_decay
        self.restore_probability = restore_probability
        self.confidence_threshold = confidence_threshold
        self.scales = scales
        self.restore_rng = torch.Generator().manual_seed(seed)
        self.steps = self.augmented_samples = 0

    def augmented_probabilities(self, image, standard):
        output_size = image.shape[-2:]
        total = torch.zeros_like(standard)
        for scale in self.scales:
            size = tuple(max(1, round(side * scale)) for side in output_size)
            view = image if size == output_size else F.interpolate(
                image, size=size, mode="bilinear", align_corners=False)
            flipped_logits = self.teacher(view.flip((-1,))).flip((-1,))
            if size == output_size:
                total.add_(standard)
            else:
                logits = F.interpolate(self.teacher(view), size=output_size, mode="bilinear", align_corners=False)
                total.add_(logits.softmax(1))
                flipped_logits = F.interpolate(flipped_logits, size=output_size, mode="bilinear", align_corners=False)
            total.add_(flipped_logits.softmax(1))
        return total / (2 * len(self.scales))

    @torch.enable_grad()
    def __call__(self, image, valid):
        if not valid.any():
            raise ValueError("CoTTA requires at least one valid pixel per update.")
        with torch.no_grad():
            confidence = self.anchor(image).softmax(1).amax(1)
            low_confidence = confidence <= self.confidence_threshold
            target = self.teacher(image).softmax(1)
            if (low_confidence & valid).any():
                averaged = self.augmented_probabilities(image, target)
                target = torch.where(low_confidence.unsqueeze(1), averaged, target)
                self.augmented_samples += 1

        self.optimizer.zero_grad(set_to_none=True)
        student_pixels = self.model(image).movedim(1, -1)[valid]
        teacher_pixels = target.movedim(1, -1)[valid]
        loss = F.cross_entropy(student_pixels, teacher_pixels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite distillation loss in CoTTA.")
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
                teacher.copy_(self.teacher_decay * teacher + (1 - self.teacher_decay) * student)
            if self.restore_probability > 0:
                for student, anchor in zip(self.model.parameters(), self.anchor.parameters()):
                    mask = torch.rand(student.shape, generator=self.restore_rng,
                                      device="cpu") < self.restore_probability
                    student.copy_(torch.where(mask.to(student.device), anchor, student))
        self.steps += 1
        return target

    def state_summary(self):
        return {"optimizer_sha256": optimizer_digest(self.optimizer), "update_steps": self.steps,
                "augmented_samples": self.augmented_samples,
                "teacher_sha256": tensor_digest(self.teacher.state_dict().items()),
                "anchor_sha256": tensor_digest(self.anchor.state_dict().items()),
                "restore_rng_sha256": tensor_digest([("rng", self.restore_rng.get_state())])}

    def verify_run(self, start, end, sample_count):
        if end["update_steps"] - start["update_steps"] != sample_count:
            raise RuntimeError("Expected one CoTTA update for each target image.")
        if end["anchor_sha256"] != start["anchor_sha256"]:
            raise RuntimeError("CoTTA changed the source anchor.")
