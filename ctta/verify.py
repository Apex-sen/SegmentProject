import hashlib
import runpy
from copy import deepcopy

import torch
from torch import nn

from .experiment import write_json
from .methods import SourceOnly, TentContinual, entropy_loss, tensor_digest
from .model import image_dice


def verify_reference(reference_path, run_dir):
    reference = runpy.run_path(str(reference_path.resolve()))
    torch.manual_seed(42)
    source = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                           nn.ReLU(), nn.Conv2d(4, 3, 1))
    model = deepcopy(source)
    tent = TentContinual(model, learning_rate=1e-4, momentum=0.9)
    reference_model = reference["configure_model"](deepcopy(source))
    params, names = reference["collect_params"](reference_model)
    assert set(names) == tent.adapted_parameter_names
    reference_optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9)
    valid = torch.ones(1, 9, 11, dtype=torch.bool)
    valid[:, :2, :] = False
    source_predict = SourceOnly(source)
    source_before = tensor_digest(source.state_dict().items())
    before = tensor_digest(model.state_dict().items())
    max_logit_difference = 0.0
    max_parameter_difference = 0.0

    for _ in range(2):
        image = torch.randn(1, 3, 9, 11)
        source_predict(image, valid)

        def reference_forward(x):
            # The official update averages [N, C] classification logits.
            # Treat the valid pixel predictions as that batch for the reference loss.
            return reference_model(x).movedim(1, -1)[valid]

        with torch.no_grad():
            expected_loss = reference["softmax_entropy"](reference_forward(image)).mean()
            actual_loss = entropy_loss(model(image), valid)
        torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-5)

        expected = reference["forward_and_adapt"](image, reference_forward, reference_optimizer).detach()
        actual = tent(image, valid).movedim(1, -1)[valid]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        max_logit_difference = max(max_logit_difference, (actual - expected).abs().max().item())

        for name, param in model.named_parameters():
            other = reference_model.get_parameter(name)
            torch.testing.assert_close(param, other, atol=1e-6, rtol=1e-5)
            if name not in names:
                torch.testing.assert_close(param, source.get_parameter(name), atol=0, rtol=0)
            max_parameter_difference = max(max_parameter_difference, (param - other).abs().max().item())
        torch.testing.assert_close(tent.optimizer.state_dict(), reference_optimizer.state_dict(),
                                   atol=1e-6, rtol=1e-5)

    assert tensor_digest(source.state_dict().items()) == source_before
    assert tensor_digest(model.state_dict().items()) != before
    assert tent.steps == 2
    # Padding is excluded and cup pixels count toward the full disc.
    assert image_dice(torch.tensor([[1, 2, 2]]), torch.tensor([[1, 2, 255]])) == [1.0, 1.0]
    assert image_dice(torch.tensor([[1, 1, 2]]), torch.tensor([[1, 2, 255]])) == [1.0, 0.0]

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "verification_reference.json"
    write_json(path, dict(
        status="passed", steps=tent.steps, max_logit_difference=max_logit_difference,
        max_parameter_difference=max_parameter_difference,
        source_state_unchanged=True, non_bn_parameters_unchanged=True,
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        torch_version=torch.__version__, device="cpu",
    ))
    print(f"Official TENT comparison passed: {path}", flush=True)
