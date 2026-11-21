import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict
from itertools import groupby
from statistics import mean

import torch
from torch.utils.data import DataLoader, Dataset

from .data import csv_writer, load_image, load_mask, read_manifest, write_csv
from .methods import METHOD_REGISTRY, tensor_digest
from .model import ResUNet34, dice_score, image_dice, image_tensor, overlap_counts, segmentation_loss


METRIC_COLUMNS = ("od_dice", "oc_dice", "mean_dice", "od_iou", "oc_iou", "miou")
IMAGE_SCORE_COLUMNS = ("method", "domain", "sample_id", "stream_index", "od_dice", "oc_dice", "mean_dice")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class SourceDataset(Dataset):
    def __init__(self, config, samples):
        self.root = config["data_root"]
        self.size = config["image_size"]
        self.mask_values = config["domains"][config["source"]]["mask_values"]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb, _, geometry = load_image(self.root / sample.image, self.size)
        mask = load_mask(self.root / sample.mask, self.size, geometry, self.mask_values)
        return image_tensor(rgb), torch.from_numpy(mask).long()


def train(config, run_dir, device, smoke, epochs_override, *, batch_size_override=None):
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    samples = read_manifest(run_dir)
    training = SourceDataset(config, [sample for sample in samples if sample.split == "train"])
    validation = SourceDataset(config, [sample for sample in samples if sample.split == "val"])
    if not len(training) or not len(validation):
        raise ValueError("The manifest needs both source training and source validation samples.")

    settings = config["train"]
    epochs = epochs_override if epochs_override is not None else 1 if smoke else settings["epochs"]
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    batch_size = 2 if smoke else settings["batch_size"]
    if batch_size_override is not None:
        batch_size = batch_size_override
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=settings["batch_size"])
    imagenet_weights = settings["imagenet_weights"] and not smoke
    model = ResUNet34(imagenet_weights=imagenet_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    ckpt_path = run_dir / "source_best.pt"
    best_score = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for image, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite source training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(image)

        model.eval()
        with torch.inference_mode():
            scores = []
            for images, targets in val_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                scores.extend(image_dice(pred, target) for pred, target in zip(predictions, targets))
        od, oc = map(mean, zip(*scores))
        average = (od + oc) / 2
        history.append(dict(epoch=epoch, train_loss=total_loss / len(training),
                            val_od_dice=od, val_oc_dice=oc, val_mean_dice=average))
        if average > best_score:
            best_score = average
            temporary = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "source": config["source"],
                        "image_size": config["image_size"], "seed": seed, "epoch": epoch,
                        "val_mean_dice": average, "smoke": smoke,
                        "imagenet_weights": imagenet_weights}, temporary)
            temporary.replace(ckpt_path)
        write_csv(run_dir / "source_training.csv", history)
        print(f"Epoch {epoch}/{epochs}  val OD {od:.4f}  val OC {oc:.4f}", flush=True)

    print(f"Best source checkpoint: {ckpt_path}", flush=True)


def score_stream(config, predict, samples, method, device, per_image):
    model = predict.model
    trainable = predict.adapted_parameter_names

    def state():
        return {"model_sha256": tensor_digest(model.state_dict().items()), **predict.state_summary()}

    def fixed_digest():
        return tensor_digest((name, param) for name, param in model.named_parameters()
                             if name not in trainable)

    start, fixed_start = state(), fixed_digest()
    domains, boundaries = [], []
    stream_index = 0
    root = config["data_root"]
    size = config["image_size"]
    for domain, domain_samples in groupby(samples, key=lambda sample: sample.domain):
        before = state()
        if boundaries and before != boundaries[-1]["after"]:
            raise RuntimeError(f"{method}: adaptation state changed at the {domain} boundary.")
        count = 0
        dice_sum = [0.0, 0.0]
        intersection_sum = [0, 0]
        union_sum = [0, 0]
        for sample in domain_samples:
            rgb, valid, geometry = load_image(root / sample.image, size)
            image = image_tensor(rgb).unsqueeze(0).to(device)
            valid_pixels = torch.from_numpy(valid).unsqueeze(0).to(device)
            prediction = predict(image, valid_pixels).argmax(dim=1)[0].cpu()
            target = load_mask(root / sample.mask, size, geometry, config["domains"][domain]["mask_values"])
            scores = []
            for index, (intersection, union) in enumerate(overlap_counts(prediction, torch.from_numpy(target))):
                score = dice_score(intersection, union)
                scores.append(score)
                dice_sum[index] += score
                intersection_sum[index] += intersection
                union_sum[index] += union
            count += 1
            stream_index += 1
            if per_image is not None:
                od, oc = scores
                per_image.writerow(dict(method=method, domain=domain, sample_id=sample.sample_id,
                                         stream_index=stream_index, od_dice=od, oc_dice=oc,
                                         mean_dice=(od + oc) / 2))

        od, oc = [total / count for total in dice_sum]
        od_iou, oc_iou = [intersection / union if union else 1.0
                          for intersection, union in zip(intersection_sum, union_sum)]
        domains.append(dict(method=method, domain=domain, n=count,
                            od_dice=od, oc_dice=oc, mean_dice=(od + oc) / 2,
                            od_iou=od_iou, oc_iou=oc_iou, miou=(od_iou + oc_iou) / 2))
        boundaries.append(dict(domain=domain, before=before, after=state()))

    end = boundaries[-1]["after"]
    if fixed_digest() != fixed_start:
        raise RuntimeError(f"{method}: parameters outside the adaptation set changed.")
    predict.verify_run(start, end, len(samples))
    report = dict(method=method, sample_count=len(samples),
                  adapted_parameters=sorted(trainable), domain_states=boundaries,
                  fixed_parameters_unchanged=True)
    return domains, report


def summarize(domain_scores):
    summary = []
    for method, method_domains in groupby(domain_scores, key=lambda row: row["method"]):
        domains = list(method_domains)
        summary.extend(domains)
        summary.append(dict(method=method, domain="macro_average", n=sum(row["n"] for row in domains),
                            **{key: mean(row[key] for row in domains) for key in METRIC_COLUMNS}))

    source_scores = {row["domain"]: row for row in summary if row["method"] == "source_only"}
    for row in summary:
        reference = source_scores.get(row["domain"])
        row["delta_vs_source_only"] = row["mean_dice"] - reference["mean_dice"] if reference else None
        row["delta_miou_vs_source_only"] = row["miou"] - reference["miou"] if reference else None
    return summary


def evaluate(config, run_dir, device, smoke, methods, output):
    checkpoint_path = run_dir / "source_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, expected in (("source", config["source"]), ("image_size", config["image_size"]), ("smoke", smoke)):
        if checkpoint[key] != expected:
            raise ValueError(f"Checkpoint {key} is {checkpoint[key]!r}, expected {expected!r}.")
    samples = [sample for sample in read_manifest(run_dir) if sample.split == "target"]
    if [domain for domain, _ in groupby(samples, key=lambda sample: sample.domain)] != config["targets"]:
        raise ValueError("Target order differs from the config. Run prepare again.")

    output.mkdir(exist_ok=True)
    save_per_image = config["report"]["save_per_image"]
    detail_path = output / "per_image.csv"
    detail_output = csv_writer(detail_path, IMAGE_SCORE_COLUMNS) if save_per_image else nullcontext()
    domain_scores, reports = [], []
    with detail_output as per_image:
        for name, settings in methods.items():
            model = ResUNet34().to(device)
            model.load_state_dict(checkpoint["state_dict"])
            predict = METHOD_REGISTRY[name](model, **settings)
            measurements, report = score_stream(config, predict, samples, name, device, per_image)
            domain_scores.extend(measurements)
            reports.append(report)
    summary = summarize(domain_scores)
    sequence = json.dumps([asdict(sample) for sample in samples], sort_keys=True)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    write_csv(output / "summary.csv", summary)
    write_json(output / "verification.json", dict(
        checkpoint_sha256=digest.hexdigest(),
        target_stream_sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        source=config["source"], targets=config["targets"], image_size=config["image_size"],
        device=str(device), torch_version=torch.__version__, smoke=smoke,
        shared_target_stream=True, method_settings=methods, methods=reports,
        per_image_csv=detail_path.name if save_per_image else None,
    ))
    for row in summary:
        if row["domain"] == "macro_average":
            print(f"{row['method']:16s} n={row['n']}  OD Dice {row['od_dice']:.4f}  "
                  f"OC Dice {row['oc_dice']:.4f}  Mean Dice {row['mean_dice']:.4f}  "
                  f"mIoU {row['miou']:.4f}", flush=True)
    print(f"Results: {output}", flush=True)
