import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict
from itertools import groupby
from statistics import mean

import torch
from torch.utils.data import DataLoader, Dataset

from .data import csv_writer, domain_settings, load_domain_mask, load_image, read_manifest, write_csv
from .methods import METHOD_REGISTRY, SourceOnly, tensor_digest
from .metrics import hd95
from .model import (NUM_CLASSES, STRUCTURES, ResUNet34, binary_overlap, dice_score, image_dice,
                    image_tensor, iou_score, segmentation_loss, structure_masks)


METRIC_COLUMNS = {
    "fundus": ("od_dice", "oc_dice", "mean_dice", "od_iou", "oc_iou", "miou", "mean_image_iou",
               "od_hd95", "oc_hd95", "mean_hd95"),
    "polyp": ("polyp_dice", "polyp_iou", "polyp_hd95"),
}
IMAGE_SCORE_COLUMNS = ("method", "domain", "sample_id", "stream_index")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_mean(scores):
    return None if None in scores else mean(scores)


PREPROCESSING_DEFAULTS = {"resize_mode": "letterbox", "image_normalization": "imagenet",
                          "benchmark_disc_roi_size": 0}


def source_holdout(config, samples):
    split_names = {"train", "test"}
    if config["train"]["validation_fraction"] > 0:
        split_names.add("val")
    rows = [sample for sample in samples if sample.split != "target"]
    if {sample.split for sample in rows} != split_names or any(sample.domain != config["source"] for sample in rows):
        raise ValueError(f"The manifest needs {sorted(split_names)} samples from the current source. Run prepare again.")
    if len({sample.sample_id.casefold() for sample in rows}) != len(rows):
        raise ValueError("Source samples must appear only once across train, val, and test.")
    payload = json.dumps([asdict(sample) for sample in rows], sort_keys=True).encode("utf-8")
    return [sample for sample in rows if sample.split == "test"], hashlib.sha256(payload).hexdigest()


class SourceDataset(Dataset):
    def __init__(self, config, samples):
        self.root = config["data_root"]
        self.size = config["image_size"]
        self.resize_mode = config.get("resize_mode", "letterbox")
        self.normalization = config.get("image_normalization", "imagenet")
        self.settings = domain_settings(config, config["source"])
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb, _, geometry = load_image(self.root / sample.image, self.size, resize_mode=self.resize_mode)
        mask = load_domain_mask(self.root / sample.mask, self.size, geometry, self.settings)
        return image_tensor(rgb, self.normalization), torch.from_numpy(mask).long()


def build_model(config, device, *, imagenet_weights=False):
    model = ResUNet34(imagenet_weights=imagenet_weights, num_classes=NUM_CLASSES[config.get("task", "fundus")])
    memory_format = torch.channels_last if config["experiment"].get("channels_last") else torch.preserve_format
    return model.to(device=device, memory_format=memory_format)


def train(config, run_dir, device, smoke, epochs_override, *, batch_size_override=None):
    seed = config["seed"]
    task = config.get("task", "fundus")
    structures = STRUCTURES[task]
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    samples = read_manifest(run_dir)
    settings = config["train"]
    preprocessing = {key: config.get(key, default) for key, default in PREPROCESSING_DEFAULTS.items()}
    training = SourceDataset(config, [sample for sample in samples if sample.split == "train"])
    validation = SourceDataset(config, [sample for sample in samples if sample.split == "val"])
    if not training or bool(validation) != (settings["validation_fraction"] > 0):
        raise ValueError("The training and validation samples must match the source configuration. Run prepare again.")

    split_metadata = {}
    if settings.get("test_fraction", 0):
        _, split_metadata["source_split_sha256"] = source_holdout(config, samples)
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
    model = build_model(config, device, imagenet_weights=imagenet_weights)
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

        train_loss = total_loss / len(training)
        validation_dice, metrics = {}, {}
        average = None
        if validation:
            model.eval()
            with torch.inference_mode():
                scores = []
                for images, targets in val_loader:
                    predictions = model(images.to(device)).argmax(dim=1).cpu()
                    scores.extend(image_dice(pred, target, structures.values()) for pred, target in zip(predictions, targets))
            validation_dice = dict(zip(structures, map(mean, zip(*scores))))
            average = mean(validation_dice.values())
            metrics = {f"val_{structure}_dice": score for structure, score in validation_dice.items()}
            if task == "fundus":
                metrics["val_mean_dice"] = average
        history.append(dict(epoch=epoch, train_loss=train_loss, **metrics))
        if average is None or average > best_score:
            if average is not None:
                best_score = average
            temporary = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "task": task, "source": config["source"],
                        "image_size": config["image_size"], "seed": seed, "epoch": epoch, **preprocessing,
                        "smoke": smoke, "imagenet_weights": imagenet_weights, **split_metadata, **metrics}, temporary)
            temporary.replace(ckpt_path)
        write_csv(run_dir / "source_training.csv", history)
        names = {"od": "OD", "oc": "OC", "polyp": "Polyp"}
        validation_log = "  ".join(f"val {names[name]} Dice {score:.4f}" for name, score in validation_dice.items())
        print(f"Epoch {epoch}/{epochs}  train loss {train_loss:.4f}  {validation_log}".rstrip(), flush=True)

    print(f"{'Best' if validation else 'Latest'} source checkpoint: {ckpt_path}", flush=True)


def score_stream(config, predict, samples, method, device, per_image):
    model = predict.model
    trainable = predict.adapted_parameter_names
    task = config.get("task", "fundus")
    structures = STRUCTURES[task]

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
    resize_mode = config.get("resize_mode", "letterbox")
    normalization = config.get("image_normalization", "imagenet")
    compute_hd95 = config.get("report", {}).get("compute_hd95", True)
    for domain, domain_samples in groupby(samples, key=lambda sample: sample.domain):
        before = state()
        if boundaries and before != boundaries[-1]["after"]:
            raise RuntimeError(f"{method}: adaptation state changed at the {domain} boundary.")
        settings = domain_settings(config, domain)
        partial_annotations = settings.get("partial_annotations", False)
        count = 0
        scored = [0] * len(structures)
        empty_predictions = [0] * len(structures)
        both_empty = [0] * len(structures)
        dice_sum = [0.0] * len(structures)
        image_iou_sum = [0.0] * len(structures)
        hd95_sum = [0.0] * len(structures)
        intersection_sum = [0] * len(structures)
        union_sum = [0] * len(structures)
        for sample in domain_samples:
            rgb, valid, geometry = load_image(root / sample.image, size, resize_mode=resize_mode)
            image = image_tensor(rgb, normalization).unsqueeze(0).to(device)
            valid_pixels = torch.from_numpy(valid).unsqueeze(0).to(device)
            prediction = predict(image, valid_pixels).argmax(dim=1)[0].cpu()
            target, labels = load_domain_mask(root / sample.mask, size, geometry, settings, include_labels=True)
            scores = [None] * len(structures)
            image_ious = [None] * len(structures)
            distances = [None] * len(structures)
            _, width, height, left, top = geometry
            region = (slice(top, top + height), slice(left, left + width))
            masks = structure_masks(prediction[region], torch.from_numpy(target[region]), structures.values())
            for index, (labels_for_structure, (pred, actual)) in enumerate(zip(structures.values(), masks)):
                if partial_annotations and labels_for_structure[0] not in labels:
                    continue
                intersection, union = binary_overlap(pred, actual)
                score = dice_score(intersection, union)
                image_iou = iou_score(intersection, union)
                distance = hd95(pred.numpy(), actual.numpy()) if compute_hd95 else None
                scores[index] = score
                image_ious[index] = image_iou
                distances[index] = distance
                scored[index] += 1
                empty_predictions[index] += not pred.any().item()
                both_empty[index] += union == 0
                dice_sum[index] += score
                image_iou_sum[index] += image_iou
                if distance is not None:
                    hd95_sum[index] += distance
                intersection_sum[index] += intersection
                union_sum[index] += union
            count += 1
            stream_index += 1
            if per_image is not None:
                row = dict(method=method, domain=domain, sample_id=sample.sample_id, stream_index=stream_index)
                row.update({f"{name}_dice": score for name, score in zip(structures, scores)})
                row.update({f"{name}_iou": score for name, score in zip(structures, image_ious)})
                row.update({f"{name}_hd95": distance for name, distance in zip(structures, distances)})
                if task == "fundus":
                    row["mean_dice"] = structure_mean(scores)
                    row["mean_iou"] = structure_mean(image_ious)
                    row["mean_hd95"] = structure_mean(distances)
                per_image.writerow(row)

        dice = [total / n if n else None for total, n in zip(dice_sum, scored)]
        distances = [total / n if compute_hd95 and n else None for total, n in zip(hd95_sum, scored)]
        iou = [iou_score(intersection, union) if n else None
               for intersection, union, n in zip(intersection_sum, union_sum, scored)]
        domain_counts = {}
        for name, n_scored, n_empty, n_both_empty in zip(structures, scored, empty_predictions, both_empty):
            domain_counts[f"n_{name}"] = n_scored
            domain_counts[f"n_{name}_empty_predictions"] = n_empty
            domain_counts[f"n_{name}_both_empty"] = n_both_empty
            domain_counts[f"n_{name}_misses"] = n_empty - n_both_empty
        after = state()
        domain_counts["n_bn_retries"] = after.get("retry_images", 0) - before.get("retry_images", 0)
        if task == "fundus":
            mean_image_iou = structure_mean([total / n if n else None for total, n in zip(image_iou_sum, scored)])
            domains.append(dict(method=method, domain=domain, n=count,
                                od_dice=dice[0], oc_dice=dice[1], mean_dice=structure_mean(dice),
                                od_iou=iou[0], oc_iou=iou[1], miou=structure_mean(iou), mean_image_iou=mean_image_iou,
                                od_hd95=distances[0], oc_hd95=distances[1], mean_hd95=structure_mean(distances),
                                **domain_counts))
        else:
            domains.append(dict(method=method, domain=domain, n=count, polyp_dice=dice[0], polyp_iou=iou[0],
                                polyp_hd95=distances[0], **domain_counts))
        boundaries.append(dict(domain=domain, n=count, counts=domain_counts, before=before, after=after))

    end = boundaries[-1]["after"]
    if fixed_digest() != fixed_start:
        raise RuntimeError(f"{method}: parameters outside the adaptation set changed.")
    predict.verify_run(start, end, len(samples))
    report = dict(method=method, sample_count=len(samples),
                  adapted_parameters=sorted(trainable), domain_states=boundaries,
                  fixed_parameters_unchanged=True)
    return domains, report


def summarize(domain_scores, task="fundus"):
    summary = []
    for method, method_domains in groupby(domain_scores, key=lambda row: row["method"]):
        domains = list(method_domains)
        summary.extend(domains)
        metrics = {}
        for key in METRIC_COLUMNS[task]:
            scores = [row[key] for row in domains if row[key] is not None]
            metrics[key] = mean(scores) if scores else None
        metrics.update({key: sum(row[key] for row in domains) for key in domains[0] if key.startswith("n_")})
        summary.append(dict(method=method, domain="macro_average", n=sum(row["n"] for row in domains), **metrics))

    source_scores = {row["domain"]: row for row in summary if row["method"] == "source_only"}
    deltas = (("polyp_dice", "delta_vs_source_only"),) if task == "polyp" else (
        ("mean_dice", "delta_vs_source_only"), ("miou", "delta_miou_vs_source_only"))
    for row in summary:
        reference = source_scores.get(row["domain"])
        for metric, delta in deltas:
            baseline = reference[metric] if reference else None
            score = row[metric]
            row[delta] = score - baseline if score is not None and baseline is not None else None
    return summary


def evaluate(config, run_dir, device, smoke, methods, output, *, checkpoint_path=None, eval_split="all"):
    task = config.get("task", "fundus")
    if checkpoint_path is None:
        checkpoint_path = run_dir / "source_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}. Run train or pass --checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_task = checkpoint.get("task", "fundus")
    if checkpoint_task != task:
        raise ValueError(f"Checkpoint task is {checkpoint_task!r}, expected {task!r}.")
    for key, expected in (("source", config["source"]), ("image_size", config["image_size"]), ("smoke", smoke)):
        if checkpoint[key] != expected:
            raise ValueError(f"Checkpoint {key} is {checkpoint[key]!r}, expected {expected!r}.")
    preprocessing = {key: config.get(key, default) for key, default in PREPROCESSING_DEFAULTS.items()}
    for key, default in PREPROCESSING_DEFAULTS.items():
        if checkpoint.get(key, default) != preprocessing[key]:
            raise ValueError(f"The checkpoint was trained with a different {key} than {preprocessing[key]!r}.")
    manifest = read_manifest(run_dir)
    samples = [sample for sample in manifest if sample.split == "target"]
    if eval_split != "source-test":
        if [domain for domain, _ in groupby(samples, key=lambda sample: sample.domain)] != config["targets"]:
            raise ValueError("Target order differs from the config. Run prepare again.")
    print(f"Using source checkpoint: {checkpoint_path.resolve()}", flush=True)

    save_per_image = config["report"]["save_per_image"]
    detail_columns = IMAGE_SCORE_COLUMNS
    for metric in ("dice", "iou", "hd95"):
        detail_columns += tuple(f"{name}_{metric}" for name in STRUCTURES[task])
        if task == "fundus":
            detail_columns += (f"mean_{metric}",)

    source_report = None
    if eval_split != "target" and config["train"].get("test_fraction", 0):
        test_samples, split_digest = source_holdout(config, manifest)
        if checkpoint.get("source_split_sha256") != split_digest:
            raise ValueError("The checkpoint was not trained with this source split. Run train again before source testing.")
        model = build_model(config, device)
        model.load_state_dict(checkpoint["state_dict"])
        source_detail = run_dir / "source_test_per_image.csv"
        with (csv_writer(source_detail, detail_columns) if save_per_image else nullcontext()) as per_image:
            measurements, source_report = score_stream(
                config, SourceOnly(model), test_samples, "source_only", device, per_image)
        source_report["source_split_sha256"] = split_digest
        source_result = run_dir / "source_test.csv"
        write_csv(source_result, measurements)
        source_report["csv_sha256"] = file_sha256(source_result)
        print(f"Source test: {config['source']}  n={len(test_samples)}  Results: {source_result}", flush=True)
        del model
    if eval_split == "source-test":
        return

    output.mkdir(exist_ok=True)
    detail_path = output / "per_image.csv"
    detail_output = csv_writer(detail_path, detail_columns) if save_per_image else nullcontext()
    domain_scores, reports = [], []
    with detail_output as per_image:
        for name, settings in methods.items():
            model = build_model(config, device)
            model.load_state_dict(checkpoint["state_dict"])
            predict = METHOD_REGISTRY[name](model, **settings)
            measurements, report = score_stream(config, predict, samples, name, device, per_image)
            domain_scores.extend(measurements)
            reports.append(report)
            del predict, model
    summary = summarize(domain_scores, task)
    sequence = json.dumps([asdict(sample) for sample in samples], sort_keys=True)
    summary_path = output / "summary.csv"
    write_csv(summary_path, summary)
    write_json(output / "verification.json", dict(
        checkpoint_sha256=file_sha256(checkpoint_path), summary_csv_sha256=file_sha256(summary_path),
        target_stream_sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        task=task, source=config["source"], targets=config["targets"], image_size=config["image_size"],
        **preprocessing,
        device=str(device), torch_version=torch.__version__, smoke=smoke,
        shared_target_stream=True, method_settings=methods, methods=reports,
        source_test=source_report,
        per_image_csv=detail_path.name if save_per_image else None,
        hd95=dict(enabled=config["report"].get("compute_hd95", True),
                  unit="px", percentile=95, direction_reduction="max", connectivity=4,
                  single_empty="valid_image_diagonal", both_empty=0.0),
    ))
    if task == "polyp":
        fields = (("Polyp Dice", "polyp_dice"), ("Polyp IoU", "polyp_iou"), ("Polyp HD95 (px)", "polyp_hd95"))
    else:
        fields = (("OD Dice", "od_dice"), ("OC Dice", "oc_dice"), ("Mean Dice", "mean_dice"),
                  ("OD IoU", "od_iou"), ("OC IoU", "oc_iou"), ("mIoU", "miou"),
                  ("Per-image IoU", "mean_image_iou"),
                  ("OD HD95 (px)", "od_hd95"), ("OC HD95 (px)", "oc_hd95"), ("Mean HD95 (px)", "mean_hd95"))
    domain_width = max(len(row["domain"]) for row in summary)
    for row in summary:
        metrics = "  ".join(f"{label} {row[key]:.4f}" if row[key] is not None else f"{label} N/A"
                            for label, key in fields)
        print(f"{row['method']:16s}  {row['domain']:<{domain_width}}  n={row['n']:<4d}  {metrics}", flush=True)
    print(f"Results: {output}", flush=True)
