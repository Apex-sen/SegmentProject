import csv
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps

from preprocess import calculate_resize_size


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
IGNORE = 255


@dataclass(frozen=True)
class Sample:
    domain: str
    sample_id: str
    image: str
    mask: str
    split: str


def index_files(directory, extensions):
    files = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.stem.casefold()
        if key in files:
            raise ValueError(f"Repeated sample ID in {directory}: {path.stem}")
        files[key] = path
    return files


def index_domain_files(root, directories, extensions):
    if isinstance(directories, str):
        directories = [directories]
    files = {}
    for name in directories:
        indexed = index_files(root / name, extensions)
        duplicate = files.keys() & indexed.keys()
        if duplicate:
            raise ValueError(f"Repeated sample IDs across directories: {sorted(duplicate)[:5]}")
        files.update((key, indexed[key]) for key in sorted(indexed))
    return files


def oriented_size(path):
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            return height, width
        return width, height


def domain_settings(config, domain):
    settings = config["domains"][domain]
    if domain == config["source"]:
        return {**settings, **settings.get("source_overrides", {})}
    return settings


def split_source(ids, seed, validation_fraction, test_fraction=0):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be non-negative and leave room for training.")
    if not 0 <= test_fraction < 1 - validation_fraction:
        raise ValueError("test_fraction must be non-negative and leave room for training and validation.")
    reserve_validation = int(validation_fraction > 0)
    reserve_test = int(test_fraction > 0)
    if len(ids) < 1 + reserve_validation + reserve_test:
        raise ValueError("The source domain needs at least one sample in each requested split.")

    random.Random(seed).shuffle(ids)
    n_validation = max(1, min(len(ids) - 1 - reserve_test, round(len(ids) * validation_fraction))) if reserve_validation else 0
    n_test = max(1, min(len(ids) - n_validation - 1, round(len(ids) * test_fraction))) if reserve_test else 0
    test_end = n_validation + n_test
    splits = {"train": ids[test_end:]}
    if reserve_validation:
        splits["val"] = ids[:n_validation]
    if reserve_test:
        splits["test"] = ids[n_validation:test_end]
    return splits


def prepare_disc_roi(image, mask, output, roi_size, image_size, mask_values):
    image_file = output / "Images" / f"{image.stem}.png"
    mask_file = output / "Masks" / f"{image.stem}.png"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as original_image, Image.open(mask) as original_mask:
        rgb = ImageOps.exif_transpose(original_image).convert("RGB")
        labels = ImageOps.exif_transpose(original_mask)
        pixels = np.asarray(labels)
        od_region = (pixels == mask_values[1]) | (pixels == mask_values[2])
        y, x = np.nonzero(od_region)
        if not len(x) or max(x.max() - x.min(), y.max() - y.min()) >= roi_size:
            raise ValueError(f"Cannot fit the optic disc into a {roi_size}-pixel crop: {mask}")
        left = int(x.min() + x.max() + 1 - roi_size) // 2
        top = int(y.min() + y.max() + 1 - roi_size) // 2
        box = (left, top, left + roi_size, top + roi_size)
        size = (image_size, image_size)
        rgb.crop(box).resize(size, Image.Resampling.BICUBIC).save(image_file, compress_level=1)
        # Pad using the dataset's background label, including REFUGE's 255.
        cropped_labels = Image.new(labels.mode, (roi_size, roi_size), mask_values[0])
        if labels.mode == "P":
            cropped_labels.putpalette(labels.getpalette())
        inner = (max(0, left), max(0, top), min(labels.width, box[2]), min(labels.height, box[3]))
        cropped_labels.paste(labels.crop(inner), (inner[0] - left, inner[1] - top))
        cropped_labels.resize(size, Image.Resampling.NEAREST).save(mask_file)
    return image_file, mask_file


def prepare_manifest(config, run_dir):
    source, targets = config["source"], config["targets"]
    domains = [source, *targets]
    if len(targets) < 2 or len(domains) != len(set(domains)):
        raise ValueError("Choose one source and at least two distinct target domains.")

    root = config["data_root"]
    roi_size = config.get("benchmark_disc_roi_size", 0)
    samples = []
    for domain in domains:
        settings = domain_settings(config, domain)
        images = index_domain_files(root, settings["images"], IMAGE_EXTENSIONS)
        mask_extensions = MASK_EXTENSIONS if settings.get("mask_encoding", "labels") == "labels" else IMAGE_EXTENSIONS
        masks = index_domain_files(root, settings["masks"], mask_extensions)
        if not images or images.keys() != masks.keys():
            missing = sorted(images.keys() - masks.keys())[:5]
            extra = sorted(masks.keys() - images.keys())[:5]
            raise ValueError(f"{domain}: missing masks={missing}, unmatched masks={extra}, images={len(images)}")
        expected_pairs = settings.get("expected_pairs")
        if expected_pairs is not None and len(images) != expected_pairs:
            raise ValueError(f"{domain}: expected {expected_pairs} image-mask pairs, found {len(images)}.")

        ids = list(images)
        if domain == source:
            training = config["train"]
            splits = split_source(ids, config["seed"], training["validation_fraction"],
                                  training.get("test_fraction", 0))
        else:
            splits = {"target": ids}

        if roi_size:
            print(f"Preparing {domain} optic-disc ROI crops", flush=True)
        for split, split_ids in splits.items():
            limit = config["sampling"].get(f"max_{split}_samples", 0)
            if limit:
                split_ids = split_ids[:limit]
            for key in split_ids:
                image, mask = images[key], masks[key]
                if oriented_size(image) != oriented_size(mask):
                    raise ValueError(f"Image and mask dimensions differ: {image}, {mask}")
                if roi_size:
                    output = root / f"Benchmark_ROI_{roi_size}_{config['image_size']}" / domain
                    image, mask = prepare_disc_roi(image, mask, output, roi_size, config["image_size"],
                                                   settings["mask_values"])
                samples.append(Sample(domain, image.stem, image.relative_to(root).as_posix(),
                                      mask.relative_to(root).as_posix(), split))

    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "manifest.csv", [asdict(sample) for sample in samples])
    return samples


def read_manifest(run_dir):
    with (run_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        return [Sample(**row) for row in csv.DictReader(stream)]


@contextmanager
def csv_writer(path, fieldnames):
    stream = NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            yield writer
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows):
    with csv_writer(path, list(rows[0])) as writer:
        writer.writerows(rows)


def load_image(path: Path, size, *, resize_mode="letterbox"):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        original_size = image.size
        if resize_mode == "stretch":
            width, height, left, top = size, size, 0, 0
            canvas = image.resize((size, size), Image.Resampling.BICUBIC)
        elif resize_mode == "letterbox":
            width, height, left, top, _, _ = calculate_resize_size(original_size, (size, size))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size))
            canvas.paste(image, (left, top))
        else:
            raise ValueError(f"Unknown resize_mode: {resize_mode}")

    valid = np.zeros((size, size), dtype=bool)
    valid[top:top + height, left:left + width] = True
    return np.array(canvas), valid, (original_size, width, height, left, top)


def load_mask(path: Path, size, geometry, mask_values=None, *, encoding="labels", include_labels=False):
    original_size, width, height, left, top = geometry
    labels = None
    with Image.open(path) as original:
        mask = ImageOps.exif_transpose(original)
        if mask.size != original_size:
            raise ValueError(f"Expected mask dimensions {original_size} for {path}, got {mask.size}")
        if encoding == "labels":
            if (mask_values is None or len(mask_values) != 3 or len(set(mask_values)) != 3
                    or not all(0 <= v <= 255 for v in mask_values)):
                raise ValueError(f"Expected three distinct byte labels for {path}, got {mask_values}")
            if mask.mode not in ("L", "P"):
                raise ValueError(f"Expected an 8-bit label mask for {path}, got {mask.mode}")
            values = {color for _, color in mask.getcolors(maxcolors=256)}
            unknown = values.difference(mask_values)
            if unknown:
                raise ValueError(f"Unexpected mask labels in {path}: {sorted(unknown)}")
            mapping = np.zeros(256, dtype=np.uint8)
            mapping[mask_values] = [0, 1, 2]
            pixels = mapping[np.asarray(mask.resize((width, height), Image.Resampling.NEAREST))]
            labels = {label for label, value in enumerate(mask_values) if value in values}
        elif encoding == "binary_gray":
            pixels = np.asarray(mask.resize((width, height), Image.Resampling.NEAREST).convert("L")) >= 128
        elif encoding == "binary_red_green":
            rgb = np.asarray(mask.resize((width, height), Image.Resampling.NEAREST).convert("RGB"))
            pixels = (rgb[..., 0] >= 128) | (rgb[..., 1] >= 128)
        else:
            raise ValueError(f"Unknown mask encoding for {path}: {encoding}")

    target = np.full((size, size), IGNORE, dtype=np.uint8)
    target[top:top + height, left:left + width] = pixels
    if include_labels:
        if labels is None:
            labels = set(np.unique(pixels).astype(int).tolist())
        return target, labels
    return target


def load_domain_mask(path, size, geometry, settings, *, include_labels=False):
    return load_mask(path, size, geometry, settings.get("mask_values"),
                     encoding=settings.get("mask_encoding", "labels"), include_labels=include_labels)


def save_previews(config, run_dir, samples, size):
    directory = run_dir / "previews"
    directory.mkdir(exist_ok=True)
    for domain in [config["source"], *config["targets"]]:
        sample = next(sample for sample in samples if sample.domain == domain)
        root = config["data_root"]
        rgb, valid, geometry = load_image(root / sample.image, size, resize_mode=config.get("resize_mode", "letterbox"))
        target = load_domain_mask(root / sample.mask, size, geometry, domain_settings(config, domain))
        overlay = rgb.copy()
        for label, color in ((1, [40, 220, 130]), (2, [255, 140, 25])):
            region = valid & (target == label)
            overlay[region] = (0.50 * rgb[region] + 0.50 * np.array(color)).astype(np.uint8)
        Image.fromarray(overlay).save(directory / f"{domain}_{sample.sample_id}.png")
