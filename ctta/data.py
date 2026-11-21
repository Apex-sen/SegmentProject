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


def oriented_size(path):
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            return height, width
        return width, height


def prepare_manifest(config, run_dir):
    source, targets = config["source"], config["targets"]
    domains = [source, *targets]
    if len(targets) < 2 or len(domains) != len(set(domains)):
        raise ValueError("Choose one source and at least two distinct target domains.")

    root = config["data_root"]
    samples = []
    for domain in domains:
        settings = config["domains"][domain]
        images = index_files(root / settings["images"], IMAGE_EXTENSIONS)
        masks = index_files(root / settings["masks"], MASK_EXTENSIONS)
        if not images or images.keys() != masks.keys():
            missing = sorted(images.keys() - masks.keys())[:5]
            extra = sorted(masks.keys() - images.keys())[:5]
            raise ValueError(f"{domain}: missing masks={missing}, unmatched masks={extra}, images={len(images)}")

        ids = sorted(images)
        if domain == source:
            if len(ids) < 2:
                raise ValueError(f"{domain}: source training and validation both need samples.")
            random.Random(config["seed"]).shuffle(ids)
            fraction = config["train"]["validation_fraction"]
            if not 0 < fraction < 1:
                raise ValueError("validation_fraction must be between zero and one.")
            n_validation = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
            splits = {"train": ids[n_validation:], "val": ids[:n_validation]}
        else:
            splits = {"target": ids}

        for split, split_ids in splits.items():
            limit = config["sampling"][f"max_{split}_samples"]
            if limit:
                split_ids = split_ids[:limit]
            for key in split_ids:
                image, mask = images[key], masks[key]
                if oriented_size(image) != oriented_size(mask):
                    raise ValueError(f"Image and mask dimensions differ: {image}, {mask}")
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


def load_image(path: Path, size):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        original_size = image.size
        width, height, left, top, _, _ = calculate_resize_size(original_size, (size, size))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size))
        canvas.paste(image, (left, top))

    valid = np.zeros((size, size), dtype=bool)
    valid[top:top + height, left:left + width] = True
    return np.array(canvas), valid, (original_size, width, height, left, top)


def load_mask(path: Path, size, geometry, mask_values):
    original_size, width, height, left, top = geometry
    if len(mask_values) != 3 or len(set(mask_values)) != 3 or not all(0 <= v <= 255 for v in mask_values):
        raise ValueError(f"Expected three distinct byte labels for {path}, got {mask_values}")
    with Image.open(path) as original:
        mask = ImageOps.exif_transpose(original)
        if mask.size != original_size or mask.mode not in ("L", "P"):
            raise ValueError(f"Expected an aligned 8-bit mask: {path} ({mask.mode}, {mask.size})")
        values = {color for _, color in mask.getcolors(maxcolors=256)}
        unknown = values.difference(mask_values)
        if unknown:
            raise ValueError(f"Unexpected mask labels in {path}: {sorted(unknown)}")
        mask = np.array(mask.resize((width, height), Image.Resampling.NEAREST))

    mapping = np.zeros(256, dtype=np.uint8)
    mapping[mask_values] = [0, 1, 2]
    target = np.full((size, size), IGNORE, dtype=np.uint8)
    target[top:top + height, left:left + width] = mapping[mask]
    return target


def save_previews(config, run_dir, samples, size):
    directory = run_dir / "previews"
    directory.mkdir(exist_ok=True)
    for domain in [config["source"], *config["targets"]]:
        sample = next(sample for sample in samples if sample.domain == domain)
        root = config["data_root"]
        rgb, valid, geometry = load_image(root / sample.image, size)
        target = load_mask(root / sample.mask, size, geometry, config["domains"][domain]["mask_values"])
        overlay = rgb.copy()
        for label, color in ((1, [40, 220, 130]), (2, [255, 140, 25])):
            region = valid & (target == label)
            overlay[region] = (0.50 * rgb[region] + 0.50 * np.array(color)).astype(np.uint8)
        Image.fromarray(overlay).save(directory / f"{domain}_{sample.sample_id}.png")
