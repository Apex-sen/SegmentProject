from pathlib import Path
from shutil import copy2

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent / "data" / "FundData"


def export_pair(image, od_path, oc_path, output, threshold, left_view=False):
    images = output / "Images"
    masks = output / "Masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    with Image.open(image) as rgb, Image.open(od_path) as od_img, Image.open(oc_path) as oc_img:
        if rgb.size != od_img.size or rgb.size != oc_img.size:
            raise ValueError(f"图像和标注的尺寸不一致: {image}")

        if left_view:
            box = (0, 0, rgb.width // 2, rgb.height)
            rgb, od_img, oc_img = (item.crop(box) for item in (rgb, od_img, oc_img))

        od = np.asarray(od_img) >= threshold
        oc = np.asarray(oc_img) >= threshold
        if not od.any() or not oc.any() or np.any(oc & ~od):
            raise ValueError(f"请检查视盘或视杯标注: {image}")

        mask = np.zeros(od.shape, dtype=np.uint8)
        mask[od] = 1
        mask[oc] = 2

        if left_view:
            rgb.save(images / f"{image.stem}.png")
        else:
            copy2(image, images / image.name)
        indexed_mask = Image.fromarray(mask)
        # Display distinct colors while keeping the pixel indices used by the loader.
        indexed_mask.putpalette([
            0, 0, 0,       # Background
            40, 220, 130,  # Optic disc outside the cup
            255, 140, 25,  # Optic cup
        ])
        indexed_mask.save(masks / f"{image.stem}.png", bits=8)


def prepare_drishti(split):
    drishti = root / "Drishti-GS"
    archive, annotations, output = {
        "train": ("Training-20211018T055246Z-001/Training", "GT", "CTTA_Train"),
        "test": ("Test-20211018T060000Z-001/Test", "Test_GT", "CTTA_Test"),
    }[split]
    folder = drishti / archive
    for image in sorted((folder / "Images").glob("*/*.png")):
        name = image.stem
        gt = folder / annotations / name / "SoftMap"
        export_pair(image, gt / f"{name}_ODsegSoftmap.png", gt / f"{name}_cupsegSoftmap.png",
                    drishti / output, threshold=191)


def prepare_rim():
    rim = root / "RIM-ONE-r3"
    for group in ("Healthy", "Glaucoma and suspects"):
        folder = rim / "RIM-ONE r3" / group
        for image in sorted((folder / "Stereo Images").glob("*.jpg")):
            name = image.stem
            gt = folder / "Average_masks"
            export_pair(image, gt / f"{name}-Disc-Avg.png", gt / f"{name}-Cup-Avg.png",
                        rim / "CTTA", threshold=1, left_view=True)


if __name__ == "__main__":
    prepare_drishti("train")
    prepare_drishti("test")
    prepare_rim()
