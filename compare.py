
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
)


# ============================================================
# 1. 数据集路径配置：只需要修改这里
# ============================================================

DATASETS = {
    "PAPILA": {
        # 我们自己转换后的数据
        "our_image_dir": r"data/GAMMA/Images_Square",
        "our_mask_dir": r"data/GAMMA/Masks_Square",

        # 官方提供的 512×512 数据
        "official_image_dir": r"data/FundData/ORIGA/Images_Square",
        "official_mask_dir": r"data/FundData/ORIGA/Masks_Square",

        # 比较结果输出位置
        "output_dir": r"data/GAMMA",
    },

    # 其他数据集可以继续添加
    # "Drishti-GS": {
    #     "our_image_dir": r"...",
    #     "our_mask_dir": r"...",
    #     "official_image_dir": r"...",
    #     "official_mask_dir": r"...",
    #     "output_dir": r"...",
    # },
}


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

MASK_EXTENSIONS = {
    ".png", ".bmp", ".tif", ".tiff"
}

# 默认黑色/数值 0 为背景。
# 如果官方 Mask 使用其他背景编码，需要根据实际标签修改。
BACKGROUND_LABEL = 0

# 差异图放大倍数
DIFF_SCALE = 4


# ============================================================
# 2. 文件读取与匹配
# ============================================================

def collect_files(directory, extensions):
    """递归读取目录下的图像文件，以相对路径（不含扩展名）为键。"""

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")

    files = {}

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        key = path.relative_to(directory).with_suffix("").as_posix()

        if key in files:
            raise ValueError(
                f"发现重复文件标识：{key}\n"
                f"文件1：{files[key]}\n"
                f"文件2：{path}"
            )

        files[key] = path

    return files


def match_files(our_dir, official_dir, extensions):
    """匹配两套数据中的同名文件，不改变文件内容。"""

    our_files = collect_files(our_dir, extensions)
    official_files = collect_files(official_dir, extensions)

    our_keys = set(our_files)
    official_keys = set(official_files)

    common_keys = sorted(our_keys & official_keys)

    missing_official = sorted(our_keys - official_keys)
    missing_ours = sorted(official_keys - our_keys)

    return (
        common_keys,
        our_files,
        official_files,
        missing_official,
        missing_ours,
    )


# ============================================================
# 3. Image 比较
# ============================================================

def calculate_image_metrics(our_path, official_path):
    """
    直接比较我们转换的 Image 和官方 Image。
    不执行 Resize、Crop、Padding 等操作。
    """

    with Image.open(our_path) as im:
        our_image = np.asarray(im.convert("RGB"))

    with Image.open(official_path) as im:
        official_image = np.asarray(im.convert("RGB"))

    # 严格检查尺寸
    if our_image.shape != official_image.shape:
        raise ValueError(
            f"Image 尺寸不一致："
            f"ours={our_image.shape}, "
            f"official={official_image.shape}"
        )

    our_float = our_image.astype(np.float64)
    official_float = official_image.astype(np.float64)

    difference = np.abs(our_float - official_float)

    mae = float(np.mean(difference))
    rmse = float(np.sqrt(np.mean(difference ** 2)))

    psnr = float(
        peak_signal_noise_ratio(
            official_image,
            our_image,
            data_range=255,
        )
    )

    ssim = float(
        structural_similarity(
            official_image,
            our_image,
            channel_axis=2,
            data_range=255,
        )
    )

    exact_match = bool(np.array_equal(our_image, official_image))

    return {
        "width": our_image.shape[1],
        "height": our_image.shape[0],
        "MAE": mae,
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": ssim,
        "Exact_Match": exact_match,
    }, our_image, official_image


# ============================================================
# 4. Mask 标签处理
# ============================================================

def load_mask(path):
    """
    保留 Mask 原始像素编码：
    - 灰度图：每个像素是一个标签值
    - P 模式：保留 palette index
    - RGB 图：每个像素由 RGB 三元组表示
    """

    with Image.open(path) as image:
        return np.asarray(image).copy()


def encode_labels(mask):
    """
    将每个像素转换成可比较的标签标识。
    RGB Mask 使用颜色三元组，不会把 RGB 通道误认为三个类别。
    """

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        return mask

    raise ValueError(f"不支持的 Mask 维度：{mask.shape}")


def get_background_label(mask):
    """根据 Mask 类型返回背景标签。"""

    if mask.ndim == 2:
        return BACKGROUND_LABEL

    if mask.ndim == 3:
        if isinstance(BACKGROUND_LABEL, tuple):
            return BACKGROUND_LABEL

        return tuple([BACKGROUND_LABEL] * mask.shape[2])

    raise ValueError(f"不支持的 Mask 形状：{mask.shape}")


def label_counts(mask):
    """统计各标签像素数量。"""

    encoded = encode_labels(mask)

    if encoded.ndim == 2:
        values, counts = np.unique(
            encoded,
            return_counts=True,
        )
        return {
            int(v): int(c)
            for v, c in zip(values, counts)
        }

    pixels = encoded.reshape(-1, encoded.shape[-1])

    values, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True,
    )

    return {
        tuple(int(x) for x in v): int(c)
        for v, c in zip(values, counts)
    }


# ============================================================
# 5. Mask 比较
# ============================================================

def calculate_mask_metrics(our_path, official_path):
    """
    比较我们转换的 Mask 和官方 Mask。
    不进行 Resize，不改变标签编码。
    """

    our_mask = load_mask(our_path)
    official_mask = load_mask(official_path)

    if our_mask.shape != official_mask.shape:
        raise ValueError(
            f"Mask 尺寸/通道不一致："
            f"ours={our_mask.shape}, "
            f"official={official_mask.shape}"
        )

    our_labels = encode_labels(our_mask)
    official_labels = encode_labels(official_mask)

    # 如果两边的标签编码不同，直接报告不一致
    if our_labels.shape != official_labels.shape:
        raise ValueError("Mask 标签数组形状不一致")

    our_counts = label_counts(our_labels)
    official_counts = label_counts(official_labels)

    our_background = get_background_label(our_mask)
    official_background = get_background_label(official_mask)

    if our_background != official_background:
        raise ValueError(
            f"背景编码不一致：ours={our_background}, "
            f"official={official_background}。"
            "请先确认两边的背景标签定义。"
        )

    background = our_background

    # 像素准确率
    pixel_accuracy = float(
        np.mean(
            np.all(our_labels == official_labels, axis=-1)
            if our_labels.ndim == 3
            else our_labels == official_labels
        )
    )

    # 所有出现过的标签
    all_labels = set(our_counts) | set(official_counts)

    class_results = []
    dice_values = []
    iou_values = []

    for label in sorted(all_labels, key=str):
        if label == background:
            continue

        if our_labels.ndim == 2:
            pred = our_labels == label
            target = official_labels == label
        else:
            label_array = np.asarray(label)
            pred = np.all(our_labels == label_array, axis=-1)
            target = np.all(official_labels == label_array, axis=-1)

        intersection = int(np.logical_and(pred, target).sum())
        pred_area = int(pred.sum())
        target_area = int(target.sum())

        union = int(np.logical_or(pred, target).sum())

        if pred_area + target_area == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / (pred_area + target_area)

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        dice_values.append(dice)
        iou_values.append(iou)

        class_results.append({
            "class": str(label),
            "our_pixels": pred_area,
            "official_pixels": target_area,
            "intersection": intersection,
            "Dice": dice,
            "IoU": iou,
        })

    macro_dice = (
        float(np.mean(dice_values))
        if dice_values else float("nan")
    )

    macro_iou = (
        float(np.mean(iou_values))
        if iou_values else float("nan")
    )

    exact_match = bool(np.array_equal(our_labels, official_labels))

    return {
        "width": our_mask.shape[1],
        "height": our_mask.shape[0],
        "Pixel_Accuracy": pixel_accuracy,
        "Macro_Dice": macro_dice,
        "Macro_IoU": macro_iou,
        "Exact_Match": exact_match,
        "Our_Labels": str(sorted(our_counts, key=str)),
        "Official_Labels": str(sorted(official_counts, key=str)),
        "Our_Label_Counts": str(our_counts),
        "Official_Label_Counts": str(official_counts),
        "Class_Results": class_results,
    }, our_mask, official_mask


# ============================================================
# 6. 可视化
# ============================================================

def mask_to_rgb(mask):
    """仅用于可视化，不参与指标计算。"""

    if mask.ndim == 2:
        image = Image.fromarray(mask.astype(np.uint8))
        return image.convert("RGB")

    if mask.shape[2] >= 3:
        return Image.fromarray(mask[:, :, :3].astype(np.uint8))

    return Image.fromarray(mask[:, :, 0].astype(np.uint8)).convert("RGB")


def save_image_visualization(our_image, official_image, save_path):
    """保存 Image 对比图及差异图。"""

    our_pil = Image.fromarray(our_image)
    official_pil = Image.fromarray(official_image)

    diff = ImageChops.difference(our_pil, official_pil)
    diff = diff.point(lambda x: min(255, x * DIFF_SCALE))

    width, height = our_pil.size

    canvas = Image.new(
        "RGB",
        (width * 3, height + 35),
        "white",
    )

    canvas.paste(official_pil, (0, 35))
    canvas.paste(our_pil, (width, 35))
    canvas.paste(diff, (width * 2, 35))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Official", fill="black")
    draw.text((width + 10, 10), "Ours", fill="black")
    draw.text((width * 2 + 10, 10), "Difference", fill="black")

    canvas.save(save_path)


def save_mask_visualization(our_mask, official_mask, save_path):
    """保存 Mask 对比图及像素不一致区域。"""

    our_rgb = mask_to_rgb(our_mask)
    official_rgb = mask_to_rgb(official_mask)

    if our_mask.ndim == 2:
        mismatch = our_mask != official_mask
    else:
        mismatch = np.any(our_mask != official_mask, axis=-1)

    height, width = mismatch.shape

    diff_array = np.zeros((height, width, 3), dtype=np.uint8)
    diff_array[mismatch] = [255, 0, 255]

    diff_image = Image.fromarray(diff_array)

    canvas = Image.new(
        "RGB",
        (width * 3, height + 35),
        "white",
    )

    canvas.paste(official_rgb, (0, 35))
    canvas.paste(our_rgb, (width, 35))
    canvas.paste(diff_image, (width * 2, 35))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Official Mask", fill="black")
    draw.text((width + 10, 10), "Our Mask", fill="black")
    draw.text((width * 2 + 10, 10), "Mismatch", fill="black")

    canvas.save(save_path)


# ============================================================
# 7. 单个数据集比较
# ============================================================

def compare_dataset(name, config):
    output_dir = Path(config["output_dir"])
    image_vis_dir = output_dir / "image_visualizations"
    mask_vis_dir = output_dir / "mask_visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_vis_dir.mkdir(parents=True, exist_ok=True)
    mask_vis_dir.mkdir(parents=True, exist_ok=True)

    image_matches = match_files(
        config["our_image_dir"],
        config["official_image_dir"],
        IMAGE_EXTENSIONS,
    )

    mask_matches = match_files(
        config["our_mask_dir"],
        config["official_mask_dir"],
        MASK_EXTENSIONS,
    )

    image_keys, our_images, official_images, image_missing_official, image_missing_ours = image_matches
    mask_keys, our_masks, official_masks, mask_missing_official, mask_missing_ours = mask_matches

    print(f"\n========== {name} ==========")
    print(f"匹配 Image：{len(image_keys)}")
    print(f"匹配 Mask：{len(mask_keys)}")

    image_rows = []
    mask_rows = []
    class_rows = []
    error_rows = []

    common_keys = sorted(set(image_keys) & set(mask_keys))

    for index, key in enumerate(common_keys, 1):
        print(f"[{index}/{len(common_keys)}] 比较：{key}")

        # Image
        try:
            metrics, our_img, official_img = calculate_image_metrics(
                our_images[key],
                official_images[key],
            )

            metrics["sample"] = key
            image_rows.append(metrics)

            save_image_visualization(
                our_img,
                official_img,
                image_vis_dir / f"{key.replace('/', '_')}.png",
            )

        except Exception as e:
            error_rows.append({
                "sample": key,
                "type": "Image",
                "error": str(e),
            })
            print(f"  Image 比较失败：{e}")

        # Mask
        try:
            metrics, our_mask, official_mask = calculate_mask_metrics(
                our_masks[key],
                official_masks[key],
            )

            class_results = metrics.pop("Class_Results")
            metrics["sample"] = key
            mask_rows.append(metrics)

            for result in class_results:
                result["sample"] = key
                class_rows.append(result)

            save_mask_visualization(
                our_mask,
                official_mask,
                mask_vis_dir / f"{key.replace('/', '_')}.png",
            )

        except Exception as e:
            error_rows.append({
                "sample": key,
                "type": "Mask",
                "error": str(e),
            })
            print(f"  Mask 比较失败：{e}")

    # 记录未匹配文件
    for key in image_missing_official:
        error_rows.append({
            "sample": key,
            "type": "Image",
            "error": "我们转换的数据存在，但官方数据中不存在",
        })

    for key in image_missing_ours:
        error_rows.append({
            "sample": key,
            "type": "Image",
            "error": "官方数据存在，但我们转换的数据中不存在",
        })

    for key in mask_missing_official:
        error_rows.append({
            "sample": key,
            "type": "Mask",
            "error": "我们转换的 Mask 存在，但官方 Mask 中不存在",
        })

    for key in mask_missing_ours:
        error_rows.append({
            "sample": key,
            "type": "Mask",
            "error": "官方 Mask 存在，但我们转换的 Mask 中不存在",
        })

    # 保存 CSV
    def save_csv(filename, rows):
        if not rows:
            return

        fieldnames = list(rows[0].keys())

        with open(
            output_dir / filename,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    save_csv("image_metrics.csv", image_rows)
    save_csv("mask_metrics.csv", mask_rows)
    save_csv("class_metrics.csv", class_rows)
    save_csv("errors.csv", error_rows)

    # 汇总报告
    with open(
        output_dir / "summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(f"数据集：{name}\n")
        f.write("=" * 50 + "\n")

        f.write(f"匹配 Image 数量：{len(image_keys)}\n")
        f.write(f"匹配 Mask 数量：{len(mask_keys)}\n")
        f.write(f"同时匹配 Image 和 Mask 的样本数：{len(common_keys)}\n")
        f.write(f"错误/未匹配记录数：{len(error_rows)}\n\n")

        if image_rows:
            f.write("【Image 指标】\n")
            for metric in ["MAE", "RMSE", "PSNR", "SSIM"]:
                values = [
                    row[metric]
                    for row in image_rows
                    if np.isfinite(row[metric])
                ]

                if values:
                    f.write(
                        f"{metric}: "
                        f"mean={np.mean(values):.6f}, "
                        f"std={np.std(values):.6f}\n"
                    )

            exact = sum(row["Exact_Match"] for row in image_rows)
            f.write(
                f"Image 完全一致数量：{exact}/{len(image_rows)}\n\n"
            )

        if mask_rows:
            f.write("【Mask 指标】\n")

            for metric in [
                "Pixel_Accuracy",
                "Macro_Dice",
                "Macro_IoU",
            ]:
                values = [
                    row[metric]
                    for row in mask_rows
                    if np.isfinite(row[metric])
                ]

                if values:
                    f.write(
                        f"{metric}: "
                        f"mean={np.mean(values):.6f}, "
                        f"std={np.std(values):.6f}\n"
                    )

            exact = sum(row["Exact_Match"] for row in mask_rows)
            f.write(
                f"Mask 完全一致数量：{exact}/{len(mask_rows)}\n"
            )

    print(f"\n{name} 比较完成！")
    print(f"结果保存位置：{output_dir}")


# ============================================================
# 8. 主函数
# ============================================================

if __name__ == "__main__":

    for dataset_name, dataset_config in DATASETS.items():
        compare_dataset(dataset_name, dataset_config)

    print("\n所有数据集比较完成！")
