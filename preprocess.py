
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# ============================================================
# 1. 全局配置
# ============================================================

TARGET_SIZE = (512, 512)

# 支持的图像格式
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# True：遍历指定目录下的所有子目录
# False：只遍历指定目录的第一层
RECURSIVE = True


# ============================================================
# 2. 数据集路径配置
# ============================================================

DATASETS = {

    "PAPILA": {
        # 原始眼底图像目录
        "image_dir": r"data/FundData/ORIGA/Images",

        # 原始 Mask 目录
        "mask_dir": r"data/FundData/ORIGA/Masks",

        # 输出目录
        "output_dir": r"data/GAMMA",
    },

    # 后续数据集按照相同格式添加
    #
    # "Drishti-GS": {
    #     "image_dir": r"D:\datasets\Drishti-GS\实际图像目录",
    #     "mask_dir": r"D:\datasets\Drishti-GS\实际标注目录",
    #     "output_dir": r"D:\processed_datasets\Drishti-GS",
    # },
    #
    # "RIM-ONE-r3": {
    #     "image_dir": r"D:\datasets\RIM-ONE-r3\实际图像目录",
    #     "mask_dir": r"D:\datasets\RIM-ONE-r3\实际标注目录",
    #     "output_dir": r"D:\processed_datasets\RIM-ONE-r3",
    # },
}


# ============================================================
# 3. 获取目录下的所有图像文件
# ============================================================

def get_image_files(directory):

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    if RECURSIVE:
        files = [
            p for p in directory.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        files = [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    return sorted(files)


# ============================================================
# 4. 建立 Image 和 Mask 的匹配关系
# ============================================================

def build_file_map(files, root):

    file_map = {}

    for file in files:

        # 使用相对路径（去掉扩展名）作为匹配键
        # 例如：
        # images/001.jpg -> 001
        # masks/001.png  -> 001
        key = file.relative_to(root).with_suffix("").as_posix()

        if key in file_map:
            raise ValueError(
                f"发现重复文件匹配键: {key}\n"
                f"{file_map[key]}\n{file}"
            )

        file_map[key] = file

    return file_map


# ============================================================
# 5. 计算等比例缩放尺寸和 Padding
# ============================================================

def calculate_resize_size(original_size, target_size):

    original_w, original_h = original_size
    target_w, target_h = target_size

    scale = min(
        target_w / original_w,
        target_h / original_h
    )

    new_w = round(original_w * scale)
    new_h = round(original_h * scale)

    new_w = max(1, min(new_w, target_w))
    new_h = max(1, min(new_h, target_h))

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    right = target_w - new_w - left
    bottom = target_h - new_h - top

    return new_w, new_h, left, top, right, bottom


# ============================================================
# 6. Image 预处理
# ============================================================

def preprocess_image(image_path):

    with Image.open(image_path) as image:

        # 处理 EXIF 方向
        image = ImageOps.exif_transpose(image)

        # 统一为 RGB
        image = image.convert("RGB")

        original_size = image.size

        new_w, new_h, left, top, right, bottom = (
            calculate_resize_size(
                original_size,
                TARGET_SIZE
            )
        )

        # 高质量缩放
        image = image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )

        # 创建黑色画布
        canvas = Image.new(
            "RGB",
            TARGET_SIZE,
            (0, 0, 0)
        )

        # 居中放置
        canvas.paste(image, (left, top))

        return canvas, original_size, (new_w, new_h)


# ============================================================
# 7. Mask 预处理
# ============================================================

def preprocess_mask(mask_path, expected_original_size):

    with Image.open(mask_path) as mask:

        # 处理 EXIF 方向
        mask = ImageOps.exif_transpose(mask)

        original_size = mask.size

        # 确保 Image 和 Mask 原始尺寸一致
        if original_size != expected_original_size:
            raise ValueError(
                f"Image 与 Mask 原始尺寸不一致！\n"
                f"Mask: {mask_path}\n"
                f"Mask尺寸: {original_size}\n"
                f"Image尺寸: {expected_original_size}"
            )

        # 不进行 RGB/L 转换，尽量保留原始标签编码
        # NEAREST 不会产生新的插值标签值

        new_w, new_h, left, top, right, bottom = (
            calculate_resize_size(
                original_size,
                TARGET_SIZE
            )
        )

        mask = mask.resize(
            (new_w, new_h),
            Image.Resampling.NEAREST
        )

        # 使用 0 作为 Padding 的背景标签
        canvas = Image.new(
            mask.mode,
            TARGET_SIZE,
            0
        )

        canvas.paste(mask, (left, top))

        return canvas, original_size, (new_w, new_h)


# ============================================================
# 8. 处理单个数据集
# ============================================================

def process_dataset(dataset_name, config):

    print("\n" + "=" * 65)
    print(f"开始处理数据集: {dataset_name}")
    print("=" * 65)

    image_dir = Path(config["image_dir"])
    mask_dir = Path(config["mask_dir"])
    output_dir = Path(config["output_dir"])

    output_image_dir = output_dir / "Images_Square"
    output_mask_dir = output_dir / "Masks_Square"

    # 检查输入目录
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image 目录不存在: {image_dir}"
        )

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Mask 目录不存在: {mask_dir}"
        )

    # 创建输出目录
    output_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 获取文件
    image_files = get_image_files(image_dir)
    mask_files = get_image_files(mask_dir)

    print(f"Image 数量: {len(image_files)}")
    print(f"Mask 数量: {len(mask_files)}")

    # 建立匹配字典
    image_map = build_file_map(image_files, image_dir)
    mask_map = build_file_map(mask_files, mask_dir)

    image_keys = set(image_map.keys())
    mask_keys = set(mask_map.keys())

    missing_masks = image_keys - mask_keys
    missing_images = mask_keys - image_keys

    if missing_masks or missing_images:

        print("\n文件匹配检查失败！")

        if missing_masks:
            print("没有对应 Mask 的 Image:")
            for key in sorted(missing_masks):
                print("  ", key)

        if missing_images:
            print("没有对应 Image 的 Mask:")
            for key in sorted(missing_images):
                print("  ", key)

        raise ValueError(
            "Image 和 Mask 无法完全一一匹配，请检查文件名及目录结构。"
        )

    print("Image-Mask 一对一匹配检查通过！")

    # 处理记录
    log_records = []

    success_count = 0
    failed_count = 0

    # ========================================================
    # 遍历处理
    # ========================================================

    for index, key in enumerate(sorted(image_keys), start=1):

        image_path = image_map[key]
        mask_path = mask_map[key]

        print(
            f"\n[{index}/{len(image_keys)}] "
            f"正在处理: {key}"
        )

        try:

            # Image 处理
            processed_image, original_size, resized_size = (
                preprocess_image(image_path)
            )

            # Mask 处理
            processed_mask, mask_original_size, mask_resized_size = (
                preprocess_mask(
                    mask_path,
                    original_size
                )
            )

            # 检查输出尺寸
            if processed_image.size != TARGET_SIZE:
                raise ValueError("Image 输出尺寸错误")

            if processed_mask.size != TARGET_SIZE:
                raise ValueError("Mask 输出尺寸错误")

            # 保留原始相对目录结构
            relative_path = Path(key)

            image_output_path = (
                output_image_dir / relative_path
            ).with_suffix(".png")

            mask_output_path = (
                output_mask_dir / relative_path
            ).with_suffix(".png")

            image_output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            mask_output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # 保存 PNG
            processed_image.save(image_output_path)

            # 保留 Mask 的模式和 Palette 信息（如果存在）
            processed_mask.save(mask_output_path)

            success_count += 1

            print(f"原始尺寸: {original_size}")
            print(f"缩放尺寸: {resized_size}")
            print(f"最终尺寸: {TARGET_SIZE}")
            print("处理成功！")

            log_records.append([
                key,
                str(image_path),
                str(mask_path),
                original_size,
                resized_size,
                TARGET_SIZE,
                "SUCCESS",
                ""
            ])

        except Exception as error:

            failed_count += 1

            print(f"处理失败: {error}")

            log_records.append([
                key,
                str(image_path),
                str(mask_path),
                "",
                "",
                TARGET_SIZE,
                "FAILED",
                str(error)
            ])

    # ========================================================
    # 保存处理日志
    # ========================================================

    log_path = output_dir / "processing_log.csv"

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "File_Key",
            "Image_Path",
            "Mask_Path",
            "Original_Size",
            "Resized_Size",
            "Target_Size",
            "Status",
            "Error"
        ])

        writer.writerows(log_records)

    # ========================================================
    # 输出统计信息
    # ========================================================

    print("\n" + "-" * 65)
    print(f"数据集: {dataset_name}")
    print(f"成功数量: {success_count}")
    print(f"失败数量: {failed_count}")
    print(f"日志位置: {log_path}")
    print(f"输出 Image: {output_image_dir}")
    print(f"输出 Mask: {output_mask_dir}")
    print("-" * 65)

    return success_count, failed_count


# ============================================================
# 9. 主程序
# ============================================================

if __name__ == "__main__":

    total_success = 0
    total_failed = 0

    for dataset_name, config in DATASETS.items():

        try:

            success, failed = process_dataset(
                dataset_name,
                config
            )

            total_success += success
            total_failed += failed

        except Exception as error:

            print(
                f"\n数据集 {dataset_name} 处理终止：{error}"
            )

            total_failed += 1

    print("\n" + "=" * 65)
    print("所有数据集处理结束")
    print(f"总成功数量: {total_success}")
    print(f"总失败数量: {total_failed}")
    print("=" * 65)
