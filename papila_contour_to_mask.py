
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

# PAPILA 解压后的数据根目录
DATA_ROOT = Path(
    r"data/FundData/PAPILA"
)

# 原始眼底图像目录
IMAGE_DIR = DATA_ROOT / "FundusImages"

# TXT 轮廓标注目录
CONTOUR_DIR = (
    DATA_ROOT
    / "ExpertsSegmentations"
    / "Contours"
)

# 输出目录
OUTPUT_ROOT = Path(
    r"data/FundData/PAPILA"
)

# 同时处理两位专家
EXPERTS = ["exp1", "exp2"]

# 是否将超出视盘区域的视杯像素裁剪掉
# 视杯在解剖结构上应位于视盘内部
CLIP_CUP_TO_DISC = True

# 是否保存轮廓可视化结果
SAVE_PREVIEW = False

# 每位专家最多保存多少张预览图
PREVIEW_LIMIT = 10


# ============================================================
# 2. 读取 TXT 轮廓坐标
# ============================================================

def load_contour(txt_path, width, height):
    """
    读取 PAPILA TXT 轮廓坐标。

    原始格式：
        x1 y1
        x2 y2
        x3 y3
        ...

    返回：
        OpenCV polygon，shape = (N, 1, 2)
    """

    if not txt_path.exists():
        raise FileNotFoundError(
            f"找不到轮廓文件: {txt_path}"
        )

    # 读取浮点坐标
    coords = np.loadtxt(
        str(txt_path),
        dtype=np.float32,
        ndmin=2
    )

    if coords.shape[1] < 2:
        raise ValueError(
            f"轮廓文件格式错误: {txt_path}"
        )

    # 只使用前两列：X、Y
    coords = coords[:, :2]

    # 删除非有限值
    coords = coords[
        np.isfinite(coords).all(axis=1)
    ]

    if len(coords) < 3:
        raise ValueError(
            f"轮廓点少于3个，无法构成多边形: {txt_path}"
        )

    # 坐标取整
    points = np.rint(coords).astype(np.int32)

    # 检查坐标是否超出图像范围
    out_of_bounds = (
        (points[:, 0] < 0)
        | (points[:, 0] >= width)
        | (points[:, 1] < 0)
        | (points[:, 1] >= height)
    )

    if np.any(out_of_bounds):
        print(
            f"[WARNING] {txt_path.name}: "
            f"{np.sum(out_of_bounds)} 个坐标超出图像范围，执行边界裁剪"
        )

        points[:, 0] = np.clip(
            points[:, 0], 0, width - 1
        )

        points[:, 1] = np.clip(
            points[:, 1], 0, height - 1
        )

    return points.reshape(-1, 1, 2)


# ============================================================
# 3. 将多边形转换为二值 Mask
# ============================================================

def polygon_to_mask(points, height, width):
    """
    将轮廓坐标转换为二值 Mask。

    0：背景
    1：多边形内部
    """

    binary_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        binary_mask,
        [points],
        color=1
    )

    return binary_mask.astype(bool)


# ============================================================
# 4. 生成 OAIMS 三分类 Mask
# ============================================================

def generate_oaims_mask(
    image,
    disc_txt,
    cup_txt,
    save_path
):
    """
    根据 Disc 和 Cup 两个 TXT 文件
    生成 OAIMS 三分类灰度 PNG Mask。

    255 = Background
    128 = Outer-ring
    0   = Cup
    """

    height, width = image.shape[:2]

    # 读取两个轮廓
    disc_points = load_contour(
        disc_txt,
        width,
        height
    )

    cup_points = load_contour(
        cup_txt,
        width,
        height
    )

    # 转换为二值区域
    disc_mask = polygon_to_mask(
        disc_points,
        height,
        width
    )

    cup_mask = polygon_to_mask(
        cup_points,
        height,
        width
    )

    # 检查视杯是否超出视盘
    outside_pixels = np.sum(
        cup_mask & (~disc_mask)
    )

    if outside_pixels > 0:
        print(
            f"[WARNING] {save_path.stem}: "
            f"Cup 有 {outside_pixels} 个像素位于 Disc 外部"
        )

        if CLIP_CUP_TO_DISC:
            cup_mask = cup_mask & disc_mask

    # 初始化为背景
    mask = np.full(
        (height, width),
        255,
        dtype=np.uint8
    )

    # 视盘区域：128
    mask[disc_mask] = 128

    # 视杯区域：0
    mask[cup_mask] = 0

    # 保存 PNG
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(save_path),
        mask
    )

    if not success:
        raise IOError(
            f"Mask 保存失败: {save_path}"
        )

    return mask, disc_points, cup_points


# ============================================================
# 5. 保存轮廓预览图
# ============================================================

def save_overlay(
    image,
    disc_points,
    cup_points,
    save_path
):
    """
    绿色：Disc 轮廓
    红色：Cup 轮廓
    """

    preview = image.copy()

    cv2.polylines(
        preview,
        [disc_points],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3
    )

    cv2.polylines(
        preview,
        [cup_points],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(save_path),
        preview
    )


# ============================================================
# 6. 主处理流程
# ============================================================

def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"图像目录不存在: {IMAGE_DIR}"
        )

    if not CONTOUR_DIR.exists():
        raise FileNotFoundError(
            f"轮廓目录不存在: {CONTOUR_DIR}"
        )

    # 读取全部 JPG 图像
    image_paths = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg"]
    ])

    print("=" * 60)
    print("PAPILA TXT -> OAIMS PNG Mask")
    print("=" * 60)

    print(f"原始图像数量: {len(image_paths)}")
    print(f"专家数量: {len(EXPERTS)}")

    if len(image_paths) != 488:
        print(
            f"[WARNING] 原始图像数量不是预期的488张，"
            f"实际为{len(image_paths)}张"
        )

    # 分别处理两位专家
    for expert in EXPERTS:

        print("\n" + "=" * 60)
        print(f"正在处理专家: {expert}")
        print("=" * 60)

        mask_dir = (
            OUTPUT_ROOT
            / f"mask_{expert}"
        )

        preview_dir = (
            OUTPUT_ROOT
            / f"preview_{expert}"
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        processed = 0
        missing = []

        for index, image_path in enumerate(
            image_paths,
            start=1
        ):

            image_id = image_path.stem

            # 对应的两个 TXT 文件
            disc_txt = (
                CONTOUR_DIR
                / f"{image_id}_disc_{expert}.txt"
            )

            cup_txt = (
                CONTOUR_DIR
                / f"{image_id}_cup_{expert}.txt"
            )

            # 检查标注是否完整
            if not disc_txt.exists():
                missing.append(
                    disc_txt.name
                )
                continue

            if not cup_txt.exists():
                missing.append(
                    cup_txt.name
                )
                continue

            # 读取原始图像
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR
            )

            if image is None:
                print(
                    f"[ERROR] 图像读取失败: {image_path}"
                )
                continue

            # 生成 Mask
            mask, disc_points, cup_points = (
                generate_oaims_mask(
                    image=image,
                    disc_txt=disc_txt,
                    cup_txt=cup_txt,
                    save_path=mask_dir / f"{image_id}.png"
                )
            )

            # 保存预览图
            if (
                SAVE_PREVIEW
                and processed < PREVIEW_LIMIT
            ):
                save_overlay(
                    image,
                    disc_points,
                    cup_points,
                    preview_dir / f"{image_id}.jpg"
                )

            processed += 1

            if index % 50 == 0:
                print(
                    f"进度: {index}/{len(image_paths)} "
                    f"| 已生成 Mask: {processed}"
                )

        # 专家处理结果
        print("\n专家处理完成:", expert)
        print(f"成功生成 Mask: {processed}")
        print(f"缺失 TXT 文件: {len(missing)}")

        if missing:
            print("缺失文件示例:")
            for name in missing[:20]:
                print("  ", name)

        if processed != len(image_paths):
            print(
                "[WARNING] Mask 数量与图像数量不一致，"
                "请检查缺失标注。"
            )

    print("\n" + "=" * 60)
    print("全部处理结束！")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
