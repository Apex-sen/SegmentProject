# 医学分割持续测试时适应

眼底实验在统一的、带 BN 的 ResUNet-34 上比较 Source Only、TENT-continual、CoTTA、SAR 和 VPTTA。默认配置 `configs/fundus_v2.toml` 按 70% 训练、10% 验证、20% 测试划分 REFUGE2 源域。PAPILA 使用专家 1 的标注，四个目标域按全量图像数从少到多排列。

| 顺序 | 目标域 | 图像数 |
|---|---|---:|
| 1 | GAMMA | 100 |
| 2 | PAPILA | 488 |
| 3 | ORIGA | 650 |
| 4 | G1020 | 1,020 |
| | **合计** | **2,258** |

使用 `--config configs/fundus_v1.toml` 可在 ORIGA 源权重和 PAPILA → GAMMA 测试流上运行五种方法。

## 本地运行

实验 TOML 保存在本地 `configs/`，数据和输出分别保存在 `data/` 和 `runs/`。这些目录由 Git 忽略。在新的工作树中运行实验前，将本地 TOML 放回 `configs/`，或用 `--config` 指向已有的配置文件。

建议使用 Python 3.12 和 PyTorch。安装时可按 [PyTorch 的 Windows 指南](https://pytorch.org/get-started/locally/)选择 CPU 或 CUDA 版。项目也支持已有的 Python 3.10 环境。

```powershell
python -m pip install -r requirements.txt
python main.py --smoke --device cpu --lr 0.0003 --batch_size 2 --epochs 1
```

小样本运行读取 8 张训练图、4 张验证图、4 张源域测试图和每个目标域各 4 张图，默认训练 1 轮。结果写到 `runs/fundus_v2_refuge2_70_10_20/smoke`。从头跑正式实验可用 `python main.py run --device auto`，结果写到 `runs/fundus_v2_refuge2_70_10_20`。传入参数但省略子命令时，默认依次执行数据准备、源域训练、评估。只输入 `python main.py` 会显示帮助。

默认用随机种子 `42` 按图像划分源域。REFUGE2 有 840 张训练图、120 张验证图和 240 张测试图。验证集平均 Dice 用于选取最佳权重，源域测试使用该权重做冻结推理，结果写到 `source_test.csv`。跨域评估的结果写到 `summary.csv`。小样本上限在 TOML 的 `[smoke]` 中，正式运行的上限在 `[sampling]` 中，`0` 表示使用该划分或目标域的全部图像。

```powershell
python main.py --smoke --device cpu --max_train_samples 16 --max_val_samples 8 --max_test_samples 8 --max_target_samples 10
```

`--max_target_samples` 按目标域分别生效，`10` 会从每个目标域各取至多 10 张。数量在生成 `manifest.csv` 时固定，`train` 和 `evaluate` 读取同一份清单。

启动参数定义在根目录的 `main.py`。命令行传入的参数覆盖 TOML 对应值，例如 `--lr`、`--batch_size`、`--epochs`、`--gpu`、`--data_root`、`--save_dir` 和 `--methods`。`--lr` 适用于源模型训练。TENT 的学习率仍由 `[methods.tent_continual]` 配置。`--gpu` 用于选择 CUDA 设备索引。

也可以逐阶段运行。`evaluate` 默认读取当前运行目录中的 `source_best.pt`，也可用 `--checkpoint` 指定文件。源域测试要求权重和本次源域划分匹配。`--eval-split source-test` 只运行源域测试，`--eval-split target` 只运行跨域评估。这两个选项也可用于 `run`。

```powershell
python main.py prepare
python main.py train --device cuda
python main.py evaluate --device cuda --eval-split source-test
python main.py evaluate --device cuda
```

两个入口 `python main.py` 和 `python -m ctta` 接受相同的子命令和参数。省略 `--config` 时读取 `configs/fundus_v2.toml`。TOML 包含数据路径、源域和目标域、随机种子、训练参数、设备和方法选择。`[experiment]` 中的 `methods` 指定评估顺序，`device` 默认自动选择。各方法的参数写在 `[methods.<方法名>]`。每个评估方法独立加载同一份源模型权重。

## ORIGA 眼底对比

已有正式 ORIGA 源权重和对应测试清单时，可直接用所有方法评估。需要从头准备数据和训练时，用 `run` 代替 `evaluate`。`--method cotta`、`--method sar` 和 `--method vptta` 可分别运行单项。用 `--smoke --device cpu` 可验证小样本清单与小样本权重。

```powershell
python main.py evaluate --config configs/fundus_v1.toml --device cuda
```

- **CoTTA** 在有效像素上按论文式 (1) 和式 (5) 使用教师的完整类别概率计算软标签交叉熵。更新学生后，以 EMA 更新教师，再随机恢复部分学生权重。增强按论文 §4.3 使用七个尺度及水平翻转，将预测对齐到分割网格后平均概率。分割移植参照 SHIFT，源锚点置信度逐像素与 `0.69` 比较，高置信像素取单尺度教师预测，其余像素取增强平均预测。ResUNet-34 的三套模型都使用源域 BN 统计，适应学习率为 `6e-5 / 8`。[论文](https://arxiv.org/pdf/2203.13591#page=4) · [作者代码](https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py) · [SHIFT 分割实现](https://github.com/zwbx/SHIFT-Continuous_Test_Time_Adaptation/blob/master/mmseg/apis/test.py#L63)
- **SAR** 将分类代码里的样本单位映射为有效像素。第一次前向保留熵低于 `0.4 × ln(C)` 的像素，`C` 是本地输出类别数。SAM 扰动后在同一批像素上再次用该阈值筛选，再更新 BN 仿射参数。沿用官方对 ResNet `layer4` 仿射参数的排除、`ρ=0.05`、SGD 动量 `0.9`、恢复阈值 `0.2` 以及批量为 1 的 ResNet 学习率。任一轮筛选为空时跳过更新，第二轮为空时同时撤销 SAM 扰动。当前图像使用第一次前向的 logits 评分。按公开实现保留恢复时的熵 EMA 和底层 SGD 动量。[SAR 更新](https://github.com/mr-eggplant/SAR/blob/main/sar.py) · [SAM](https://github.com/mr-eggplant/SAR/blob/main/sam.py) · [ResNet 学习率](https://github.com/mr-eggplant/SAR/blob/main/main.py)
- **VPTTA** 沿用官方 `OPTIC` 的 FFT 低频乘性提示、余弦近邻初始化、编码器与解码器 BN 预热统计以及 Adam 更新。每张图更新一次提示，再用适应后的输入评分。源模型权重和 BN 缓冲区保持冻结，Adam 状态、记忆库和预热样本计数贯穿测试流。保留官方 FIFO 插入前 `len > 40` 的条件，所以最多记录 41 个不同键。本地使用已有的三通道分割头、预处理和 OD/OC 指标。官方眼底实验使用双通道 sigmoid 头和转置卷积解码器。[OPTIC 入口](https://github.com/Chen-Ziyang/VPTTA/blob/main/OPTIC/vptta.py) · [BN 预热](https://github.com/Chen-Ziyang/VPTTA/blob/main/OPTIC/utils/convert.py) · [记忆库](https://github.com/Chen-Ziyang/VPTTA/blob/main/OPTIC/utils/memory.py) · [官方网络](https://github.com/Chen-Ziyang/VPTTA/blob/main/OPTIC/networks/ResUnet_TTA.py)

## 息肉二分类实验

使用 `configs/polyp_v1.toml` 选择息肉实验。Kvasir-SEG 固定用 800 张训练、200 张验证。全量测试按 ETIS-LaribPolypDB（196 张）→ CVC-ClinicDB PNG（612 张）→ BKAI（1,000 张）的顺序处理 1,808 张已配对图像。

训练使用带 BN 的两通道 ResUNet-34、交叉熵和前景 Dice loss。图像和掩码在线等比缩放并补齐到 `352×352`，损失和评分排除补边。Kvasir、ETIS 和 CVC 用灰度阈值解码，BKAI 将红、绿两种息肉标注合并为前景。最佳权重按 Kvasir 验证集的前景 Dice 选出。

```bash
python main.py run --config configs/polyp_v1.toml --smoke --device cpu
python main.py run --config configs/polyp_v1.toml --device cuda
```

第一条命令在 `runs/polyp_v1/smoke` 做一轮小样本检查。第二条命令执行 50 轮正式训练和全量评估。两条命令都会运行 Source Only、TENT-continual、CoTTA、SAR 和 VPTTA。各方法独立加载同一份 Kvasir-SEG 最佳源权重，并按同一顺序读取目标图像。已有与清单和运行模式匹配的 `source_best.pt` 时，可用 `evaluate` 复跑五种方法。

所有方法共用本地的双通道 ResUNet-34。SAR 的有效像素熵阈值随输出类别数变为 `0.4 × ln(2)`。VPTTA 采用作者息肉代码中的 `0.01` 提示学习率，每图更新一次，按编码器 BN 统计对齐。作者的息肉实验使用 PraNet 和单通道 sigmoid 输出。[作者息肉代码](https://github.com/Chen-Ziyang/VPTTA/blob/main/POLYP/vptta.py) · [论文实验设置](https://arxiv.org/html/2311.18363#S4.SS2)

`runs/polyp_v1/summary.csv` 包含分域前景 Dice、IoU 和 HD95、跨域等权平均，以及 `delta_vs_source_only` 中的 Dice 差值。域内 Dice 和 HD95 按图平均，IoU 按该域所有有效像素累计。

## 按源域并行运行正式实验

眼底有五轮，息肉有四轮。每轮有独立的配置和输出目录，可以在不同终端中指定不同 GPU。每类任务的 `common.toml` 定义共享设置，每轮配置指定源域、目标流顺序和输出目录。

| 任务 | 配置目录 | 源域配置文件 |
| --- | --- | --- |
| 眼底 | `configs/fundus_v3_rotation` | `refuge2.toml`、`gamma.toml`、`papila.toml`、`origa.toml`、`g1020.toml` |
| 息肉 | `configs/polyp_v2_rotation` | `kvasir_seg.toml`、`etis_laribpolypdb.toml`、`cvc_clinicdb.toml`、`bkai.toml` |

```bash
# 各命令在独立终端执行
python -u main.py run --config configs/fundus_v3_rotation/refuge2.toml --device cuda --gpu 0
python -u main.py run --config configs/fundus_v3_rotation/gamma.toml --device cuda --gpu 1
python -u main.py run --config configs/polyp_v2_rotation/kvasir_seg.toml --device cuda --gpu 2

# 等对应任务的所有源域跑完后执行
python merge_rotation.py fundus
python merge_rotation.py polyp
```

每轮以种子 `42` 将源域按 70%／10%／20% 划分，训练 50 轮，按验证 Dice 保存最佳 checkpoint，并将源域留出测试结果写入 `source_test.csv`。其余域使用全量数据。眼底从 `data/FundData` 读取原图和掩码，G1020 作源域和目标域都使用 `G1020/Images_790` 与 `G1020/Masks_790`。息肉从 `data/PolypData` 读取原图和掩码。

新配置的 CoTTA 在学生、EMA 教师和源锚点上使用当前测试批次的 BN 统计。目标评估还加入 `empty_prediction_bn_retry` 作为独立对照。它先做冻结的 Source Only 推理，有效区域内的视杯或息肉预测为空时，用当前图像 BN 重推一次。各域的 `n_*_misses` 记录有标注前景而预测为空的图像数，`n_bn_retries` 记录重推图像数，两者同时写入 `summary.csv` 和 `verification.json`。

汇总读取 `runs/fundus_v3_rotation` 或 `runs/polyp_v2_rotation` 下的正式结果，并在同一目录写入三个文件。

- `rotation_summary.csv` 的每个源域列是这一轮的目标域等权均值，`average` 再对各源域轮次等权平均。
- `rotation_domains.csv` 保留每一轮各方法的分目标域指标和漏分计数。
- `rotation_source_test.csv` 收集每轮源域留出测试成绩，供单独查阅。

## 扩展方法

1. 在 `ctta` 下实现适应类。构造函数接收源模型和该方法的配置，`__call__(image, valid)` 返回可按通道取 `argmax` 的分割分数。通过 `adapted_parameter_names` 声明可更新参数，通过 `state_summary()` 和 `verify_run(start, end, sample_count)` 提供跨域状态摘要和方法自己的验收检查。
2. 将类加入 `METHOD_REGISTRY`。在 TOML 的 `experiment.methods` 中加入对应名称，把该方法的参数放进 `[methods.<方法名>]`。

评估器会按配置初始化新方法，使用同一 checkpoint、测试流和评分规则。

## 标签与结果

读取原始掩码并映射成背景、盘环、视杯。缩放后保留有效区域，训练损失、TENT 熵和评分均排除几何补边。OD 是盘环与视杯的并集，OC 是视杯。Dice 在域内按图平均。OD IoU 和 OC IoU 分别在域内累计有效像素的交集和并集后相除，mIoU 是两者的均值。逐图 IoU（`mean_image_iou`）先分别对有效标注图的 OD IoU 和 OC IoU 求均值，再对两个结构等权平均。最后对各目标域做等权平均，`macro_average` 行的 `n` 是总样本数。

旧版 `fundus_v2.toml` 中，G1020 作目标域时，1,020 张图全部进入跨域测试流。OD 和 OC 分别在有对应轮廓的 1,019 张、791 张图上计分。G1020 作源域时，配置读取 `G1020/Images_Square_790` 和 `G1020/Masks_Square_790`，其中应放置原始 JSON 同时含有 `disc`、`cup` 轮廓的 790 对方形文件。这一源域按 553 张训练、79 张验证、158 张测试划分。`summary.csv` 的 `n_od` 和 `n_oc` 记录各指标的有效样本数。逐图 CSV 中缺少标注的结构对应的 Dice 和 HD95 留空，逐图 Mean Dice 和 Mean HD95 在两个结构均有标注时填写。

HD95 在去除补边的二维评估网格上计算，单位为像素。使用四邻域提取轮廓，对两个方向的最近欧氏距离分别取第 95 百分位，再取较大值。有效标注中单侧掩码为空时，HD95 取有效区域内像素中心的最大距离。双侧为空时记 0。域内 HD95 按图平均，跨域等权平均。`n_*_empty_predictions` 记录有效标注中的空预测数，`n_*_both_empty` 记录双空数并包含在前者中。

- `manifest.csv` 记录固定的数据划分及目标流顺序，`previews/` 保存标签叠加检查图。
- 有源域验证集时，`source_best.pt` 保存验证平均 Dice 最佳的 checkpoint。验证比例为 0 时保存最后一轮的 checkpoint。`source_training.csv` 记录训练过程。
- `summary.csv` 保存分域 Dice、IoU、HD95、跨域等权均值、现有的 Source Only 差值、有效样本数和空掩码计数。评估结束时，终端按方法显示各目标域及 `macro_average`，同时列出样本数。Dice 和 IoU 的分数为 0–1，HD95 越低越好。
- 加 `--save_per_image` 或设定 TOML 的 `[report].save_per_image = true` 时，额外生成含逐图 Dice、IoU 和 HD95 的 `per_image.csv`，源域测试对应 `source_test_per_image.csv`。眼底逐图 `mean_iou` 在 OD 和 OC 均参与评分时填写。
- `verification.json` 记录 checkpoint、测试流和域边界状态。CoTTA 还记录教师、优化器、恢复随机数状态和增强次数。SAR 记录优化器、熵 EMA、更新、跳过和恢复次数。VPTTA 记录提示、Adam、记忆库和命中次数。

TENT 使用当前样本的 BN 统计和像素预测熵，只优化 BN 的仿射参数。当前样本用优化前的 logits 评分，更新留给下一张图。跨域沿用模型和优化器状态。评估阶段在方法返回预测后读取目标掩码并评分。开启 `benchmark_disc_roi_size` 的眼底实验还会在离线数据准备阶段使用标注中的视盘外接框中心裁剪 ROI。

## 官方实现逐步对齐

在 PowerShell 中把 [TENT 作者的 tent.py](https://github.com/DequanWang/tent/blob/master/tent.py) 下载到临时目录，再运行以下检查。它会对齐两次更新的损失、logits、参数和优化器状态，并生成 `verification_reference.json`。

```powershell
$tentReference = Join-Path $env:TEMP 'tent_reference.py'
Invoke-WebRequest 'https://raw.githubusercontent.com/DequanWang/tent/master/tent.py' -OutFile $tentReference
python main.py verify-reference --reference $tentReference --smoke
```
