# GlyphWeave · 字织

**融合笔画与结构的手写字符识别网络。**

[English](README.md) | 简体中文

GlyphWeave 支持识别 **7,185 个汉字、26 个大写字母、26 个小写字母和 10 个数字，共 7,247 类**。网络结合笔画梯度、分层卷积、空间注意力、多尺度融合与余弦分类器，所有可学习参数均从随机初始化开始训练。

本项目自主实现并组合成熟的网络组件。首次单随机种子实验已完成：完整 870,895 张测试图像上的 **Top-1 为 95.95%、Top-5 为 99.65%、宏平均 Top-1 为 97.64%**。实验协议、字符分组与局限见文末的[实验结果](#experiments)，当前不宣称达到最先进水平。

## 项目特点

- 笔画感知输入：灰度图像与固定的水平、垂直 Sobel 梯度。
- 分层特征编码：大核深度卷积与低分辨率空间注意力。
- 训练与推理共用预处理，并保留字符长宽比。
- 支持 Apple Silicon MPS 与 Windows 单卡 CUDA 训练、EMA、混合精度及断点恢复。
- 可检查的数据划分与评估：提供整体、各类别及字符分组指标。

<a id="architecture"></a>

## 网络结构

GlyphWeave 将表示字符字形的 *glyph* 与表示编织的 *weave* 结合，体现笔画细节与空间结构的融合，中文名为**字织**。当前模型实现仍为 `handwriting/model.py` 中的 `HandwritingNet`，配置标识为 `handwriting_net`。

编码器采用具有通道扩张与全局响应归一化（GRN）的残差深度卷积块，设计参考 [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)。空间注意力作用于末级特征网格，在控制 token 数量的同时建模字符结构。中间卷积特征、末级卷积特征与注意力特征经池化后使用可学习权重融合。

```mermaid
flowchart TD
    A[单字图片] --> B[保留长宽比的归一化]
    B --> C[灰度 + 水平/垂直 Sobel 梯度]
    C --> D[分层卷积编码器]
    D --> E[空间结构注意力]
    D --> F[中间 + 末级卷积特征池化]
    E --> G[可学习的多尺度融合]
    F --> G
    G --> H[LayerNorm + Dropout]
    H --> I[余弦分类器: 7247 类]
```

| 配置 | 输入 | 参数量 | 通道数 | 各阶段深度 | 注意力块数 |
|---|---|---:|---|---|---:|
| `mac.yaml` / `windows_cuda.yaml` | 128 × 128 | 7,820,044 | 40 / 80 / 160 / 320 | 2 / 2 / 6 / 2 | 2 |
| `quality.yaml` | 160 × 160 | 14,053,588 | 48 / 96 / 192 / 384 | 2 / 3 / 8 / 3 | 3 |

参数量包含分类头。扩大配置用于进一步实验，是否提高准确率需要实测。模块定义与设计依据见[网络设计说明](docs/网络设计.md)。

<a id="dataset-downloads"></a>

## 数据与下载

数据集文件和模型检查点与源码仓库分别发布。模型包已在 v1.0.0 提供，处理后的数据集将在单独上传后补充下载链接。

<!-- DATASET_RELEASE_LINKS: 获得真实发布地址后，同步更新两份 README。 -->

| 文件 | 内容 | 下载 | 版本 / SHA-256 |
|---|---|---|---|
| 处理后的数据集 | `metadata.json`、`index.npy` 及引用的全部 `raw/` 文件 | **待上传** | 待补充 |
| 推理模型 | EMA 权重与实验记录 | [推理 ZIP](https://github.com/13536309143/GlyphWeave/releases/download/v1.0.0/glyphweave-v1.0.0-inference.zip) | v1.0.0 / [SHA-256](https://github.com/13536309143/GlyphWeave/releases/download/v1.0.0/SHA256SUMS.txt) |
| 完整训练检查点 | 原始最佳检查点与实验记录 | [训练 ZIP](https://github.com/13536309143/GlyphWeave/releases/download/v1.0.0/glyphweave-v1.0.0-training.zip) | v1.0.0 / [SHA-256](https://github.com/13536309143/GlyphWeave/releases/download/v1.0.0/SHA256SUMS.txt) |

上游来源：

- [EMNIST — NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)：ByClass 保留大小写字母的独立标签。
- [CASIA-HWDB 下载页面](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)：离线手写单字数据。
- [本项目使用的 CASIA 镜像](https://huggingface.co/datasets/OrkaZeta/HWDB1/tree/main)：原服务器无法访问时使用的第三方来源。

数据集的获取和再分发仍遵循各上游条款，包括 [CASIA 使用协议](https://nlpr.ia.ac.cn/databases/handwriting/Application_form.html)。

下载处理后的数据时，应将**整个**目录解压到 `data/processed/`，仅有索引不足以训练。上传完整 `processed/` 就足够训练和评估，原始 ZIP/GZ 无需重复上传。当前目录约 23 GiB，使用 GitHub Release 分发时需分卷，每个文件小于 2 GiB，或使用外部数据托管平台。[GitHub 附件限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。使用原始文件时，恢复 `data/CASIA-HWDB/` 与 `data/EMNIST/`，安装依赖后运行 `prepare_data.py`。打包与校验方法见[数据集说明](docs/数据集.md)。

| 划分 | 图像数量 | 类别数量 |
|---|---:|---:|
| 训练 | 3,353,310 | 7,247 |
| 验证 | 374,648 | 7,247 |
| 测试 | 870,895 | 7,247 |

CASIA 保留官方测试集，并从官方训练书写者中划出验证书写者，三个划分的书写者互不重叠。EMNIST 保留官方测试集，训练与验证按类别分层抽样；由于没有书写者 ID，无法证明这两个划分的书写者隔离。数据准备过滤 135 条零尺寸记录和 1 张空白图，排除非目标符号，并在加载时对 EMNIST 图像转置一次。

<a id="released-model"></a>

## 发布模型

[v1.0.0 Release](https://github.com/13536309143/GlyphWeave/releases/tag/v1.0.0) 提供包含选定 EMA 权重的推理包，以及独立的完整训练检查点包。两者均附配置、类别映射、历史、测试报告、图片与清单；加载下载文件前请核对 `SHA256SUMS.txt`。

识别时，在安装依赖后下载 `glyphweave-v1.0.0-inference.zip` 并解压到项目根目录。**单张图片推理不需要数据集**。

```bash
.venv/bin/python predict.py path/to/glyph.png --checkpoint glyphweave-v1.0.0-inference/best.pt
.venv/bin/python app.py --checkpoint glyphweave-v1.0.0-inference/best.pt
```

Windows 将 `.venv/bin/python` 替换为 `.\.venv-win\Scripts\python.exe`。推理检查点不能续训。完整检查点保留原始优化器与调度器状态，但 80 轮调度已经完成；延长训练需要新实验，不能在恢复时直接改变 `epochs`。包内容见[发布说明](docs/releases/v1.0.0.md)。

<a id="quick-start"></a>

## 快速开始

克隆仓库，并在训练前下载上述数据集。

```bash
git clone https://github.com/13536309143/GlyphWeave.git
cd GlyphWeave
```

### Apple Silicon / MPS

创建环境、安装依赖、检查设备和数据，然后开始训练。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_environment.py --device auto
.venv/bin/python train.py --config configs/mac.yaml
```

如果下载的是原始文件，请在环境检查前运行 `.venv/bin/python prepare_data.py`。MPS 训练使用 FP32。若在 `import torch` 时出现 `KeyboardInterrupt`，说明训练开始前取消了启动，重新运行命令即可。

### Windows / NVIDIA CUDA

**训练电脑提供的配置（硬件信息采集于 2026-10-09；80 轮训练已完成，但未记录完整运行库版本）：**

| 项目 | 提供的数值 |
|---|---|
| 项目目录 | `E:\write` |
| Python | 3.14.7，64 位 |
| GPU | NVIDIA GeForce RTX 4070 系列；提供的输出截断了完整型号 |
| 显存 | 总计 8,188 MiB；截图时已使用 5,209 MiB |
| NVIDIA 驱动 | 616.64 |
| 本机 CUDA Toolkit（`nvcc`） | 13.4.92 |

沿用现有的标准 Python 3.14 环境。版本检查命令为 `python --version`，需要两个短横线。以下安装固定为 **PyTorch 2.11.0 + torchvision 0.26.0，CUDA 12.8**：这是[官方版本配对](https://pytorch.org/get-started/previous-versions/)，且 [torch](https://download.pytorch.org/whl/cu128/torch/) 与 [torchvision](https://download.pytorch.org/whl/cu128/torchvision/) 索引均提供 Python 3.14 的 Windows wheel。这是明确选定的安装组合，不代表最新版本。

`nvcc` 显示的 CUDA Toolkit、`nvidia-smi` 显示的驱动能力，以及 PyTorch 使用的 CUDA 运行时属于不同版本。较新的 NVIDIA 驱动通过[向后兼容](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html)支持较旧的 CUDA 运行时。保留已安装的 Toolkit 即可；本项目使用预编译 wheel，不编译 CUDA 扩展。不要因为本机 Toolkit 是 13.4 就把安装索引改为 `cu134`。选择其他安装组合时使用[官方安装选择器](https://pytorch.org/get-started/locally/)。

在 PowerShell 中进入项目目录，无需激活环境即可安装：

```powershell
cd E:\write
python --version
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --upgrade pip
.\.venv-win\Scripts\python.exe -m pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv-win\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv-win\Scripts\python.exe -m pip check
```

PyTorch 与 `requirements-windows.txt` 分开安装；Mac 的 `requirements.txt` 不作为 Windows 安装方案。将完整处理数据复制到 `data/processed/`。使用原始文件时，请在环境检查前运行 `.\.venv-win\Scripts\python.exe prepare_data.py`。

检查完整显卡名称、已安装的运行时、实际 CUDA 运算与数据路径：

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
.\.venv-win\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
.\.venv-win\Scripts\python.exe check_environment.py --device cuda
```

预期包版本与运行时为 `2.11.0+cu128`、`12.8`；CUDA 可用性应为 `True`，设备检查应报告 `gpu_kernel_check: passed`。它们与本机 Toolkit 版本不同是正常现象。仅凭提供的系统输出，还不能证明 PyTorch 已能执行 CUDA kernel。

**约 8 GB 显存的起始设置：**先关闭不必要的 GPU 应用。截图时仅剩约 2.9 GiB 空闲，因此总显存不等于训练可用显存。复制默认配置，建立独立实验：

```powershell
Copy-Item configs\windows_cuda.yaml configs\windows_8gb.yaml
```

开始全新训练前，在 `configs/windows_8gb.yaml` 中修改以下字段，其余设置保留：

```yaml
run_dir: runs/windows_8gb
batch_size: 8
accumulation: 8
```

保留 `precision: fp16` 与 `image_size: 128`，有效 batch 仍为 64。这是较保守的起始设置，不保证一定能容纳。若 CUDA 报显存不足，在新训练前降低为 `batch_size: 4`、`accumulation: 16`。已有检查点续训时不能改变 batch 与累积次数。

先在独立输出目录运行短流程检查，再从随机初始化开始训练：

```powershell
.\.venv-win\Scripts\python.exe train.py --config configs/windows_8gb.yaml --run-dir runs/windows_8gb_smoke --smoke-steps 8
.\.venv-win\Scripts\python.exe train.py --config configs/windows_8gb.yaml
```

短流程检查点仅供诊断，正常推理会拒绝使用。若加载进程启动失败，可在短流程命令中添加 `--workers 0` 排查。正式训练中断并成功保存后，使用以下命令恢复：

```powershell
.\.venv-win\Scripts\python.exe train.py --config configs/windows_8gb.yaml --resume runs/windows_8gb/last.pt
```

通用的 `configs/windows_cuda.yaml` 仍采用 batch 16、累积 4 次，输出到 `runs/windows_cuda`。上述配置评估与识别时使用 `runs/windows_8gb/best.pt`。更多安装与恢复细节见 [Windows 训练说明](docs/Windows训练.md)。

<a id="training"></a>

## 训练协议

默认 Mac 与 Windows 配置采用以下共同的优化设置。

| 设置 | 数值 |
|---|---|
| 优化器 / 学习率 / 权重衰减 | AdamW / 0.0005 / 0.05 |
| 有效 batch | 64 = 16 × 4 次梯度累积 |
| 抽样 | 每轮有放回抽取 250,000 次；每个样本的权重 ∝ 类别样本数平方根的倒数 |
| 学习率调度 | 2 轮预热、余弦衰减，最多 80 轮 |
| 正则化 | 标签平滑 0.05、DropPath、dropout、轻度几何增强 |
| EMA / 梯度范数裁剪 | 0.999 / 1.0 |
| 模型选择 / 提前停止 | EMA 权重的验证集宏平均 Top-1 / 耐心值 12 |

一轮代表固定的抽样预算，**并非遍历整个训练集**。默认验证子集包含 72,470 张图像，每类 10 张。在新实验中设置 `validation_limit: null` 可使用全量验证集。配置选择使用验证集，测试集留作最终评估。

`last.pt` 保存恢复状态；`best.pt` 保存验证指标最佳的 EMA 检查点。程序每 1,000 次成功优化器更新、每轮结束及正常处理的中断时保存。续训使用相同配置：

```bash
.venv/bin/python train.py --config configs/mac.yaml --resume runs/mac/last.pt
```

恢复时允许改变设备、加载进程数、精度和路径；不兼容的网络、batch 或调度修改会被拒绝。程序没有保存全部随机数生成器状态，因此断点恢复不保证逐位复现。CUDA AMP 溢出时，会同时跳过优化器、调度器与 EMA 更新。

新实验通过 `--run-dir` 指定新的输出目录。扩大模型使用 `configs/quality.yaml`，评估时应显式指定对应检查点，不使用默认路径。

<a id="evaluation"></a>

## 评估与识别

评估完整独立测试集、识别单张图片，或启动本地上传页面：

```bash
.venv/bin/python evaluate.py --checkpoint runs/mac/best.pt --output runs/mac/test_metrics.json
.venv/bin/python predict.py path/to/glyph.png --checkpoint runs/mac/best.pt
.venv/bin/python app.py --checkpoint runs/mac/best.pt
```

Windows 上将 `.venv/bin/python` 替换为 `.\.venv-win\Scripts\python.exe`，并使用 `runs/windows_cuda/best.pt` 与 `runs/windows_cuda/test_metrics.json`。启动 `app.py` 后打开[本地页面](http://127.0.0.1:7860)。已知字符属于汉字时，可在识别命令中添加 `--group chinese`。

使用上述 8 GB 配置时，选择它自己的检查点与报告目录：

```powershell
.\.venv-win\Scripts\python.exe evaluate.py --checkpoint runs/windows_8gb/best.pt --output runs/windows_8gb/test_metrics.json --device cuda
.\.venv-win\Scripts\python.exe predict.py path/to/glyph.png --checkpoint runs/windows_8gb/best.pt --device cuda
.\.venv-win\Scripts\python.exe app.py --checkpoint runs/windows_8gb/best.pt --device cuda
```

评估报告包含 Top-1、Top-5、宏平均 Top-1、汉字/数字/大小写字母分组指标与混淆对。完整测试集包含 870,895 张图像。`--limit` 仅用于子集检查，必须明确标注。

输入应为裁剪清晰的单个字符。整行、文档、范围外字符与复杂背景需要额外模型或数据。当前分数未经校准；`O/0`、`l/I/1` 等外观相近的写法可能需要上下文。由于汉字类别占绝大多数，报告整体成绩时也应提供字符分组指标。

<a id="repository"></a>

## 仓库结构与检查

```text
GlyphWeave/
├── handwriting/
├── configs/
├── docs/
│   └── assets/
├── scripts/check_readme_sync.py
├── tests/
├── prepare_data.py
├── check_environment.py
├── train.py
├── evaluate.py
├── predict.py
├── app.py
├── README.md
├── README.zh-CN.md
└── AGENTS.md
```

`handwriting/` 包含模型、数据集、预处理、运行时与指标实现；`configs/` 定义可复现的训练配置；`docs/` 保存网络结构、平台、数据与实验文档。数据集、环境、检查点和本地训练输出通过 `.gitignore` 排除。

英文为默认 README。[英文 README](README.md) 与中文版必须在内容、结构、命令、链接和结果上保持对应，同步规则已写入 [AGENTS.md](AGENTS.md)。检查文档一致性；安装处理后的数据后，再运行流程测试：

```bash
python3 scripts/check_readme_sync.py
.venv/bin/python -m unittest discover -s tests -v
```

文档检查验证结构与共同的技术内容，翻译质量仍需人工核对。流程测试与小批次拟合用于验证实现行为，不代表泛化能力。CUDA 专项测试需要 NVIDIA GPU，跳过测试不能视为 CUDA 验证通过。发布前可参考 [GitHub 发布检查表](docs/GitHub发布.md)。

## 参考资料

- [ConvNeXt V2 — 网络设计参考与 GRN](https://github.com/facebookresearch/ConvNeXt-V2)
- [EMNIST — 数据集来源](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [CASIA 手写数据库 — 数据集来源](https://nlpr.ia.ac.cn/databases/handwriting/home.html)
- [PyTorch — 训练框架与安装](https://pytorch.org/get-started/locally/)

<a id="experiments"></a>

## 实验结果——单实验

**v1.0.0** 报告默认 GlyphWeave 模型在 Windows CUDA 上从零训练的结果。本节仅包含随机种子为 42 的一个已完成实验。扩大配置、基线和消融尚未评估，不作为已完成结果列出。

### 实验设置

| 项目 | 记录值 |
|---|---|
| 配置 / 初始化 | `configs/windows_cuda.yaml` / 随机初始化，无预训练权重 |
| 参数量 / 输入 / 随机种子 | 7,820,044 / 128 × 128 / 42 |
| 精度 / 有效 batch | FP16 AMP / 64 = 16 × 4 次梯度累积 |
| 优化器 / 学习率 / 权重衰减 | AdamW / 0.0005 / 0.05 |
| 学习率调度 / EMA | 2 轮预热、余弦衰减 / 0.999 |
| 已完成预算 | 80 轮 × 每轮抽取 250,000 张 = 20,000,000 次有放回抽样 |
| 成功更新 / 跳过更新 | 312,445 / 115 |
| 模型选择 | 按验证集宏平均 Top-1 选择最佳 EMA；最佳为第 80 轮 |
| 验证协议 | 固定 72,470 张子集，每类 10 张 |
| 测试协议 | 完整独立划分：870,895 张、7,247 类；`subset: false`、`verification_only: false` |
| 提供的硬件信息 | RTX 4070 系列，显存 8,188 MiB；未记录完整显卡型号、CPU、内存与实测计时 |
| 提供的软件信息 | Windows、64 位 Python 3.14.7；实验文件未记录实际 PyTorch 与 CUDA 运行时版本 |
| 数据集索引 SHA-256 | `ab95cdd227094c2e2b221ff958adce0d42b6b30a76089473316cbf4ec6b380b3` |

原始文件未记录训练代码提交和测试评估检查点的身份。Release 清单通过哈希标识分发文件，不补造缺失的历史来源信息。已保留[配置](docs/experiments/windows_cuda_v1/config.json)、[训练历史](docs/experiments/windows_cuda_v1/history.jsonl)和[测试报告](docs/experiments/windows_cuda_v1/test_metrics.json)，便于检查。

### 验证与完整测试结果

| 划分 | 图像数量 | Top-1 (%) | Top-5 (%) | 宏平均 Top-1 (%) |
|---|---:|---:|---:|---:|
| 验证子集，最佳第 80 轮 | 72,470 | 97.66 | 99.70 | 97.66 |
| 完整独立测试集 | 870,895 | **95.95** | **99.65** | **97.64** |

### 完整测试集字符分组

| 分组 | 测试图像数量 | Top-1 (%) | 典型混淆 |
|---|---:|---:|---|
| 汉字 | 742,069 | **97.59** | 汆 / 氽、谭 / 潭 |
| 数字 | 59,934 | **93.71** | `0 / O`、`1 / I / l` |
| 大写字母 | 36,590 | **83.72** | `O / 0`、`I / 1`、`C / c` |
| 小写字母 | 32,302 | **76.32** | `l / 1`、`c / C`、`s / S` |

评估器未记录分组 Top-5。整体验证/测试差距不应直接解释为过拟合：字母与数字占验证子集的 0.86%，但占测试集的 14.79%。宏平均 Top-1 按 7,247 个字符类别平均，并非四个分组等权。验证分组样本量较少，且组内字符频次不同，也限制了分组成绩的直接比较。

### 训练趋势

验证 Top-1 在第 3 轮达到 88.12%、第 8 轮达到 95.48%、第 20 轮达到 96.68%、第 50 轮达到 97.25%，第 80 轮达到 97.66%。最后 10 轮约增加 0.095 个百分点，收益逐渐减小。最终训练损失为 0.7731，验证损失为 0.1231；由于增强、标签平滑与 EMA 的差异，不应直接将两者之差解释为训练/验证泛化差距。

### 错误分析与适用范围

| 真值 → 预测 | 测试误判图像数量 |
|---|---:|
| `O → 0` | 1,672 |
| `l → 1` | 1,600 |
| `0 → O` | 1,258 |
| `I → 1` | 803 |

这四种有方向的混淆占全部 Top-1 错误的 15.14%。小写字母仍是最弱分组。后续实验应检查字形归一化，并在固定验证协议下比较字符分组采样、标准基线和各网络组件。本版本不宣称完成基线优势证明、消融收益验证、多种子稳健性、延迟、分数校准或外部照片准确率评估。

### 学习曲线与混淆总览

![GlyphWeave 单实验学习曲线与完整测试分析](docs/assets/windows-cuda-v1-overview.png)

[矢量图](docs/assets/windows-cuda-v1-overview.svg) · [可读取的实验摘要](docs/experiments/windows_cuda_v1/experiment-summary.json)。定性图片样例与其他实验完成后可以继续补充，两份 README 必须同步更新。
