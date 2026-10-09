# Windows 单张 NVIDIA GPU：从零重新训练

Windows 配置为 `configs/windows_cuda.yaml`。它强制使用 CUDA，默认 FP16 混合精度，batch 16、累积 4 次，有效 batch 64；模型与 Mac 默认版相同。所有可学习参数从随机初始化开始，输出保存在 **`runs/windows_cuda`**。

## 1. 复制项目和数据

把项目复制到 Windows，例如 `D:\write`。保留 Python 文件、`handwriting/`、`configs/`、`docs/`、`tests/`、说明和依赖文件，并复制 **完整 `data/processed/`**：

```text
D:\write\
  train.py
  check_environment.py
  requirements-windows.txt
  handwriting\
  configs\windows_cuda.yaml
  data\processed\
    metadata.json
    index.npy
    raw\...所有提取的数据文件...
```

处理数据约 23 GB，metadata 内是相对路径，可以直接跨系统读取。只复制 index.npy/metadata.json 不够。Windows 重新训练无需复制旧 `runs/mac/`；Mac 的 `.venv/`、`.venv-win/`、`__pycache__/` 和诊断用 `work/` 不用复制。若只复制了原始 CASIA/EMNIST 压缩文件，安装完成后先执行 `prepare_data.py`。

建议项目和处理数据放本地 SSD。原始 ZIP/GZ 可另外备份；已经复制完整 processed 后，正式训练不依赖原始压缩包。

## 2. 安装 Windows 环境

需要 Windows 64 位、NVIDIA CUDA 显卡及适配的驱动，安装标准 64 位 CPython。下面以已安装 Python 3.12 为例；也可使用有相应 PyTorch Windows wheel 的其他受支持版本。

在 PowerShell 中运行，无需激活虚拟环境：

```powershell
cd D:\write
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --upgrade pip
.\.venv-win\Scripts\python.exe -m pip install "torch>=2.5,<3" "torchvision>=0.20,<1" --index-url https://download.pytorch.org/whl/cu128
.\.venv-win\Scripts\python.exe -m pip install -r requirements-windows.txt
```

CUDA 12.8 官方 wheel 仓库提供 Windows 构建，这里作为安装示例。显卡/驱动不匹配时，按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择 Windows、Pip 和适合的 CUDA 构建，不要把网址中的 CUDA 数字随意猜改。安装参考来自 [官方 cu128 wheel 仓库](https://download.pytorch.org/whl/cu128/torch/)。

`requirements.txt` 是 Mac 环境版本记录，Windows 使用上述两步安装，先装 CUDA 版 torch/torchvision，再装 `requirements-windows.txt` 中的其他依赖；后者不会替换已安装的 CUDA 包。项目不需要自行编译 CUDA 扩展。实际安装版本会显示在训练启动日志，可记录 `pip freeze` 以便复现。

## 3. 检查环境和短流程

先检查 CUDA 能否执行计算、显卡信息、数据文件是否齐全：

```powershell
.\.venv-win\Scripts\python.exe check_environment.py --device cuda
```

应看到 `device: cuda` 和 `gpu_kernel_check: passed`。此检查不训练、不生成检查点。若 CUDA 不可用，程序会明确失败，不会默默切换到 CPU。显卡不受当前 wheel 支持时，也会在真实 kernel 检查中报错。

接着跑一次独立的短流程，包括数据加载、网络前向/反向、验证和保存：

```powershell
.\.venv-win\Scripts\python.exe train.py --config configs/windows_cuda.yaml --run-dir runs/windows_smoke --smoke-steps 8
```

这是流程验证模型，无法用于正式识别，输出不会进入正式目录。默认 `workers: 4`；如果子进程启动失败，可在短流程命令末尾加 `--workers 0` 排查，再尝试 2/4 个进程。[PyTorch Windows 多进程说明](https://docs.pytorch.org/docs/2.14/notes/windows.html#usage-multiprocessing)

可运行回归检查：

```powershell
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

其中 CUDA 测试会在有 NVIDIA GPU 时实际运行 FP16 网络更新；没有 CUDA 时会标记跳过，不能把跳过当作 CUDA 测试通过。

## 4. 正式从零训练

```powershell
.\.venv-win\Scripts\python.exe train.py --config configs/windows_cuda.yaml
```

**不要添加 `--resume`，也不用加载 Mac 模型。** 开始日志应包含 `device: cuda`、`precision: fp16`、`verification_only: false` 和显卡名称。已有 Windows 检查点时，程序会拒绝覆盖；新的从零实验可加 `--run-dir runs/windows_cuda_v2`。

每轮有放回抽取 250,000 个样本，最多 80 轮，固定验证子集 72,470 张覆盖 7,247 类；连续 12 轮未刷新最佳各类别平均 Top-1 会提前停止。最佳 EMA 模型为 `runs/windows_cuda/best.pt`，最新进度为 `last.pt`。

CUDA FP16 的梯度溢出会输出 `amp_overflow`，缩小缩放值并跳过该次参数更新，学习率调度与 EMA 也不会推进。少量启动期溢出不等于失败；若持续溢出或损失非有限，先排查，必要时开始新实验用 `precision: fp32`，或在支持的 GPU 上用 `bf16`。[PyTorch AMP 示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)

如果显存不足，在新训练开始前将配置改成 `batch_size: 8`、`accumulation: 8`，保持有效 batch 64。显存较多可另开 batch 32、累积 2 的对照实验，实际速度与效果需要测量，不能只按显存推算。

## 5. Windows 暂停与继续

按 Ctrl+C 后等待“已保存中断前的完整更新”提示，再关闭终端。恢复 Windows 本次训练：

```powershell
.\.venv-win\Scripts\python.exe train.py --config configs/windows_cuda.yaml --resume runs/windows_cuda/last.pt
```

模型、优化器、调度器和 EMA 会恢复。设备、进程数、精度与路径允许改变；batch、累积次数、轮数和模型结构等训练设置仍需匹配。恢复中途批次时会重新遍历并跳过该轮已处理部分，开始计算前可能等待一段时间。未保存全部随机状态，不保证逐位复现。

跨 Mac/CUDA 续训也已兼容空 GradScaler 状态，但本次“重新训练”应使用第 4 节命令。FP32/MPS 检查点迁移到 FP16 时初始化新的 GradScaler，不沿用空状态。

## 6. 完整测试和图片识别

训练方案确定后评估完整独立测试集，评估可能需要较长时间：

```powershell
.\.venv-win\Scripts\python.exe evaluate.py --checkpoint runs/windows_cuda/best.pt --output runs/windows_cuda/test_metrics.json --device cuda
.\.venv-win\Scripts\python.exe predict.py 2.png --checkpoint runs/windows_cuda/best.pt --device cuda
.\.venv-win\Scripts\python.exe app.py --checkpoint runs/windows_cuda/best.pt --device cuda
```

页面地址为 <http://127.0.0.1:7860>。所有命令显式指定 Windows 模型路径，避免默认使用旧 Mac 模型。分组查看汉字、数字、大小写字母成绩，不要以主要由汉字构成的整体成绩代表所有类别效果。

本次适配已在 Mac 上验证 UTF-8、真实 GradScaler 溢出/恢复机制和原有 MPS 管线；目前没有 Windows/NVIDIA 实机，因此 Windows 安装和 CUDA kernel 测试需要在目标机器完成。
