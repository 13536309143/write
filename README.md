# 手写单字识别：自主网络，从零训练

支持 **7,185 个汉字、26 个大写字母、26 个小写字母和 10 个数字，共 7,247 类**。已实现数据准备、Apple Silicon MPS / Windows 单卡 CUDA 训练、断点续训、独立测试、图片识别和本地上传页面。

**Windows 从零重新训练：先看 [Windows 操作说明](docs/Windows训练.md)，使用 `configs/windows_cuda.yaml`，输出到 `runs/windows_cuda`。**

网络为自行实现的 HandwritingNet：笔画梯度输入 → 四级卷积特征 → 全局结构注意力 → 多尺度融合 → 余弦分类器。借鉴公开研究中的模块，但没有加载任何预训练权重，也没有宣称新的学术原创或最优识别率。详细结构与调研见 [网络设计](docs/网络设计.md)。

## 当前状态

- 原始数据保存在 `data/CASIA-HWDB/`、`data/EMNIST/`，已生成 `data/processed/` 索引。
- 本机 `.venv` 已安装依赖，MPS 可用；完成真实样本梯度测试、多进程加载、模型保存/加载和续训检查。
- Mac 已完成前 5 轮验证，第 5 轮验证子集 Top-1 94.11%、Top-5 98.99%，第 6 轮暂停进度保存在 `runs/mac/last.pt`；尚未完成独立测试。Windows 配置另开从零训练。
- `runs/verification*/`、`runs/windows_smoke/` 只用于流程检查，默认识别程序会拒绝这些模型。
- 默认配置约 782 万参数；扩大配置约 1,405 万参数。后者需要更多计算，准确率是否提高要实测。

## 开始训练

在终端运行：

```bash
cd /Volumes/OUT/write
.venv/bin/python train.py --config configs/mac.yaml
```

默认使用 MPS、FP32、物理 batch 16、梯度累积 4 次，有效 batch 64。每轮有放回抽取 250,000 个训练样本，最多 80 轮；**这里一轮不等于完整遍历 335 万训练样本**。类别采样对样本较少的类别适度加权。日志显示训练损失、速度、验证进度和准确率。

启动时会依次显示依赖加载、数据索引、网络初始化和数据加载器提示，首次批次还需初始化 GPU 运算。若终端出现 `^C` 和 `KeyboardInterrupt`，表示按下 `Ctrl+C` 取消了程序；发生在 `import torch` 时，训练尚未开始，也不会生成本次训练检查点。此时重新运行原命令即可；正常训练中断并保存后才需要 `--resume`。

首次在新环境安装或重新准备数据时：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_data.py
```

原始压缩包保留，处理目录使用内存映射读取，避免将所有图像放入内存；当前处理目录约 23 GB。正常训练无需再次解压。

按 `Ctrl+C` 会保存已完成的优化器更新。继续时使用同一配置：

```bash
.venv/bin/python train.py --config configs/mac.yaml --resume runs/mac/last.pt
```

`last.pt` 用于续训，`best.pt` 是按验证集各类别平均 Top-1 选择的 EMA 模型。每 1,000 次成功优化器更新和每轮结束会保存检查点。续训校验数据索引、类别顺序和训练配置；允许改变设备、加载进程、精度与路径，中途更改 batch、网络、轮数等参数会被拒绝。数据抽样顺序可恢复，但未保存全部随机数状态，续训的增强/Dropout 随机结果不保证逐位一致。

若出现内存不足，开始新实验前把 `batch_size` 改为 8、`accumulation` 改为 8，保持有效 batch 64。加载进程消耗较高时降低 `workers`。新实验使用新目录，避免覆盖检查点：

```bash
.venv/bin/python train.py --config configs/mac.yaml --run-dir runs/experiment2
```

扩大配置另开训练，从随机参数开始：

```bash
.venv/bin/python train.py --config configs/quality.yaml
```

当前机器短流程测试约 28–59 张/秒，多进程启动对短测试影响明显；这些数值不能当成完整训练速度承诺。250,000 样本一轮仅训练部分按此范围约 71–149 分钟，还要加验证和读盘时间；长训练应预留数天，并保持电源、外置盘连接和系统唤醒。

## 测试与识别

训练完成后，先评估完整独立测试集：

```bash
.venv/bin/python evaluate.py --checkpoint runs/mac/best.pt
```

结果保存在 `runs/mac/test_metrics.json`，包括整体 Top-1、Top-5、各类别平均 Top-1、汉字/数字/大小写字母分组准确率、易错字符和混淆对。默认测试全部 870,895 张图片；`--limit` 只做抽样检查，不能作为完整测试结果。

识别一张图片：

```bash
.venv/bin/python predict.py /绝对路径/单字.png --checkpoint runs/mac/best.pt
# 已知图片是汉字时可添加 --group chinese
```

启动本地上传页面：

```bash
.venv/bin/python app.py --checkpoint runs/mac/best.pt
```

浏览器打开 <http://127.0.0.1:7860>。上传图片后显示识别结果和前 5 个候选；图片在内存处理。模型分数没有概率校准，不能理解为真实正确率。

输入必须是清晰的单个字符。支持黑白背景识别、透明背景和等比例归一化；照片请先裁剪，并尽量使用干净背景。整行文字、整页文档、范围外字符识别需要另外设计检测、分割或序列模型。某些手写 `O/0`、`l/I/1`、`C/c` 本身缺少可区分信息，单字模型无法凭空恢复上下文。

扩大配置的评估/页面需显式指定 `runs/quality/best.pt`；评估可同时指定 `--output runs/quality/test_metrics.json`。

## 数据划分与检查

| 划分 | 图像数量 | 覆盖类别 |
|---|---:|---:|
| 训练 | 3,353,310 | 7,247 |
| 验证 | 374,648 | 7,247 |
| 测试 | 870,895 | 7,247 |

CASIA 保留官方测试集，从官方训练书写者中划出验证书写者，三个划分的书写者互不重叠。EMNIST 保留官方测试集，训练/验证按类别分层抽样；原始文件没有书写者 ID，无法证明 EMNIST 训练/验证按书写者隔离。

训练保留 CASIA 中目标汉字及英文字母/数字，排除其他符号；过滤 135 条零尺寸记录和 1 张空白图。EMNIST 原始 IDX 图像在加载时转置一次，已做方向预览检查。默认每轮验证 72,470 张分层子集，覆盖全部类别，避免每轮全量验证时间过长；如需全量验证，在新实验配置中设 `validation_limit: null`。

运行必要的验证：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

这些检查包含真实训练样本的小批次记忆测试，证明梯度和保存链路可用，**不等于泛化准确率**。

## 主要文件

| 文件 | 用途 |
|---|---|
| `handwriting/model.py` | 自主网络与模块 |
| `prepare_data.py`、`handwriting/data.py` | GNT/IDX 数据解析、划分、内存映射 |
| `handwriting/images.py` | 训练和推理共用预处理 |
| `train.py`、`configs/` | 训练、优化、EMA、断点恢复 |
| `evaluate.py`、`predict.py`、`app.py` | 独立评估、命令行识别、本地上传 |
| `docs/网络设计.md` | 结构、设计依据、研究来源和后续改进方法 |
