# GitHub 发布准备

仓库地址为 `https://github.com/13536309143/GlyphWeave.git`。源码、数据与模型分别发布；首次完整单实验模型见 [v1.0.0 发布说明](releases/v1.0.0.md)。

源码、配置、测试、文档与未来实验图片进入 Git；`data/`、`runs/`、`work/`、虚拟环境、模型二进制、原始数据和压缩包由 `.gitignore` 排除。实际数据与模型继续保留在本机。

## 提交前检查

在项目根目录运行：

```bash
git status --short
git diff --check
python3 scripts/check_readme_sync.py
git ls-files
git check-ignore data/processed/index.npy runs/mac/best.pt .venv/bin/python
```

`.gitignore` 不会自动撤销之前已跟踪的文件，需检查 `git ls-files`。本次整理时数据集、环境与训练输出未被跟踪；后续每次上传前仍应核对暂存区。

确认改动后，按实际文件选择暂存并查看：

```bash
git add README.md README.zh-CN.md LICENSE AGENTS.md .gitignore .gitattributes docs scripts
git diff --cached --stat
git diff --cached --check
```

确认暂存内容再自行提交和推送。源码与项目原创文档采用根目录的 [MIT 许可证](../LICENSE)；数据集及单独发布的模型检查点不属于该源码许可范围，第三方依赖保留各自许可证。

## 数据与实验发布

数据单独上传，依照[数据集说明](数据集.md)附上版本和校验值。发布后在两份 README 的“数据与下载”部分同步填入真实下载链接，不使用本机路径或尚不存在的地址。

首次 Windows CUDA 完整实验已经写入两份 README 末尾，原始记录存放在 `docs/experiments/windows_cuda_v1/`，图片存放在 `docs/assets/`。后续实验依据[报告模板](实验报告模板.md)补充，保留已完成与未完成实验的区别。模型附件通过 Release 单独发布，附配置、文件校验值及完整测试报告；原始记录缺失的训练代码提交不得事后补造。

默认首页为英文 `README.md`，中文版为 `README.zh-CN.md`；所有后续 README 修改遵循 [AGENTS.md](../AGENTS.md) 的双语同步规则。
