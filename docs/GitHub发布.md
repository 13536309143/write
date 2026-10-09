# GitHub 发布准备

当前仓库的远程地址为 `https://github.com/13536309143/write.git`。本次整理仅准备本地内容，没有提交或推送。

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
git add README.md README.zh-CN.md AGENTS.md .gitignore .gitattributes docs scripts
git diff --cached --stat
git diff --cached --check
```

确认暂存内容再自行提交和推送。公开源码前应由项目所有者确定许可证；本次整理未代选许可证，也不改变数据集授权。

## 数据与实验发布

数据单独上传，依照[数据集说明](数据集.md)附上版本和校验值。发布后在两份 README 的 `DATASET_RELEASE_LINKS` 位置同步填入真实下载链接，不使用本机路径或尚不存在的地址。

完成实验后，依据[报告模板](实验报告模板.md)补充两份 README 末尾。将实际图像放入 `docs/assets/`，再启用 `EXPERIMENT_FIGURES` 图片位置。实验权重单独发布，附对应配置、代码提交与完整测试报告。

默认首页为英文 `README.md`，中文版为 `README.zh-CN.md`；所有后续 README 修改遵循 [AGENTS.md](../AGENTS.md) 的双语同步规则。
