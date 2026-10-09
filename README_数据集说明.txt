手写字数据集

下载目标：26 个大写字母、26 个小写字母、数字 0–9、7,185 个汉字。

data/EMNIST/
来源：https://www.nist.gov/itl/products-and-services/emnist-dataset
包含官方完整 gzip ZIP 下载包，并提取了 ByClass 的训练图像、训练标签、测试图像、测试标签及类别映射。
ByClass：697,932 个训练样本、116,323 个测试样本；62 类，28×28 灰度图像。
verification.json 记录验证结果及每类样本数量。

data/CASIA-HWDB/
原始来源：https://nlpr.ia.ac.cn/databases/handwriting/Download.html
下载镜像：https://huggingface.co/datasets/OrkaZeta/HWDB1/tree/main
已下载并校验全部 10 个 Gnt1.0–1.2 Train/Test ZIP 数据包，总计 8,651,931,573 字节。
实际检查：1,020 位书写者文件，共 3,895,135 个样本、7,356 类字符。
其中 7,185 种汉字，共 3,721,874 个汉字样本；另含大小写英文字母、数字和符号。
ZIP 内是原始 .gnt 灰度图像与标签数据，不是已经导出的 PNG。
.part 后缀表示尚未完成的下载，不要当作完整数据使用。
download_manifest.json 记录镜像文件大小和 SHA-256 校验码；download_results.json 记录下载结果。
verification.json 记录实际样本与字符覆盖情况。
chinese_characters.txt 是 7,185 个汉字的完整清单。
chinese_character_counts.json 记录每个汉字的样本数。23
SHA256SUMS.txt 记录 10 个 ZIP 的 SHA-256 校验码。
全部 ZIP 均已通过镜像 SHA-256 和 ZIP CRC 完整性验证，并完整扫描 GNT 样本记录。

标签和训练划分
EMNIST 的数字编号需要按 emnist-byclass-mapping.txt 转换为字符。
CASIA GNT 标签使用 GB 编码，合并两套数据时需统一转为字符标签，避免直接混用编号。
官方测试集保持独立；训练脚本从官方训练数据中额外划分验证集。CASIA 还包含英文字母、数字和其他符号。

训练用处理数据（data/processed/）
统一类别：0–9、A–Z、a–z、7,185 个汉字，总计 7,247 类。
CASIA 中非目标符号不进入训练；135 条零尺寸记录及 1 张空白图已过滤。
有效样本：训练 3,353,310；验证 374,648；测试 870,895，三个划分均覆盖全部类别。
CASIA 训练、验证、测试书写者互不重叠。
EMNIST 验证从官方训练集按类别分层抽取，文件未提供书写者 ID，不保证其训练/验证书写者隔离。
EMNIST 原始 IDX 图像需要转置一次，已在数据加载器中处理。
metadata.json 保存类别映射、划分统计与索引校验值；index.npy 为样本内存映射索引。
原始 ZIP/GZ 文件保留，训练操作不会修改它们。

自主神经网络、训练与识别操作见 README.md（英文）与 README.zh-CN.md（中文）；两份 README 后续同步维护。
数据发布与打包要求见 docs/数据集.md；详细结构见 docs/网络设计.md。
