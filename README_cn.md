# 🔀 SymSelect：输入依赖有效性条件下的对称性对齐

🌍 语言：[English](README.md) | [中文](README_cn.md)

👉 **概要。** 同一种候选结构变换，对部分样本保留标签，对其他样本可能改变标签。SymSelect 使用投影表示的余弦相似度，在一个 mini-batch 内选择得分最高的 `k` 个候选关系进行对齐，无需预先获取变换有效性标签。

**📄 ICLR 2027 投稿** · [英文说明](README.md)

> **发布范围：** 本仓库提供独立的 PyTorch 参考实现与离线教学 Demo，而非 MNS、I-RAVEN 全实验复现。论文主结果来自所提供的当前版本论文，不来自小规模演示。

## 🎥 Overview

每个原始样本采样一个候选变换。原始输入与变换输入经过共享骨干网络和独立的投影头；在整个 batch 上计算候选关系的余弦相似度并选择 top-k。分类交叉熵仅针对原始样本，选中关系用于对齐损失。

## ✨ 核心特点

- 🎯 同一个结构变换可以对不同输入具有不同的标签有效性。
- 🧮 在 batch 的候选关系之间选择 top-k，而不是每个样本内部重复筛选。
- 🛡️ 选择与训练阶段不使用候选变换有效性标签。
- 📈 当前论文报告，在研究的 MNS 和 I-RAVEN 设置下恢复了监督有效关系对齐收益的 77%–92%。
- 🔬 即使选中有效率是 100%，也不能因此保证保留完整有效关系群体的有益梯度。
- ⚡ 一个独立的 [`symselect.py`](symselect.py) 文件与两个不下载数据集的 Demo。

## 📐 方法

```text
z_i = E(f(x_i))              z'_i = E(f(T_i(x_i)))
S_i = cosine(z_i, z'_i)
K = top_k(S_1, ..., S_B)
L_align = 1 - mean(S_i for i in K)
L_total = L_CE(original anchors) + lambda_align * L_align
```

离散索引不通过梯度传播，但选中关系的余弦分数可以正常反向传播。论文采用 `B=64`、`k=6`、`lambda_align=1` 的参考设置。

理论同时考虑被选中关系中的无效污染，以及被保留的有效关系对完整有效梯度的代表性偏差，不能将选择精度当作全部评价依据。

## ⚙️ 安装与运行

```bash
git clone https://github.com/AtheavenMa/SymSelect.git
cd SymSelect
python -m pip install -e '.[demo,test]'
python -m pytest -q
python demo/demo.py
python demo/theory.py
```

主 Demo 生成合成算术表达式及操作数交换后的候选关系，训练一个小型共享网络，再完成一次真实的联合损失梯度更新；事后展示的有效性标签绝不会送入选择器。

理论 Demo 基于精确的三个候选关系反例：top-1 的有效率为 100%，其更新对目标的局部一阶作用为 -1，而完整有效关系 Oracle 的作用为 +1。

## 📊 论文主要结果

| 数据集 | Base | 有效关系监督对齐 | SymSelect | 恢复的收益 |
|---|---:|---:|---:|---:|
| MNS · CCT | 35.74% | 40.23% | 39.18% | 77% |
| I-RAVEN · CCT | 75.62% | 79.34% | 78.97% | 90% |

这张表是论文中报告的 10 个随机种子结果的均值，而不是 Demo 的执行结果。详情及完整 Quickstart 见 [英文 README](README.md)。

## 📜 许可证

原创参考代码使用 MIT 许可证。第三方数据集、模型和图件仍需遵循各自授权。
