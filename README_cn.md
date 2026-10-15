# 🔀 SymSelect: Learning Which Transformations to Trust for Systematic Reasoning

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 TL;DR. 同一种结构变换可能保留某些样本的标签，却改变另一些样本的标签。**SymSelect** 根据表征相似度选择需要对齐的候选样本对，在不知道变换有效性标签的情况下，恢复监督有效关系对齐的大部分收益。核心方法位于 [PyTorch 单文件](symselect.py)，并提供可运行的[算术 Demo](demo/demo.py) 与[理论 Demo](demo/theory.py)。

<p align="center">
  <b>📄 arXiv 论文即将发布</b> &nbsp; | &nbsp;
  <a href="#-quickstart"><b>🚀 Quickstart</b></a> &nbsp; | &nbsp;
  <a href="#-demo"><b>🧪 Demo</b></a>
</p>

## 🎥 Overview

<p align="center">
  <img src="assets/figures/fig1.png" alt="SymSelect 论文原图：候选样本对、投影相似度、Top-k 对齐与局部继承" width="900"/>
</p>

**图 1.** 每个样本生成一个候选变换，在投影空间计算相似度，选出整个 mini-batch 中得分最高的 `k` 个样本对。选中更新和有效关系 Oracle 之间的差异来自无效关系污染与被保留有效关系的偏差。

---

## ✨ Highlights

- 🎯 **输入依赖的有效性。** 同一种变换可能对一些样本保持标签不变，对另一些样本改变标签。
- 🧮 **选择性对齐。** 每个候选关系计算一个余弦分数，在整个 mini-batch 中选择 top-`k`。
- 🧭 **无需有效性标签。** 使用模型的投影表示进行选择，不需要知道变换是否真正有效。
- 📈 **恢复 Oracle 收益。** MNS 和 I-RAVEN 的实验中，恢复了监督有效关系对齐收益的 **77%–92%**。
- 🔬 **两类偏差。** 理论区分无效关系污染与保留下来的有效关系带来的梯度偏差。
- ⚡ **轻量且可运行。** [`symselect.py`](symselect.py) 实现核心 PyTorch 目标函数，Demo 展示选择、梯度和理论反例。

---

## 💡 Why SymSelect?

交换 `2 + 5` 的操作数不会改变结果；交换 `2 − 5` 的操作数通常会改变结果。对齐所有变换样本会引入错误约束，而忽略所有变换又浪费了有效的结构监督。Supervised Alignment 可以借助真实有效性标签筛选，但实际训练往往无法获得这些标签。

SymSelect 使用可观测的表示相似度来选择样本对。核心问题是：**不观察有效性标签时，能够保留多少有效关系对齐的收益？**

---

## 📐 What SymSelect Computes

🎯 **候选样本对。** 对每个原始样本 $x_i$ 生成一个候选变换 $T_i$，经过共享 backbone $f_\theta$ 和投影头 $E$ 得到：

$$
z_i = E(f_\theta(x_i)),\qquad z_i' = E(f_\theta(T_i(x_i))).
$$

🧮 **相似度选择。** 在 mini-batch 的 $B$ 个候选样本对之间排序：

$$
S_i=\cos(z_i,z_i'),\qquad \mathcal K=\mathrm{TopK}(S_1,\dots,S_B;k).
$$

🧩 **训练目标。** 分类损失只用于原始样本，选中的样本对用于对齐损失：

$$
L_{\mathrm{align}}=1-\frac{1}{k}\sum_{i\in\mathcal K}S_i,\qquad
L_{\mathrm{SymSelect}}=L_{\mathrm{CE}}+\lambda_{\mathrm{align}}L_{\mathrm{align}}.
$$

⚙️ **实现细节。** Top-`k` 索引在反向传播时固定，但选中相似度仍然可以求导。论文使用 $B=64$、$k=6$、$\lambda_{\mathrm{align}}=1$。

📏 **局部继承条件。** 记 $\rho_k$ 为选中关系的有效性比例，$\epsilon_{\mathrm{sel}}$ 为保留有效关系的梯度偏差。满足论文的平滑性与步长条件时，一个充分条件是：

$$
\mu_J > H_J\bigl(\epsilon_{\mathrm{sel}}+2(1-\rho_k)G\bigr).
$$

即使被选中的样本全部有效，也不能因此消除 $\epsilon_{\mathrm{sel}}$。详见 [Method & Theory](docs/METHOD.md) 和[理论反例](demo/theory.py)。

---

## 🔎 Comparison of Methods

| 方法 | 候选变换的处理 | 需要有效性标签？ |
|---|---|:---:|
| Base | 仅使用原始样本训练分类 | 否 |
| Sym. Sample | 添加有效变换样本进行分类监督，不添加对齐损失 | 是 |
| Align All | 对齐全部候选关系 | 否 |
| Random-`k` | 随机选择 `k` 个关系对齐 | 否 |
| Bottom-`k` | 选择相似度最低的 `k` 个关系对齐 | 否 |
| Validity-matched random-`k` | 保持有效/无效关系的数量，随机化被选中样本的身份 | 是 |
| Sup. Align (Oracle) | 对齐所有真实有效关系 | 是 |
| 🔀 **SymSelect** | **对齐投影余弦相似度最高的 `k` 个关系** | **否** |

---

## ⚙️ Install

Python 3.10 或更新版本：

```bash
git clone https://github.com/AtheavenMa/SymSelect.git
cd SymSelect
python -m pip install -e '.[demo,test]'
python -m pytest -q
```

核心 [`symselect.py`](symselect.py) 仅依赖 PyTorch；可视化 Demo 还需要 Matplotlib。

---

## 🚀 Quickstart

```python
import torch
import torch.nn.functional as F
from symselect import symselect_loss

x = torch.randn(64, 5)
x_t = x[:, [1, 0, 2, 3, 4]]
y = (x[:, 0] + x[:, 1] > 0).long()

backbone = torch.nn.Linear(5, 32)
classifier = torch.nn.Linear(32, 2)
projector = torch.nn.Linear(32, 16)

h, h_t = backbone(x), backbone(x_t)
loss_ce = F.cross_entropy(classifier(h), y)
selection = symselect_loss(projector(h), projector(h_t), k=6)
loss = loss_ce + selection.loss
loss.backward()

print(selection.indices.tolist())
print(selection.loss.item())
```

`selection` 还包含全部候选关系的相似度及选中关系的相似度。完整训练演示见 [Demo Guide](docs/DEMO.md)。

👉 运行算术 Demo：

```bash
python demo/demo.py
```

---

## 📊 Main Results

**测试准确率（%），10 个随机种子的均值 ± 标准差。** SymSelect 在 MNS 的四种 backbone 和 I-RAVEN 的 CCT 上接近监督有效关系对齐的效果。

| 数据集 | Backbone | Base | Sym. Sample | Sup. Align | 🔀 SymSelect |
|---|---|---:|---:|---:|---:|
| MNS | ResNet-MLP | 25.29±0.44 | 27.03±0.52 | 30.05±0.56 | **28.94±0.40** |
| MNS | ViT-B/16 | 22.78±0.34 | 24.33±0.27 | 27.63±0.23 | **27.22±0.39** |
| MNS | CLIP ViT-B/16 | 23.46±0.32 | 26.77±0.38 | 28.36±0.37 | **27.59±0.37** |
| MNS | CCT | 35.74±0.46 | 37.86±0.45 | 40.23±0.41 | **39.18±0.49** |
| I-RAVEN | CCT | 75.62±0.42 | 77.25±0.46 | 79.34±0.53 | **78.97±0.58** |

CCT 在 MNS 上恢复监督对齐收益的 **77%**，在 I-RAVEN 上恢复 **90%**。全部五组设置的恢复范围为 77%–92%。

<p align="center">
  <img src="assets/figures/fig2.png" alt="论文原图：MNS 和 I-RAVEN 中原始样本与有效变换样本的训练准确率" width="900"/>
</p>

**图 2.** MNS 和 I-RAVEN 训练过程中原始样本及有效变换样本的准确率。Base 模型出现了有效变换样本上的性能差距，而监督有效关系对齐能够缩小这一差距。

#### 🎯 匹配选择器的对照实验

| 选择方法 | MNS 精度 / 保留率 | MNS 准确率 | I-RAVEN 精度 / 保留率 | I-RAVEN 准确率 |
|---|:---:|---:|:---:|---:|
| Random-`k` | .28 / .10 | 36.27±0.44 | .29 / .08 | 75.96±0.45 |
| Validity-matched random-`k` | .74 / .23 | 37.52±0.41 | .81 / .29 | 77.21±0.39 |
| 🔀 **SymSelect** | **.74 / .23** | **39.18±0.49** | **.81 / .29** | **78.97±0.58** |

在相同的选中有效性组成下，SymSelect 在 MNS 和 I-RAVEN 上仍分别领先 **1.66** 与 **1.76** 个百分点。

#### 🔬 Selector Dynamics 与局部收益

<p align="center">
  <img src="assets/figures/fig3.png" alt="论文原图：选中有效率、有效关系保留率、梯度偏差及局部更新" width="900"/>
</p>

**图 3.** 在一条 I-RAVEN 训练轨迹上，选中关系的有效性提高、保留有效关系的梯度偏差降低，同时选中对齐更新带来正向的局部诊断改善。

更多对照结果见 [Paper Results](docs/PAPER_RESULTS.md)，原论文图件见 [Figure Map](docs/FIGURE_MAP.md)。

---

## 🧪 Demo

👉 **算术样本对选择。** 小型 PyTorch 网络学习合成算术表达式，计算交换操作数后的候选关系相似度，并完成一次分类加对齐的梯度更新。

```bash
python demo/demo.py
python demo/demo.py --seed 2 --batch-size 64 --k 6
```

默认结果输出到 `outputs/demo/`，包括 `selection.png`、`pairs.json` 和 `metrics.json`。有效性标签只用于选择后的诊断。

👉 **理论反例。** 选中关系的有效率为 100%，但选中关系和 Oracle 对局部诊断目标的一阶作用方向相反。

```bash
python demo/theory.py
# 选中更新的局部一阶作用：-1
# Oracle 的局部一阶作用：+1
```

---

## 🗺️ Figure Map

| 图件 | 内容 |
|---|---|
| [图 1](assets/figures/fig1.png) | 候选样本对、相似度、Top-`k` 对齐与局部继承 |
| [图 2](assets/figures/fig2.png) | 原始样本与有效变换样本的训练准确率 |
| [图 3](assets/figures/fig3.png) | I-RAVEN 选择器动态与局部对齐作用 |
| [补充图](assets/figures/fig4.png) | MNS 中有效对称样本与等量随机增强的比较 |

---

## 🙏 Appreciation

感谢 MNS 和 I-RAVEN 的研究者，以及结构化泛化与表征学习领域的贡献者。

📄 **论文。** *SymSelect: Learning Which Transformations to Trust for Systematic Reasoning*. arXiv 论文即将发布。

## 📜 License

[MIT License](LICENSE).
