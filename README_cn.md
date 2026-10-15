# 🔀 SymSelect：输入依赖有效性条件下的对称性对齐

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 TL;DR. 同一种结构变换可能对某些样本保留标签，却改变另一些样本的标签。**SymSelect** 利用表征相似度选择值得对齐的候选样本对，在无需已知变换有效性标签的情况下恢复监督对齐的大部分收益。

<p align="center">
  <b>📄 ICLR 2027 投稿</b> &nbsp; | &nbsp;
  <a href="#-quickstart"><b>🚀 Quickstart</b></a> &nbsp; | &nbsp;
  <a href="#-demo"><b>🧪 Demo</b></a>
</p>

## 🎥 Overview

<p align="center">
  <img src="assets/figures/overview.svg" alt="SymSelect：候选变换、共享编码器、批内相似度选择与训练" width="900"/>
</p>

**图 1.** 为每个原始样本生成一个候选变换，计算两者表征的余弦相似度，在整个 mini-batch 中选择 top-`k` 关系进行对齐；分类损失仍针对原始样本。

---

## ✨ Highlights

- 🎯 **输入依赖的有效性。** 同一种结构变换在不同样本上可能产生不同的标签变化。
- 🧮 **选择性对齐。** 在整个 mini-batch 中计算相似度并选出 top-`k` 候选关系。
- 🧭 **无需有效性标注。** 选择对齐对象时不使用变换是否有效的标签。
- 📈 **接近监督对齐。** 论文在 MNS 和 I-RAVEN 的多个设置上恢复了 77%–92% 的监督对齐收益。
- 🔬 **不仅是选择精度。** 选中的样本全部有效，仍不保证保留完整有效样本群的有益梯度。
- ⚡ **轻量 PyTorch 接口。** [`symselect.py`](symselect.py) 可以直接接入具有投影表征的模型。

---

## 💡 Why SymSelect?

交换 `2 + 5` 的操作数不会改变结果，交换 `2 − 5` 的操作数通常会改变结果。对齐全部变换可能引入错误约束，完全不使用变换则浪费有效的监督信号。

SymSelect 根据模型产生的表示相似度挑选候选关系。理论进一步区分无效关系的污染，以及只保留部分有效关系所造成的梯度偏差。

---

## 📐 How SymSelect Works

对一个 mini-batch 中的原始样本 `x_i`，每个样本生成一个候选变换 `T_i`。使用共享 backbone `f_θ` 和投影头 `E`：

$$
z_i=E(f_\theta(x_i)),\qquad z'_i=E(f_\theta(T_i(x_i))).
$$

计算相似度并选择最高的 `k` 个样本对：

$$
S_i=\cos(z_i,z'_i),\qquad
\mathcal K=\operatorname{TopK}(S_1,\ldots,S_B;k).
$$

优化目标为

$$
L_{\mathrm{align}}=1-\frac{1}{k}\sum_{i\in\mathcal K}S_i,\qquad
L=L_{\mathrm{CE}}+\lambda_{\mathrm{align}}L_{\mathrm{align}}.
$$

🎯 **选择。** Top-`k` 的离散索引不参与求导，但选中的相似度正常传播梯度。

🧩 **监督。** 原始样本用于分类监督，选中的候选关系用于对齐。

⚙️ **论文设置。** `B=64`、`k=6`、`λ_align=1`。

📏 **理论结果。** 除了选中有效率 `ρ_k`，还需要考虑保留的有效关系所产生的梯度偏差 `ε_sel`：

$$
\mu_J>H_J\bigl(\epsilon_{\mathrm{sel}}+2(1-\rho_k)G\bigr).
$$

详细定义见 [Method and Theory](docs/METHOD.md)，精确反例见 [`demo/theory.py`](demo/theory.py)。

---

## 🔎 Comparison of Methods

| 方法 | 候选变换的处理方式 |
|---|---|
| Base | 仅在原始样本上训练分类 |
| Sym. Sample | 把已知有效的变换样本加入分类监督 |
| Align All | 对齐全部候选样本对 |
| Random-`k` | 随机选择 `k` 个样本对 |
| Bottom-`k` | 选择相似度最低的 `k` 个样本对 |
| Validity-matched random-`k` | 匹配每个 batch 的有效/无效组成后随机选择 |
| Sup. Align | 利用有效性标签对齐全部有效关系 |
| 🔀 **SymSelect** | **无需有效性标签，选择相似度最高的 `k` 个样本对** |

---

## ⚙️ Install

```bash
git clone https://github.com/AtheavenMa/SymSelect.git
cd SymSelect
python -m pip install -e '.[demo,test]'
python -m pytest -q
```

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
```

运行完整算术 Demo 见 [`demo/demo.py`](demo/demo.py)。

---

## 📊 Main Results

**论文报告的测试准确率（%），10 个随机种子的均值 ± 标准差。**

| 数据集 | Backbone | Base | Sym. Sample | Sup. Align | 🔀 SymSelect |
|---|---|---:|---:|---:|---:|
| MNS | ResNet-MLP | 25.29±0.44 | 27.03±0.52 | 30.05±0.56 | **28.94±0.40** |
| MNS | ViT-B/16 | 22.78±0.34 | 24.33±0.27 | 27.63±0.23 | **27.22±0.39** |
| MNS | CLIP ViT-B/16 | 23.46±0.32 | 26.77±0.38 | 28.36±0.37 | **27.59±0.37** |
| MNS | CCT | 35.74±0.46 | 37.86±0.45 | 40.23±0.41 | **39.18±0.49** |
| I-RAVEN | CCT | 75.62±0.42 | 77.25±0.46 | 79.34±0.53 | **78.97±0.58** |

<p align="center">
  <img src="assets/figures/recovery.svg" alt="五组模型设置中恢复的监督对齐收益" width="850"/>
</p>

**图 2.** 以 `(SymSelect − Base) / (Sup. Align − Base)` 计算恢复的监督对齐收益比例。

| CCT 方法 | MNS 准确率 | I-RAVEN 准确率 |
|---|---:|---:|
| Random-`k` | 36.27±0.44 | 75.96±0.45 |
| Validity-matched random-`k` | 37.52±0.41 | 77.21±0.39 |
| 🔀 **SymSelect** | **39.18±0.49** | **78.97±0.58** |

SymSelect 在有效性组成相同的情况下，分别高出 1.66 和 1.76 个百分点，说明选中哪些关系本身也很重要。

---

## 🧪 Demo

👉 **算术变换 Demo。** 生成加减乘表达式、交换操作数，用共享神经网络计算相似度、选择 top-`k`，并完成一次真实的联合损失反向传播：

```bash
python demo/demo.py
```

输出保存在 `outputs/demo/`，包括相似度排序可视化、样本对信息和训练诊断。

👉 **理论反例 Demo。** 展示选中有效率 100% 时，局部更新方向仍可能与完整有效关系 Oracle 相反。

```bash
python demo/theory.py
```

详见 [Demo Guide](docs/DEMO.md)。

---

## 🗺️ Figure Map

| 图 | 内容 |
|---|---|
| [Overview](assets/figures/overview.svg) | 变换候选、相似度选择与训练 |
| [Recovery](assets/figures/recovery.svg) | 五组设置的监督对齐收益恢复比例 |
| [Demo](demo/demo.py) | 相似度排序、变换关系和训练指标 |

---

## 🙏 Appreciation

感谢 MNS、I-RAVEN 数据集的开发者，以及在结构化泛化和表征学习领域作出贡献的研究者。

## 📜 License

MIT. 详见 [LICENSE](LICENSE)。

