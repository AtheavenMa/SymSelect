# 🔀 SymSelect: Recovering Symmetry Alignment When Validity Depends on the Sample

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 **TL;DR.** A candidate symmetry transformation may preserve labels for some samples but not others. SymSelect selects the top-`k` within-batch candidate pairs by projected representation cosine similarity, without seeing transformation-validity labels. This repository provides a small, reusable PyTorch loss and two offline demonstrations.

**📄 ICLR 2027 submission** · [🚀 Quickstart](#-quickstart) · [🧪 Demo](#-demo)

> **Release scope:** This is a working reference implementation and offline educational demo, **not a complete MNS/I-RAVEN benchmark reproduction**. The reported benchmark numbers below are from the current manuscript and are not outputs of the bundled demo. The original benchmark training code and checkpoints were not present in the provided archive.

## 🎥 Overview

A structural transformation creates a candidate pair whose label validity is unknown. A shared backbone and projector map both views to representations. SymSelect sorts pairwise cosine similarities across the **entire mini-batch** and aligns only the top-`k` pairs; classification is computed on original anchors only.

## ✨ Highlights

- 🎯 **Hidden pair validity:** One transformation can be valid for some inputs and invalid for others.
- 🧮 **Mini-batch selection:** One candidate transformation per anchor, cosine ranking across candidates, and top-`k` alignment.
- 🛡️ **No leakage:** No validity labels or transformed target labels enter SymSelect training.
- 📈 **Paper-reported result:** 77%–92% recovery of the supervised valid-pair alignment gain on the reported MNS and I-RAVEN configurations.
- 🔬 **Beyond validity precision:** Even selecting only valid pairs need not preserve the oracle gradient's beneficial direction.
- ⚡ **Runnable and reusable:** [`symselect.py`](symselect.py) depends only on PyTorch. Both offline demos use synthetic data or an exact theoretical construction.

## 💡 Why SymSelect?

Swapping the operands of `2 + 5` preserves the result; swapping those of `2 − 5` does not. Aligning every pair injects invalid constraints; discarding all pairs sacrifices useful supervisory structure. A validity-label oracle is generally unavailable.

SymSelect uses the model's projected cosine similarity to choose candidates, but similarity is not a perfect validity oracle. The theory separates invalid-pair contamination from a second issue: whether the retained valid subset faithfully represents the full valid-pair gradient.

## 📐 What SymSelect Computes

For original inputs `x_i` and one sampled transformation `T_i` per input, the shared backbone `f` and projection head `E` produce

```text
z_i = E(f(x_i))       z'_i = E(f(T_i(x_i)))
S_i = cosine(z_i, z'_i)
K = top_k(S_1, ..., S_B)
L_align = 1 - mean(S_i for i in K)
L_total = CE(classifier(f(x)), y) + lambda_align * L_align
```

The selection indices are treated as fixed for backpropagation while the selected similarities retain their gradient. The reported paper setting uses mini-batch size `B=64`, `k=6`, and `lambda_align=1`.

The sufficient local oracle-inheritance inequality is

```text
mu_J > H_J * (epsilon_sel + 2 * (1 - rho_k) * G)
```

where `rho_k` is selected validity precision and `epsilon_sel` measures the valid-subset gradient discrepancy. Precision alone cannot eliminate the latter.

## 🔎 Comparison of Methods

| Method | Validity labels? | Alignment |
|---|:---:|---|
| Base | No | None |
| Sym. Sample | Yes | Supervise valid transformed examples; no alignment term |
| Align All | No | All candidates |
| Random-k | No | Uniform k candidates |
| Bottom-k | No | Lowest-similarity k |
| Validity-matched random-k | Yes, diagnostic | Random identities while preserving each mini-batch's selected validity composition |
| Sup. Align (oracle) | Yes | Valid candidate relations |
| **SymSelect** | **No** | **Top-k projected cosine relations** |

These control methods are described for interpreting the paper's experiments; only SymSelect itself is implemented as the core public API.

## ⚙️ Install

```bash
git clone https://github.com/AtheavenMa/SymSelect.git
cd SymSelect
python -m pip install -e '.[demo,test]'
python -m pytest -q
```

No benchmark datasets or checkpoints are downloaded.

## 🚀 Quickstart

```python
import torch
import torch.nn.functional as F
from symselect import symselect_loss

B = 64
x = torch.randn(B, 5)
x_transformed = torch.randn(B, 5)  # Replace with one T(x) per original
y = torch.randint(0, 3, (B,))

backbone = torch.nn.Linear(5, 32)
classifier = torch.nn.Linear(32, 3)
projector = torch.nn.Sequential(
    torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16)
)

h = backbone(x)
h_t = backbone(x_transformed)   # Shared backbone weights
loss_ce = F.cross_entropy(classifier(h), y)   # Original anchors only
selection = symselect_loss(projector(h), projector(h_t), k=6)
loss = loss_ce + selection.loss
loss.backward()

print("top-k indices:", selection.indices)
print("alignment loss:", float(selection.loss.detach()))
```

## 📊 Main Results

**Paper-reported numbers, not demo metrics.** Test accuracy across 10 seeds:

| Benchmark | Base | Supervised alignment | SymSelect | Oracle gain recovered |
|---|---:|---:|---:|---:|
| MNS · CCT | 35.74% | 40.23% | 39.18% | 77% |
| I-RAVEN · CCT | 75.62% | 79.34% | 78.97% | 90% |

Validity-matched random-`k` reaches 37.52% against SymSelect's 39.18% on MNS, and 77.21% against 78.97% on I-RAVEN. Thus merely matching selected validity composition does not reproduce selection's downstream performance.

These values are copied from the supplied current paper rather than newly rerun. Do not interpret the small synthetic demo as the benchmark experiment.

## 🧪 Demo

```bash
# Full offline arithmetic example: warmup + one actual gradient step
python demo/demo.py --output outputs/arithmetic

# Exact theoretical counterexample using PyTorch autograd
python demo/theory.py --output outputs/theory
```

The arithmetic demo generates `metrics.json`, `pairs.json` and `selection.png`. Its validity flags are used **after** selection for diagnostics only. It is intentionally not the original Machine Number Sense benchmark.

The theory demo demonstrates **100% selected validity precision** but an update effect of **-1**, whereas the full valid-pair oracle has effect **+1** on the example's linear diagnostic objective.

## 🗺️ Figure Map and Release Scope

The complete locally validated release package includes the paper's original illustration assets and extended bilingual documentation. Its included images are derived from the current submission manuscript; legacy or redundant manuscript figures were excluded. The current GitHub transfer is tracked separately from the paper's reported benchmark values.

## 🙏 Appreciation

SymSelect is research about selective alignment under input-dependent transformation validity. Please credit original benchmark datasets, pretrained backbones and third-party components when extending the reference release.

## 📜 License

MIT for the original reference code; third-party materials retain their own rights. See [LICENSE](LICENSE).
