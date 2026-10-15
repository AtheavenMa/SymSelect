"""SymSelect: within-mini-batch top-k cosine-similarity alignment.

For each original anchor, sample one candidate transformation. The caller
computes classification loss on the original anchors and adds a weighted
SymSelect alignment loss. Validity labels never enter this method.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F

@dataclass(frozen=True)
class SymSelectResult:
    loss: Tensor
    indices: Tensor
    similarities: Tensor
    selected_similarities: Tensor

def symselect_loss(z: Tensor, z_transformed: Tensor, k: int = 6, *, eps: float = 1e-8) -> SymSelectResult:
    """Select k anchor-transformation pairs by projected cosine similarity.

    Inputs must be aligned (B, d) float tensors. The indices are chosen from
    detached scores but the loss retains gradients through selected pairs.
    """
    if not isinstance(z, Tensor) or not isinstance(z_transformed, Tensor):
        raise TypeError("z and z_transformed must both be torch.Tensor")
    if z.ndim != 2 or z_transformed.shape != z.shape:
        raise ValueError(f"expected matching (batch, features) tensors; got {tuple(z.shape)} and {tuple(z_transformed.shape)}")
    if z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError("batch size and feature dimension must both be positive")
    if not z.is_floating_point() or not z_transformed.is_floating_point():
        raise TypeError("projected representations must have a floating-point dtype")
    if z.device != z_transformed.device:
        raise ValueError("z and z_transformed must be on the same device")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > z.shape[0]:
        raise ValueError(f"k must be an integer in [1, {z.shape[0]}], got {k!r}")
    if not isinstance(eps, (int, float)) or not (0.0 < eps < float("inf")):
        raise ValueError("eps must be a finite positive number")
    similarities = F.cosine_similarity(z, z_transformed, dim=1, eps=eps)
    indices = torch.argsort(similarities.detach(), descending=True, stable=True)[:k]
    selected_similarities = similarities.index_select(0, indices)
    return SymSelectResult(1.0 - selected_similarities.mean(), indices, similarities, selected_similarities)

__all__ = ["symselect_loss", "SymSelectResult"]
