"""Synthetic arithmetic; NOT the paper's MNS dataset."""
from dataclasses import dataclass
import torch

OPERATORS = ("+", "×", "−")

@dataclass(frozen=True)
class ArithmeticBatch:
    operation: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

def sample_batch(n, generator):
    if n < 1:
        raise ValueError("n must be positive")
    op = torch.multinomial(torch.tensor([.15, .15, .70]), n, replacement=True, generator=generator)
    a = torch.randint(1, 7, (n,), generator=generator)
    b = torch.randint(1, 7, (n,), generator=generator)
    b = torch.where((op == 2) & (a == b), b % 6 + 1, b)
    return ArithmeticBatch(op, a, b)

def encode(batch, *, swap=False):
    a, b = (batch.right, batch.left) if swap else (batch.left, batch.right)
    op = batch.operation
    return torch.stack([a.float()/6, b.float()/6, (op==0).float(), (op==1).float(), (op==2).float()], 1)

def labels(batch, *, swap=False):
    a, b = (batch.right, batch.left) if swap else (batch.left, batch.right)
    v = torch.where(batch.operation == 0, a+b, torch.where(batch.operation==1, a*b, a-b))
    return v+5

def validity_for_display(batch):
    return labels(batch)==labels(batch,swap=True)

def expression(batch,i,*,swap=False):
    a,b=(batch.right[i],batch.left[i]) if swap else (batch.left[i],batch.right[i])
    return f"{int(a)} {OPERATORS[int(batch.operation[i])]} {int(b)}"
