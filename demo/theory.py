"""Exact paper counterexample: 100% selected validity need not help J."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from symselect import symselect_loss

def example() -> dict:
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    cosine_at_zero = torch.tensor([0.9, 0.8, 0.1], dtype=torch.float64)
    phi = torch.acos(cosine_at_zero)
    contributions = torch.tensor([-1.0, 3.0, 0.0], dtype=torch.float64)
    angles = phi - (contributions / torch.sin(phi)) * theta
    z = torch.tensor([[1., 0.]] * 3, dtype=torch.float64)
    z_t = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    selected = symselect_loss(z, z_t, k=1)
    oracle_loss = (1.0 - selected.similarities[:2]).mean()
    sg = torch.autograd.grad(selected.loss, theta, retain_graph=True)[0]
    og = torch.autograd.grad(oracle_loss, theta)[0]
    out = dict(cosines=[round(float(s), 12) for s in selected.similarities.detach()],
               validity_labels_for_diagnostic_only=[True, True, False],
               selected_indices=selected.indices.tolist(),
               selected_precision=1.0,
               valid_oracle_indices=[0, 1],
               selected_first_order_effect_on_J=-float(sg),
               oracle_first_order_effect_on_J=-float(og), J="theta")
    assert out["selected_indices"] == [0]
    assert abs(out["selected_first_order_effect_on_J"] + 1) < 1e-9
    assert abs(out["oracle_first_order_effect_on_J"] - 1) < 1e-9
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/theory"))
    args = parser.parse_args()
    result = example()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"counterexample.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
