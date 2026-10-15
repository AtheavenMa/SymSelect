# Offline demonstrations

## Arithmetic selection
```bash
python demo/demo.py --output outputs/demo
```
A randomly sampled batch of synthetic arithmetic expressions generates one candidate operand swap per anchor. The training warmup uses classification on original expressions only. The subsequent SymSelect step combines original-anchor cross-entropy with top-k projected cosine alignment.

Outputs: `metrics.json`, `pairs.json`, `selection.png`. Validity labels are computed for **post-hoc visualization**, never passed to the alignment selector.

The arithmetic expressions are not the actual MNS benchmark; this is an educational demonstration, not a reproduced paper accuracy.

## Theoretical counterexample
```bash
python demo/theory.py --output outputs/theory
```
This evaluates a three-pair construction where top-1 achieves perfect validity precision but the local first-order effect on J is -1 rather than the oracle's +1. The check uses actual PyTorch automatic differentiation.

The `--no-plot` option to the arithmetic demo allows a headless smoke test without plotting.
