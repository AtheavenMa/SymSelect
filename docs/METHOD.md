# SymSelect method

SymSelect addresses **sample-dependent transformation validity**. Each anchor x has one candidate transformed view T(x), whose label preservation is unknown to the selector. A shared backbone and projection head produce z and z'. Each candidate pair contributes one cosine similarity score S_i.

Select the top-k similarities **within the mini-batch** and use

```text
L_align = 1 - (1/k) sum_{i in top-k} cosine(z_i, z'_i)
L_total = cross_entropy(original_inputs) + lambda_align * L_align
```

Top-k membership is discrete. The selected similarity scores remain differentiable through the original tensors.

Supervised alignment is a validity-label-aware oracle comparator, not an inference requirement of SymSelect. Validity-matched random-k is also diagnostic-only and matches the valid/invalid composition at each batch rather than just globally.

## Inheritance caution

Selection precision alone does not guarantee oracle-like improvement. The theorem controls both invalid-pair contamination and retained-valid gradient bias. A sufficient local bound is

```text
mu_J > H_J * [epsilon_sel + 2(1 - rho_k) G]
```

with the regularity and step-size conditions stipulated in the current manuscript.

See `demo/theory.py` for the exact three-pair PyTorch counterexample.
