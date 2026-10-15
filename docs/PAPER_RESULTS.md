# Paper Results

The tables below collect the main reported accuracies and selector controls from SymSelect.

## Main accuracy

Test accuracy (%), mean ± standard deviation across 10 seeds.

| Dataset | Backbone | Base | Sym. Sample | Sup. Align | SymSelect |
|---|---|---:|---:|---:|---:|
| MNS | ResNet-MLP | 25.29 ± 0.44 | 27.03 ± 0.52 | 30.05 ± 0.56 | **28.94 ± 0.40** |
| MNS | ViT-B/16 | 22.78 ± 0.34 | 24.33 ± 0.27 | 27.63 ± 0.23 | **27.22 ± 0.39** |
| MNS | CLIP ViT-B/16 | 23.46 ± 0.32 | 26.77 ± 0.38 | 28.36 ± 0.37 | **27.59 ± 0.37** |
| MNS | CCT | 35.74 ± 0.46 | 37.86 ± 0.45 | 40.23 ± 0.41 | **39.18 ± 0.49** |
| I-RAVEN | CCT | 75.62 ± 0.42 | 77.25 ± 0.46 | 79.34 ± 0.53 | **78.97 ± 0.58** |

## Selector controls

| Selector | MNS precision | MNS retention | MNS accuracy | I-RAVEN precision | I-RAVEN retention | I-RAVEN accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Bottom-`k` | .12 | .04 | 35.91 ± 0.38 | .07 | .02 | 75.48 ± 0.37 |
| Random-`k` | .28 | .10 | 36.27 ± 0.44 | .29 | .08 | 75.96 ± 0.45 |
| Validity-matched random-`k` | .74 | .23 | 37.52 ± 0.41 | .81 | .29 | 77.21 ± 0.39 |
| **SymSelect** | **.74** | **.23** | **39.18 ± 0.49** | **.81** | **.29** | **78.97 ± 0.58** |

The validity-matched control retains the same number of valid and invalid pairs at each selection step but randomly assigns the selected identities within those groups.

## Align All

| MNS backbone | Align All | SymSelect |
|---|---:|---:|
| ResNet-MLP | 26.03 | **28.94** |
| ViT-B/16 | 23.11 | **27.22** |
| CLIP ViT-B/16 | 24.98 | **27.59** |
| CCT | 37.54 | **39.18** |

