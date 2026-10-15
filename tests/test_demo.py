"""Run educational demos in CI on CPU."""
from demo.theory import example

def test_theory_counterexample():
    x=example()
    assert x["selected_precision"]==1.0
    assert abs(x["selected_first_order_effect_on_J"]+1)<1e-9
    assert abs(x["oracle_first_order_effect_on_J"]-1)<1e-9

def test_toy_validity():
    import torch
    from demo.toy_data import sample_batch, validity_for_display
    b=sample_batch(64,torch.Generator().manual_seed(0))
    v=validity_for_display(b)
    assert v.shape==(64,)
    assert v.dtype==torch.bool
