"""CPU checks for the top-k objective."""
import math
import pytest
import torch
from symselect import symselect_loss

def test_manual_topk():
    z=torch.tensor([[1.,0.]]*4,dtype=torch.float64)
    a=torch.tensor([0.,math.pi/3,math.pi/2,math.pi],dtype=torch.float64)
    zt=torch.stack((a.cos(),a.sin()),dim=1)
    r=symselect_loss(z,zt,k=2)
    assert r.indices.tolist()==[0,1]
    assert float(r.loss)==pytest.approx(.25,abs=1e-12)

def test_selected_gradient():
    angles=torch.tensor([.4,.8,1.2],dtype=torch.float64,requires_grad=True)
    z=torch.tensor([[1.,0.]]*3,dtype=torch.float64)
    zt=torch.stack((angles.cos(),angles.sin()),1)
    r=symselect_loss(z,zt,k=1)
    r.loss.backward()
    assert r.indices.tolist()==[0]
    assert angles.grad[0]==pytest.approx(math.sin(.4),abs=1e-12)
    assert not angles.grad[1:].abs().sum()

def test_bad_input():
    z=torch.eye(2)
    for k in (0,3,True,1.1):
        with pytest.raises(ValueError):symselect_loss(z,z,k=k)
    with pytest.raises(ValueError):symselect_loss(z,torch.ones(3,2),k=1)

def test_ties():
    z=torch.ones(4,2)
    assert symselect_loss(z,z,k=2).indices.tolist()==[0,1]

def test_zero_vectors():
    z=torch.zeros(3,2,requires_grad=True)
    r=symselect_loss(z,torch.ones_like(z),k=2)
    r.loss.backward()
    assert torch.isfinite(z.grad).all()
