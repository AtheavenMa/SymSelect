"""Offline arithmetic selection demo with real PyTorch gradient update.

Educational synthetic data only, not a reproduction of MNS/I-RAVEN.
Run: python demo/demo.py --output outputs/demo
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from symselect import symselect_loss
from demo.toy_data import sample_batch, encode, labels, validity_for_display, expression

class ArithmeticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(5,96), nn.ReLU(), nn.Linear(96,96), nn.ReLU())
        self.classifier = nn.Linear(96,42)
        self.projector = nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Linear(64,32))
    def forward(self,x):
        h=self.backbone(x)
        return self.classifier(h), self.projector(h)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs/demo"))
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--k",type=int,default=6)
    p.add_argument("--warmup",type=int,default=250)
    p.add_argument("--no-plot",action="store_true")
    args=p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(args.seed)
    net=ArithmeticNet()
    opt=torch.optim.Adam(net.parameters(),lr=0.003)
    for _ in range(args.warmup):
        batch=sample_batch(256,g)
        opt.zero_grad()
        logits,_=net(encode(batch))
        loss=F.cross_entropy(logits,labels(batch))
        loss.backward()
        opt.step()
    batch=sample_batch(args.batch_size,g)
    x,xt=encode(batch),encode(batch,swap=True)
    y=labels(batch)
    opt.zero_grad()
    logits,z=net(x)
    _,zt=net(xt)
    selection=symselect_loss(z,zt,k=args.k)
    ce=F.cross_entropy(logits,y)
    total=ce+selection.loss
    total.backward()
    grad_norm=sum(float(p.grad.norm()) for p in net.projector.parameters() if p.grad is not None)
    valid=validity_for_display(batch)
    picked=selection.indices.tolist()
    scores=selection.similarities.detach().tolist()
    pairs=[{"original":expression(batch,i),"swapped":expression(batch,i,swap=True),
            "similarity":round(scores[i],6),"valid_for_display_only":bool(valid[i]),"selected":i in picked}
           for i in range(args.batch_size)]
    before=float(selection.selected_similarities.mean().detach())
    opt.step()
    with torch.no_grad():
        _,za=net(x)
        _,zta=net(xt)
        all_after=F.cosine_similarity(za,zta,dim=1)
        after=float(all_after[picked].mean())
    result={"source":"synthetic arithmetic demo; not paper benchmark",
            "environment":{"torch":torch.__version__,"python":platform.python_version()},
            "selection":{"batch_size":args.batch_size,"k":args.k,"indices":picked},
            "diagnostics":{"validity_prior":float(valid.float().mean()),"selected_precision":float(valid[picked].float().mean())},
            "optimization":{"cross_entropy_before":float(ce.detach()),"alignment_loss_before":float(selection.loss.detach()),
                            "projector_gradient_norm":grad_norm,"selected_cosine_before":before,
                            "selected_cosine_after":after}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.output/"pairs.json").write_text(json.dumps(pairs,indent=2)+"\n")
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
        axes[0].bar(range(len(scores)),[scores[i] for i in order],
                    color=["#26877c" if bool(valid[i]) else "#ce7356" for i in order])
        axes[0].axvline(args.k-.5,color="#333",linestyle="--")
        axes[0].set(xlabel="Cosine similarity rank",ylabel="Cosine similarity",title="Within-batch selection")
        axes[1].bar(["All candidate pairs","Top-k"],[float(valid.float().mean()),float(valid[picked].float().mean())],
                    color=["#777","#26877c"])
        axes[1].set(ylim=(0,1),ylabel="Validity fraction (post hoc)",title="Selection diagnostics")
        fig.suptitle("SymSelect arithmetic demo · synthetic examples, not MNS")
        fig.savefig(args.output/"selection.png",dpi=170)
        plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
