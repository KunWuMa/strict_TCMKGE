# -*- coding: utf-8 -*-
import torch
from compositional_kge import make_strict_inductive, read_compositions
from pair_interaction_kge import PairCompositionalComplEx
from train_baselines import Dataset


def main():
    data=Dataset("composition_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
    model=PairCompositionalComplEx(data,"relation_pair_residual",arrays)
    h,r,t=map(int,data.dev[0])
    before=model.score(torch.tensor([h]),torch.tensor([r]),torch.tensor([t])).detach()
    with torch.no_grad(): model.er.weight[h].fill_(123.0); model.ei.weight[h].fill_(-456.0)
    after=model.score(torch.tensor([h]),torch.tensor([r]),torch.tensor([t])).detach()
    assert torch.equal(before,after),"formula ID leakage"
    # With the correction gate effectively zero, PairComp must reduce to uniform mean.
    base=PairCompositionalComplEx(data,"relation_pair_residual",arrays)
    base.load_state_dict(model.state_dict())
    with torch.no_grad(): base.interaction_gate.fill_(-100.0)
    mean_r,mean_i=base.er(base.ingredients[torch.tensor([h])]),base.ei(base.ingredients[torch.tensor([h])])
    w=base.uniform_weight[torch.tensor([h])][...,None]
    got_r,got_i=base.head_embedding(torch.tensor([h]),torch.tensor([r]))
    assert torch.allclose(got_r,(w*mean_r).sum(1),atol=1e-6)
    assert torch.allclose(got_i,(w*mean_i).sum(1),atol=1e-6)
    print("pair interaction tests passed")


if __name__ == "__main__": main()
