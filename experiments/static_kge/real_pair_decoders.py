# -*- coding: utf-8 -*-
"""PairComp encoder transfer to DistMult and TransE decoders."""

from __future__ import annotations

import json, math, random, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compositional_kge import (BATCH, DIM, EPOCHS, EVAL_EVERY, NEGATIVES, OUT,
                               PATIENCE_EVALS, make_strict_inductive, read_compositions)
from train_baselines import Dataset, evaluate, sample_negative_tails

SEEDS = (20260915, 20260916, 20260917)
DECODERS = ("distmult", "transe")
ENCODERS = ("mean", "pair")


class RealPairKGE(nn.Module):
    def __init__(self, data, arrays, decoder, encoder):
        super().__init__()
        if decoder not in DECODERS or encoder not in ENCODERS: raise ValueError((decoder, encoder))
        self.decoder, self.encoder = decoder, encoder
        self.entity = nn.Embedding(data.n_entities, DIM)
        self.relation = nn.Embedding(data.n_relations, DIM)
        nn.init.xavier_uniform_(self.entity.weight); nn.init.xavier_uniform_(self.relation.weight)
        ingredients, mask, uniform, *_ = arrays
        self.register_buffer("ingredients", torch.from_numpy(ingredients), persistent=False)
        self.register_buffer("ingredient_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("uniform_weight", torch.from_numpy(uniform), persistent=False)
        self.register_buffer("is_formula", torch.from_numpy(mask.any(1)), persistent=False)
        if encoder == "pair":
            self.pair_transform = nn.Linear(DIM, DIM, bias=False)
            nn.init.xavier_uniform_(self.pair_transform.weight)
            self.pair_gate = nn.Parameter(torch.tensor(-2.0))

    def head_embedding(self, h):
        out = self.entity(h); rows = self.is_formula[h]
        if not bool(rows.any()): return out
        fh = h[rows]; herbs = self.ingredients[fh]; valid = self.ingredient_mask[fh]
        values = self.entity(herbs); weights = self.uniform_weight[fh]
        mean = (weights[..., None]*values).sum(1)
        if self.encoder == "pair":
            m = valid[..., None].to(values.dtype); total = (values*m).sum(1)
            square = (values.square()*m).sum(1); n = valid.sum(1).to(values.dtype)
            pair = (total.square()-square)/(n*(n-1)).clamp_min(1.0)[:, None]
            pair = pair*(n > 1)[:, None]
            mean = mean+torch.sigmoid(self.pair_gate)*torch.tanh(self.pair_transform(pair))
        out = out.clone(); out[rows] = mean; return out

    def score(self, h, r, t):
        he, re, te = self.head_embedding(h), self.relation(r), self.entity(t)
        if self.decoder == "distmult": return (he*re*te).sum(-1)
        return 12.0-(F.normalize(he, dim=-1)+re-F.normalize(te, dim=-1)).abs().sum(-1)

    def score_tails(self, h, r, tails):
        he, re, te = self.head_embedding(h.reshape(1)), self.relation(r).reshape(1, -1), self.entity(tails)
        if self.decoder == "distmult": return (he*re*te).sum(-1)
        return 12.0-(F.normalize(he, dim=-1)+re-F.normalize(te, dim=-1)).abs().sum(-1)


def train(data, arrays, decoder, encoder, seed, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = RealPairKGE(data, arrays, decoder, encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(seed); best=None; stale=0; history=[]; start=time.time()
    checkpoint = OUT/f"decoder_{decoder}_{encoder}_seed{seed}.pt"
    for epoch in range(1, EPOCHS+1):
        model.train(); losses=[]; order=rng.permutation(len(data.train))
        for offset in range(0, len(order), BATCH):
            raw=data.train[order[offset:offset+BATCH]]; neg_np=sample_negative_tails(data,raw,rng)
            batch=torch.as_tensor(raw,device=device); neg=torch.as_tensor(neg_np,device=device)
            h,r,t=batch[:,0],batch[:,1],batch[:,2]; pos=model.score(h,r,t)
            ns=model.score(h[:,None].expand_as(neg).reshape(-1),r[:,None].expand_as(neg).reshape(-1),neg.reshape(-1)).reshape(len(batch),NEGATIVES)
            loss=F.softplus(-pos).mean()+F.softplus(ns).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch==1 or epoch%EVAL_EVERY==0:
            dev=evaluate(model,data,data.dev,device,True); record={"epoch":epoch,"loss":float(np.mean(losses)),"dev_typed":dev}
            history.append(record); print(decoder,encoder,seed,record,flush=True)
            if best is None or dev["mrr"]>best["dev_typed"]["mrr"]:
                best=record; stale=0; torch.save({"state_dict":model.state_dict()},checkpoint)
            else:
                stale+=1
                if stale>=PATIENCE_EVALS: break
    return {"selected":best,"history":history,"checkpoint":checkpoint.name,
            "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.time()-start}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda"); selections={}
    for decoder in DECODERS:
        selections[decoder]={}
        for encoder in ENCODERS:
            selections[decoder][encoder]={}
            for seed in SEEDS:
                data=Dataset("hybrid_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
                selections[decoder][encoder][str(seed)]=train(data,arrays,decoder,encoder,seed,device)
                (OUT/"decoder_transfer_dev.json").write_text(json.dumps(selections,indent=2),encoding="utf-8")
    results={}
    for decoder in DECODERS:
        results[decoder]={}
        for encoder in ENCODERS:
            results[decoder][encoder]={}
            for seed in SEEDS:
                data=Dataset("hybrid_holdout"); arrays=read_compositions(data); make_strict_inductive(data)
                model=RealPairKGE(data,arrays,decoder,encoder).to(device)
                chosen=selections[decoder][encoder][str(seed)]
                model.load_state_dict(torch.load(OUT/chosen["checkpoint"],map_location=device,weights_only=True)["state_dict"])
                results[decoder][encoder][str(seed)]={"typed":evaluate(model,data,data.test,device,True),"selected":chosen["selected"]}
            results[decoder][encoder]["summary"]={metric:{"mean":float(np.mean([results[decoder][encoder][str(s)]["typed"][metric] for s in SEEDS])),
                                                                    "sample_std":float(np.std([results[decoder][encoder][str(s)]["typed"][metric] for s in SEEDS],ddof=1))}
                                                          for metric in ("mrr","hits1","hits3","hits10")}
            print("DECODER TEST",decoder,encoder,results[decoder][encoder]["summary"],flush=True)
    (OUT/"decoder_transfer_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__": main()
