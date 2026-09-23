# -*- coding: utf-8 -*-
"""Global-negative training sensitivity for unrestricted entity ranking."""
from __future__ import annotations
import json,random,time
import numpy as np
import torch
import torch.nn.functional as F
from compositional_kge import BATCH,DIM,EPOCHS,NEGATIVES,OUT,make_strict_inductive,read_compositions
from expanded_kge_baselines import initialise_model
from hybrid_pair_kge import HybridPairComplEx
from train_baselines import Dataset,evaluate

SEEDS=(20260915,20260916,20260917)
MODELS=("paircomp","mean_complex","matched")
EVAL_EPOCHS={1,10,20,30,40}

def sample_global(data,batch,rng):
    neg=rng.integers(0,data.n_entities,size=(len(batch),NEGATIVES),dtype=np.int64)
    for rid in np.unique(batch[:,1]):
        mask=batch[:,1]==rid;heads=batch[mask,0];sampled=neg[mask]
        for _ in range(8):
            codes=heads[:,None]*data.n_entities+sampled
            bad=np.isin(codes,data.train_codes[int(rid)],assume_unique=False)
            if not bad.any():break
            sampled[bad]=rng.integers(0,data.n_entities,size=int(bad.sum()),dtype=np.int64)
        neg[mask]=sampled
    return neg

def make_model(name,data,arrays):
    if name=="paircomp":return HybridPairComplEx(data,"hybrid_pair",arrays)
    if name=="matched":return HybridPairComplEx(data,"dual_mean_control",arrays)
    return initialise_model(data,arrays,"mean_complex")

def negative_scores(model,h,r,tails):
    if isinstance(model,HybridPairComplEx):
        hr,hi=model.head_embedding(h,r);rr,ri=model.rr(r),model.ri(r);tr,ti=model.er(tails),model.ei(tails)
        return (hr[:,None]*rr[:,None]*tr+hi[:,None]*rr[:,None]*ti+hr[:,None]*ri[:,None]*ti-hi[:,None]*ri[:,None]*tr).sum(-1)
    return model.score_negative_tails(h,r,tails)

def train_one(name,seed,data,arrays,device):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=make_model(name,data,arrays).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-7)
    rng=np.random.default_rng(seed);best=None;stale=0;hist=[];start=time.time()
    ckpt=OUT/f"globalneg_{name}_seed{seed}.pt"
    for epoch in range(1,EPOCHS+1):
        model.train();order=rng.permutation(len(data.train));losses=[]
        for off in range(0,len(order),BATCH):
            raw=data.train[order[off:off+BATCH]];neg=sample_global(data,raw,rng)
            b=torch.as_tensor(raw,device=device);nt=torch.as_tensor(neg,device=device)
            h,r,t=b[:,0],b[:,1],b[:,2]
            loss=F.softplus(-model.score(h,r,t)).mean()+F.softplus(negative_scores(model,h,r,nt)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if epoch in EVAL_EPOCHS:
            dev=evaluate(model,data,data.dev,device,False)
            rec={"epoch":epoch,"loss":float(np.mean(losses)),"dev_untyped":dev};hist.append(rec);print(name,seed,rec,flush=True)
            if best is None or dev["mrr"]>best["dev_untyped"]["mrr"]:
                best=rec;stale=0;torch.save({"state_dict":model.state_dict(),"name":name,"seed":seed},ckpt)
            else:
                stale+=1
                if stale>=3:break
    return {"selected":best,"history":hist,"checkpoint":ckpt.name,"parameters":sum(p.numel() for p in model.parameters()),"seconds":time.time()-start}

def summary(rows,metric,kind):
    vals=[rows[str(s)][kind][metric] for s in SEEDS]
    return {"mean":float(np.mean(vals)),"sample_std":float(np.std(vals,ddof=1))}

def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");data=Dataset("hybrid_holdout");arrays=read_compositions(data);make_strict_inductive(data)
    path=OUT/"global_negative_training_dev.json";sel=json.loads(path.read_text()) if path.exists() else {"models":{}}
    for name in MODELS:
        sel["models"].setdefault(name,{})
        for seed in SEEDS:
            old=sel["models"][name].get(str(seed))
            if old and (OUT/old["checkpoint"]).exists():print("RESUME",name,seed,flush=True);continue
            sel["models"][name][str(seed)]=train_one(name,seed,data,arrays,device)
            path.write_text(json.dumps(sel,indent=2),encoding="utf-8")
    result={"protocol":{"negative_sampling":"uniform over all entities; training positives filtered","selection":"untyped filtered development MRR","seeds":list(SEEDS),"dimension":DIM},"models":{}}
    for name in MODELS:
        result["models"][name]={}
        for seed in SEEDS:
            model=make_model(name,data,arrays).to(device)
            state=torch.load(OUT/sel["models"][name][str(seed)]["checkpoint"],map_location=device,weights_only=True)
            model.load_state_dict(state["state_dict"])
            result["models"][name][str(seed)]={"typed":evaluate(model,data,data.test,device,True),"untyped":evaluate(model,data,data.test,device,False),"selected":sel["models"][name][str(seed)]["selected"]}
        result["models"][name]["summary"]={k:{m:summary(result["models"][name],m,k) for m in ("mrr","hits1","hits3","hits10")} for k in ("typed","untyped")}
        print("GLOBAL TEST",name,result["models"][name]["summary"],flush=True)
    (OUT/"global_negative_training_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
if __name__=="__main__":main()
