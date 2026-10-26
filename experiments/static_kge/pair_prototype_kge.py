# -*- coding: utf-8 -*-
"""Train-derived pair prototypes composed into the frozen ComplEx head space.

Formula IDs are lookup conveniences for ingredients; compose() accepts herb IDs
directly. Counts, priors, candidate ranges and prototypes use training data only.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch
from torch import nn

from compositional_kge import DIM, CompositionalComplEx
from pair_memory_dev import PairMemory, ingredients
from pair_prototype_dev import tail_features


class PairPrototypeKGE(nn.Module):
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self,value):
        if min(value)<0 or sum(value)>1:
            raise ValueError(value)
        self._weights=tuple(value)
        if hasattr(self,"cache"):
            self.cache.clear()

    def __init__(self, data, base, weights=(0., .25), ridge=.01,
                 shrinkage=20, min_support=3, relations=None):
        super().__init__()
        if min(weights) < 0 or sum(weights) > 1:
            raise ValueError(weights)
        if type(base) is not CompositionalComplEx or base.variant != "uniform_mean":
            raise ValueError("This composer currently requires the uniform-mean ComplEx backbone")
        self.data, self.base, self.weights = data, base, tuple(weights)
        self.device = base.er.weight.device
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.memories, self.phi, self.prototypes = {}, {}, {}
        self.cache = {}
        started = time.time()
        for r in (data.range_candidates if relations is None else relations):
            r = int(r)
            memory = PairMemory(data, r, shrinkage, min_support)
            phi = tail_features(base, r, data.range_candidates[r])
            centered = phi - phi.mean(0, keepdims=True)
            gram = centered.T @ centered
            strength = ridge * np.trace(gram) / gram.shape[0]
            projector = np.linalg.solve(gram + max(strength,1e-10)*np.eye(2*DIM), centered.T).T
            projected = {}
            for name, counts, totals in (("herb", memory.herb_counts, memory.herb_n),
                                         ("pair", memory.pair_counts, memory.pair_n)):
                values = np.empty((counts.shape[0],2*DIM),dtype=np.float64)
                for start in range(0,len(values),512):
                    ids = list(range(start,min(start+512,len(values))))
                    lift = memory.lift(counts,ids,totals).astype(np.float64)
                    values[start:start+len(ids)] = (lift-lift.mean(1,keepdims=True)) @ projector
                projected[name] = values
            self.memories[r], self.phi[r], self.prototypes[r] = memory, centered, projected
        self.build_seconds = time.time()-started
        self.prototype_bytes = sum(v.nbytes for entry in self.prototypes.values() for v in entry.values())

    @torch.inference_mode()
    def compose(self, herb_ids, relation):
        """Compose from an unordered herb set; no formula-specific vector is read."""
        herbs = tuple(sorted(set(int(h) for h in herb_ids)))
        key = herbs, int(relation)
        if key in self.cache:
            return self.cache[key]
        if not herbs:
            raise ValueError("Formula composition is empty")
        r = int(relation)
        ids = torch.as_tensor(herbs,device=self.device)
        mu = torch.cat((self.base.er(ids).mean(0),self.base.ei(ids).mean(0))).cpu().numpy().astype(np.float64)
        memory, proto, phi = self.memories[r], self.prototypes[r], self.phi[r]
        hids = [memory.herb_vocab[h] for h in herbs if h in memory.herb_vocab]
        pids = [memory.pair_vocab[p] for p in itertools.combinations(herbs,2) if p in memory.pair_vocab]
        herb = proto["herb"][hids].mean(0) if hids else np.zeros(2*DIM)
        pair = proto["pair"][pids].mean(0) if pids else np.zeros(2*DIM)
        normalize = lambda vector: vector/max(float(np.std(phi@vector)),1e-8)
        hw,pw = self.weights
        combined = (1-hw-pw)*normalize(mu)+hw*normalize(herb)+pw*normalize(pair)
        value = torch.as_tensor(combined,dtype=self.base.er.weight.dtype,device=self.device)
        self.cache[key] = value
        return value

    @torch.inference_mode()
    def head_embedding(self,h,r):
        hr,hi = self.base.head_embedding(h,r)
        hr,hi = hr.clone(),hi.clone()
        for i,(head,relation) in enumerate(zip(h.tolist(),r.tolist())):
            if self.data.arrays[1][int(head)].any():
                vector = self.compose(ingredients(self.data,head),int(relation))
                hr[i],hi[i] = vector[:DIM],vector[DIM:]
        return hr,hi

    @torch.inference_mode()
    def score_tails(self,h,r,tails):
        hr,hi = self.head_embedding(h.reshape(1),r.reshape(1))
        rr,ri = self.base.rr(r),self.base.ri(r)
        tr,ti = self.base.er(tails),self.base.ei(tails)
        return (hr*rr*tr+hi*rr*ti+hr*ri*ti-hi*ri*tr).sum(-1)

    @torch.inference_mode()
    def score(self,h,r,t):
        hr,hi = self.head_embedding(h,r)
        rr,ri = self.base.rr(r),self.base.ri(r)
        tr,ti = self.base.er(t),self.base.ei(t)
        return (hr*rr*tr+hi*rr*ti+hr*ri*ti-hi*ri*tr).sum(-1)
