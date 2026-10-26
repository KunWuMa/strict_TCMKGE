"""Deep Sets phi -> masked sum -> rho, followed by the common ComplEx decoder.

Zaheer et al., Deep Sets, NeurIPS 2017, Section 3.1. This implements the
published architecture family for this task; herb embeddings and ComplEx supply
the task-specific input/output interface. There is no additional mean residual.
"""
import torch
from torch import nn
from compositional_kge import CompositionalComplEx,DIM


class FullDeepSetsComplEx(CompositionalComplEx):
    def __init__(self,data,arrays):
        super().__init__(data,'uniform_mean',arrays)
        width=2*DIM
        self.phi=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width))
        self.rho=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width))
        for module in (self.phi,self.rho):
            for layer in module:
                if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)

    def head_embedding(self,h,r):
        real,imag=self.er(h),self.ei(h)
        rows=self.is_formula[h]
        if not bool(rows.any()):return real,imag
        ids=self.ingredients[h[rows]];mask=self.ingredient_mask[h[rows]]
        tokens=torch.cat((self.er(ids),self.ei(ids)),dim=-1)
        pooled=(self.phi(tokens)*mask[...,None]).sum(1)
        encoded=self.rho(pooled)
        a,b=encoded.chunk(2,dim=-1)
        real,imag=real.clone(),imag.clone();real[rows]=a;imag[rows]=b
        return real,imag
