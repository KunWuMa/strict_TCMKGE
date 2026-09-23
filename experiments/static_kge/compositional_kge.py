# -*- coding: utf-8 -*-
"""Strict inductive, composition-aware ComplEx experiments on TCM-MKG."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_baselines import Dataset, evaluate, sample_negative_tails

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MKG = next(ROOT.glob("01_*")) / "TCM-MKG"
OUT = HERE / "results"
SEED = 20260915
DIM = 128
BATCH = 2048
NEGATIVES = 8
EPOCHS = 40
EVAL_EVERY = 5
PATIENCE_EVALS = 4
VARIANTS = ("uniform_mean", "dose_mean", "relation_attention", "dose_relation_attention")
SUPPORTED_VARIANTS = VARIANTS + ("residual_attention", "dose_residual_attention")


def read_compositions(data: Dataset):
    formulas = {}
    doses = {}
    path = MKG / "D4_CPM_CHP.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            f = data.entity_to_id.get("cpm:" + row["CPM_ID"].strip())
            h = data.entity_to_id.get("chp:" + row["CHP_ID"].strip())
            if f is None or h is None:
                continue
            formulas.setdefault(f, []).append(h)
            value = row.get("Dosage_ratio", "").strip()
            doses.setdefault(f, []).append(float(value) if value else None)

    n = data.n_entities
    max_len = max(map(len, formulas.values()))
    ingredients = np.zeros((n, max_len), dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.bool_)
    uniform = np.zeros((n, max_len), dtype=np.float32)
    dosage = np.zeros((n, max_len), dtype=np.float32)
    complete_dose = np.zeros(n, dtype=np.bool_)
    for formula, herbs in formulas.items():
        length = len(herbs)
        ingredients[formula, :length] = herbs
        mask[formula, :length] = True
        uniform[formula, :length] = 1.0 / length
        values = doses[formula]
        if all(value is not None for value in values):
            weights = np.asarray(values, dtype=np.float32)
            weights /= weights.sum()
            dosage[formula, :length] = weights
            complete_dose[formula] = True
        else:
            dosage[formula, :length] = 1.0 / length
    return ingredients, mask, uniform, dosage, complete_dose


def make_strict_inductive(data: Dataset):
    contains_id = data.relation_to_id["cpm_contains_chp"]
    held_heads = set(map(int, data.dev[:, 0])) | set(map(int, data.test[:, 0]))
    keep = np.asarray([
        int(r) != contains_id and int(h) not in held_heads and int(t) not in held_heads
        for h, r, t in data.train
    ], dtype=np.bool_)
    original = len(data.train)
    data.train = data.train[keep]
    # Negative sampling and filtered evaluation must reflect the strict graph.
    for relation_id in range(data.n_relations):
        tails = sorted({int(t) for h, r, t in data.train if int(r) == relation_id})
        data.range_candidates[relation_id] = np.asarray(tails, dtype=np.int64)
        codes = [int(h) * data.n_entities + int(t) for h, r, t in data.train if int(r) == relation_id]
        data.train_codes[relation_id] = np.asarray(sorted(set(codes)), dtype=np.int64)
    for key in list(data.all_true):
        # Retain evaluation truths for filtered metrics; remove only obsolete train-only keys for held heads.
        if key[0] in held_heads:
            eval_tails = {
                int(t) for array in (data.dev, data.test)
                for h, r, t in array if int(h) == key[0] and int(r) == key[1]
            }
            data.all_true[key] = eval_tails
    return {"original_train": original, "strict_train": len(data.train), "held_formula_count": len(held_heads)}


def drop_training_relations(data: Dataset, relation_names):
    """Drop auxiliary relations after strict splitting and rebuild sampling indexes."""
    ids = {data.relation_to_id[name] for name in relation_names}
    data.train = data.train[np.asarray([int(r) not in ids for _, r, _ in data.train], dtype=np.bool_)]
    for relation_id in range(data.n_relations):
        tails = sorted({int(t) for h, r, t in data.train if int(r) == relation_id})
        data.range_candidates[relation_id] = np.asarray(tails, dtype=np.int64)
        codes = [int(h) * data.n_entities + int(t) for h, r, t in data.train if int(r) == relation_id]
        data.train_codes[relation_id] = np.asarray(sorted(set(codes)), dtype=np.int64)


class CompositionalComplEx(nn.Module):
    def __init__(self, data: Dataset, variant: str, arrays):
        super().__init__()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.er = nn.Embedding(data.n_entities, DIM)
        self.ei = nn.Embedding(data.n_entities, DIM)
        self.rr = nn.Embedding(data.n_relations, DIM)
        self.ri = nn.Embedding(data.n_relations, DIM)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        ingredients, mask, uniform, dosage, complete_dose = arrays
        self.register_buffer("ingredients", torch.from_numpy(ingredients), persistent=False)
        self.register_buffer("ingredient_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("uniform_weight", torch.from_numpy(uniform), persistent=False)
        self.register_buffer("dosage_weight", torch.from_numpy(dosage), persistent=False)
        self.register_buffer("complete_dose", torch.from_numpy(complete_dose), persistent=False)
        self.register_buffer("is_formula", torch.from_numpy(mask.any(axis=1)), persistent=False)
        if "attention" in variant:
            self.key_r = nn.Linear(DIM, DIM, bias=False)
            self.key_i = nn.Linear(DIM, DIM, bias=False)
            self.query = nn.Embedding(data.n_relations, DIM)
            nn.init.xavier_uniform_(self.key_r.weight)
            nn.init.xavier_uniform_(self.key_i.weight)
            nn.init.xavier_uniform_(self.query.weight)
        if variant == "dose_relation_attention":
            self.dose_strength = nn.Parameter(torch.zeros(data.n_relations))
        if variant in {"residual_attention", "dose_residual_attention"}:
            # Start near the stable mean aggregator and learn only a supported correction.
            self.attention_gate = nn.Parameter(torch.full((data.n_relations,), -2.0))

    def head_embedding(self, h, r):
        hr, hi = self.er(h), self.ei(h)
        formula_rows = self.is_formula[h]
        if not bool(formula_rows.any()):
            return hr, hi
        fh = h[formula_rows]
        fr = r[formula_rows]
        herbs = self.ingredients[fh]
        valid = self.ingredient_mask[fh]
        her, hei = self.er(herbs), self.ei(herbs)
        if self.variant == "uniform_mean":
            weights = self.uniform_weight[fh]
        elif self.variant == "dose_mean":
            weights = self.dosage_weight[fh]
        else:
            keys = torch.tanh(self.key_r(her) + self.key_i(hei))
            logits = (keys * self.query(fr)[:, None, :]).sum(-1) / math.sqrt(DIM)
            if self.variant == "dose_relation_attention":
                base = self.dosage_weight[fh]
                count = valid.sum(-1, keepdim=True).clamp_min(1)
                log_prior = torch.log((base * count).clamp_min(1e-8))
                strength = F.softplus(self.dose_strength[fr])[:, None]
                logits = logits + strength * log_prior
            logits = logits.masked_fill(~valid, -1e9)
            attention = torch.softmax(logits, dim=-1)
            if self.variant in {"residual_attention", "dose_residual_attention"}:
                base = self.dosage_weight[fh] if self.variant == "dose_residual_attention" else self.uniform_weight[fh]
                gate = torch.sigmoid(self.attention_gate[fr])[:, None]
                weights = (1.0 - gate) * base + gate * attention
            else:
                weights = attention
        aggregated_r = (weights[..., None] * her).sum(1)
        aggregated_i = (weights[..., None] * hei).sum(1)
        hr = hr.clone(); hi = hi.clone()
        hr[formula_rows] = aggregated_r
        hi[formula_rows] = aggregated_i
        return hr, hi

    def score(self, h, r, t):
        hr, hi = self.head_embedding(h, r)
        rr, ri = self.rr(r), self.ri(r)
        tr, ti = self.er(t), self.ei(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    def score_tails(self, h, r, tails):
        # Avoid repeating the composition encoder once per candidate.
        hr, hi = self.head_embedding(h.reshape(1), r.reshape(1))
        rr, ri = self.rr(r), self.ri(r)
        tr, ti = self.er(tails), self.ei(tails)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    @torch.inference_mode()
    def ingredient_weights(self, formula, relation):
        h = torch.tensor([formula], device=self.er.weight.device)
        r = torch.tensor([relation], device=self.er.weight.device)
        herbs = self.ingredients[h]
        valid = self.ingredient_mask[h]
        if self.variant == "uniform_mean":
            weights = self.uniform_weight[h]
        elif self.variant == "dose_mean":
            weights = self.dosage_weight[h]
        else:
            her, hei = self.er(herbs), self.ei(herbs)
            keys = torch.tanh(self.key_r(her) + self.key_i(hei))
            logits = (keys * self.query(r)[:, None, :]).sum(-1) / math.sqrt(DIM)
            if self.variant == "dose_relation_attention":
                base = self.dosage_weight[h]
                count = valid.sum(-1, keepdim=True).clamp_min(1)
                logits += F.softplus(self.dose_strength[r])[:, None] * torch.log((base * count).clamp_min(1e-8))
            logits = logits.masked_fill(~valid, -1e9)
            attention = torch.softmax(logits, -1)
            if self.variant in {"residual_attention", "dose_residual_attention"}:
                base = self.dosage_weight[h] if self.variant == "dose_residual_attention" else self.uniform_weight[h]
                gate = torch.sigmoid(self.attention_gate[r])[:, None]
                weights = (1.0 - gate) * base + gate * attention
            else:
                weights = attention
        return herbs[0, valid[0]], weights[0, valid[0]]


def train_one(data, variant, arrays, device, seed=SEED, tag=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = CompositionalComplEx(data, variant, arrays).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(seed)
    best = None; stale = 0; history = []; start = time.time()
    checkpoint = OUT / (tag or f"strict_composition_{variant}_seed{seed}.pt")
    for epoch in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(data.train)); losses = []
        for offset in range(0, len(order), BATCH):
            batch_np = data.train[order[offset:offset+BATCH]]
            negatives_np = sample_negative_tails(data, batch_np, rng)
            batch = torch.as_tensor(batch_np, device=device)
            negatives = torch.as_tensor(negatives_np, device=device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            pos = model.score(h, r, t)
            flat_h = h[:, None].expand_as(negatives).reshape(-1)
            flat_r = r[:, None].expand_as(negatives).reshape(-1)
            neg = model.score(flat_h, flat_r, negatives.reshape(-1)).reshape(len(batch), NEGATIVES)
            loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            dev = evaluate(model, data, data.dev, device, typed=True)
            record = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_typed": dev}
            history.append(record); print(variant, record, flush=True)
            if best is None or dev["mrr"] > best["dev_typed"]["mrr"]:
                best = record; stale = 0
                torch.save({"state_dict": model.state_dict(), "variant": variant, "seed": seed}, checkpoint)
            else:
                stale += 1
                if stale >= PATIENCE_EVALS:
                    break
    return {"selected": best, "history": history, "checkpoint": checkpoint.name,
            "seconds": time.time()-start, "parameters": sum(p.numel() for p in model.parameters())}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(exist_ok=True)
    data = Dataset("formula_holdout")
    arrays = read_compositions(data)
    strict = make_strict_inductive(data)
    # Eval target ranges must remain non-empty and include every gold target.
    for array in (data.dev, data.test):
        for _, r, t in array:
            assert int(t) in set(map(int, data.range_candidates[int(r)]))
    device = torch.device("cuda")
    selections = {"config": {"seed": SEED, "dim": DIM, "batch": BATCH, "negatives": NEGATIVES,
                              "epochs": EPOCHS, "lr": 0.003, "strict": strict}, "models": {}}
    for variant in VARIANTS:
        selections["models"][variant] = train_one(data, variant, arrays, device)
        (OUT / "composition_dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    results = {"config": selections["config"], "models": {}}
    for variant in VARIANTS:
        chosen = selections["models"][variant]
        model = CompositionalComplEx(data, variant, arrays).to(device)
        checkpoint = torch.load(OUT / chosen["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        result = {"typed": evaluate(model, data, data.test, device, True),
                  "untyped": evaluate(model, data, data.test, device, False),
                  "selected": chosen["selected"], "parameters": chosen["parameters"]}
        results["models"][variant] = result
        print("TEST", variant, result, flush=True)
    (OUT / "composition_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("COMPOSITION SCREEN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
