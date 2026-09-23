# -*- coding: utf-8 -*-
"""Train reproducible KGE feasibility baselines on the frozen benchmark."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
OUT = HERE / "results"
SEED = 20260915
DIM = 128
BATCH = 4096
NEGATIVES = 8
EPOCHS = 40
EVAL_EVERY = 5
PATIENCE_EVALS = 4


def read(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [tuple(row) for row in csv.reader(handle, delimiter="\t") if row]


class Dataset:
    def __init__(self, split: str, drop_relations=()):
        directory = BENCH / split
        raw_train, raw_dev, raw_test = (read(directory / f"{name}.tsv") for name in ("train", "dev", "test"))
        drop_relations = set(drop_relations)
        raw_train = [edge for edge in raw_train if edge[1] not in drop_relations]
        entities = sorted({x for triples in (raw_train, raw_dev, raw_test) for h, _, t in triples for x in (h, t)})
        relations = sorted({r for triples in (raw_train, raw_dev, raw_test) for _, r, _ in triples})
        self.entity_to_id = {value: index for index, value in enumerate(entities)}
        self.relation_to_id = {value: index for index, value in enumerate(relations)}
        self.entities = entities; self.relations = relations
        self.train = self.encode(raw_train); self.dev = self.encode(raw_dev); self.test = self.encode(raw_test)
        self.n_entities = len(entities); self.n_relations = len(relations)
        self.range_candidates = {}
        for relation_id in range(self.n_relations):
            tails = sorted({int(t) for h, r, t in self.train if int(r) == relation_id})
            self.range_candidates[relation_id] = np.asarray(tails, dtype=np.int64)
        # Frozen benchmark files must already satisfy candidate coverage. Evaluation
        # never filters queries at load time; fail loudly if a future split violates it.
        self.candidate_coverage = {}
        for split_name, triples in (("dev", self.dev), ("test", self.test)):
            missing = [
                (int(h), int(r), int(t)) for h, r, t in triples
                if int(t) not in self.range_candidates[int(r)]
            ]
            if missing:
                raise ValueError(f"{split_name} has {len(missing)} targets outside training-derived relation ranges")
            self.candidate_coverage[split_name] = {"queries": int(len(triples)), "excluded_at_load": 0}
        self.train_codes = {}
        for relation_id in range(self.n_relations):
            codes = [int(h) * self.n_entities + int(t) for h, r, t in self.train if int(r) == relation_id]
            self.train_codes[relation_id] = np.asarray(sorted(set(codes)), dtype=np.int64)
        self.all_true = defaultdict(set)
        for array in (self.train, self.dev, self.test):
            for h, r, t in array:
                self.all_true[(int(h), int(r))].add(int(t))

    def encode(self, triples):
        return np.asarray([(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) for h, r, t in triples], dtype=np.int64)

    def target_relation(self, relation_id: int) -> bool:
        return self.relations[relation_id].startswith("cpm_has_tcmt::")


class KGE(nn.Module):
    def score(self, h, r, t):
        raise NotImplementedError

    def score_tails(self, h, r, tails):
        heads = h.expand_as(tails); relations = r.expand_as(tails)
        return self.score(heads, relations, tails)


class TransE(KGE):
    def __init__(self, n_entities, n_relations, dim):
        super().__init__(); self.entity = nn.Embedding(n_entities, dim); self.relation = nn.Embedding(n_relations, dim)
        nn.init.uniform_(self.entity.weight, -6 / math.sqrt(dim), 6 / math.sqrt(dim)); nn.init.uniform_(self.relation.weight, -6 / math.sqrt(dim), 6 / math.sqrt(dim))

    def score(self, h, r, t):
        he = F.normalize(self.entity(h), dim=-1); te = F.normalize(self.entity(t), dim=-1)
        return 12.0 - (he + self.relation(r) - te).abs().sum(-1)


class DistMult(KGE):
    def __init__(self, n_entities, n_relations, dim):
        super().__init__(); self.entity = nn.Embedding(n_entities, dim); self.relation = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight); nn.init.xavier_uniform_(self.relation.weight)

    def score(self, h, r, t):
        return (self.entity(h) * self.relation(r) * self.entity(t)).sum(-1)


class ComplEx(KGE):
    def __init__(self, n_entities, n_relations, dim):
        super().__init__(); self.er = nn.Embedding(n_entities, dim); self.ei = nn.Embedding(n_entities, dim); self.rr = nn.Embedding(n_relations, dim); self.ri = nn.Embedding(n_relations, dim)
        for parameter in self.parameters(): nn.init.xavier_uniform_(parameter)

    def score(self, h, r, t):
        hr, hi = self.er(h), self.ei(h); rr, ri = self.rr(r), self.ri(r); tr, ti = self.er(t), self.ei(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


MODELS = {"transe": TransE, "distmult": DistMult, "complex": ComplEx}


def sample_negative_tails(data: Dataset, batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    negative = np.empty((len(batch), NEGATIVES), dtype=np.int64)
    for relation_id in np.unique(batch[:, 1]):
        mask = batch[:, 1] == relation_id; heads = batch[mask, 0]
        candidates = data.range_candidates[int(relation_id)]
        sampled = rng.choice(candidates, size=(len(heads), NEGATIVES), replace=True)
        for _ in range(8):
            codes = heads[:, None] * data.n_entities + sampled
            bad = np.isin(codes, data.train_codes[int(relation_id)], assume_unique=False)
            if not bad.any(): break
            sampled[bad] = rng.choice(candidates, size=int(bad.sum()), replace=True)
        negative[mask] = sampled
    return negative


@torch.inference_mode()
def evaluate(model: KGE, data: Dataset, triples: np.ndarray, device, typed: bool) -> dict:
    model.eval(); ranks = []; all_entities = np.arange(data.n_entities, dtype=np.int64)
    for h, r, t in triples:
        candidates = data.range_candidates[int(r)] if typed else all_entities
        # All target tails occur in training, hence are in the typed candidate set.
        candidate_tensor = torch.as_tensor(candidates, device=device)
        scores = model.score_tails(torch.tensor(int(h), device=device), torch.tensor(int(r), device=device), candidate_tensor).float().cpu().numpy()
        known = data.all_true[(int(h), int(r))] - {int(t)}
        if known:
            scores[np.isin(candidates, np.fromiter(known, dtype=np.int64))] = -np.inf
        target_pos = int(np.searchsorted(candidates, int(t))) if typed else int(t)
        target_score = scores[target_pos]
        greater = int(np.sum(scores > target_score)); tied = int(np.sum(scores == target_score)) - 1
        ranks.append(1.0 + greater + 0.5 * max(0, tied))
    ranks = np.asarray(ranks, dtype=np.float64)
    return {"n": len(ranks), "mrr": float(np.mean(1.0 / ranks)), "hits1": float(np.mean(ranks <= 1)), "hits3": float(np.mean(ranks <= 3)), "hits10": float(np.mean(ranks <= 10)), "mean_rank": float(np.mean(ranks))}


def frequency_baseline(data: Dataset, triples: np.ndarray, typed: bool) -> dict:
    frequencies = defaultdict(Counter)
    for _, r, t in data.train:
        frequencies[int(r)][int(t)] += 1
    ranks = []; all_entities = np.arange(data.n_entities, dtype=np.int64)
    for h, r, t in triples:
        candidates = data.range_candidates[int(r)] if typed else all_entities
        scores = np.asarray([frequencies[int(r)][int(candidate)] for candidate in candidates], dtype=np.float64)
        known = data.all_true[(int(h), int(r))] - {int(t)}
        if known: scores[np.isin(candidates, np.fromiter(known, dtype=np.int64))] = -np.inf
        target_pos = int(np.searchsorted(candidates, int(t))) if typed else int(t)
        target_score = scores[target_pos]
        rank = 1.0 + np.sum(scores > target_score) + 0.5 * max(0, int(np.sum(scores == target_score)) - 1)
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {"n": len(ranks), "mrr": float(np.mean(1 / ranks)), "hits1": float(np.mean(ranks <= 1)), "hits3": float(np.mean(ranks <= 3)), "hits10": float(np.mean(ranks <= 10)), "mean_rank": float(np.mean(ranks))}


def train_one(split: str, model_name: str, data: Dataset, device, tag: str | None = None) -> dict:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = MODELS[model_name](data.n_entities, data.n_relations, DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(SEED)
    best = None; stale = 0; start = time.time()
    checkpoint = OUT / f"{tag or split}_{model_name}.pt"
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); permutation = rng.permutation(len(data.train)); losses = []
        for offset in range(0, len(permutation), BATCH):
            batch_np = data.train[permutation[offset:offset + BATCH]]
            negatives_np = sample_negative_tails(data, batch_np, rng)
            batch = torch.as_tensor(batch_np, device=device); negatives = torch.as_tensor(negatives_np, device=device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            positive_score = model.score(h, r, t)
            negative_score = model.score(h[:, None].expand_as(negatives).reshape(-1), r[:, None].expand_as(negatives).reshape(-1), negatives.reshape(-1)).reshape(len(batch), NEGATIVES)
            loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            dev = evaluate(model, data, data.dev, device, typed=True)
            record = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_typed": dev}; history.append(record)
            print(split, model_name, record, flush=True)
            if best is None or dev["mrr"] > best["dev_typed"]["mrr"]:
                best = record; stale = 0
                torch.save({"state_dict": model.state_dict(), "model": model_name, "dim": DIM}, checkpoint)
            else:
                stale += 1
                if stale >= PATIENCE_EVALS: break
    return {"selected": best, "history": history, "seconds": time.time() - start, "checkpoint": checkpoint.name, "parameters": sum(p.numel() for p in model.parameters())}


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for this baseline run")
    OUT.mkdir(exist_ok=True); device = torch.device("cuda")
    selections = {}; datasets = {}
    for split in ("random", "formula_holdout"):
        data = Dataset(split); datasets[split] = data; selections[split] = {}
        for model_name in MODELS:
            selections[split][model_name] = train_one(split, model_name, data, device)
            (OUT / "dev_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    # Test remains untouched until all model classes and both split selections are frozen.
    results = {"config": {"seed": SEED, "dim": DIM, "batch": BATCH, "negatives": NEGATIVES, "epochs": EPOCHS, "lr": 0.003, "weight_decay": 1e-7, "device": str(device), "torch": torch.__version__}}
    for split, data in datasets.items():
        results[split] = {"frequency": {"typed": frequency_baseline(data, data.test, True), "untyped": frequency_baseline(data, data.test, False)}}
        for model_name, cls in MODELS.items():
            checkpoint = torch.load(OUT / selections[split][model_name]["checkpoint"], map_location=device, weights_only=True)
            model = cls(data.n_entities, data.n_relations, checkpoint["dim"]).to(device); model.load_state_dict(checkpoint["state_dict"])
            results[split][model_name] = {"typed": evaluate(model, data, data.test, device, True), "untyped": evaluate(model, data, data.test, device, False), "selected": selections[split][model_name]["selected"]}
            print("TEST", split, model_name, results[split][model_name], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("KGE BASELINES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
