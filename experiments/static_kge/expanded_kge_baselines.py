# -*- coding: utf-8 -*-
"""Expanded KGE baselines under the strict unseen-formula protocol."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compositional_kge import (
    BATCH,
    DIM,
    EPOCHS,
    EVAL_EVERY,
    NEGATIVES,
    OUT,
    PATIENCE_EVALS,
    make_strict_inductive,
    read_compositions,
)
from train_baselines import (
    ComplEx,
    Dataset,
    DistMult,
    TransE,
    evaluate,
    sample_negative_tails,
)

SEEDS = (20260915, 20260916, 20260917)
STRICT_ID_MODELS = {"transe": TransE, "distmult": DistMult, "complex": ComplEx}
COMPOSITIONAL_MODELS = ("mean_complex", "rotate", "pairre", "nodepiece_transformer", "set_transformer")


class CompositionalKGE(nn.Module):
    """Generate unseen formula heads from herbs, then apply a standard decoder."""

    def __init__(self, data, arrays, model_name):
        super().__init__()
        if model_name not in COMPOSITIONAL_MODELS:
            raise ValueError(model_name)
        self.model_name = model_name
        ingredients, mask, uniform, *_ = arrays
        self.register_buffer("ingredients", torch.from_numpy(ingredients), persistent=False)
        self.register_buffer("ingredient_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("uniform_weight", torch.from_numpy(uniform), persistent=False)
        self.register_buffer("is_formula", torch.from_numpy(mask.any(1)), persistent=False)

        if model_name in {"mean_complex", "rotate", "nodepiece_transformer", "set_transformer"}:
            self.er = nn.Embedding(data.n_entities, DIM)
            self.ei = nn.Embedding(data.n_entities, DIM)
            nn.init.xavier_uniform_(self.er.weight)
            nn.init.xavier_uniform_(self.ei.weight)
        else:
            self.entity = nn.Embedding(data.n_entities, DIM)
            nn.init.xavier_uniform_(self.entity.weight)

        if model_name in {"mean_complex", "nodepiece_transformer", "set_transformer"}:
            self.rr = nn.Embedding(data.n_relations, DIM)
            self.ri = nn.Embedding(data.n_relations, DIM)
            nn.init.xavier_uniform_(self.rr.weight)
            nn.init.xavier_uniform_(self.ri.weight)
        elif model_name == "rotate":
            self.phase = nn.Embedding(data.n_relations, DIM)
            nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        else:
            self.relation_head = nn.Embedding(data.n_relations, DIM)
            self.relation_tail = nn.Embedding(data.n_relations, DIM)
            nn.init.xavier_uniform_(self.relation_head.weight)
            nn.init.xavier_uniform_(self.relation_tail.weight)

        if model_name == "nodepiece_transformer":
            # Domain-specific NodePiece: herb entities are anchor tokens.
            layer = nn.TransformerEncoderLayer(
                d_model=2 * DIM,
                nhead=4,
                dim_feedforward=2 * DIM,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.token_norm = nn.LayerNorm(2 * DIM)
        if model_name == "set_transformer":
            width = 2 * DIM
            self.set_attn = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
            self.set_ln1 = nn.LayerNorm(width)
            self.set_ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
            self.set_ln2 = nn.LayerNorm(width)
            self.pma_seed = nn.Parameter(torch.empty(1, 1, width))
            self.pma_attn = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
            self.pma_ln = nn.LayerNorm(width)
            self.set_residual_gate = nn.Parameter(torch.tensor(-2.0))
            nn.init.xavier_uniform_(self.pma_seed)

    def complex_head(self, h):
        hr, hi = self.er(h), self.ei(h)
        rows = self.is_formula[h]
        if not bool(rows.any()):
            return hr, hi
        fh = h[rows]
        herbs = self.ingredients[fh]
        valid = self.ingredient_mask[fh]
        ar, ai = self.er(herbs), self.ei(herbs)
        if self.model_name == "nodepiece_transformer":
            tokens = torch.cat([ar, ai], dim=-1)
            encoded = self.token_encoder(tokens, src_key_padding_mask=~valid)
            weights = valid.to(encoded.dtype)
            pooled = (encoded * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
            pooled = self.token_norm(pooled)
            out_r, out_i = pooled.chunk(2, dim=-1)
        elif self.model_name == "set_transformer":
            tokens = torch.cat([ar, ai], dim=-1)
            attended, _ = self.set_attn(tokens, tokens, tokens, key_padding_mask=~valid, need_weights=False)
            encoded = self.set_ln1(tokens + attended)
            encoded = self.set_ln2(encoded + self.set_ff(encoded))
            query = self.pma_seed.expand(len(encoded), -1, -1)
            pooled, _ = self.pma_attn(query, encoded, encoded, key_padding_mask=~valid, need_weights=False)
            pooled = self.pma_ln(query + pooled)[:, 0]
            weights = valid.to(tokens.dtype)
            base = (tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
            pooled = base + torch.sigmoid(self.set_residual_gate) * pooled
            out_r, out_i = pooled.chunk(2, dim=-1)
        else:
            weights = self.uniform_weight[fh]
            out_r = (weights[..., None] * ar).sum(1)
            out_i = (weights[..., None] * ai).sum(1)
        hr, hi = hr.clone(), hi.clone()
        hr[rows], hi[rows] = out_r, out_i
        return hr, hi

    def real_head(self, h):
        result = self.entity(h)
        rows = self.is_formula[h]
        if not bool(rows.any()):
            return result
        fh = h[rows]
        herbs = self.ingredients[fh]
        weights = self.uniform_weight[fh]
        value = (weights[..., None] * self.entity(herbs)).sum(1)
        result = result.clone()
        result[rows] = value
        return result

    def score(self, h, r, t):
        if self.model_name in {"mean_complex", "nodepiece_transformer", "set_transformer"}:
            hr, hi = self.complex_head(h)
            rr, ri = self.rr(r), self.ri(r)
            tr, ti = self.er(t), self.ei(t)
            return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)
        if self.model_name == "rotate":
            hr, hi = self.complex_head(h)
            tr, ti = self.er(t), self.ei(t)
            phase = self.phase(r)
            rr, ri = torch.cos(phase), torch.sin(phase)
            dr = hr * rr - hi * ri - tr
            di = hr * ri + hi * rr - ti
            return 12.0 - torch.sqrt(dr.square() + di.square() + 1e-9).sum(-1)
        he = F.normalize(self.real_head(h), dim=-1)
        te = F.normalize(self.entity(t), dim=-1)
        rh = self.relation_head(r)
        rt = self.relation_tail(r)
        return 12.0 - (he * rh - te * rt).abs().sum(-1)

    def score_negative_tails(self, h, r, tails):
        """Encode each head once and score a B x K negative-tail matrix."""
        if self.model_name in {"mean_complex", "nodepiece_transformer", "set_transformer"}:
            hr, hi = self.complex_head(h)
            rr, ri = self.rr(r), self.ri(r)
            tr, ti = self.er(tails), self.ei(tails)
            return (
                hr[:, None, :] * rr[:, None, :] * tr
                + hi[:, None, :] * rr[:, None, :] * ti
                + hr[:, None, :] * ri[:, None, :] * ti
                - hi[:, None, :] * ri[:, None, :] * tr
            ).sum(-1)
        if self.model_name == "rotate":
            hr, hi = self.complex_head(h)
            tr, ti = self.er(tails), self.ei(tails)
            phase = self.phase(r)
            rr, ri = torch.cos(phase), torch.sin(phase)
            dr = hr[:, None, :] * rr[:, None, :] - hi[:, None, :] * ri[:, None, :] - tr
            di = hr[:, None, :] * ri[:, None, :] + hi[:, None, :] * rr[:, None, :] - ti
            return 12.0 - torch.sqrt(dr.square() + di.square() + 1e-9).sum(-1)
        he = F.normalize(self.real_head(h), dim=-1)
        te = F.normalize(self.entity(tails), dim=-1)
        rh = self.relation_head(r)
        rt = self.relation_tail(r)
        return 12.0 - (he[:, None, :] * rh[:, None, :] - te * rt[:, None, :]).abs().sum(-1)

    def score_tails(self, h, r, tails):
        h = h.reshape(1)
        r = r.reshape(1)
        if self.model_name in {"mean_complex", "nodepiece_transformer", "set_transformer"}:
            hr, hi = self.complex_head(h)
            rr, ri = self.rr(r), self.ri(r)
            tr, ti = self.er(tails), self.ei(tails)
            return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)
        if self.model_name == "rotate":
            hr, hi = self.complex_head(h)
            tr, ti = self.er(tails), self.ei(tails)
            phase = self.phase(r)
            rr, ri = torch.cos(phase), torch.sin(phase)
            dr = hr * rr - hi * ri - tr
            di = hr * ri + hi * rr - ti
            return 12.0 - torch.sqrt(dr.square() + di.square() + 1e-9).sum(-1)
        he = F.normalize(self.real_head(h), dim=-1)
        te = F.normalize(self.entity(tails), dim=-1)
        rh = self.relation_head(r)
        rt = self.relation_tail(r)
        return 12.0 - (he * rh - te * rt).abs().sum(-1)


def initialise_model(data, arrays, name):
    if name in STRICT_ID_MODELS:
        return STRICT_ID_MODELS[name](data.n_entities, data.n_relations, DIM)
    return CompositionalKGE(data, arrays, name)


def train_one(data, arrays, name, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = initialise_model(data, arrays, name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(seed)
    best, stale, history, started = None, 0, [], time.time()
    checkpoint = OUT / f"expanded_{name}_seed{seed}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(data.train))
        losses = []
        for offset in range(0, len(order), BATCH):
            raw = data.train[order[offset:offset + BATCH]]
            negative_np = sample_negative_tails(data, raw, rng)
            batch = torch.as_tensor(raw, device=device)
            negative = torch.as_tensor(negative_np, device=device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            positive_score = model.score(h, r, t)
            if isinstance(model, CompositionalKGE):
                negative_score = model.score_negative_tails(h, r, negative)
            else:
                negative_score = model.score(
                    h[:, None].expand_as(negative).reshape(-1),
                    r[:, None].expand_as(negative).reshape(-1),
                    negative.reshape(-1),
                ).reshape(len(batch), NEGATIVES)
            loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            dev = evaluate(model, data, data.dev, device, True)
            record = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_typed": dev}
            history.append(record)
            print(name, seed, record, flush=True)
            if best is None or dev["mrr"] > best["dev_typed"]["mrr"]:
                best, stale = record, 0
                torch.save({"state_dict": model.state_dict(), "name": name, "seed": seed}, checkpoint)
            else:
                stale += 1
                if stale >= PATIENCE_EVALS:
                    break
    return {
        "selected": best,
        "history": history,
        "checkpoint": checkpoint.name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.time() - started,
    }


def summarize(seed_results):
    return {
        metric: {
            "mean": float(np.mean([seed_results[str(seed)]["typed"][metric] for seed in SEEDS])),
            "sample_std": float(np.std([seed_results[str(seed)]["typed"][metric] for seed in SEEDS], ddof=1)),
        }
        for metric in ("mrr", "hits1", "hits3", "hits10", "mean_rank")
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    names = [f"strict_id_{name}" for name in STRICT_ID_MODELS] + list(COMPOSITIONAL_MODELS)
    dev_path = OUT / "expanded_kge_baselines_dev.json"
    selections = json.loads(dev_path.read_text(encoding="utf-8")) if dev_path.exists() else {"protocol": {}, "models": {}}
    for full_name in names:
        strict_id = full_name.startswith("strict_id_")
        name = full_name.removeprefix("strict_id_") if strict_id else full_name
        selections["models"].setdefault(full_name, {})
        for seed in SEEDS:
            existing = selections["models"][full_name].get(str(seed))
            if existing is not None and (OUT / existing["checkpoint"]).exists():
                print("RESUME SKIP", full_name, seed, flush=True)
                continue
            data = Dataset("hybrid_holdout")
            arrays = read_compositions(data)
            integrity = make_strict_inductive(data)
            if strict_id:
                arrays_for_model = None
            else:
                arrays_for_model = arrays
            selections["protocol"] = {
                "split": "hybrid_holdout",
                "strict_entity_holdout": integrity,
                "seeds": list(SEEDS),
                "dimension": DIM,
                "max_epochs": EPOCHS,
                "selection": "typed filtered dev MRR",
            }
            selections["models"][full_name][str(seed)] = train_one(
                data, arrays_for_model, name, seed, device
            )
            (OUT / "expanded_kge_baselines_dev.json").write_text(
                json.dumps(selections, indent=2), encoding="utf-8"
            )

    results = {"protocol": selections["protocol"], "models": {}}
    for full_name in names:
        strict_id = full_name.startswith("strict_id_")
        name = full_name.removeprefix("strict_id_") if strict_id else full_name
        results["models"][full_name] = {}
        for seed in SEEDS:
            data = Dataset("hybrid_holdout")
            arrays = read_compositions(data)
            make_strict_inductive(data)
            model = initialise_model(data, None if strict_id else arrays, name).to(device)
            chosen = selections["models"][full_name][str(seed)]
            state = torch.load(OUT / chosen["checkpoint"], map_location=device, weights_only=True)
            model.load_state_dict(state["state_dict"])
            results["models"][full_name][str(seed)] = {
                "typed": evaluate(model, data, data.test, device, True),
                "selected": chosen["selected"],
                "parameters": chosen["parameters"],
                "seconds": chosen["seconds"],
            }
        results["models"][full_name]["summary"] = summarize(results["models"][full_name])
        print("EXPANDED TEST", full_name, results["models"][full_name]["summary"], flush=True)
    (OUT / "expanded_kge_baselines_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
