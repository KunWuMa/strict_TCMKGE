# -*- coding: utf-8 -*-
"""Adapt the official InGram implementation to the strict TCM formula holdout."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compositional_kge import OUT, make_strict_inductive, read_compositions
from train_baselines import Dataset

HERE = Path(__file__).resolve().parent
INGRAM = HERE / "third_party" / "InGram"
sys.path.insert(0, str(INGRAM))
from dataset import TrainData  # noqa: E402
from model import InGram  # noqa: E402
from relgraph import generate_relation_triplets  # noqa: E402
from utils import generate_neg  # noqa: E402

SEEDS = (20260915, 20260916, 20260917)
D_E = 32
D_R = 32
HDR_E = 8
HDR_R = 4
BINS = 10
NUM_HEAD = 8
LAYERS_E = 2
LAYERS_R = 2
NEGATIVES = 10
LR = 5e-4
MARGIN = 2.0
MAX_EPOCHS = 200
EVAL_EVERY = 10
PATIENCE_EVALS = 100
MESSAGE_FRACTION = 0.75


def write_ascii_train(data):
    directory = HERE / "benchmark" / "ingram_tcm"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train.txt"
    with path.open("w", encoding="ascii", newline="") as handle:
        for h, r, t in data.train:
            handle.write(f"{int(h)}\tr{int(r)}\t{int(t)}\n")
    return directory


def add_inverse(forward, num_rel):
    inverse = forward[:, [2, 1, 0]].copy()
    inverse[:, 1] += num_rel
    return np.concatenate([forward, inverse], axis=0)


def global_composition_edges(data, arrays, heads):
    ingredients, mask, *_ = arrays
    contains = data.relation_to_id["cpm_contains_chp"]
    rows = []
    for head in sorted(set(map(int, heads))):
        for herb in ingredients[head, mask[head]]:
            rows.append((head, contains, int(herb)))
    return np.asarray(rows, dtype=np.int64)


def local_training_compositions(train, data, arrays):
    global_to_local = {int(name): idx for name, idx in train.ent2id.items()}
    ingredients, mask, *_ = arrays
    held = set(map(int, data.dev[:, 0])) | set(map(int, data.test[:, 0]))
    rows = []
    for formula in np.flatnonzero(mask.any(1)):
        formula = int(formula)
        if formula in held or formula not in global_to_local:
            continue
        for herb in ingredients[formula, mask[formula]]:
            herb = int(herb)
            if herb in global_to_local:
                rows.append((global_to_local[formula], train.num_rel, global_to_local[herb]))
    return np.asarray(rows, dtype=np.int64)


def make_eval_message(data, arrays, triples):
    train_formulae = [
        int(entity)
        for entity in np.unique(data.train[:, 0])
        if arrays[1][int(entity)].any()
    ]
    held_formulae = list(map(int, np.unique(triples[:, 0])))
    compositions = global_composition_edges(data, arrays, train_formulae + held_formulae)
    forward = np.concatenate([data.train, compositions], axis=0)
    return add_inverse(forward, data.n_relations)


def initialise_graph(num_ent, num_rel, message, seed):
    torch.manual_seed(seed)
    ent = torch.empty((num_ent, D_E), device="cuda")
    rel = torch.empty((2 * num_rel, D_R), device="cuda")
    gain = torch.nn.init.calculate_gain("relu")
    torch.nn.init.xavier_normal_(ent, gain=gain)
    torch.nn.init.xavier_normal_(rel, gain=gain)
    relation_triplets = generate_relation_triplets(message, num_ent, num_rel, BINS)
    return (
        ent,
        rel,
        torch.as_tensor(message, dtype=torch.long, device="cuda"),
        torch.as_tensor(relation_triplets, dtype=torch.long, device="cuda"),
    )


@torch.inference_mode()
def typed_ranks(model, data, triples, cache):
    model.eval()
    init_ent, init_rel, message, relation_triplets = cache
    emb_ent, emb_rel = model(init_ent, init_rel, message, relation_triplets)
    ranks = []
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        candidates = data.range_candidates[r]
        candidate_tensor = torch.as_tensor(candidates, device="cuda")
        relation = model.rel_proj(emb_rel[r])
        scores = (emb_ent[h] * relation * emb_ent[candidate_tensor]).sum(-1).float().cpu().numpy()
        known = data.all_true[(h, r)] - {t}
        if known:
            scores[np.isin(candidates, np.fromiter(known, dtype=np.int64))] = -np.inf
        target_pos = int(np.searchsorted(candidates, t))
        target_score = scores[target_pos]
        ranks.append(
            1.0 + np.sum(scores > target_score)
            + 0.5 * max(0, int(np.sum(scores == target_score)) - 1)
        )
    return np.asarray(ranks, dtype=np.float64)


def metrics(ranks):
    return {
        "n": int(len(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "hits1": float((ranks <= 1).mean()),
        "hits3": float((ranks <= 3).mean()),
        "hits10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
    }


def build_model():
    return InGram(
        dim_ent=D_E,
        hid_dim_ratio_ent=HDR_E,
        dim_rel=D_R,
        hid_dim_ratio_rel=HDR_R,
        num_bin=BINS,
        num_layer_ent=LAYERS_E,
        num_layer_rel=LAYERS_R,
        num_head=NUM_HEAD,
    ).cuda()


def train_one(train, local_compositions, data, dev_cache, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MarginRankingLoss(margin=MARGIN, reduction="mean")
    best = None
    stale = 0
    history = []
    started = time.time()
    checkpoint = OUT / f"ingram_tcm_full200_seed{seed}.pt"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        split_message, supervision = train.split_transductive(MESSAGE_FRACTION)
        forward = split_message[: len(split_message) // 2]
        forward = np.concatenate([forward, local_compositions], axis=0)
        num_rel = train.num_rel + 1
        message = add_inverse(forward, num_rel)
        init_ent, init_rel, message_tensor, relation_triplets = initialise_graph(
            train.num_ent, num_rel, message, seed + epoch * 1009
        )
        supervision_tensor = torch.as_tensor(supervision, dtype=torch.long, device="cuda")
        emb_ent, emb_rel = model(init_ent, init_rel, message_tensor, relation_triplets)
        positive = model.score(emb_ent, emb_rel, supervision_tensor)
        negative_triples = generate_neg(supervision_tensor, train.num_ent, num_neg=NEGATIVES)
        negative = model.score(emb_ent, emb_rel, negative_triples)
        loss = loss_fn(positive.repeat(NEGATIVES), negative, torch.ones_like(negative))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=False)
        optimizer.step()

        if epoch == 1 or epoch % EVAL_EVERY == 0:
            ranks = typed_ranks(model, data, data.dev, dev_cache)
            dev = metrics(ranks)
            record = {"epoch": epoch, "loss": float(loss.detach()), "dev_typed": dev}
            history.append(record)
            print("ingram", seed, record, flush=True)
            if best is None or dev["mrr"] > best["dev_typed"]["mrr"]:
                best = record
                stale = 0
                torch.save({"state_dict": model.state_dict(), "seed": seed}, checkpoint)
            else:
                stale += 1
                if stale >= PATIENCE_EVALS:
                    break
        del init_ent, init_rel, message_tensor, relation_triplets
        del supervision_tensor, emb_ent, emb_rel, positive, negative, negative_triples, loss
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
            "sample_std": float(np.std(
                [seed_results[str(seed)]["typed"][metric] for seed in SEEDS], ddof=1
            )),
        }
        for metric in ("mrr", "hits1", "hits3", "hits10", "mean_rank")
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    integrity = make_strict_inductive(data)
    train_directory = write_ascii_train(data)
    train = TrainData(str(train_directory) + os.sep)
    local_compositions = local_training_compositions(train, data, arrays)
    dev_message = make_eval_message(data, arrays, data.dev)
    test_message = make_eval_message(data, arrays, data.test)
    graph = SimpleNamespace(num_ent=data.n_entities, num_rel=data.n_relations)

    dev_path = OUT / "ingram_tcm_full200_dev.json"
    selections = json.loads(dev_path.read_text(encoding="utf-8")) if dev_path.exists() else {
        "protocol": {
            "official_repository_commit": "f2554dd0c684bad71d3e01d534f2f7814cf28644",
            "split": "hybrid_holdout",
            "strict": integrity,
            "composition_edges_are_message_only": True,
            "typed_filtered_dev_selection": True,
            "seeds": list(SEEDS),
            "max_epochs": MAX_EPOCHS,
        },
        "models": {},
    }
    for seed in SEEDS:
        existing = selections["models"].get(str(seed))
        if existing is not None and (OUT / existing["checkpoint"]).exists():
            print("RESUME SKIP ingram", seed, flush=True)
            continue
        dev_cache = initialise_graph(
            graph.num_ent, graph.num_rel, dev_message, seed + 700000
        )
        selections["models"][str(seed)] = train_one(
            train, local_compositions, data, dev_cache, seed
        )
        dev_path.write_text(json.dumps(selections, indent=2), encoding="utf-8")
        del dev_cache
        torch.cuda.empty_cache()

    results = {"protocol": selections["protocol"], "models": {}}
    rank_rows = []
    for seed in SEEDS:
        model = build_model()
        state = torch.load(
            OUT / selections["models"][str(seed)]["checkpoint"],
            map_location="cuda",
            weights_only=True,
        )
        model.load_state_dict(state["state_dict"])
        test_cache = initialise_graph(
            graph.num_ent, graph.num_rel, test_message, seed + 700000
        )
        ranks = typed_ranks(model, data, data.test, test_cache)
        rank_rows.append(ranks)
        results["models"][str(seed)] = {
            "typed": metrics(ranks),
            "selected": selections["models"][str(seed)]["selected"],
            "parameters": selections["models"][str(seed)]["parameters"],
            "seconds": selections["models"][str(seed)]["seconds"],
        }
        print("INGRAM TEST", seed, results["models"][str(seed)]["typed"], flush=True)
        del model, test_cache
        torch.cuda.empty_cache()
    results["models"]["summary"] = summarize(results["models"])
    results["per_query_ranks"] = np.stack(rank_rows).tolist()
    (OUT / "ingram_tcm_full200_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("INGRAM COMPLETE", results["models"]["summary"], flush=True)


if __name__ == "__main__":
    main()
