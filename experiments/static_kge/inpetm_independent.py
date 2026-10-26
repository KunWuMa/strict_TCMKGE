# -*- coding: utf-8 -*-
"""Recipe-grouped, formula-inductive InPETM prescription-phenotype benchmark.

The original InPETM files are downloaded separately and are never committed.
This script only builds splits from the original CSVs; it does not create labels
from herb-level phenotype links.  Formula identities and composition sets are
held out jointly to reduce near-duplicate leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from compositional_kge import BATCH, DIM, NEGATIVES, OUT, CompositionalComplEx
from hybrid_pair_kge import HybridPairComplEx
from candidate_pair_kge import CandidatePairComplEx
from relation_sparse_pair_kge import RelationSparsePairComplEx
from train_baselines import evaluate, sample_negative_tails
from tcm_data_paths import inpetm_directory


HERE = Path(__file__).resolve().parent
DATA = inpetm_directory()


class InPETMDataset:
    def __init__(self, source: str, seed: int, eval_groups: int):
        folder = (HERE / "independent_data" / "dragontcm" / "normalized"
                  if source == "DRAGON" else DATA)
        prescriptions = pd.read_csv(folder / "prescription.csv", usecols=["presid", "exdb"],
                                    dtype=str, encoding_errors="replace").dropna()
        compositions = pd.read_csv(folder / "prescription_herb.csv",
                                   usecols=["presid", "herbid"],
                                   dtype=str, encoding_errors="replace").dropna().drop_duplicates()
        links = pd.read_csv(folder / "prescription_phenotype.csv",
                            usecols=["presid", "phenid", "relationtype"],
                            dtype=str, encoding_errors="replace").dropna().drop_duplicates()
        source_ids = set(prescriptions.loc[prescriptions.exdb == source, "presid"])
        compositions = compositions[compositions.presid.isin(source_ids)]
        links = links[links.presid.isin(source_ids)]
        formula_herbs = compositions.groupby("presid").herbid.agg(
            lambda values: tuple(sorted(set(values))))
        formula_herbs = formula_herbs[formula_herbs.index.isin(links.presid)]
        links = links[links.presid.isin(formula_herbs.index)]
        recipe_groups = defaultdict(list)
        for formula, herbs in formula_herbs.items():
            recipe_groups[herbs].append(formula)
        groups = list(recipe_groups.values())
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n_eval = min(eval_groups, max(1, len(groups) // 10))
        dev_groups, test_groups = groups[:n_eval], groups[n_eval:2*n_eval]
        train_groups = groups[2*n_eval:]
        formula_ids = {
            "train": {f for group in train_groups for f in group},
            "dev": {f for group in dev_groups for f in group},
            "test": {f for group in test_groups for f in group},
        }
        if not all(formula_ids.values()):
            raise ValueError("Source too small for a disjoint train/dev/test split")
        train_herbs = set(compositions.loc[
            compositions.presid.isin(formula_ids["train"]), "herbid"])
        train_tails = {
            relation: set(frame.phenid)
            for relation, frame in links[links.presid.isin(formula_ids["train"])].groupby("relationtype")
        }
        audit = {"source": source, "split_seed": seed,
                 "unique_recipe_groups": len(groups),
                 "source_formulas": len(formula_herbs),
                 "source_links_deduplicated": len(links),
                 "split_formula_counts_before_coverage": {
                     key: len(value) for key, value in formula_ids.items()},
                 "source_relation_counts": links.relationtype.value_counts().to_dict(),
                 "coverage": {}}
        accepted = {}
        for split in ("train", "dev", "test"):
            subset = links[links.presid.isin(formula_ids[split])].copy()
            if split == "train":
                accepted[split] = subset
                continue
            ingredient_ok = {f for f in formula_ids[split]
                             if set(formula_herbs[f]) <= train_herbs}
            herb_missing = subset[~subset.presid.isin(ingredient_ok)]
            after_herb = subset[subset.presid.isin(ingredient_ok)]
            range_ok = [tail in train_tails.get(rel, set())
                        for tail, rel in zip(after_herb.phenid, after_herb.relationtype)]
            accepted[split] = after_herb[np.asarray(range_ok, dtype=bool)]
            audit["coverage"][split] = {
                "raw_queries": len(subset),
                "removed_for_unseen_herb": len(herb_missing),
                "removed_for_training_range": len(after_herb) - len(accepted[split]),
                "retained_queries": len(accepted[split]),
                "retained_formulas": accepted[split].presid.nunique(),
            }
        formulas = sorted(set(formula_herbs.index))
        herbs = sorted(set(compositions.herbid))
        tails = sorted(set(links.phenid))
        entities = (["formula:" + value for value in formulas] +
                    ["herb:" + value for value in herbs] +
                    ["phenotype:" + value for value in tails])
        self.entities = entities
        self.entity_to_id = {value: i for i, value in enumerate(entities)}
        self.relations = sorted(links.relationtype.unique())
        self.relation_to_id = {value: i for i, value in enumerate(self.relations)}
        self.n_entities = len(entities)
        self.n_relations = len(self.relations)

        def encode(frame):
            return np.asarray([
                (self.entity_to_id["formula:" + f], self.relation_to_id[r],
                 self.entity_to_id["phenotype:" + t])
                for f, t, r in frame[["presid", "phenid", "relationtype"]].itertuples(
                    index=False, name=None)], dtype=np.int64).reshape(-1, 3)

        self.train = encode(accepted["train"])
        self.dev = encode(accepted["dev"])
        self.test = encode(accepted["test"])
        self.range_candidates = {}
        self.train_codes = {}
        for relation in range(self.n_relations):
            triples = self.train[self.train[:, 1] == relation]
            self.range_candidates[relation] = np.unique(triples[:, 2])
            self.train_codes[relation] = np.unique(
                triples[:, 0] * self.n_entities + triples[:, 2])
        self.all_true = defaultdict(set)
        for triples in (self.train, self.dev, self.test):
            for h, r, t in triples:
                self.all_true[(int(h), int(r))].add(int(t))
        length = max(map(len, formula_herbs.values))
        ingredients = np.zeros((self.n_entities, length), dtype=np.int64)
        mask = np.zeros_like(ingredients, dtype=bool)
        uniform = np.zeros_like(ingredients, dtype=np.float32)
        for formula, herb_set in formula_herbs.items():
            index = self.entity_to_id["formula:" + formula]
            herb_ids = [self.entity_to_id["herb:" + h] for h in herb_set]
            ingredients[index, :len(herb_ids)] = herb_ids
            mask[index, :len(herb_ids)] = True
            uniform[index, :len(herb_ids)] = 1.0 / len(herb_ids)
        self.arrays = (ingredients, mask, uniform, uniform.copy(),
                       np.zeros(self.n_entities, dtype=bool))
        self.audit = audit
        self.audit["train_queries"] = len(self.train)
        self.audit["training_range_sizes"] = {
            self.relations[r]: len(values)
            for r, values in self.range_candidates.items()}
        self.audit["distinct_training_herbs"] = len(train_herbs)
        if not len(self.dev) or not len(self.test):
            raise ValueError("No covered development or test queries")


def make_model(data, model_name: str, top_k: int, blend: float):
    if model_name == "mean":
        return CompositionalComplEx(data, "uniform_mean", data.arrays)
    if model_name == "pair":
        return HybridPairComplEx(data, "hybrid_pair", data.arrays)
    if model_name == "second_moment":
        return HybridPairComplEx(data, "second_moment_control", data.arrays)
    if model_name == "sparse":
        return RelationSparsePairComplEx(data, data.arrays, top_k, blend)
    if model_name == "unconditioned":
        return RelationSparsePairComplEx(data, data.arrays, top_k, blend,
                                         selection="unconditioned")
    if model_name == "candidate":
        return CandidatePairComplEx(data, data.arrays, shortlist=12,
                                    top_k=top_k)
    raise ValueError(model_name)


def score_bundle(model, h, r, tails):
    """Mathematically identical ComplEx scores with a shared head encoding."""
    hr, hi = model.head_embedding(h, r)
    rr, ri = model.rr(r), model.ri(r)
    tr, ti = model.er(tails), model.ei(tails)
    return ((hr * rr)[:, None, :] * tr +
            (hi * rr)[:, None, :] * ti +
            (hr * ri)[:, None, :] * ti -
            (hi * ri)[:, None, :] * tr).sum(-1)


def run(args):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = InPETMDataset(args.source, args.split_seed, args.eval_groups)
    model = make_model(data, args.model, args.top_k, args.blend).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-7)
    rng = np.random.default_rng(args.seed)
    tag = (f"inpetm_{args.source.lower()}_g{args.eval_groups}_"
           f"{args.model}_k{args.top_k}_b{args.blend:g}_seed{args.seed}")
    if args.split_seed != 20261021:
        tag += f"_split{args.split_seed}"
    checkpoint = OUT / f"{tag}.pt"
    if args.evaluate_only:
        report_path = OUT / f"{tag}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["audit"] != data.audit:
            raise ValueError("Dataset audit changed since development training")
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(state["state_dict"])
        report["role"] = "independent test after development selection"
        report["test"] = evaluate(model, data, data.test, torch.device("cuda"), True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"tag": tag, "test": report["test"]}, indent=2),
              flush=True)
        return
    best, stale, history = None, 0, []
    start = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = rng.permutation(len(data.train))
        losses = []
        for offset in range(0, len(order), BATCH):
            raw = data.train[order[offset:offset+BATCH]]
            negatives = sample_negative_tails(data, raw, rng)
            batch = torch.as_tensor(raw, device="cuda")
            tails = torch.as_tensor(np.concatenate(
                (raw[:, 2:3], negatives), axis=1), device="cuda")
            scores = (model.score_bundle(batch[:, 0], batch[:, 1], tails)
                      if args.model == "candidate" else
                      score_bundle(model, batch[:, 0], batch[:, 1], tails))
            loss = F.softplus(-scores[:, 0]).mean() + F.softplus(scores[:, 1:]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0:
            dev = evaluate(model, data, data.dev, torch.device("cuda"), True)
            record = {"epoch": epoch, "loss": float(np.mean(losses)), "dev": dev}
            history.append(record)
            print(tag, record, flush=True)
            if best is None or dev["mrr"] > best["dev"]["mrr"]:
                best, stale = record, 0
                torch.save({"state_dict": model.state_dict(), "args": vars(args)},
                           checkpoint)
            else:
                stale += 1
                if stale >= 4:
                    break
    report = {"role": "development only" if args.dev_only else "independent test",
              "audit": data.audit, "model": args.model, "config": vars(args),
              "parameters": sum(p.numel() for p in model.parameters()),
              "selected": best, "history": history,
              "seconds": time.time() - start,
              "checkpoint": checkpoint.name}
    if not args.dev_only:
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(state["state_dict"])
        report["test"] = evaluate(model, data, data.test, torch.device("cuda"), True)
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"tag": tag, "selected": best, "test": report.get("test"),
                      "seconds": report["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["KTKP", "TCM-ID", "KAMPO", "DRAGON"],
                        default="KTKP")
    parser.add_argument("--split-seed", type=int, default=20261021)
    parser.add_argument("--eval-groups", type=int, default=400)
    parser.add_argument("--model", choices=["mean", "pair", "second_moment",
                                            "sparse", "unconditioned", "candidate"],
                        required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--blend", type=float, default=0.25)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    run(parser.parse_args())
