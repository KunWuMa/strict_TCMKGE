# -*- coding: utf-8 -*-
"""Resumable primary + KTKP experiments. Protocol is fixed separately from results.

Stages: fixed, search, selected, replicate, sensitivity, summarize, all.
Search never scores test queries. Selected configurations are saved before scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import os
import time
from functools import lru_cache
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from compositional_kge import DIM, OUT, CompositionalComplEx, make_strict_inductive, read_compositions
from expanded_kge_baselines import CompositionalKGE, STRICT_ID_MODELS
from hybrid_pair_kge import HybridPairComplEx
from inpetm_independent import InPETMDataset, score_bundle
from pair_interaction_kge import PairCompositionalComplEx
from pair_memory_dev import ingredients, standardized
from pair_prototype_kge import PairPrototypeKGE
from pairproto_cache_lock import serialized_cache
from train_baselines import Dataset, sample_negative_tails

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "pair_prototype_full_protocol.json"
P = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
AMENDMENT_PATH = HERE / "pair_prototype_compute_amendment.json"
A = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
PREFIX = "prototype_suite"


def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def seeds_for(name):
    return P["seeds"] if name in A["five_seed_models"] else P["replication_seeds"]


def wait_priority_recipe():
    """Avoid two local workers writing the same recipe-split checkpoints."""
    marker=OUT/"prototype_suite_priority_recipe_process.json"
    if os.name!="nt" or not marker.exists():
        return
    identity=json.loads(marker.read_text(encoding="utf-8"))
    pid=int(identity["pid"])
    if pid==os.getpid():
        return
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        return
    try:
        created,exited,kernel_time,user_time=(wintypes.FILETIME() for _ in range(4))
        if kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel_time),ctypes.byref(user_time)):
            birth=(created.dwHighDateTime<<32)|created.dwLowDateTime
            if identity.get("creation_filetime")!=birth:
                return  # A stale PID must never wait for an unrelated process.
        code=wintypes.DWORD()
        announced=False
        while kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value==259:
            if not announced:
                print("WAITING for existing recipe worker",pid,flush=True);announced=True
            time.sleep(5)
    finally:
        kernel.CloseHandle(handle)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=8)
def load_data(source, split):
    if source == "PRIMARY":
        if split.startswith("recipe_"):
            # Rebuild directly from public raw tables; historical processed files
            # are not prerequisites for the new exact-recipe benchmark.
            data = Dataset.__new__(Dataset)
            return recipe_grouped_primary(data, split)
        data = Dataset(split)
        data.arrays = read_compositions(data)
        strict = make_strict_inductive(data)
        audit = {"source": source, "split": split, "strict": strict,
                 "coverage": {}, "primary_split_note": "Historical formula holdout; exact-recipe overlap audited separately"}
        for name in ("dev", "test"):
            triples = getattr(data, name)
            keep = np.asarray([int(t) in data.range_candidates[int(r)] for _, r, t in triples])
            if not keep.all():
                raise AssertionError("Frozen primary split lost strict training range coverage")
            audit["coverage"][name] = {"raw_queries": len(triples), "retained_queries": len(triples),
                                       "removed_for_training_range": 0,
                                       "scope": "Already-filtered frozen benchmark files, not preconstruction source queries"}
        upstream = OUT / "candidate_coverage_audit.json"
        if split == "hybrid_holdout" and upstream.exists():
            audit["upstream_candidate_filtering"] = json.loads(upstream.read_text(encoding="utf-8"))["splits"]
        # Audit, never silently alter historical splits to improve metrics.
        groups = {}
        for name, triples in (("train", data.train), ("dev", data.dev), ("test", data.test)):
            groups[name] = {ingredients(data, h) for h in np.unique(triples[:, 0])
                            if data.arrays[1][int(h)].any()}
        audit["exact_recipe_overlap"] = {a + "_" + b: len(groups[a] & groups[b])
                                          for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
        data.audit = audit
    else:
        data = InPETMDataset("KTKP", int(split), 400)
    return data


def recipe_grouped_primary(data, split):
    """Group the unfiltered raw graph, without redistributing source triples."""
    from build_benchmark import collect
    seed = int(split.split("_")[-1])
    raw_target, raw_context, _ = collect()
    raw_full = raw_target + raw_context
    data.entities = sorted({v for h,_,t in raw_full for v in (h,t)})
    data.relations = sorted({r for _,r,_ in raw_full})
    data.entity_to_id = {v:i for i,v in enumerate(data.entities)}
    data.relation_to_id = {v:i for i,v in enumerate(data.relations)}
    data.n_entities, data.n_relations = len(data.entities), len(data.relations)
    data.arrays = read_compositions(data)
    full = np.unique(data.encode(raw_full), axis=0)
    target = np.array([data.relations[int(r)].startswith("cpm_has_tcmt::") for _,r,_ in full])
    groups = defaultdict(list)
    labelled = set(map(int,full[target,0]))
    for h in np.unique(full[:,0]):
        if data.arrays[1][int(h)].any():
            groups[ingredients(data,h)].append(int(h))
    groups = {key:heads for key,heads in groups.items() if labelled.intersection(heads)}
    keys = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    n = round(.1*len(keys))
    dev_heads = {h for i in order[:n] for h in groups[keys[i]]}
    test_heads = {h for i in order[n:2*n] for h in groups[keys[i]]}
    held = dev_heads | test_heads
    contains = data.relation_to_id["cpm_contains_chp"]
    data.train = full[np.array([int(h) not in held and int(t) not in held and int(r)!=contains
                               for h,r,t in full])]
    seen = set(data.train[:,0]) | set(data.train[:,2])
    data.range_candidates = {r:np.unique(data.train[data.train[:,1]==r,2]) for r in range(data.n_relations)}
    data.train_codes = {r:np.unique(data.train[data.train[:,1]==r,0]*data.n_entities+
                                   data.train[data.train[:,1]==r,2]) for r in range(data.n_relations)}
    coverage = {}
    for name, heads in (("dev",dev_heads),("test",test_heads)):
        raw = full[target & np.isin(full[:,0],list(heads))]
        ingredient_ok = np.array([set(ingredients(data,h)) <= seen for h,_,_ in raw])
        after = raw[ingredient_ok]
        range_ok = np.array([int(t) in data.range_candidates[int(r)] for _,r,t in after])
        kept = after[range_ok]
        setattr(data,name,kept)
        coverage[name] = {"raw_queries":len(raw),"removed_for_unseen_herb":int((~ingredient_ok).sum()),
                          "removed_for_training_range":int((~range_ok).sum()),"retained_queries":len(kept),
                          "retained_formulas":len(np.unique(kept[:,0]))}
    data.all_true = defaultdict(set)
    for h,r,t in full:
        data.all_true[int(h),int(r)].add(int(t))
    # Include all training formula groups, even those with only context relations.
    train_recipes = {ingredients(data,h) for h in np.unique(data.train[:,0]) if data.arrays[1][int(h)].any()}
    dev_recipes = {ingredients(data,h) for h in dev_heads}
    test_recipes = {ingredients(data,h) for h in test_heads}
    overlaps = {"train_dev":len(train_recipes&dev_recipes),"train_test":len(train_recipes&test_recipes),
                "dev_test":len(dev_recipes&test_recipes)}
    if any(overlaps.values()):
        raise AssertionError(("Recipe-group overlap",overlaps))
    data.audit = {"source":"PRIMARY","split":split,"split_seed":seed,"eligible_recipe_groups":len(keys),
                  "train_queries":len(data.train),"coverage":coverage,"exact_recipe_overlap":overlaps,
                  "source_graph":"Unfiltered target and context edges reconstructed from source via build_benchmark.collect",
                  "definition":"Exact-composition-disjoint local split replication, not an independent source"}
    return data


def make_model(data, name, heads=4):
    if name=='full_deepsets':
        from full_deepsets_kge import FullDeepSetsComplEx
        return FullDeepSetsComplEx(data,data.arrays)
    if name == "mean":
        return CompositionalComplEx(data, "uniform_mean", data.arrays)
    if name in ("pair", "second_moment", "matched"):
        variant = {"pair": "hybrid_pair", "second_moment": "second_moment_control",
                   "matched": "dual_mean_control"}[name]
        return HybridPairComplEx(data, variant, data.arrays)
    if name == "deepsets":
        return PairCompositionalComplEx(data, "deepsets_elementwise", data.arrays)
    if name in STRICT_ID_MODELS:
        return STRICT_ID_MODELS[name](data.n_entities, data.n_relations, DIM)
    model = CompositionalKGE(data, data.arrays, name)
    if name == "set_transformer" and heads != 4:
        # Same projections and parameter count; only attention partition changes.
        model.set_attn.num_heads = heads
        model.set_attn.head_dim = 2 * DIM // heads
        model.pma_attn.num_heads = heads
        model.pma_attn.head_dim = 2 * DIM // heads
    return model


def legacy_checkpoint(source, split, name, seed):
    if source == "KTKP" and name in ("mean", "pair", "second_moment"):
        tag = f"inpetm_ktkp_g400_{name}_k3_b0.25_seed{seed}"
        if int(split) != 20261021:
            tag += f"_split{split}"
        return OUT / (tag + ".pt")
    if source != "PRIMARY":
        return None
    if split != "hybrid_holdout":
        short = {"mean": "mean", "pair": "paircomp", "matched": "matched"}.get(name)
        return OUT / f"review_{split}_{short}_seed{seed}.pt" if short else None
    if name == "mean":
        tag = "expanded_mean_complex"
    elif name in ("pair", "matched"):
        tag = ({"pair": "hybridconfirm_hybrid", "matched": "hybridconfirm_parameter_control"}[name]
               if seed <= 20260917 else {"pair": "review5_paircomp", "matched": "review5_matched"}[name])
    elif name == "second_moment":
        tag = "review_second_moment"
    elif name == "deepsets":
        tag = "review_true_deepsets"
    else:
        tag = "expanded_" + name
    return OUT / f"{tag}_seed{seed}.pt"


def neural_tag(source, split, name, seed, lr=.003, heads=4):
    return f"{PREFIX}_{source.lower()}_{split}_{name}_lr{lr:g}_h{heads}_seed{seed}"


@torch.inference_mode()
def reciprocal_ranks(model, data, triples, all_entities=False):
    """Cache scores for multi-label queries; average rank ties, filtered truths."""
    model.eval()
    cache, rr = {}, np.empty(len(triples), dtype=np.float64)
    for i, (h, r, t) in enumerate(triples):
        h, r, t = int(h), int(r), int(t)
        key = h, r
        candidates = np.arange(data.n_entities) if all_entities else data.range_candidates[r]
        if key not in cache:
            values = model.score_tails(torch.tensor(h, device="cuda"), torch.tensor(r, device="cuda"),
                                       torch.as_tensor(candidates, device="cuda"))
            cache[key] = values.float().cpu().numpy()
        score = cache[key]
        allowed = ~np.isin(candidates, list(data.all_true[key] - {t}))
        pos = int(np.searchsorted(candidates, t))
        if pos == len(candidates) or candidates[pos] != t:
            raise AssertionError("Target outside frozen candidate set")
        rank = 1 + np.sum((score > score[pos]) & allowed) + .5 * max(0, int(np.sum((score == score[pos]) & allowed)) - 1)
        rr[i] = 1. / rank
    return rr


def metrics(rr):
    return {"mrr": float(np.mean(rr)), "hits1": float(np.mean(rr == 1)),
            "hits3": float(np.mean(rr >= 1/3)), "hits10": float(np.mean(rr >= .1))}


@serialized_cache('checkpoint',lambda v:neural_tag(v['source'],v['split'],v['name'],v['seed'],v['lr'],v['heads']))
def train_or_load(data, source, split, name, seed, lr=.003, heads=4):
    OUT.mkdir(parents=True,exist_ok=True)
    tag = neural_tag(source, split, name, seed, lr, heads)
    path, record = OUT / (tag + ".pt"), OUT / (tag + "_dev.json")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = make_model(data, name, heads).cuda()
    if record.exists():
        info = json.loads(record.read_text(encoding="utf-8"))
        saved = OUT / info["checkpoint"]
        if saved.exists():
            model.load_state_dict(torch.load(saved, map_location="cuda", weights_only=False)["state_dict"])
            return model.eval(), info
    legacy = legacy_checkpoint(source, split, name, seed) if lr == .003 and heads == 4 else None
    if legacy and legacy.exists():
        model.load_state_dict(torch.load(legacy, map_location="cuda", weights_only=False)["state_dict"])
        rr = reciprocal_ranks(model, data, data.dev)
        info = {"role": "development only; historical checkpoint reused", "checkpoint": legacy.name,
                "dev": metrics(rr), "config": {"lr": lr, "heads": heads, "seed": seed, "model": name},
                "parameters": sum(p.numel() for p in model.parameters()), "seconds": None, "history": []}
        # Persist alias metadata; load legacy on subsequent invocations, no large checkpoint copy.
        save(record, info)
        return model.eval(), info
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    rng = np.random.default_rng(seed)
    best, stale, history, started = None, 0, [], time.perf_counter()
    micro = 128 if name == "set_transformer" else 512
    for epoch in range(1, 41):
        model.train(); order = rng.permutation(len(data.train)); losses = []
        for offset in range(0, len(order), 2048):
            raw = data.train[order[offset:offset+2048]]
            negative = sample_negative_tails(data, raw, rng)
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for begin in range(0, len(raw), micro):
                part = raw[begin:begin+micro]
                batch = torch.as_tensor(part, device="cuda")
                tails = torch.as_tensor(negative[begin:begin+micro], device="cuda")
                if isinstance(model, CompositionalKGE):
                    positive = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
                    neg = model.score_negative_tails(batch[:, 0], batch[:, 1], tails)
                elif hasattr(model, "head_embedding"):
                    bundle = score_bundle(model, batch[:, 0], batch[:, 1],
                                          torch.cat((batch[:, 2:3], tails), dim=1))
                    positive, neg = bundle[:, 0], bundle[:, 1:]
                else:
                    positive = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
                    neg = model.score(batch[:, 0, None].expand_as(tails).reshape(-1),
                                      batch[:, 1, None].expand_as(tails).reshape(-1), tails.reshape(-1)).reshape_as(tails)
                loss = F.softplus(-positive).mean() + F.softplus(neg).mean()
                weight = len(part) / len(raw)
                (loss * weight).backward(); value += float(loss.detach()) * weight
            optimizer.step(); losses.append(value)
        if epoch == 1 or epoch % 5 == 0:
            dev = metrics(reciprocal_ranks(model, data, data.dev))
            row = {"epoch": epoch, "loss": float(np.mean(losses)), "dev": dev}
            history.append(row)
            print("DEV", tag, epoch, round(dev["mrr"], 6), flush=True)
            if best is None or dev["mrr"] > best["dev"]["mrr"]:
                best, stale = row, 0
                torch.save({"state_dict": model.state_dict(), "tag": tag}, path)
            else:
                stale += 1
                if stale >= 4:
                    break
    info = {"role": "development only", "checkpoint": path.name, "selected": best, "dev": best["dev"],
            "config": {"lr": lr, "heads": heads, "seed": seed, "model": name, "micro_batch": micro},
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": time.perf_counter()-started, "history": history}
    save(record, info)
    model.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["state_dict"])
    return model.eval(), info


@serialized_cache('evaluation',lambda v:neural_tag(v['source'],v['split'],v['name'],v['seed'],v['lr'],v['heads']))
def run_neural(source, split, name, seed, lr=.003, heads=4, test=True, boundary=False):
    tag = neural_tag(source, split, name, seed, lr, heads)
    result_path = OUT / (tag + "_test.json")
    rank_path = OUT / (tag + "_rr.npz")
    if test and result_path.exists() and rank_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    data = load_data(source, split)
    model, info = train_or_load(data, source, split, name, seed, lr, heads)
    if not test:
        return info
    rr = reciprocal_ranks(model, data, data.test)
    arrays = {"rr": rr, "heads": data.test[:, 0], "relations": data.test[:, 1]}
    report = {"source": source, "split": split, "seed": seed, "model": name,
              "config": info["config"], "dev": info["dev"], "test": metrics(rr), "audit": data.audit,
              "parameters": info["parameters"], "training_seconds": info["seconds"], "checkpoint": info["checkpoint"]}
    if boundary:
        arr = reciprocal_ranks(model, data, data.test, all_entities=True)
        arrays["all_entity_rr"] = arr; report["all_entity_test"] = metrics(arr)
    save_npz(rank_path, **arrays); save(result_path, report)
    print("TEST", source, split, name, seed, report["test"], flush=True)
    return report


def bootstrap(delta, heads, unit="formula"):
    unique, inverse = np.unique(heads, return_inverse=True)
    sums = np.bincount(inverse, weights=delta); counts = np.bincount(inverse)
    rng = np.random.default_rng(20261022); samples = []
    for _ in range(50):
        draw = rng.integers(0, len(unique), size=(100, len(unique)))
        samples.extend(sums[draw].sum(1) / counts[draw].sum(1))
    return {"difference": float(np.mean(delta)), unit+"_cluster_ci95": np.quantile(samples, [.025, .975]).tolist()}


def weighted_ranks(data, triples, cache, weights):
    rr = []
    for h, r, t in triples:
        key = int(h), int(r)
        values = cache[key]
        score = sum(weight * part for weight, part in zip(weights, values))
        candidates = data.range_candidates[int(r)]
        allowed = ~np.isin(candidates, list(data.all_true[key] - {int(t)}))
        target = score[np.searchsorted(candidates, int(t))]
        rank = 1 + np.sum((score > target) & allowed) + .5*max(0, int(np.sum((score == target) & allowed))-1)
        rr.append(1./rank)
    return np.asarray(rr)


@torch.inference_mode()
def profile_cache(model, data, triples, raw=False):
    cache = {}
    for h, r, _ in triples:
        h, r = int(h), int(r)
        if (h, r) in cache:
            continue
        herbs = ingredients(data, h)
        memory = model.memories[r]
        ids = torch.as_tensor(herbs, device="cuda")
        mu = torch.cat((model.base.er(ids).mean(0), model.base.ei(ids).mean(0))).cpu().numpy().astype(float)
        phi = model.phi[r]
        if raw:
            hi = [memory.herb_vocab[x] for x in herbs if x in memory.herb_vocab]
            pi = [memory.pair_vocab[x] for x in itertools.combinations(herbs, 2) if x in memory.pair_vocab]
            hs = memory.lift(memory.herb_counts, hi, memory.herb_n)
            ps = memory.lift(memory.pair_counts, pi, memory.pair_n)
            herb = hs.mean(0) if len(hs) else np.zeros(len(phi))
            pair = ps.mean(0) if len(ps) else np.zeros(len(phi))
        else:
            hi = [memory.herb_vocab[x] for x in herbs if x in memory.herb_vocab]
            pi = [memory.pair_vocab[x] for x in itertools.combinations(herbs, 2) if x in memory.pair_vocab]
            proto = model.prototypes[r]
            herb = phi @ proto["herb"][hi].mean(0) if hi else np.zeros(len(phi))
            pair = phi @ proto["pair"][pi].mean(0) if pi else np.zeros(len(phi))
        cache[h, r] = tuple(standardized(v) for v in (phi @ mu, herb, pair))
    return cache


@serialized_cache('prototype',lambda v:f"{PREFIX}_{v['source'].lower()}_{v['split']}_prototype_seed{v['seed']}_lr{v['backbone_lr']:g}_sensitivity{int(v['sensitivity'])}")
def prototype_run(source, split, seed, sensitivity=False, backbone_lr=.003):
    tag = f"{PREFIX}_{source.lower()}_{split}_prototype_seed{seed}"
    if backbone_lr != .003:
        tag += f"_backbone_lr{backbone_lr:g}"
    result = OUT / (tag + ("_sensitivity.json" if sensitivity else ".json"))
    if result.exists() and (sensitivity or (OUT/(tag+'_rr.npz')).exists()):
        return json.loads(result.read_text(encoding="utf-8"))
    data = load_data(source, split)
    base, info = train_or_load(data, source, split, "mean", seed, lr=backbone_lr)
    relations = np.unique(np.concatenate((data.dev[:, 1], data.test[:, 1])))
    config = P["fixed_prototype"]
    model = PairPrototypeKGE(data, base, relations=relations, ridge=config["ridge_relative"],
                             shrinkage=config["shrinkage"], min_support=config["minimum_pair_support"])
    if sensitivity:
        records = []
        hw, pw = P["datasets"][source]["prototype_weights"]
        candidates = [dict(config)]
        for key in config:
            values = P["search"][key]
            for value in values:
                cfg = {**config, key: value}
                if cfg not in candidates:
                    candidates.append(cfg)
        for cfg in candidates:
            m = model if cfg == config else PairPrototypeKGE(data, base, relations=relations,
                      ridge=cfg["ridge_relative"], shrinkage=cfg["shrinkage"], min_support=cfg["minimum_pair_support"])
            cache = profile_cache(m, data, data.dev)
            scores = {}
            for gamma in P["search"]["prototype_weights_grid"]:
                scores[str(gamma)] = metrics(weighted_ranks(data, data.dev, cache, (1-gamma, 0, gamma)))
            records.append({"config": cfg, "dev_weights": scores, "prototype_bytes": m.prototype_bytes,
                            "build_seconds": m.build_seconds})
        report = {"role": "development-only one-factor sensitivity; no main setting changes", "source": source,
                  "split": split, "seed": seed, "records": records}
        save(result, report); print("SENSITIVITY DONE", source, split, flush=True); return report
    weights = {"mean": (0., 0.), "herb_prototype": tuple(P["datasets"][source]["herb_weights"]),
               "pair_prototype": tuple(P["datasets"][source]["prototype_weights"])}
    arrays = {"heads": data.test[:, 0], "relations": data.test[:, 1]}
    report = {"source": source, "split": split, "seed": seed, "audit": data.audit, "weights": weights,
              "checkpoint": info["checkpoint"], "dev": {}, "test": {},
              "backbone_learning_rate":backbone_lr,
              "build_seconds": model.build_seconds, "prototype_bytes": model.prototype_bytes,
              "memory": {str(r): {"recipes": m.n_recipes, "herbs": len(m.herb_vocab), "pairs": len(m.pair_vocab),
                                  "sparse_count_bytes": sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (m.herb_counts,m.pair_counts))}
                         for r, m in model.memories.items()}}
    dev_cache, test_cache = profile_cache(model, data, data.dev), profile_cache(model, data, data.test)
    for name, (hw, pw) in weights.items():
        report["dev"][name] = metrics(weighted_ranks(data, data.dev, dev_cache, (1-hw-pw, hw, pw)))
        rr = weighted_ranks(data, data.test, test_cache, (1-hw-pw, hw, pw))
        arrays[name] = rr; report["test"][name] = metrics(rr)
    # Compare cached linear scores against actual compositional KGE API, not only statistical reranking.
    model.weights = weights["pair_prototype"]
    actual = reciprocal_ranks(model, data, data.test)
    if not np.allclose(actual, arrays["pair_prototype"], atol=1e-8):
        raise AssertionError("Projection/composition rank mismatch")
    raw_cache = profile_cache(model, data, data.test, raw=True)
    hw, pw = weights["pair_prototype"]
    arrays["raw_profiles"] = weighted_ranks(data, data.test, raw_cache, (1-hw-pw, hw, pw))
    report["test"]["raw_profiles"] = metrics(arrays["raw_profiles"])
    # Same storage and support strata; identity is the ablated factor.
    rng = np.random.default_rng(20261022)
    for r, memory in model.memories.items():
        support = np.floor(np.log2(memory.pair_n)).astype(int)
        original = model.prototypes[r]["pair"].copy()
        for level in np.unique(support):
            ids = np.flatnonzero(support == level)
            model.prototypes[r]["pair"][ids] = original[rng.permutation(ids)]
    shuffled = profile_cache(model, data, data.test)
    arrays["shuffled_pairs"] = weighted_ranks(data, data.test, shuffled, (1-hw-pw, hw, pw))
    report["test"]["shuffled_pairs"] = metrics(arrays["shuffled_pairs"])
    prior_cache = {(int(h), int(r)): (model.memories[int(r)].prior,)
                   for h, r, _ in data.test}
    arrays["frequency_prior"] = weighted_ranks(data, data.test, prior_cache, (1.,))
    report["test"]["frequency_prior"] = metrics(arrays["frequency_prior"])
    coverage = []
    for h, r, _ in data.test:
        pairs = list(itertools.combinations(ingredients(data, h), 2)); vocab = model.memories[int(r)].pair_vocab
        coverage.append(sum(p in vocab for p in pairs)/max(1, len(pairs)))
    arrays["coverage"] = np.asarray(coverage)
    arrays["lengths"] = data.arrays[1].sum(1)[data.test[:, 0]]
    save_npz(OUT / (tag + "_rr.npz"), **arrays)
    save(result, report)
    print("PROTOTYPE TEST", source, split, seed, report["test"], flush=True)
    return report


def search(source, split=None, selection_tag=PREFIX, models=None):
    split = split or P["datasets"][source]["main_split"]
    path = OUT / f"{selection_tag}_{source.lower()}_selected_configs.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    report = {"role": "Configurations selected on development only before their test evaluation",
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "compute_amendment_sha256":hashlib.sha256(AMENDMENT_PATH.read_bytes()).hexdigest(),
              "source":source,"split":split,"models": {}}
    for name in models or A["dev_search_models"]:
        candidates = [(.003, 4), (.001, 4), (.01, 4)]
        if name == "set_transformer":
            candidates += [(.003, h) for h in P["search"]["set_transformer_heads_extra"]]
        rows = []
        for lr, heads in candidates:
            info = run_neural(source, split, name, P["search"]["seed"], lr, heads, test=False)
            rows.append({"lr": lr, "heads": heads, "dev": info["dev"], "checkpoint": info["checkpoint"]})
        chosen = max(rows, key=lambda row: row["dev"]["mrr"])
        report["models"][name] = {"chosen": chosen, "candidates": rows}
        save(OUT / f"{selection_tag}_{source.lower()}_search_progress.json", report)
    save(path, report)
    print("SEARCH LOCKED", source, path.name, flush=True)
    return report


def summarize():
    report = {"protocol": P, "compute_amendment": A, "datasets": {}}
    for source, spec in P["datasets"].items():
        datasets = {}
        split_names = [spec["main_split"]] + spec["replications"]
        if source == "PRIMARY":
            split_names += [json.loads((HERE/"pair_prototype_recipe_protocol.json").read_text(encoding="utf-8"))["split"]]
        for split in split_names:
            seeds = P["seeds"] if split == spec["main_split"] else P["replication_seeds"]
            models, matrices, heads, meta = {}, {}, None, None
            for seed in seeds:
                proto = OUT / f"{PREFIX}_{source.lower()}_{split}_prototype_seed{seed}_rr.npz"
                if proto.exists():
                    z = np.load(proto); heads = z["heads"]; meta = z
                    for name in z.files:
                        if name not in ("heads", "relations", "coverage", "lengths"):
                            matrices.setdefault(name, []).append(z[name])
                for name in P["main_neural_models"]:
                    tag = neural_tag(source, split, name, seed)
                    path = OUT / (tag + "_rr.npz")
                    if path.exists():
                        z = np.load(path)
                        if heads is not None and not np.array_equal(heads,z["heads"]):
                            raise AssertionError("Query alignment mismatch")
                        heads = z["heads"]
                        matrices.setdefault("neural_"+name, []).append(z["rr"])
            for name, values in matrices.items():
                rows = [metrics(v) for v in values]
                models[name] = {"runs": len(rows), "summary": {m: {"mean": float(np.mean([v[m] for v in rows])),
                                  "sample_std": float(np.std([v[m] for v in rows], ddof=1)) if len(rows)>1 else None}
                                  for m in rows[0]}}
            selection = OUT / f"{PREFIX}_{source.lower()}_selected_configs.json"
            if split == spec["main_split"] and selection.exists():
                locked = json.loads(selection.read_text(encoding="utf-8"))
                for name, settings in locked["models"].items():
                    chosen = settings["chosen"]
                    values = []
                    for seed in seeds:
                        tag = neural_tag(source, split, name, seed, chosen["lr"], chosen["heads"])
                        path = OUT / (tag + "_rr.npz")
                        if path.exists():
                            values.append(np.load(path)["rr"])
                    if values:
                        label = "dev_selected_" + name
                        matrices[label] = values
                        rows = [metrics(v) for v in values]
                        models[label] = {"runs":len(rows), "config":chosen,
                                         "summary":{m:{"mean":float(np.mean([v[m] for v in rows])),
                                         "sample_std":float(np.std([v[m] for v in rows],ddof=1)) if len(rows)>1 else None}
                                                    for m in rows[0]}}
            comparisons = {}
            selected_backbone = {}
            if split == spec["main_split"] and selection.exists():
                selected_lr = json.loads(selection.read_text(encoding="utf-8"))["models"]["mean"]["chosen"]["lr"]
                for seed in seeds:
                    extra = f"_backbone_lr{selected_lr:g}" if selected_lr != .003 else ""
                    path=OUT/f"{PREFIX}_{source.lower()}_{split}_prototype_seed{seed}{extra}_rr.npz"
                    if path.exists():
                        z=np.load(path)
                        for label in ("mean","pair_prototype","herb_prototype","raw_profiles"):
                            selected_backbone.setdefault(label,[]).append(z[label])
            if "pair_prototype" in matrices:
                full = np.stack(matrices["pair_prototype"])
                current = load_data(source,split)
                recipe_to_id = {}
                recipe_ids = []
                for h in heads:
                    key = ingredients(current,int(h))
                    recipe_ids.append(recipe_to_id.setdefault(key,len(recipe_to_id)))
                recipe_ids = np.asarray(recipe_ids)
                if meta is not None:
                    relations = meta["relations"]
                    for name, values in matrices.items():
                        macro = [np.mean([v[relations==r].mean() for r in np.unique(relations)]) for v in values]
                        models[name]["relation_macro_mrr"] = {"mean":float(np.mean(macro)),
                                         "sample_std":float(np.std(macro,ddof=1)) if len(macro)>1 else None}
                for name, values in matrices.items():
                    if name == "pair_prototype" or len(values) > len(full):
                        continue
                    control = np.stack(values)
                    common = full[:len(control)]
                    comparisons[name] = {"matched_seed_count":len(control),
                                         "mrr": bootstrap((common-control).mean(0), heads),
                                         "hits1": bootstrap(((common==1)-(control==1).astype(float)).mean(0), heads),
                                         "recipe_cluster_mrr": bootstrap((common-control).mean(0), recipe_ids, "recipe"),
                                         "recipe_cluster_hits1": bootstrap(((common==1)-(control==1).astype(float)).mean(0), recipe_ids, "recipe")}
                if meta is not None:
                    strata = {"pair_coverage_zero": meta["coverage"] == 0,
                              "pair_coverage_0_to_half": (meta["coverage"]>0)&(meta["coverage"]<.5),
                              "pair_coverage_half_plus": meta["coverage"]>=.5,
                              "herbs_1_4": meta["lengths"]<=4,
                              "herbs_5_9": (meta["lengths"]>=5)&(meta["lengths"]<=9),
                              "herbs_10_plus": meta["lengths"]>=10}
                    strata.update({"relation_"+str(r):meta["relations"]==r for r in np.unique(meta["relations"])})
                    strata = {label:{"queries":int(mask.sum()), "pair_mrr":float(full[:,mask].mean()),
                                     "mean_mrr":float(np.stack(matrices["mean"])[:,mask].mean())}
                              for label,mask in strata.items() if mask.any()}
                else:
                    strata = {}
            else:
                strata = {}
            datasets[split] = {"intended_seeds": seeds, "models": models, "comparisons": comparisons, "strata": strata,
                               "bootstrap_note":"Formula and exact-recipe cluster intervals both conditional on current seed-averaged predictions",
                               "audit":load_data(source,split).audit if matrices else None}
            if selected_backbone:
                rows={name:[metrics(v) for v in values] for name,values in selected_backbone.items()}
                block={"backbone_lr":selected_lr,"weights":"Unchanged fixed prototype and herb weights",
                       "models":{name:{"runs":len(values),"mrr_mean":float(np.mean([v['mrr'] for v in values])),
                                        "mrr_sample_std":float(np.std([v['mrr'] for v in values],ddof=1)) if len(values)>1 else None}
                                 for name,values in rows.items()},"comparisons":{}}
                chosen=np.stack(selected_backbone["pair_prototype"])
                for name,values in selected_backbone.items():
                    if name!="pair_prototype" and len(values)==len(chosen):
                        control=np.stack(values)
                        block["comparisons"][name]={"mrr":bootstrap((chosen-control).mean(0),heads),
                            "hits1":bootstrap(((chosen==1)-(control==1).astype(float)).mean(0),heads)}
                for name,values in matrices.items():
                    if name.startswith("dev_selected_") and len(values)<=len(chosen):
                        control=np.stack(values);common=chosen[:len(control)]
                        block["comparisons"][name]={"matched_seed_count":len(control),
                            "mrr":bootstrap((common-control).mean(0),heads),
                            "hits1":bootstrap(((common==1)-(control==1).astype(float)).mean(0),heads)}
                datasets[split]["dev_selected_backbone_audit"]=block
        report["datasets"][source] = datasets
    save(OUT / f"{PREFIX}_summary.json", report)
    lines = ["# 药对原型 KGE：完整实验进度与结果", "", "自动生成；数值来自保存的逐查询排名。未完成项不会标为完成。", ""]
    for source, splits in report["datasets"].items():
        for split, result in splits.items():
            lines += [f"## {source} / {split}", "", "| 方法 | 已完成种子 | MRR 均值±标准差 | Hits@1 |", "|---|---:|---:|---:|"]
            for name,row in result["models"].items():
                m=row["summary"];sd=m["mrr"]["sample_std"]
                lines.append(f"| {name} | {row['runs']} | {m['mrr']['mean']:.4f} ± {sd or 0:.4f} | {m['hits1']['mean']:.4f} |")
            lines += ["", "药对原型相对各对照的方剂聚类区间：", ""]
            for name,row in result["comparisons"].items():
                v=row["mrr"];ci=v["formula_cluster_ci95"]
                lines.append(f"- {name}: ΔMRR={v['difference']:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            lines.append("")
    (HERE / "PAIR_PROTOTYPE_FULL_RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY SAVED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fixed", "search", "selected", "replicate", "recipe", "sensitivity", "summarize", "all"])
    parser.add_argument("--source", choices=["PRIMARY", "KTKP"])
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    sources = [args.source] if args.source else list(P["datasets"])
    stages = ["fixed", "search", "selected", "replicate", "recipe", "sensitivity", "summarize"] if args.stage == "all" else [args.stage]
    with threadpool_limits(limits=4):
        for stage in stages:
            if stage == "summarize":
                summarize(); continue
            if stage == "recipe":
                wait_priority_recipe()
                plan=json.loads((HERE/"pair_prototype_recipe_protocol.json").read_text(encoding="utf-8"))
                for seed in plan["seeds"]:
                    prototype_run("PRIMARY",plan["split"],seed)
                for name in plan["models"]:
                    for seed in plan["seeds"]:
                        run_neural("PRIMARY",plan["split"],name,seed)
                continue
            for source in sources:
                spec = P["datasets"][source]; split = spec["main_split"]
                if stage == "fixed":
                    for seed in P["seeds"]:
                        prototype_run(source, split, seed)
                    for name in args.models or P["main_neural_models"]:
                        for seed in seeds_for(name):
                            run_neural(source, split, name, seed)
                elif stage == "search":
                    search(source)
                elif stage == "selected":
                    selected = search(source)
                    for name, row in selected["models"].items():
                        for seed in seeds_for(name):
                            run_neural(source, split, name, seed, row["chosen"]["lr"], row["chosen"]["heads"])
                    # Paired fairness check: both Mean and prototype share exactly the selected Mean backbone.
                    # Prototype mixing weights remain fixed; no test-driven recalibration.
                    for seed in P["seeds"]:
                        prototype_run(source,split,seed,backbone_lr=selected["models"]["mean"]["chosen"]["lr"])
                elif stage == "replicate":
                    for split in spec["replications"]:
                        for seed in P["replication_seeds"]:
                            prototype_run(source, split, seed)
                        for name in P["replication_models"]:
                            for seed in P["replication_seeds"]:
                                run_neural(source, split, name, seed)
                elif stage == "sensitivity":
                    prototype_run(source, split, P["search"]["prototype_sensitivity_seed"], sensitivity=True)
    print("SUITE STAGE COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
