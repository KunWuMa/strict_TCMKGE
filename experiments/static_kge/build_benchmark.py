# -*- coding: utf-8 -*-
"""Build a leakage-audited TCM-MKG link-prediction benchmark.

The prediction target is CPM -> TCMT.  Context relations remain in training.
Two target splits are produced: edge-random and formula-held-out.  In the
formula-held-out split, a held-out CPM keeps non-target context such as its
ingredient herbs, so the task measures cross-relation transfer rather than a
fully unseen-entity setting.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PART1 = next(ROOT.glob("01_*"))
MKG = PART1 / "TCM-MKG"
OUT = Path(__file__).with_name("benchmark")
SEED = 20260915


def rows(name: str):
    with (MKG / name).open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def value(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def valid(text: str) -> bool:
    return bool(text and text != "NA")


def node(kind: str, identifier: str) -> str:
    return f"{kind}:{identifier}"


def triple(head: str, relation: str, tail: str) -> tuple[str, str, str]:
    return head, relation, tail


def collect() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], dict]:
    target = set()
    groups = {}
    for row in rows("D3_CPM_TCMT.tsv"):
        h, t, group = value(row, "CPM_ID"), value(row, "TCMT_ID"), value(row, "Chinese_group")
        if valid(h) and valid(t) and valid(group):
            relation = "cpm_has_tcmt::" + group
            target.add(triple(node("cpm", h), relation, node("tcmt", t)))
            groups[relation] = group

    context = set()
    specs = [
        ("D4_CPM_CHP.tsv", "CPM_ID", "cpm", "cpm_contains_chp", "CHP_ID", "chp"),
        ("D5_CPM_ICD11.tsv", "CPM_ID", "cpm", "cpm_treats_icd11", "ICD11_code", "icd11"),
        ("D7_CHP_Medicinal_properties.tsv", "CHP_ID", "chp", "chp_has_property", "Medicinal_properties", "property"),
        ("D8_CHP_NP.tsv", "CHP_ID", "chp", "chp_from_species", "species_ID", "species"),
        ("D9_CHP_InChIKey.tsv", "CHP_ID", "chp", "chp_contains_compound", "InChIKey", "compound"),
    ]
    for filename, head_col, head_type, relation, tail_col, tail_type in specs:
        for row in rows(filename):
            h, t = value(row, head_col), value(row, tail_col)
            if valid(h) and valid(t):
                context.add(triple(node(head_type, h), relation, node(tail_type, t)))

    return sorted(target), sorted(context), groups


def split_random(target: list[tuple[str, str, str]], context, rng: np.random.Generator):
    by_relation = defaultdict(list)
    for edge in target:
        by_relation[edge[1]].append(edge)
    train, dev, test = [], [], []
    for relation, edges in sorted(by_relation.items()):
        indices = rng.permutation(len(edges))
        n_test = max(1, int(round(len(edges) * 0.10))) if len(edges) >= 10 else 0
        n_dev = max(1, int(round(len(edges) * 0.10))) if len(edges) >= 10 else 0
        test.extend(edges[index] for index in indices[:n_test])
        dev.extend(edges[index] for index in indices[n_test:n_test + n_dev])
        train.extend(edges[index] for index in indices[n_test + n_dev:])

    # TCMT nodes have no context relation.  Every evaluation tail must therefore
    # retain at least one target edge in train for a transductive KGE benchmark.
    train_entities = {h for h, _, _ in context} | {t for _, _, t in context}
    train_entities |= {h for h, _, _ in train} | {t for _, _, t in train}
    repaired_dev, repaired_test = [], []
    for source, destination in ((dev, repaired_dev), (test, repaired_test)):
        for edge in source:
            if edge[0] not in train_entities or edge[2] not in train_entities:
                train.append(edge)
                train_entities.add(edge[0]); train_entities.add(edge[2])
            else:
                destination.append(edge)
    return sorted(train), sorted(repaired_dev), sorted(repaired_test)


def split_formula_holdout(target, context, rng):
    target_by_head = defaultdict(list)
    for edge in target:
        target_by_head[edge[0]].append(edge)
    context_heads = {h for h, _, _ in context if h.startswith("cpm:")}
    eligible = sorted(set(target_by_head) & context_heads)
    shuffled = np.array(eligible, dtype=object)[rng.permutation(len(eligible))].tolist()
    n_test = int(round(len(shuffled) * 0.10))
    n_dev = int(round(len(shuffled) * 0.10))
    test_heads = set(shuffled[:n_test])
    dev_heads = set(shuffled[n_test:n_test + n_dev])
    train_heads = set(shuffled[n_test + n_dev:]) | (set(target_by_head) - set(eligible))

    train = [edge for head in train_heads for edge in target_by_head[head]]
    dev = [edge for head in dev_heads for edge in target_by_head[head]]
    test = [edge for head in test_heads for edge in target_by_head[head]]

    # Remove held-out edges whose relation or tail cannot be learned from target
    # training edges. This is declared filtering, not test-driven model tuning.
    train_relations = {r for _, r, _ in train}
    train_tails = {t for _, _, t in train}
    dev = [edge for edge in dev if edge[1] in train_relations and edge[2] in train_tails]
    test = [edge for edge in test if edge[1] in train_relations and edge[2] in train_tails]
    return sorted(train), sorted(dev), sorted(test), sorted(dev_heads), sorted(test_heads)


def write_triples(path: Path, triples: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(triples)


def describe(name, target_train, dev, test, context, dev_heads=None, test_heads=None):
    all_train = sorted(set(context) | set(target_train))
    train_entities = {h for h, _, _ in all_train} | {t for _, _, t in all_train}
    eval_entities = {h for h, _, _ in dev + test} | {t for _, _, t in dev + test}
    assert not (set(target_train) & set(dev))
    assert not (set(target_train) & set(test))
    assert not (set(dev) & set(test))
    assert eval_entities <= train_entities
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    write_triples(directory / "train.tsv", all_train)
    write_triples(directory / "dev.tsv", dev)
    write_triples(directory / "test.tsv", test)
    return {
        "train_all": len(all_train), "train_target": len(target_train),
        "train_context": len(context), "dev_target": len(dev), "test_target": len(test),
        "entities": len(train_entities), "relations": len({r for _, r, _ in all_train}),
        "dev_heads": len(dev_heads or {h for h, _, _ in dev}),
        "test_heads": len(test_heads or {h for h, _, _ in test}),
        "all_eval_entities_seen_in_train": True,
        "target_edge_disjoint": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target, context, groups = collect()
    rng = np.random.default_rng(SEED)
    random_train, random_dev, random_test = split_random(target, context, rng)
    cold_train, cold_dev, cold_test, cold_dev_heads, cold_test_heads = split_formula_holdout(
        target, context, np.random.default_rng(SEED + 1)
    )
    manifest = {
        "seed": SEED,
        "target_relation": "CPM -> typed TCMT relation",
        "target_edges_total": len(target),
        "context_edges_total": len(context),
        "relation_labels": groups,
        "random": describe("random", random_train, random_dev, random_test, context),
        "formula_holdout": describe(
            "formula_holdout", cold_train, cold_dev, cold_test, context,
            cold_dev_heads, cold_test_heads
        ),
        "formula_holdout_definition": (
            "All target CPM-TCMT edges of held-out CPM heads are absent from target training; "
            "non-target context edges remain. This is cross-relation cold-start, not unseen-entity induction."
        ),
        "source_sha256": {
            name: hashlib.sha256((MKG / name).read_bytes()).hexdigest()
            for name in ["D3_CPM_TCMT.tsv", "D4_CPM_CHP.tsv", "D5_CPM_ICD11.tsv",
                         "D7_CHP_Medicinal_properties.tsv", "D8_CHP_NP.tsv", "D9_CHP_InChIKey.tsv"]
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
