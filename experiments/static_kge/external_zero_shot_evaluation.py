# -*- coding: utf-8 -*-
"""Zero-shot external validation on de-identified outpatient prescriptions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from analyze_hybrid_confirmatory import FAMILIES, load
from compositional_kge import MKG, OUT, make_strict_inductive, read_compositions
from external_mapping_audit import EXT, rows
from hybrid_pair_kge import HybridPairComplEx
from pair_interaction_kge import PairCompositionalComplEx
from train_baselines import Dataset

SEEDS = (20260915, 20260916, 20260917)


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=float)
    return {
        "n": int(ranks.size), "mrr": float(np.mean(1.0 / ranks)),
        "hits1": float(np.mean(ranks <= 1)), "hits3": float(np.mean(ranks <= 3)),
        "hits10": float(np.mean(ranks <= 10)), "mean_rank": float(np.mean(ranks)),
    }


def seed_summary(ranks):
    """Report query count once and variability across independently trained seeds."""
    per_seed = [metrics(row) for row in ranks]
    return {
        "queries": int(ranks.shape[1]), "model_seeds": int(ranks.shape[0]),
        **{
            key: {
                "mean": float(np.mean([row[key] for row in per_seed])),
                "sample_std": float(np.std([row[key] for row in per_seed], ddof=1)),
            }
            for key in ("mrr", "hits1", "hits3", "hits10", "mean_rank")
        },
    }


def generated_embedding(model, herb_ids, relation_id):
    """Run the trained formula encoder directly on a new external herb set."""
    device = model.er.weight.device
    herbs = torch.as_tensor(herb_ids, dtype=torch.long, device=device)
    ar, ai = model.er(herbs), model.ei(herbs)
    mu_r, mu_i = ar.mean(0, keepdim=True), ai.mean(0, keepdim=True)
    n = torch.tensor([len(herb_ids)], dtype=ar.dtype, device=device)
    if isinstance(model, HybridPairComplEx):
        unary_r, unary_i = model._transform(mu_r, mu_i, model.unary_r, model.unary_i)
        valid = torch.ones((1, len(herb_ids)), dtype=torch.bool, device=device)
        pair_r, pair_i, n = model._pair_moment(ar[None], ai[None], valid)
        if model.hybrid_variant == "dual_mean_control":
            pair_r, pair_i = mu_r, mu_i
        branch_r, branch_i = model._transform(pair_r, pair_i, model.branch_r, model.branch_i)
        pair_gate = torch.sigmoid(
            model.pair_gate_intercept + model.pair_gate_log_size * (torch.log(n.clamp_min(1.0)) - np.log(5.0))
        )[:, None]
        return (
            mu_r + torch.sigmoid(model.unary_gate) * unary_r + pair_gate * branch_r,
            mu_i + torch.sigmoid(model.unary_gate) * unary_i + pair_gate * branch_i,
        )
    if isinstance(model, PairCompositionalComplEx):
        if model.pair_variant == "deepsets_residual":
            corr_r, corr_i = model._complex_transform(mu_r, mu_i)
        else:
            sr, si = ar.sum(0, keepdim=True), ai.sum(0, keepdim=True)
            denominator = max(1, len(herb_ids) * (len(herb_ids) - 1))
            pair_r = (sr.square() - ar.square().sum(0, keepdim=True) - si.square() + ai.square().sum(0, keepdim=True)) / denominator
            pair_i = 2.0 * (sr * si - (ar * ai).sum(0, keepdim=True)) / denominator
            if len(herb_ids) == 1:
                pair_r.zero_(); pair_i.zero_()
            corr_r, corr_i = model._complex_transform(pair_r, pair_i)
        gate = torch.sigmoid(model.interaction_gate)[None, :]
        return mu_r + gate * corr_r, mu_i + gate * corr_i
    raise TypeError(type(model))


@torch.inference_mode()
def rank_query(model, herb_ids, relation_id, target_id, candidates, filtered_targets):
    device = model.er.weight.device
    hr, hi = generated_embedding(model, herb_ids, relation_id)
    r = torch.tensor(relation_id, device=device)
    tails = torch.as_tensor(candidates, device=device)
    rr, ri = model.rr(r), model.ri(r)
    tr, ti = model.er(tails), model.ei(tails)
    scores = (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1).float().cpu().numpy()
    other = set(filtered_targets) - {target_id}
    if other:
        scores[np.isin(candidates, np.fromiter(other, dtype=np.int64))] = -np.inf
    target_position = int(np.searchsorted(candidates, target_id))
    target_score = scores[target_position]
    return 1.0 + int(np.sum(scores > target_score)) + 0.5 * max(0, int(np.sum(scores == target_score)) - 1)


def cluster_bootstrap(a, b, patients, seed=20261520, repetitions=20000):
    """a,b are model_seed x query rank matrices; resampling unit is patient."""
    delta_mrr = (1.0 / a).mean(0) - (1.0 / b).mean(0)
    delta_h1 = (a <= 1).mean(0) - (b <= 1).mean(0)
    unique = np.unique(patients)
    groups = [np.flatnonzero(patients == patient) for patient in unique]
    weights = np.asarray([len(group) for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    samples_mrr, samples_h1 = np.empty(repetitions), np.empty(repetitions)
    for i in range(repetitions):
        chosen = rng.integers(0, len(groups), len(groups))
        samples_mrr[i] = np.average([delta_mrr[groups[j]].mean() for j in chosen], weights=weights[chosen])
        samples_h1[i] = np.average([delta_h1[groups[j]].mean() for j in chosen], weights=weights[chosen])
    def result(delta, samples):
        return {
            "difference": float(delta.mean()),
            "patient_cluster_bootstrap_95ci": [float(x) for x in np.quantile(samples, (0.025, 0.975))],
            "probability_difference_le_zero": float(np.mean(samples <= 0)),
            "repetitions": repetitions,
        }
    return {"patients": len(unique), "queries": len(patients), "mrr": result(delta_mrr, samples_mrr), "hits1": result(delta_h1, samples_h1)}


def load_external(data):
    audit = json.loads((OUT / "external_mapping_audit.json").read_text(encoding="utf-8"))
    herb_mapping = audit["herbs"]["mapping"]
    syndrome_mapping = audit["syndromes"]["mapping"]
    d1 = {row["TCMT_ID"].strip(): row for row in rows(MKG / "D1_TCM_terminology.tsv")}
    raw = rows(EXT / "ZhaoHanqing_TCM_Prescriptions_2025.csv")
    by_rx = defaultdict(list); metadata = {}
    for row in raw:
        by_rx[row["prescription_id"]].append(row)
        metadata[row["prescription_id"]] = row
    records, exclusion = [], Counter()
    for rx, rx_rows in sorted(by_rx.items()):
        mapped_herbs = sorted({
            data.entity_to_id["chp:" + herb_mapping[row["herb_name_chinese"].strip()]]
            for row in rx_rows
            if row["herb_name_chinese"].strip() in herb_mapping
            and "chp:" + herb_mapping[row["herb_name_chinese"].strip()] in data.entity_to_id
        })
        if len(mapped_herbs) < 2:
            exclusion["fewer_than_two_mapped_herbs"] += 1
            continue
        source = metadata[rx]["syndrome_pattern"].strip()
        if source not in syndrome_mapping:
            exclusion["syndrome_not_exactly_mapped"] += 1
            continue
        gold = []
        for tcmt_id in syndrome_mapping[source]:
            entity = data.entity_to_id.get("tcmt:" + tcmt_id)
            term = d1.get(tcmt_id)
            if entity is None or term is None:
                continue
            # The external column is explicitly a syndrome-pattern field. Keep its
            # primary endpoint within the WHO terminology pattern group and avoid
            # turning disease fragments from numbered compound strings into golds.
            if term["English_group"].strip() != "Traditional medicine patterns":
                continue
            relation = data.relation_to_id.get("cpm_has_tcmt::" + term["Chinese_group"].strip())
            if relation is None or entity not in set(map(int, data.range_candidates[relation])):
                continue
            gold.append((relation, entity, tcmt_id))
        gold = sorted(set(gold))
        if not gold:
            exclusion["no_mapped_pattern_in_trained_candidate_range"] += 1
            continue
        records.append({
            "prescription_id": rx, "patient_id": metadata[rx]["patient_id"],
            "visit_half_year": metadata[rx]["visit_half_year"],
            "mapped_herbs": mapped_herbs, "mapped_herb_count": len(mapped_herbs),
            "original_herb_count": len(rx_rows), "source_syndrome": source, "gold": gold,
        })
    queries = []
    for record_index, record in enumerate(records):
        for relation, target, tcmt_id in record["gold"]:
            queries.append((record_index, relation, target, tcmt_id))
    return audit, records, queries, exclusion


def frequency_ranks(data, records, queries):
    frequencies = defaultdict(Counter)
    for _, relation, tail in data.train:
        frequencies[int(relation)][int(tail)] += 1
    external_true = defaultdict(set)
    for record_index, relation, target, _ in queries:
        external_true[(record_index, relation)].add(target)
    ranks = []
    for record_index, relation, target, _ in queries:
        candidates = data.range_candidates[relation]
        scores = np.asarray([frequencies[relation][int(candidate)] for candidate in candidates], dtype=float)
        other = external_true[(record_index, relation)] - {target}
        if other:
            scores[np.isin(candidates, np.fromiter(other, dtype=np.int64))] = -np.inf
        target_position = int(np.searchsorted(candidates, target)); value = scores[target_position]
        ranks.append(1.0 + np.sum(scores > value) + 0.5 * max(0, int(np.sum(scores == value)) - 1))
    return np.asarray(ranks)


def uniform_random_expectation(data, queries, external_true):
    values = defaultdict(list)
    candidate_counts = []
    for record_index, relation, target, _ in queries:
        effective = len(data.range_candidates[relation]) - len(external_true[(record_index, relation)] - {target})
        candidate_counts.append(effective)
        values["mrr"].append(sum(1.0 / rank for rank in range(1, effective + 1)) / effective)
        values["hits1"].append(1.0 / effective)
        values["hits3"].append(min(3, effective) / effective)
        values["hits10"].append(min(10, effective) / effective)
        values["mean_rank"].append((effective + 1) / 2.0)
    return {
        "metrics": {key: float(np.mean(value)) for key, value in values.items()},
        "candidate_count": {
            "min": int(np.min(candidate_counts)), "median": float(np.median(candidate_counts)),
            "max": int(np.max(candidate_counts)), "mean": float(np.mean(candidate_counts)),
        },
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    audit, records, queries, exclusion = load_external(data)
    if not queries:
        raise RuntimeError("No exact-mapped external queries")
    external_true = defaultdict(set)
    for record_index, relation, target, _ in queries:
        external_true[(record_index, relation)].add(target)
    ranks = {}
    for family in FAMILIES:
        per_seed = []
        for seed in SEEDS:
            model = load(family, data, arrays, seed, device)
            values = []
            for record_index, relation, target, _ in queries:
                record = records[record_index]
                values.append(rank_query(
                    model, record["mapped_herbs"], relation, target,
                    data.range_candidates[relation], external_true[(record_index, relation)],
                ))
            per_seed.append(values)
        ranks[family] = np.asarray(per_seed, dtype=float)
        print("EXTERNAL", family, seed_summary(ranks[family]), flush=True)
    frequency = frequency_ranks(data, records, queries)
    patients = np.asarray([records[index]["patient_id"] for index, _, _, _ in queries])
    sizes = np.asarray([records[index]["mapped_herb_count"] for index, _, _, _ in queries])
    report = {
        "role": "zero-shot feasibility external validation",
        "data_source": "10.5281/zenodo.21951692 v1.1",
        "model_state": "three frozen TCM-MKG confirmatory checkpoints; no external fitting or tuning",
        "coverage": {
            "total_prescriptions": 335, "included_prescriptions": len(records),
            "included_patients": len(set(record["patient_id"] for record in records)),
            "evaluation_queries": len(queries), "exclusion_counts": dict(exclusion),
            "herb_unique_exact_coverage": audit["herbs"]["exact_unique_coverage"],
            "herb_record_exact_coverage": audit["herbs"]["record_coverage"],
            "syndrome_unique_exact_coverage": audit["syndromes"]["exact_unique_coverage"],
        },
        "overall": {family: seed_summary(value) for family, value in ranks.items()},
        "per_seed": {family: [metrics(row) for row in value] for family, value in ranks.items()},
        "frequency": metrics(frequency),
        "uniform_random_expectation": uniform_random_expectation(data, queries, external_true),
        "patient_cluster_bootstrap": {
            "hybrid_vs_parameter_control": cluster_bootstrap(ranks["hybrid"], ranks["parameter_control"], patients),
            "hybrid_vs_deepsets": cluster_bootstrap(ranks["hybrid"], ranks["deepsets"], patients, seed=20261521),
            "hybrid_vs_pair_only": cluster_bootstrap(ranks["hybrid"], ranks["pair_only"], patients, seed=20261522),
        },
        "by_mapped_herb_count": {},
        "by_relation": {},
        "limitations": [
            "single-practitioner, single-clinic dataset",
            "recorded diagnoses rather than independent adjudication",
            "strict terminology mapping covers less than half of unique recorded syndrome strings",
            "source domain is individualized decoctions/granules whereas training formulas are Chinese patent medicines",
        ],
    }
    for label, subset in (("2_to_5", sizes <= 5), ("6_to_10", (sizes >= 6) & (sizes <= 10)), ("over_10", sizes > 10)):
        report["by_mapped_herb_count"][label] = {"n": int(subset.sum())}
        if subset.any():
            report["by_mapped_herb_count"][label].update({family: seed_summary(value[:, subset]) for family, value in ranks.items()})
    query_relations = np.asarray([relation for _, relation, _, _ in queries])
    for relation in sorted(set(map(int, query_relations))):
        subset = query_relations == relation
        report["by_relation"][data.relations[relation]] = {
            "queries": int(subset.sum()),
            **{family: seed_summary(value[:, subset]) for family, value in ranks.items()},
        }
    (OUT / "external_zero_shot_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Save the exact included cohort and mappings for audit without embedding model outputs in the source data.
    cohort = [{**{k: v for k, v in record.items() if k != "mapped_herbs"}, "mapped_herb_entity_ids": record["mapped_herbs"]} for record in records]
    (OUT / "external_zero_shot_cohort.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
