# -*- coding: utf-8 -*-
"""Audit the TCM-centred, explicit relation tables in TCM-MKG."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PART1 = next(ROOT.glob("01_*"))
MKG = PART1 / "TCM-MKG"


def rows(name: str):
    path = MKG / name
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def clean(value: str | None) -> str:
    return (value or "").strip()


def main() -> None:
    groups: Counter[str] = Counter()
    group_examples: dict[str, list[str]] = defaultdict(list)
    tcmt_name: dict[str, str] = {}
    for row in rows("D1_TCM_terminology.tsv"):
        group = clean(row["Chinese_group"]) or "<missing>"
        name = clean(row["Chinese_term"])
        groups[group] += 1
        tcmt_name[clean(row["TCMT_ID"])] = name
        if name and len(group_examples[group]) < 8:
            group_examples[group].append(name)

    relation_specs = {
        "cpm_indication_tcmt": ("D3_CPM_TCMT.tsv", "CPM_ID", "TCMT_ID"),
        "cpm_contains_chp": ("D4_CPM_CHP.tsv", "CPM_ID", "CHP_ID"),
        "cpm_treats_icd11": ("D5_CPM_ICD11.tsv", "CPM_ID", "ICD11_code"),
        "chp_has_property": ("D7_CHP_Medicinal_properties.tsv", "CHP_ID", "Medicinal_properties"),
        "chp_from_species": ("D8_CHP_NP.tsv", "CHP_ID", "species_ID"),
        "chp_contains_compound": ("D9_CHP_InChIKey.tsv", "CHP_ID", "InChIKey"),
    }
    relation_stats = {}
    nodes_by_type: dict[str, set[str]] = defaultdict(set)
    type_map = {
        "CPM_ID": "cpm", "TCMT_ID": "tcmt", "CHP_ID": "chp",
        "ICD11_code": "icd11", "Medicinal_properties": "property",
        "species_ID": "species", "InChIKey": "compound",
    }
    cpm_relations: dict[str, set[str]] = defaultdict(set)
    for relation, (filename, head_col, tail_col) in relation_specs.items():
        total = 0
        unique = set()
        heads, tails = set(), set()
        for row in rows(filename):
            head, tail = clean(row[head_col]), clean(row[tail_col])
            if not head or not tail or head == "NA" or tail == "NA":
                continue
            total += 1
            unique.add((head, tail))
            heads.add(head); tails.add(tail)
            nodes_by_type[type_map[head_col]].add(head)
            nodes_by_type[type_map[tail_col]].add(tail)
            if head_col == "CPM_ID":
                cpm_relations[head].add(relation)
        relation_stats[relation] = {
            "source": filename,
            "valid_rows": total,
            "unique_edges": len(unique),
            "unique_heads": len(heads),
            "unique_tails": len(tails),
        }

    syndrome_vocab = {
        line.strip() for line in (PART1 / "TCM-SD" / "syndrome_vocab.txt").read_text(
            encoding="utf-8-sig"
        ).splitlines() if line.strip()
    }
    term_to_ids: dict[str, list[str]] = defaultdict(list)
    for identifier, name in tcmt_name.items():
        term_to_ids[name].append(identifier)
    exact_overlap = sorted(syndrome_vocab & set(term_to_ids))

    all_cpm = nodes_by_type["cpm"]
    cpm_coverage = Counter(len(cpm_relations[cpm]) for cpm in all_cpm)
    result = {
        "tcmt_groups": [
            {"group": group, "count": count, "examples": group_examples[group]}
            for group, count in groups.most_common()
        ],
        "relation_stats": relation_stats,
        "unique_nodes_by_type": {key: len(value) for key, value in nodes_by_type.items()},
        "cpm_relation_type_coverage": dict(sorted(cpm_coverage.items())),
        "tcm_sd_syndrome_vocab": len(syndrome_vocab),
        "exact_tcm_sd_to_tcmt_overlap": len(exact_overlap),
        "overlap_examples": exact_overlap[:50],
        "limitations": [
            "D8_CHP_NP and D8_CHP_PO are duplicate releases; only D8_CHP_NP is counted.",
            "Predicted compound-target edges and PPI edges are excluded from this explicit TCM-centred audit.",
            "Exact string overlap is not entity alignment and underestimates synonym matches.",
        ],
    }
    out = Path(__file__).with_name("tcm_subgraph_audit.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
