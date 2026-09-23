# -*- coding: utf-8 -*-
"""Audit deterministic herb and syndrome mappings before external model evaluation."""

from __future__ import annotations

import csv
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from compositional_kge import MKG, HERE

EXT = HERE / "external_data"
OUT = HERE / "results"


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def norm_zh(value):
    value = unicodedata.normalize("NFKC", value or "").strip()
    return re.sub(r"[\s()（）,，、·]", "", value)


def norm_en(value):
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = value.replace("insufficiency", "deficiency")
    value = value.replace("pattern", "syndrome")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def aliases(value):
    if not value or value.strip().upper() == "NA":
        return []
    return [x.strip() for x in re.split(r"[;|；]", value) if x.strip()]


def has_cjk(value):
    return bool(re.search(r"[\u3400-\u9fff]", value or ""))


def syndrome_components(value):
    """Separate numbered compound labels while retaining the original label."""
    if not has_cjk(value) or not re.search(r"\d+\s*[,，]?", value):
        return [value.strip()]
    return [part.strip(" ,，;；") for part in re.split(r"\d+\s*[,，]?", value) if part.strip(" ,，;；")]


def main():
    herb_rows = rows(EXT / "ZhaoHanqing_TCM_Prescriptions_2025.csv")
    d6 = rows(MKG / "D6_Chinese_herbal_pieces.tsv")
    d1 = rows(MKG / "D1_TCM_terminology.tsv")

    herb_index = defaultdict(set)
    for row in d6:
        identifier = row["CHP_ID"].strip()
        for name in [row["Chinese_herbal_pieces"], *aliases(row.get("Chinese_synonyms", ""))]:
            herb_index[norm_zh(name)].add(identifier)
    unique_herbs = sorted({row["herb_name_chinese"].strip() for row in herb_rows})
    herb_map, unmatched_herbs, ambiguous_herbs = {}, [], {}
    for name in unique_herbs:
        candidates = sorted(herb_index.get(norm_zh(name), ()))
        if len(candidates) == 1:
            herb_map[name] = candidates[0]
        elif len(candidates) > 1:
            ambiguous_herbs[name] = candidates
        else:
            unmatched_herbs.append(name)

    official_en, synonym_en = defaultdict(set), defaultdict(set)
    official_zh, synonym_zh = defaultdict(set), defaultdict(set)
    term_meta = {}
    syndrome_groups = Counter()
    for row in d1:
        identifier = row["TCMT_ID"].strip()
        syndrome_groups[(row["Chinese_group"], row["English_group"])] += 1
        term_meta[identifier] = row
        official_en[norm_en(row["English_term"])].add(identifier)
        official_zh[norm_zh(row["Chinese_term"])].add(identifier)
        for name in aliases(row.get("Synonyms", "")):
            synonym_en[norm_en(name)].add(identifier)
        for name in aliases(row.get("Chinese_synonyms", "")):
            synonym_zh[norm_zh(name)].add(identifier)
    external_syndromes = sorted({row["syndrome_pattern"].strip() for row in herb_rows if row["syndrome_pattern"].strip() != "Not recorded"})
    syndrome_map, syndrome_details, unmatched_syndromes = {}, {}, []
    for name in external_syndromes:
        matches = set(); detail = []
        for component in syndrome_components(name):
            if has_cjk(component):
                official = official_zh.get(norm_zh(component), set())
                selected = official or synonym_zh.get(norm_zh(component), set())
                mode = "official_chinese" if official else "chinese_synonym"
            else:
                official = official_en.get(norm_en(component), set())
                selected = official or synonym_en.get(norm_en(component), set())
                mode = "official_english" if official else "english_synonym"
            if selected:
                matches.update(selected)
                detail.append({"component": component, "mode": mode, "tcmt_ids": sorted(selected)})
        if matches:
            syndrome_map[name] = sorted(matches)
            syndrome_details[name] = detail
        elif not has_cjk(name):
            external_tokens = set(norm_en(name).split())
            scored = []
            for key, identifiers in official_en.items():
                target_tokens = set(key.split())
                jaccard = len(external_tokens & target_tokens) / max(1, len(external_tokens | target_tokens))
                ratio = difflib.SequenceMatcher(None, norm_en(name), key).ratio()
                score = 0.6 * jaccard + 0.4 * ratio
                if score >= 0.45:
                    for identifier in identifiers:
                        scored.append({"tcmt_id": identifier, "english_term": term_meta[identifier]["English_term"], "english_group": term_meta[identifier]["English_group"], "score": score})
            unmatched_syndromes.append({"source": name, "suggestions": sorted(scored, key=lambda x: x["score"], reverse=True)[:5]})
        else:
            unmatched_syndromes.append({"source": name, "suggestions": []})

    prescription_herbs = defaultdict(list)
    prescription_patient = {}
    prescription_syndrome = {}
    for row in herb_rows:
        rx = row["prescription_id"]
        prescription_herbs[rx].append(row["herb_name_chinese"].strip())
        prescription_patient[rx] = row["patient_id"]
        prescription_syndrome[rx] = row["syndrome_pattern"].strip()
    eligible_exact = [
        rx for rx, herbs in prescription_herbs.items()
        if sum(name in herb_map for name in herbs) >= 2 and prescription_syndrome[rx] in syndrome_map
    ]
    report = {
        "mapping_protocol": {
            "herbs": "Unicode NFKC + punctuation/space removal; exact Chinese official name or listed synonym only",
            "syndromes": "numbered compound labels split deterministically; exact Chinese/English official term first, otherwise exact listed synonym; English additionally normalizes insufficiency=>deficiency and pattern=>syndrome",
            "fuzzy_suggestions": "audit only; never automatically included in evaluation",
        },
        "herbs": {
            "unique_external": len(unique_herbs), "exact_unique_mapped": len(herb_map),
            "exact_unique_coverage": len(herb_map) / len(unique_herbs),
            "record_coverage": sum(row["herb_name_chinese"].strip() in herb_map for row in herb_rows) / len(herb_rows),
            "ambiguous": ambiguous_herbs, "unmatched": unmatched_herbs, "mapping": herb_map,
        },
        "syndromes": {
            "unique_external_recorded": len(external_syndromes), "exact_unique_mapped": len(syndrome_map),
            "exact_unique_coverage": len(syndrome_map) / len(external_syndromes),
            "unmatched_with_nonbinding_suggestions": unmatched_syndromes,
            "mapping": syndrome_map, "mapping_details": syndrome_details,
        },
        "prescriptions": {
            "total": len(prescription_herbs), "eligible_exact_mapping": len(eligible_exact),
            "eligible_fraction": len(eligible_exact) / len(prescription_herbs),
            "eligible_unique_patients": len({prescription_patient[rx] for rx in eligible_exact}),
        },
        "tcm_term_group_counts": [
            {"chinese_group": zh, "english_group": en, "count": count}
            for (zh, en), count in syndrome_groups.most_common()
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "external_mapping_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "herbs": {k: report["herbs"][k] for k in ("unique_external", "exact_unique_mapped", "exact_unique_coverage", "record_coverage")},
        "syndromes": {k: report["syndromes"][k] for k in ("unique_external_recorded", "exact_unique_mapped", "exact_unique_coverage")},
        "prescriptions": report["prescriptions"],
        "groups": report["tcm_term_group_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
