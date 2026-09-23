# -*- coding: utf-8 -*-
"""Audit dosage fields used by the composition encoder before model design."""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MKG = next(ROOT.glob("01_*")) / "TCM-MKG"


def main():
    path = MKG / "D4_CPM_CHP.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    report = {"fields": fields, "rows": len(rows)}
    for field in fields:
        values = [(row.get(field) or "").strip() for row in rows]
        report[field] = {
            "missing": sum(v in {"", "NA"} for v in values),
            "unique": len(set(values)),
            "top": Counter(values).most_common(12),
        }
    for candidate in ("Dosage_ratio", "Dosage", "Dose", "Ratio"):
        if candidate not in fields:
            continue
        values = [(row.get(candidate) or "").strip() for row in rows]
        numeric = []
        malformed = []
        for value in values:
            if value in {"", "NA"}:
                continue
            try:
                number = float(value)
                if math.isfinite(number): numeric.append(number)
            except ValueError:
                malformed.append(value)
        report[candidate]["numeric_count"] = len(numeric)
        report[candidate]["malformed_top"] = Counter(malformed).most_common(20)
        if numeric:
            numeric.sort()
            report[candidate]["numeric_summary"] = {
                "min": numeric[0], "p25": numeric[len(numeric)//4],
                "median": numeric[len(numeric)//2], "p75": numeric[3*len(numeric)//4],
                "max": numeric[-1], "mean": sum(numeric)/len(numeric),
            }
    by_formula = defaultdict(list)
    for row in rows:
        by_formula[row["CPM_ID"]].append((row.get("Dosage_ratio") or "").strip())
    complete = {f: values for f, values in by_formula.items() if all(v not in {"", "NA"} for v in values)}
    partial = {f: values for f, values in by_formula.items() if any(v not in {"", "NA"} for v in values) and f not in complete}
    absent = {f: values for f, values in by_formula.items() if all(v in {"", "NA"} for v in values)}
    sums = [sum(map(float, values)) for values in complete.values()]
    report["formula_dosage_coverage"] = {
        "formulas": len(by_formula), "complete": len(complete), "partial": len(partial), "absent": len(absent),
        "complete_fraction": len(complete) / len(by_formula),
        "complete_sum_near_one": sum(abs(total - 1.0) <= 0.02 for total in sums),
        "complete_sum_min": min(sums), "complete_sum_median": sorted(sums)[len(sums)//2], "complete_sum_max": max(sums),
    }
    output = Path(__file__).with_name("dosage_audit.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
