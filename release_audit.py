"""Reject manuscripts and source/record-level data from the staged public release."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
tracked = subprocess.check_output(
    ["git", "ls-files", "-z"], cwd=ROOT
).decode("utf-8").split("\0")
tracked = [name.replace("\\", "/") for name in tracked if name]
for name in tracked:
    lower = name.lower()
    assert "/manuscript/" not in lower and "manuscript" not in Path(name).name.lower(), name
    assert Path(name).suffix.lower() not in {".tsv", ".csv", ".xlsx", ".xls", ".zip", ".7z", ".rar", ".jsonl", ".docx"}, name
    assert not name.startswith("experiments/static_kge/benchmark/") or Path(name).name in {"README.md", "manifest.json", "composition_manifest.json", "hybrid_manifest.json", "pair_manifest.json", "reviewer_splits_manifest.json"}, name
    if name.startswith("01_data/") or name.startswith("experiments/static_kge/external_data/"):
        assert Path(name).name == "README.md", name
    assert Path(name).name not in {"external_mapping_audit.json", "external_zero_shot_cohort.json", "pair_attribution_cases.json", "ingram_tcm_full200_results.json"}, name
print(f"Release audit passed: {len(tracked)} tracked files; no manuscript or source/record-level data.")
