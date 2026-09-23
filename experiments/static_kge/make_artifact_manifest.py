"""Create SHA-256 checksums for frozen splits, result JSON, scripts, and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    files = []
    files.extend(sorted((HERE / "benchmark" / "hybrid_holdout").glob("*.tsv")))
    files.extend(sorted((HERE / "results").glob("*.json")))
    files.extend(sorted((HERE / "figures").glob("*.pdf")))
    files.extend(sorted(HERE.glob("*.py")))
    files.extend(sorted(HERE.glob("*.md")))
    files.append(HERE / "external_data" / "ZhaoHanqing_TCM_Dataset_2025_v1.1.zip")
    files.append(HERE / "external_data" / "LICENSE")
    files.append(HERE / "environment.yml")
    records = [
        {"path": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files if path.exists()
    ]
    target = HERE / "EXPERT_SYSTEMS_ARTIFACT_MANIFEST.json"
    target.write_text(json.dumps({"algorithm": "SHA-256", "files": records}, indent=2), encoding="utf-8")
    print(f"{len(records)} files -> {target}")


if __name__ == "__main__":
    main()
