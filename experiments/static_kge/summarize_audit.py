# -*- coding: utf-8 -*-
import json
from pathlib import Path


data = json.loads(Path(__file__).with_name("source_audit.json").read_text(encoding="utf-8"))
print("TCM-MKG")
for item in data["tcm_mkg"]:
    print(item["file"], item["rows_including_header"] - 1, " | ".join(item["header"]))
print("SYMMAP")
for item in data["symmap"]:
    sample = item.get("sample", [])
    print(item["file"], item.get("max_row"), " | ".join(str(v) for v in (sample[0] if sample else [])))
