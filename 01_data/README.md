# Dataset access and placement

No dataset is distributed here. Download from original providers and observe their terms.

| Use | Source | Place files in |
| --- | --- | --- |
| Main benchmark | [TCM-MKG v3](https://zenodo.org/records/19804367) | `01_data/TCM-MKG/` |
| External stress test | [Zhao Hanqing prescriptions v1.1](https://zenodo.org/records/21951692) | `experiments/static_kge/external_data/` |
| Exploratory audit | [TCM-SD / ZY-BERT](https://github.com/Borororo/ZY-BERT) | `01_data/TCM-SD/` |
| Exploratory audit | [SymMap v2](http://www.symmap.org/download/) | `01_data/SymMap_v2/` |

The main benchmark reads TCM-MKG tables `D3_CPM_TCMT.tsv`, `D4_CPM_CHP.tsv`, `D5_CPM_ICD11.tsv`, `D7_CHP_Medicinal_properties.tsv`, `D8_CHP_NP.tsv`, and `D9_CHP_InChIKey.tsv`. Other exploratory scripts may require further tables.

The scripts create processed triples and held-out splits under `experiments/static_kge/benchmark/`. These derived files are not redistributed. Regenerate them from original data, or request the exact processed partitions and deterministic external mappings from the corresponding authors, subject to original-provider permissions.
