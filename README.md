# PairComp-KGE code and aggregate results

This repository releases the PairComp-KGE implementation and frozen aggregate experiment results. It excludes the paper manuscript, third-party source datasets, derived graph triples, train/dev/test TSV splits, individual prescriptions, and patient-level outputs.

- `experiments/static_kge/*.py`: benchmark construction, models, baselines, training, evaluation, analysis and plotting.
- `experiments/static_kge/results/`: aggregate metrics, development selections, uncertainty estimates and audits.
- `experiments/static_kge/figures/paper/`: final figures.
- `experiments/static_kge/benchmark/`: split manifests; no triples.
- `01_data/`: original-data links and placement instructions.

Set up the environment from the repository root:

```bash
conda env create -f experiments/static_kge/environment.yml
conda activate paircomp-kge
```

Download TCM-MKG v3 as described in [01_data/README.md](01_data/README.md), then run:

```bash
python experiments/static_kge/build_benchmark.py
python experiments/static_kge/build_composition_benchmark.py
python experiments/static_kge/build_hybrid_benchmark.py
python experiments/static_kge/run_hybrid_confirmatory.py
python experiments/static_kge/analyze_hybrid_confirmatory.py
```

See [experiment protocol](experiments/static_kge/EXPERT_SYSTEMS_EXPERIMENT_PROTOCOL.md) for further runs and evaluation details. Some analyses additionally require locally regenerated splits and checkpoints. The InGram comparison uses the official implementation at commit `f2554dd0c684bad71d3e01d534f2f7814cf28644`.

The exact processed splits and external terminology mappings may be requested from the corresponding authors, subject to the source providers' permissions. To check a staged release for excluded content, run `python release_audit.py`. This is methodological research, not clinical advice.
