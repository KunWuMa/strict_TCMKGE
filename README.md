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


### Reviewer-controlled second-order experiments

After regenerating the local benchmark and obtaining the source datasets under their original terms, run the exactly parameter-matched second-moment comparison and corrected elementwise Deep Sets baseline:

```bash
python experiments/static_kge/test_second_moment_review.py
python experiments/static_kge/run_second_moment_review.py
python experiments/static_kge/run_true_deepsets_review.py
python experiments/static_kge/analyze_second_moment_review.py
python experiments/static_kge/evaluate_second_moment_external.py
python experiments/static_kge/evaluate_true_deepsets_external.py
python experiments/static_kge/benchmark_review_encoders.py
python experiments/static_kge/make_review_core_figures.py
```

The released JSON files contain aggregate scores, development selections, and clustered uncertainty. Query-level ranks, checkpoints, individual prescriptions, mappings, and patient-level outputs are excluded. The external evaluation is an exploratory domain-shift stress test; its mapping is incomplete and does not establish clinical validity. The second-moment and pair summaries are algebraically equivalent given mean and formula size for formulas containing more than one herb, so the comparison tests parameterization rather than unique pair information.

See [experiment protocol](experiments/static_kge/EXPERT_SYSTEMS_EXPERIMENT_PROTOCOL.md) for further runs and evaluation details. Some analyses additionally require locally regenerated splits and checkpoints. The InGram comparison uses the official implementation at commit `f2554dd0c684bad71d3e01d534f2f7814cf28644`.

The exact processed splits and external terminology mappings may be requested from the corresponding authors, subject to the source providers' permissions. To check a staged release for excluded content, run `python release_audit.py`. This is methodological research, not clinical advice.
