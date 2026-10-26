# PairProto-KGE: code and aggregate results

Current study: relation-conditioned pair prototypes for formula-inductive
knowledge-association retrieval. TCM-MKG and the KTKP subset of InPETM are trained
and evaluated separately. This is not clinical validation or pharmacological synergy.

## Current evidence (26 October 2026)

The main comparisons and all stochastic evaluation groups use five matched seeds
(20260915–20260919). The training cap is 500 native rounds, development
MRR is checked every round, and patience is 15 non-improving checks. Native
rounds differ by architecture; equal caps are not equal compute.
Two development-selected follow-ups update the current main table:
TCM-MKG Set Transformer (learning rate 0.003, 2 heads) and
KTKP Mean-PairRE (learning rate 0.001).
Original configurations and adverse controls remain in the aggregate record.

- Code: `experiments/static_kge/`.
- Current aggregate metrics and uncertainty:
  `experiments/static_kge/published_results/scientific_reports_20261026/`.
- Current figures (record-level case figure excluded):
  `experiments/static_kge/figures/scientific_reports_20261026/`.
- Frozen histories: `experiments/static_kge/outputs/*/protocol.json`.
- Older PairComp implementation/results remain for historical provenance;
  they are not the proposed PairProto method.

## Data access

Download the original sources using [01_data/README.md](01_data/README.md) and
[the independent-source instructions](experiments/static_kge/independent_data/README.md).
Set `TCM_MKG_DIR` and `INPETM_DATA_DIR` to original table directories.
No source datasets, processed triples, query records, ranks, patient records,
checkpoints or manuscripts are distributed here.
The complete Excel includes record-level queries and is not public.
Request exact processed partitions and record-level evidence from the corresponding
author, subject to provider terms. Reconstruction code and aggregate hashes are public.

## Environment and upstream models

Use Python 3.9 and PyTorch 2.3.1 with CUDA 12.1. Analysis packages are in
`experiments/static_kge/requirements-pairproto.txt`.
Install official PyG / torch-scatter wheels compatible with that PyTorch/CUDA
combination: torch-geometric 2.5.3 and torch-scatter 2.1.2+pt23cu121.
Fetch pinned upstream code, without its benchmark data:

```bash
python experiments/static_kge/fetch_official_kge_pinned.py
```

NodePiece, InGram, ULTRA and TRIX are protocol adapters, trained from scratch;
these results do not reproduce every capability or best score in their papers.
No external graph pretraining is used.

## Reproduction entry points

Train either locked follow-up freshly, including its first seed,
without requiring the authors' reused checkpoint:

```bash
python experiments/static_kge/reproduce_sr_baseline_fresh.py --source PRIMARY --seed 20260915
python experiments/static_kge/reproduce_sr_baseline_fresh.py --source KTKP --seed 20260915
```

Repeat for all five seeds. Fresh runs have a separate directory and identity.
The archived `run_scientific_reports_followup.py` mirrors the authors'
checkpoint reuse and deliberately requires the original audit checkpoint.
Author-local paths are not downloadable data.

`budget500_training.py`, `budget500_paper_worker.py` and the frozen protocols
contain the model/evaluation implementations and experiment matrix.
The original `run_budget500_comparisons.py` / `run_budget500_paper.py` schedulers
require earlier artifacts and author-local paths in their preflights.
They are provenance scripts, not a tested clean-machine one-command benchmark.
Reconstruct data and restore configuration locks before running them.

Numerical paper analysis reads Excel ranks and typed metadata.
Displayed SeedMetrics are cross-checked against ranks; editing only those
display values does not legitimately change experimental evidence.
With complete evidence obtained under the data terms, run
`scientific_reports_results.py --recompute-only --statistics-only`.
Its default author-only manuscript export is intentionally excluded here.

For aggregate-only plots, without source data:

```bash
python experiments/static_kge/plot_sr_aggregate_release.py
```

Source-table reconstruction checks cover all five reported partitions.
The author machine and source reconstructions were verified; independent
end-to-end execution of the whole matrix on a fresh machine is not claimed.

## Interpretation

The proposed configuration improves over the shared mean backbone on both main
partitions, with positive directions in all five partitions. This does not imply
uniform superiority, lowest variance, clinical causality or unrestricted ranking
superiority. Competitive or better raw-score and support ablations are retained.
