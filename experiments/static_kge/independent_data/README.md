# Independent data for the sparse-pair follow-up

The experiments use the authors' [InPETM release](https://github.com/bmil-jnu/InPETM)
([data record and licence](https://data.mendeley.com/datasets/kpvgtm6hh6/1)).
Download `InPETM.zip` from the authors and extract these original files to
`inpetm/full_data/`:

- `prescription.csv`
- `prescription_herb.csv`
- `prescription_phenotype.csv`

The original ZIP and CSVs are ignored by Git. The benchmark builder reads
these files directly and logs composition-group splits, relation ranges and
query exclusions. No processed data or inferred target labels are distributed
here.

The exploratory Chinese-formula stress test uses
[DragonTCM](https://huggingface.co/datasets/f-galkin/DragonTCM) (CC BY-NC 4.0).
Download its `formulas` and `relations` Parquet subsets to
`dragontcm/formulas.parquet` and `dragontcm/relations.parquet`, then run
`prepare_dragontcm.py` to create the local normalized tables. The dataset card
warns that automated parsing may contain errors; the conversion only uses
explicit source relation rows and does not certify clinical correctness.
