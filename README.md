# DEL-ML workflow for PGK2 (DREAM x CACHE Target 2035): Active Learning rounds AL1 and AL2

Code that regenerates the two Active Learning rounds (50 compounds each) from the provided raw files. The descriptions of the workflow are in `WRITEUP_AL1.md` and `WRITEUP_AL2.md`.

## Contents

| Path | What it is |
|---|---|
| `pipeline/run_pipeline.py` | Single script, stage by stage: inputs, DEL and label components, series, AL1, AL2, ablation, DEL ranking of the full library, verification |
| `pipeline/run_original_chain.py` | Runs the original chain of scripts (`exp10`, `exp16`, `exp43`, `exp48`, `exp51`, `exp60`) in an isolated folder and compares the result |
| `pipeline/lineage.py`, `pipeline/dag.py` | Optional: classification of the 150+ scripts by the data they read and the dependency graph between scripts |
| `scripts/`, `earlier_workflow/` | The scripts of the whole development, including the original chain |
| `active_learning_candidates/` | The two delivered files (`Team_candeleroj_AL1_20261005.csv`, `Team_candeleroj_AL2_newgroups_20261005.csv`) |
| `requirements-pipeline.txt`, `requirements.txt` | Pinned packages for the pipeline; full development list |

## Data (not included)

The raw files are provided by the challenge organizers and are not redistributed. Place them as follows (md5 checked by the first stage):

| File | Path | md5 |
|---|---|---|
| DEL selection data | `data/raw/PGK2_selection.parquet` | `f1832ca598e647eb66ccd1f38a9d7ea5` |
| Active Learning labels | `data/raw/active_learning_data.csv` | `470533959ac0427504e71042231cbf9b` |
| File of compounds to score | `data/raw/Val-Test-set/PGK2_Test_split.csv` | `43d5f11f644893d77704d0c9aa38f5c8` |
| Structure library | `data/raw/Val-Test-set/PGK2_Validation_split.csv` | `348f76e1c4603358a3ac572d12c35020` |

## Reproduce

Run everything from the repository root. About 11 minutes on 8 threads with 16 GB of memory.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-pipeline.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python pipeline/run_pipeline.py --stage all
```

Outputs go to `pipeline_out/`: the two rounds (`Team_candeleroj_AL1.csv`, `Team_candeleroj_AL2.csv`), the selections, `ablation.json`, `library_positions.json` (the DEL ranking of all 428,960 compounds and where the 100 chosen compounds sit in it) and `verification.json`. The last stage compares both rounds with the delivered files in `active_learning_candidates/`.

With the pinned packages above, both rounds reproduce the 50 selected compounds of the delivered files, and the original chain reproduces their score column exactly. With other versions of scikit-learn the 50th compound of AL1 can change.

To run the original chain instead (about 16 minutes, in an isolated folder):

```bash
.venv/Scripts/python pipeline/run_original_chain.py
```

Single stages can be run with `--stage inputs|signals|series|al1|al2|ablation|library|verify`. Optional stages: `lineage` (needs the leaderboard export in `pipeline_in/synapse_results.csv`) and `delml` (needs the 2026-10-22 candidate files).

## License

Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE`. Copyright (c) 2026 Juan Antonio Candelero Cervantes.
