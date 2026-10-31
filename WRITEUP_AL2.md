# DEL-ML workflow: 428,960 compounds to 100. Active Learning round 2 (AL2)

Active Learning ID: 9783567 (`Team_candeleroj_AL2_newgroups_20261005.csv`)

Blind validation: the submissions of the validation phase were used for model testing only.

## Team

Team name: Team_candeleroj

### People
* Juan Antonio Candelero Cervantes, Independent researcher

### Institution
* Independent researcher, 97219, Merida, Yucatan, Mexico

## Summary

This document describes one workflow. It starts from the provided DEL selection data and the 428,960-compound library and ends in 100 selected compounds, delivered in two rounds (AL1 and AL2). The workflow was built by adding one component at a time to the previous ones; each component is described with what it adds, the role the DEL data play in it, and whether it was kept, replaced or dropped in the final workflow. The two rounds are the two outputs of the same workflow and are reported side by side under *DEL-ML Model training and selection*. This document accompanies AL2, the second round. It differs from round 1 in the weight of the DEL component, in the place where that component acts (the selection, not only the score) and in how the 50 compounds are chosen.

## Dataset

**AIRCHECK datasets.** The DEL selection data and the test and validation splits come from AIRCHECK as two datasets: PGK2, UUID `aLNFc0QKh6hLUXlADmM8-V1` (`PGK2_selection.parquet`), and PGK2_CACHE_Val_Test_Set, UUID `Uc1csiT2szUqBJIKA8uH-V1` (`PGK2_Test_split.csv` and `PGK2_Validation_split.csv`). Two further AIRCHECK datasets were downloaded and read only by scripts of the development work, and neither enters the models that generate the two rounds: OPEN DEL, UUID `HHzlNK0ydopW2aymGG5w-V1` (building-block SMILES, read by `scripts/disynthon_score.py`), and PGK2_NTC_Supplement, UUID `b2VuvfPOyqWSn6oPXalq-V1` (241,769 rows, read by `scripts/gp_step1_prepare.py` to widen the NTC background; the positive rule of this workflow uses the `count_NTC` column of `PGK2_selection.parquet` instead).

**Files read by the two rounds.** Four provided files, no others, described by the role each plays: the DEL selection data (`PGK2_selection.parquet`; 7,703,070 rows, 7,487,570 unique SMILES, not deduplicated); the Active Learning labels (`active_learning_data.csv`; 6,580 compounds, 39 positives); the compounds to score (`PGK2_Test_split.csv`; 184,632, the scored pool being these minus the labeled compounds, 178,052); and a structure library of 428,960 SMILES (`PGK2_Validation_split.csv`), without labels, used only for the common-substructure criterion. The submission template `active_learning_submission_example.csv` is read for the output format only. No label other than the Active Learning labels was used to train or score the compounds of the pool in either round.

**Resources outside AIRCHECK.** Earlier stages of the workflow also used two: CACHE #7 co-crystal structures, for docking (`PGK2_cmp21.pdb`, `PGK2_cmp47.pdb`, `PGK1_cmp45.pdb`), and a pretrained DEL transformer (DELBERT). Neither enters the models that generate the two rounds.

## Data preparation

**DEL labels.** Positives: `count_PGK2 >= 5` and `count_NTC == 0` (10,551 rows, 5,797 unique SMILES). Negatives: 50,000 rows drawn at random from `count_PGK2 <= 1` (seed 0).

**Descriptors.** Morgan count fingerprints, 2048 bits, `log1p`. Radius 2 for the DEL component, radius 3 for the label component.

**Scored pool.** The 178,052 compounds without an Active Learning label. Both component scores are converted to z-scores over this pool before being combined. The label component uses 50,000 background negatives drawn at random from the half of the pool it does not score (seed 48).

## DEL-ML Model architecture

Two instances of one algorithm, trained on different sources: L2-regularized logistic regression, `C = 0.05`, balanced class weights.

1. **DEL component.** Trained only on the DEL selection data.
2. **Label component.** Trained on the 6,580 Active Learning labels plus 50,000 background negatives, cross-fitted over two halves of the pool.

The score of a round is `z(label component) + w * z(DEL component)`, with w = 0.1 in round 1 and 0.3 in round 2. The weight 0.3 was chosen so that the DEL component would change which compounds are selected, not by cross-validation.

The workflow is not trained on Active Learning data alone. The DEL component learns from 10,551 positive rows (5,797 unique SMILES) and 50,000 negatives of the DEL selection data; the label component learns from 39 positives and 6,541 labeled negatives plus background negatives; the earlier stages come from DEL-trained rankings.

On labels it never saw, the DEL component reaches ROC-AUC 0.770 (bootstrap 95% CI 0.707-0.826) on the Active Learning labels (PR-AUC 0.0214, chance 0.0059; 9 of 39 positives in the top 5%, 3 in the top 1%) and ROC-AUC 0.606 (0.533-0.674) on the confirmed labels of the earlier stages (32 positives, 2,592 negatives). The two figures come from different, non-random samples and are not compared with each other.

## DEL-ML Model training and selection

**Training.** Scaffold-grouped cross-validation on the 6,580 labels. Label component alone, plus 0.1 DEL and plus 0.3 DEL: ROC-AUC 0.910, 0.911 and 0.912; PR-AUC 0.386, 0.396 and 0.395 (the difference between the label component alone and plus 0.1 DEL is +0.0016 in ROC-AUC, 95% CI -0.0013 to +0.0048).

**Selection of the two rounds.**

| | Round 1 (AL1) | Round 2 (AL2) |
|---|---|---|
| Selection | 17 series members (eight series with at least two known positives; quota of 3 per series with at least 25 candidates, otherwise 2, capped at 17) and 33 by label-component score (pairwise Tanimoto below 0.6 and one InChIKey block per compound) | 32 by score (at most 2 per group at Tanimoto 0.32) and 18 from groups at Tanimoto below 0.32 to every round 1 compound, one per group; the 50 of round 1 are excluded |
| DEL weight in the score | 0.1 (9% of the blend) | 0.3 (23% of the blend) |
| DEL term in the selection | No | Yes |
| Of the 50, how many change if the DEL term is removed | 0 | 18 (13 of the 32 by score, 5 of the 18 forced) |
| Effect of the DEL term on the ranking | Spearman 0.995 with the label-only ranking; the top 100 / 1,000 / 10,000 share 95 / 929 / ~9,355 | the top 100 / 1,000 / 10,000 share 88 / 787 / 8,263 with the ranking without the term |
| Position of the 50 in the DEL ranking of the 428,960 compounds | 629 to 200,518, median 21,781; 4 within the first 1,000, 17 within 10,000, 36 within 50,000, all 50 within the first half | 33 to 111,488, median 8,885; 8 within the first 1,000, 29 within 10,000, 44 within 50,000, all 50 within the first half |
| Position of the 50 in the DEL ranking of the 178,052-compound pool | 94 to 79,508, median 7,338; all 50 within the first half (89,026) | 3 to 42,429, median 2,720; all 50 within the first half |
| Positions in the label ranking of the pool | all 50 within the first 119 | all 50 within the first 116 of the pool without round 1 |

The score file of round 2 is built on a base in which the 50 compounds of round 1 already sit at the top (at least 0.999), so those 50 keep high scores in it. Across the 100 compounds, 18 depend on the DEL term, all in round 2. Repeating the round 1 rule inside a pool restricted to the top half of the DEL signal (89,026 compounds) returns the same 50, and 47, 45, 44 and 36 for the top 40, 30, 20 and 10%; this follows from the agreement between the two components, and the DEL signal was not used as a filter. Had round 2 taken the next 50 of the round 1 ranking, 46-47 of them would have been neighbors of the round 1 compounds (Tanimoto 0.32 or more) and the 100 compounds would span 15-17 similarity groups instead of 26; the DEL term would have changed 0-5 of them instead of 18. In round 2, 32 of the 50 remain within Tanimoto 0.32 of a round 1 compound; the dissimilar groups apply to the 18 forced compounds.

## Workflow

The steps are listed in the order in which they were run. Each step names the tool, the molecules it was applied to and, where the tool is published, its version and DOI. Steps 1 to 4 belong to the earlier stages and none of them enters the two rounds; steps 5 to 9 produce them. What each step added and whether it was kept is summarized in *Stages* below.

### Step 1. DEL-trained classifiers

XGBoost 3.4.1 with asymmetric loss and LightGBM 4.7.0, trained on the DEL selection data (7,703,070 rows) and applied to the whole 428,960-compound library to rank it. XGBoost: DOI 10.1145/2939672.2939785. LightGBM: Ke et al., NIPS 2017 proceedings, no DOI found. Those are the latest releases; neither package is pinned in `requirements.txt`, and lightgbm 4.7.0 is also the version present in the development environment. Replaced in the final workflow: the ranking of the library is produced by the logistic DEL component of step 7.

### Step 2. DEL fragment enrichment

Own computation on the DEL selection data (7,703,070 rows), applied to the whole library as one term of the composite scores of the first stages. No published tool, no DOI. Not part of the two rounds.

### Step 3. Docking

AutoDock Vina 1.2.7 (DOI 10.1021/acs.jcim.1c00203; original method 10.1002/jcc.21334), `vina` scoring function, exhaustiveness 4, 24 A box, seed 42. Rigid receptors were built from the CACHE #7 co-crystals `PGK2_cmp21.pdb` and `PGK1_cmp45.pdb` (chain A, ligand LIG in both), with the box centred on the centroid of the co-crystal ligand. Ligands were prepared with Meeko 0.8.0 (`MoleculePreparation`, `PDBQTWriterLegacy`; no DOI found). Applied to subsets, not to the full library: a control of 158 molecules (ATP-competed DEL hits against size-matched DEL non-hits), a screen of 300 pool candidates and a retrospective set of 143. Those are the latest releases of `vina` and `meeko`; neither is pinned in `requirements.txt` and the versions of the development environment are not recorded. Not part of the two rounds.

### Step 4. Pharmacophore scoring

Own scoring, independent of the DEL data, built from the 4-aminoquinazoline hinge motif of the CACHE #7 PGK2 inhibitors and of terazosin bound to the PGK1 nucleotide site (PDB 4O33), with bioisosteric replacements of that core and quinazoline itself excluded. Applied to the pool. No published tool, no DOI. Not part of the two rounds.

### Step 5. Family expansion by Tanimoto similarity

RDKit 2026.03.6 (DOI 10.5281/zenodo.22140358), Morgan fingerprints and Tanimoto similarity, applied to the library to find neighbors of the first confirmed active; each neighbor was checked against the DEL data. Kept in the final workflow as the notion of family used for grouping and exclusion.

### Step 6. Series by maximum common substructure

The organizers' series algorithm (Kruger et al., DOI 10.1021/acs.jcim.0c00204), applied to the known positives, and the resulting maximum common substructures matched against the unlabeled pool with RDKit 2026.03.6. Eight series with at least two known positives were obtained, and 147 unlabeled compounds of the pool contain those substructures. Supplies the 17 series members of round 1.

### Step 7. DEL component

L2-regularized logistic regression (scikit-learn 1.9.1, DOI 10.48550/arXiv.1201.0490), `C = 0.05`, balanced class weights, on Morgan count fingerprints of radius 2 and 2048 bits with `log1p` (RDKit 2026.03.6). Trained on the DEL selection data (10,551 positive rows, 5,797 unique SMILES, against 50,000 negatives drawn at random from `count_PGK2 <= 1`, seed 0) and applied both to the 178,052-compound pool, where it is the DEL term of the score, and to the whole 428,960-compound library, for the positions reported above. Used in both rounds.

### Step 8. Label component

The same algorithm and the same packages, on Morgan count fingerprints of radius 3. Trained on the 6,580 Active Learning labels (39 positives) plus 50,000 background negatives drawn at random from the half of the pool it does not score (seed 48), cross-fitted over two halves, and applied to the 178,052-compound pool. Used in both rounds.

### Step 9. Blend, diversity and grouping

Both component scores are converted to z-scores over the 178,052-compound pool and combined as `z(label component) + w * z(DEL component)`. Diversity and grouping use Tanimoto similarity over Morgan fingerprints and Butina clustering (DOI 10.1021/ci9803381), both from RDKit 2026.03.6, with the thresholds of each round. Applied to the pool; its output is the 50 compounds of each round.

### Stages

| Stage | Component added | What it adds | Role of the DEL data | In the final workflow |
|---|---|---|---|---|
| 0 | DEL-trained models (XGBoost with asymmetric loss, fragment enrichment, LightGBM) trained on the DEL selection data (7,703,070 rows) | A ranking of the whole library from the DEL data alone | The training data | The DEL data remain the base; a logistic regression on the same data is the DEL component |
| 1 | Family expansion from the first confirmed active: Tanimoto neighbors, checked against the DEL data | More compounds in the families of the first | Each neighbor is checked against the DEL data | Kept as the concept of family used for grouping and exclusion |
| 2 | Bank of confirmed negatives | Compounds known not to be active: padding for identification rounds, negatives for later models, scaffolds to avoid | The first stage supplied 1,121 of the 2,592 | Kept |
| 3 | Identification of individual actives (bisection, binary pools, isolating compounds) | Individual actives and families with their cluster structure | None directly | Method knowledge only; not part of the final selection |
| 4 | Model trained on the confirmed labels of the earlier stages (logistic regression, ECFP6 counts, background negatives) | A ranking for compounds not yet tested | The actives and negatives it learns from were found through the earlier DEL-based stages | Kept: the label component |
| 5 | Analogs and series by maximum common substructure (the organizers' series algorithm) | Members of the series of the known actives | None | Series kept; plain analogs of isolated actives dropped |
| 6 | Diversity rule, Tanimoto < 0.6 | Spread of the chosen compounds over chemistry | None | Kept; the stricter 0.32 variant replaced |
| 7 | DEL logistic component as an explicit term | The DEL signal inside the ranking | The DEL component itself | Kept (weight 0.1 in round 1, 0.3 in round 2) |
| 8 | Released Active Learning labels (6,580 compounds, 39 positives) | The label component for the pool to be scored | The DEL component stays in the score; the labels refine it | Used in both rounds |
| 9 | Round 2 | Exclusion of the round 1 compounds, 18 compounds forced into dissimilar groups, DEL weight raised to 0.3 and inside the selection | The DEL term now changes the selection | Used in round 2 |

The components do not add up linearly: some were replaced or dropped after being tested, and that is how the final workflow was reached.

### DEL weight by stage

The weight is what each stage put in; what translates is where the compounds chosen at that stage sit in the DEL signal, and, where the DEL term is explicit, how much the selection changes when it is removed. Position is the percentile in the final DEL component over the 428,960-compound library (chance: median 0.50, 10% in the top 10%, 50% in the top 50%); only compounds first chosen at the stage are counted, so known-negative padding is excluded.

| Stage | Nominal DEL weight | Compounds first chosen | Median percentile | In top 10% | In top 50% | Selection change when the DEL term is removed |
|---|---|---|---|---|---|---|
| Composite scores | 0.45, 0.35, 0.5 | 258 | 0.99 | 74% | 96% | not measured |
| Initial DEL-trained ranking and probes | 1.0 | 444 | 0.64 | 16% | 64% | not applicable (the ranking is the DEL model) |
| Families and ensemble | not recorded | 117 | 0.85 | 38% | 92% | not measured |
| LightGBM ranking (DEL-trained) | 1.0 | 250 | 0.68 | 22% | 63% | not applicable |
| Recipe on confirmed labels | 0 | 219 | 0.86 | 42% | 88% | not applicable (no DEL term) |
| Explicit blend | 0.1 | 131 | 0.98 | 70% | 93% | not measured |
| Round 1 (AL1) | 0.1 (9% of the blend) | 50 | 0.95 | 68% | 100% | 0 of 50 |
| Round 2 (AL2) | 0.3 (23% of the blend) | 50 | 0.98 | 88% | 100% | 18 of 50 |

The DEL-derived element is one line throughout the workflow: the same DEL data, refined stage by stage, with the DEL component of the final stages as its most recent form. The position of each stage's choices measures how closely they agree with that most recent form (Spearman 0.27-0.30 for the earlier elements; 0.37 between the DEL component and the label component). Agreement is partial because the element kept being refined as the workflow evolved, and because the initial ranking was sampled at increasing depth by design (from the top 50 to rank 100,000, plus a random control) and the LightGBM ranking in bands (top 50, top 1%, 1-5%), which lowers their averages. The weights of the first stages belong to different scoring schemes (the composite scores weigh a DEL fragment-enrichment term among docking and pharmacophore terms; the rankings use the DEL model alone; the last stages weigh z-scores), so the column records what each stage used, not a single scale. The recipe on confirmed labels has no DEL term; its position is reported as measured, without attributing a cause.

### Dependency on the DEL data

Without the DEL selection data this workflow cannot be executed: they are read to train the DEL component, which enters the score of both rounds and the selection of round 2. The same holds for each of the other three inputs (the Active Learning labels, the file of compounds to score and the structure library); removing any one of them stops the pipeline at the first stage that reads it.

### Tools and reproducibility

* Code: https://github.com/jacandelerocervantes-cmd/DEL-ML-CANDELEROJ
* Python 3.12.0
* scikit-learn 1.9.1 (DOI 10.48550/arXiv.1201.0490)
* numpy 2.5.3 (DOI 10.1038/s41586-020-2649-2)
* pandas 3.0.6 (DOI 10.5281/zenodo.3509134)
* scipy 1.18.1 (DOI 10.1038/s41592-019-0686-2)
* pyarrow 25.0.1 (no DOI found)
* RDKit 2026.03.6 (DOI 10.5281/zenodo.22140358)
* Series algorithm, Kruger et al. (DOI 10.1021/acs.jcim.0c00204)
* Butina clustering (DOI 10.1021/ci9803381)

A single script (`pipeline/run_pipeline.py`) and the original script chain both reproduce the 100 compounds of the two rounds; the original chain reproduces the score column of the delivered files exactly.

## Final selection

The 50 compounds of round 2 were selected from the same pool with the 50 compounds of round 1 excluded, scored with `z(label component) + 0.3 * z(DEL component)`, in two parts. Thirty-two are the highest-scoring compounds, at most 2 per similarity group at Tanimoto 0.32. The remaining 18 are taken one per group among the compounds whose Tanimoto similarity to every round 1 compound is below 0.32. All 50 lie within the first 116 positions of the label ranking of the pool without round 1, and removing the DEL term changes 18 of them (13 of the 32 by score, 5 of the 18 forced). The delivered file is `Team_candeleroj_AL2_newgroups_20261005.csv`.

## Scope of the evidence

The 39 and 32 positives are about 9% and 7% of the 431 confirmed actives of the library and are not a random sample: they were found through lists whose rankings were DEL-informed. The compounds of the earlier stages and of the rounds share almost no scaffold (0.1%) and no pair of positives reaches Tanimoto 0.32; what the earlier stages provided is method-level knowledge, not information about specific compounds of the rounds. How the 431 actives and their clusters divide between them is not known, and the 39 released positives are not all those of the pool. Intervals reflect resampling of observed compounds only.
