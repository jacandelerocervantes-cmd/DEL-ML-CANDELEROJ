# DEL-ML workflow: 428,960 compounds to 100. Active Learning round 2 (AL2)

Active Learning ID: 9783567 (`Team_candeleroj_AL2_newgroups_20261005.csv`)

## Team

Team name: Team_candeleroj

### People
* Juan Antonio Candelero Cervantes, Independent researcher

### Institution
* Independent researcher

## Summary

This document describes one workflow. It starts from the provided DEL selection data and the 428,960-compound library and ends in 100 selected compounds, delivered in two rounds (AL1 and AL2). The workflow was built by adding one component at a time to the previous ones; each component is described with what it adds, the role the DEL data play in it, and whether it was kept, replaced or dropped in the final workflow. The two rounds are the two outputs of the same workflow and are reported side by side under *DEL-ML Model training and selection*. This document accompanies AL2, the second round. It differs from round 1 in the weight of the DEL component, in the place where that component acts (the selection, not only the score) and in how the 50 compounds are chosen.

## Dataset

Four provided files, no others, described by the role each plays: the DEL selection data (`PGK2_selection.parquet`; 7,703,070 rows, 7,487,570 unique SMILES, not deduplicated); the Active Learning labels (`active_learning_data.csv`; 6,580 compounds, 39 positives); the compounds to score (184,632; the scored pool is these minus the labeled compounds, 178,052); and a structure library of 428,960 SMILES, without labels, used only for the common-substructure criterion. No label other than the Active Learning labels was used to train or score the compounds of the pool in either round.

Earlier stages of the workflow also used two external resources: CACHE #7 co-crystal structures, for docking, and a pretrained DEL transformer (DELBERT). Neither enters the models that generate the two rounds.

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

Python 3.12.0, scikit-learn 1.9.1 (10.48550/arXiv.1201.0490), numpy 2.5.3 (10.1038/s41586-020-2649-2), pandas 3.0.6 (10.5281/zenodo.3509134), scipy 1.18.1 (10.1038/s41592-019-0686-2), pyarrow 25.0.1 (no DOI found), RDKit 2026.03.6 (10.5281/zenodo.22140358), series algorithm (Kruger et al., 10.1021/acs.jcim.0c00204), Butina clustering (10.1021/ci9803381). A single script (`pipeline/run_pipeline.py`) and the original script chain both reproduce the 100 compounds of the two rounds; the original chain reproduces the score column of the delivered files exactly.

## Scope of the evidence

The 39 and 32 positives are about 9% and 7% of the 431 confirmed actives of the library and are not a random sample: they were found through lists whose rankings were DEL-informed. The compounds of the earlier stages and of the rounds share almost no scaffold (0.1%) and no pair of positives reaches Tanimoto 0.32; what the earlier stages provided is method-level knowledge, not information about specific compounds of the rounds. How the 431 actives and their clusters divide between them is not known, and the 39 released positives are not all those of the pool. Intervals reflect resampling of observed compounds only.
