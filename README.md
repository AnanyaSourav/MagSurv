# MagSurv

### Tetranucleotide compositional signatures of magnetotactic bacteria genomic sequences discriminate planetary analog environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Analysis Complete](https://img.shields.io/badge/Status-Analysis%20Complete-green.svg)]()
[![Astrobiology](https://img.shields.io/badge/Domain-Astrobiology-purple.svg)]()
[![Genomics](https://img.shields.io/badge/Field-Computational%20Genomics-orange.svg)]()

---

## Overview

Magnetotactic bacteria (MTB) are extremotolerant, phylogenetically ancient prokaryotes capable of biomineralising intracellular magnetic nanocrystals called magnetosomes. Their documented survival across irradiation, hypomagnetic exposure, microgravity, hypersalinity, and extremes of pH and temperature makes them among the most astrobiologically relevant microorganisms on Earth. Magnetofossil nanocrystals similar to MTB magnetosomes have been reported in the Martian meteorite ALH84001, and MTB habitats — suboxic or oxic-anoxic interfaces in aquatic sediments — are directly analogous to proposed ancient crater lake environments on Mars (Gale crater, Jezero crater).

**MagSurv** asks a new question: do the **whole-genome tetranucleotide (4-mer) composition profiles** of MTB genomes encode distinguishable signatures that correspond to the planetary analog class of their isolation environment on Earth?

To answer this, we use the newly released **GdbMTB database** (Ji et al. 2026 — the first dedicated, manually curated MTB genomic database, 365 genomes), map isolation environments to three planetary analog classes, and train machine learning classifiers with SHAP-based attribution to identify which compositional features are class-discriminative. We further rigorously test whether the signal is driven by phylogenetic co-segregation or reflects genuine environmental genomic adaptation.

---

## Planetary analog class scheme

| Class | Earth isolation environments | Planetary body | Rationale |
|---|---|---|---|
| `Mars_lake_analog` | Freshwater, brackish, bog (freshwater/brackish) | Mars — Gale & Jezero crater paleolakes | Suboxic-anoxic interface sediments; low salinity; moderate pH; analogous to early Mars lacustrine environments |
| `Europa_ocean_analog` | Marine, hypersaline, marine oxygen minimum zone | Europa — subsurface ocean | Salt-rich, redox-stratified, cold, low-light; consistent with Europa's saline subsurface ocean under ice shell |
| `Mars_acidic_analog` | Acidic bog / peatland (pH < 4) | Mars — volcanic / acidic crater floors | Low pH, high dissolved organics, nutrient-poor; consistent with proposed acidic surface chemistry of early and present Mars |

Five genomes from hydrothermal/subsurface environments (Enceladus-analog) were identified but excluded due to insufficient class size for cross-validation (n = 5).

---

## Dataset

- **Source:** GdbMTB — Genomic Database of Magnetotactic Bacteria (Ji et al. 2026)
- **Starting genomes:** 365
- **Quality filter:** CheckM2 (NN model) completeness ≥ 70%, contamination ≤ 10%, GUNC pass = True
- **Genomes passing QC:** 295 / 365 (80.8%)
- **Final analytical cohort:** 290 genomes (3 classes, after Enceladus exclusion)

| Class | n |
|---|---|
| Mars lake analog | 228 |
| Europa ocean analog | 31 |
| Mars acidic analog | 31 |

---

## Methods

### Feature extraction
Normalised 4-mer (tetranucleotide) relative frequencies computed over all contigs of each genome assembly using BioPython. Ambiguous bases excluded. Each genome represented as a 256-dimensional vector of 4-mer frequencies, standardised to zero mean and unit variance prior to modelling.

### Classification models
Two models evaluated under stratified 5-fold cross-validation (random seed 42, balanced class weighting):

- **Random Forest** — 500 trees, `class_weight='balanced'`, scikit-learn
- **MLP** — three hidden layers (256→128→64 units), batch normalisation, ReLU activations, dropout 0.3, Adam optimiser (lr = 0.001, weight decay = 1×10⁻⁴), cosine annealing LR schedule, 100 epochs, weighted cross-entropy loss, PyTorch

Primary evaluation metric: **balanced accuracy** (accounts for class imbalance).

### Phylogenetic confound analysis
Four sequential tests to determine whether classification signal reflects environmental genomic composition or taxonomic co-segregation:

1. Can k-mers predict GTDB phylum membership?
2. Within-Pseudomonadota classification — does signal survive within a single phylum?
3. k-mers alone vs k-mers + phylum one-hot encoding — does adding taxonomy improve predictions?
4. Phylum-only classifier as the taxonomy-only null baseline

### SHAP attribution
SHAP TreeExplainer applied to the Random Forest model trained on the full dataset. Mean absolute SHAP values computed per k-mer per class. Class-unique k-mers defined as those appearing in the top-50 SHAP-ranked features of one class but absent from the top-50 of both other classes.

---

## Results

### Classification performance

| Model | Balanced accuracy (5-fold CV) |
|---|---|
| Random chance baseline | 0.333 |
| Phylum-only null | 0.544 |
| Random Forest (4-mers) | 0.575 |
| **MLP (4-mers)** | **0.746** |

The MLP achieves a 20-point improvement over the taxonomy-only null, and 41 points over random chance.

### Phylogenetic confound tests

| Test | Result |
|---|---|
| k-mers → phylum prediction (balanced acc) | 0.436 |
| Within-Pseudomonadota classification (balanced acc) | 0.456 |
| k-mers only balanced acc | 0.575 |
| k-mers + phylum one-hot balanced acc | 0.575 |
| Delta (adding phylum) | **0.000** |
| Phylum-only null balanced acc | 0.544 |

Adding phylum labels to the k-mer model produces **zero improvement (Δ = 0.000)**, demonstrating that 4-mer composition fully subsumes phylogenetic information. The MLP's 20-point advantage over the phylum-only null reflects genuine environmental compositional signal.

### Genome-level GC content

| Class | Mean GC% | SD |
|---|---|---|
| Mars acidic analog | 47.16 | 6.78 |
| Europa ocean analog | 50.77 | 7.44 |
| Mars lake analog | 52.94 | 9.23 |

Mars acidic analog genomes are consistently GC-depleted relative to the other classes.

### SHAP-identified class-discriminative k-mers

| Class | Unique top-50 k-mers | Mean GC of unique k-mers |
|---|---|---|
| Europa ocean analog | 8 | 0.562 |
| Mars acidic analog | 7 | 0.429 |
| Mars lake analog | 0 | — |

Europa ocean analog genomes carry GC-enriched unique k-mers. Mars acidic analog genomes carry GC-depleted unique k-mers. Mars lake analog genomes have no exclusively unique top-50 k-mers — consistent with freshwater/brackish environments representing a compositional intermediate between the more selective saline and acidic conditions.

---

## Figures

| Figure | Description |
|---|---|
| `fig1_pca.png` | PCA of 256-dimensional 4-mer composition space, coloured by planetary analog class |
| `fig2_model_comparison.png` | Balanced accuracy comparison: random chance / phylum null / RF / MLP |
| `fig3_confusion_matrices.png` | Side-by-side confusion matrices for RF and MLP (5-fold CV) |
| `fig4_shap_kmers.png` | Top-15 SHAP-ranked k-mers per class, bar colour encodes k-mer GC content |
| `fig5_gc_violin.png` | Genome GC% violin plots per planetary analog class |
| `fig6_phylum_composition.png` | Stacked bar chart of phylum composition per class |
| `figS1_mlp_folds.png` | Per-fold MLP balanced accuracy across 5 CV folds |
| `figS2_shap_vs_gc.png` | Scatter of mean absolute SHAP value vs k-mer GC content for all 256 k-mers |

---

## Tables

| Table | Description |
|---|---|
| `table1_classification_metrics.csv` | Precision, recall, F1-score, balanced accuracy per model and class |
| `table2_phylo_confound_tests.csv` | Results of all four phylogenetic confound tests |
| `table3_unique_kmers.csv` | Class-exclusive top-50 SHAP k-mers with GC content and SHAP values |
| `tableS1_cohort_metadata.csv` | Full metadata for all 290 genomes (accession, taxonomy, environment, QC metrics) |
| `tableS2_qc_summary.csv` | CheckM2 completeness, contamination, GC%, N50 summary per class |
| `tableS3_phylum_composition.csv` | Phylum counts and proportions per planetary analog class |
| `tableS4_shap_full.csv` | Mean absolute SHAP value for all 256 k-mers × 3 classes |
| `tableS5_kmer_overlap.csv` | Top-50 SHAP k-mer overlap counts between every class pair |

---
