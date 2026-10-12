# Data and Code Repository

## Explicit Sound–Vision Correspondences Do Not Necessarily Produce Automatic Cross-Sensory Interference

This repository contains the anonymized behavioral datasets and MATLAB analysis code used for the analyses reported in the manuscript.

The study examined whether explicit sound–vision correspondences reflect automatic cross-sensory interactions. Participants completed:

1. An explicit correspondence rating task assessing preferences across 15 pairwise combinations of visual and auditory features.
2. A speeded audiovisual classification task measuring congruency effects in reaction time and accuracy.
3. A Big Five personality inventory assessing individual differences.

All analyses reported in the manuscript can be reproduced using the files provided in this repository.

---

# Repository Structure

```
.
├── README.md
├── analyze_CMC_full.m
├── CMC_rate_merged.csv
├── CMC_test_merged.csv
└── FFI_merged_dimension_scores.csv
```

---

# Analysis Code

## `analyze_CMC_full.m`

This MATLAB script contains the complete analysis pipeline.

The script integrates:

- Explicit correspondence rating analyses
- Audiovisual classification analyses
- Personality analyses
- Bayesian estimation
- Figure generation

The analysis pipeline includes:

- Multiple-comparison corrected hypothesis testing
- Equivalence testing
- Bayesian estimation using a BEST-style t model
- Markov chain Monte Carlo (MCMC) sampling
- Convergence diagnostics (R-hat and effective sample size)
- Bayes factor estimation using Savage–Dickey density ratios
- ROPE-based inference
- Big Five personality regression analyses

A fixed random seed is used to ensure reproducible Bayesian sampling.

---

# Data Files

## `CMC_rate_merged.csv`

Data from the explicit correspondence rating task.

Sample size:

```
N = 49
```

## `CMC_test_merged.csv`

Trial-level data from the speeded audiovisual classification task.

Primary audiovisual pairings:

1. Shape–timbre
2. Shape–pitch
3. Lightness–pitch
4. Lightness–timbre

Final sample size:

```
N = 47
```

## `FFI_merged_dimension_scores.csv`

Big Five personality scores:

- Neuroticism
- Extraversion
- Openness
- Agreeableness
- Conscientiousness

---

# Software Requirements

MATLAB R2022b or later

Required toolbox:

```
Statistics and Machine Learning Toolbox
```

---

# Running the Analysis

Place all files in the same directory and run:

```matlab
analyze_CMC_full
```

The script generates statistical tables and figures in the output directory.

---

# Output Files

## Rating analyses

- `T1_rating_tests.csv`
- `T2_rating_TOST.csv`

## Classification analyses

- `T3_classification_effects.csv`
- `S1_classification_by_taskmodality.csv`
- `S2_participant_defined_congruency.csv`

## Personality analyses

- `T4_personality_correlations.csv`
- `T6_personality_regression.csv`

## Bayesian analyses

- `T5_bayesian_estimates.csv`

---

# Bayesian Analysis Details

Bayesian models use a BEST-style Student-t framework.

Reported quantities include:

- Posterior mean
- 95% highest density interval (HDI)
- Probability of positive effects
- BF10
- R-hat
- Effective sample size (ESS)
- ROPE probability

---

# Generated Figures

Figures are saved in:

```
output/figures/
```

including:

```
fig1_group_ratings.png
fig2_individual.png
fig3_congruency.png
fig4_posterior_rt.png
fig5_traces.png
fig6_posterior_traits.png
```

---

# Reproducibility Notes

- Random number generation is initialized with a fixed seed.
- Data preprocessing, exclusion procedures, statistical analyses, Bayesian modeling, and figure generation are implemented within the analysis script.
- Running the MATLAB script reproduces the reported statistical tables and figures.

---

# Contact

For questions regarding the dataset or analysis code, please contact the corresponding author.
