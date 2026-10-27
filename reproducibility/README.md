# Primary Results Reproducibility

## Study
Sequence–Structure Evidence Conflict as a Reliability Signal for Protein Function Prediction

## Scope
This script independently recomputes primary reliability and arbitration
metrics from frozen TEST prediction arrays. It does not retrain models,
reconstruct datasets, or refit calibration or arbitration models.

## Evaluation
- Paired TEST proteins: 4,731
- GO molecular-function labels: 494
- Bootstrap: 1,000 whole-protein resamples, seed 42
- Confidence intervals: percentile 95%

## Requirements
Python, NumPy, and scikit-learn.

The current environment is not a verified historical training lockfile.

## Usage
From the repository root, run:

    python reproducibility/verify_primary_results.py --quick

For the complete bootstrap calculation:

    python reproducibility/verify_primary_results.py --bootstrap

Quick-mode output:

    reproducibility/outputs/primary_quick_report.json

Full-bootstrap output:

    reproducibility/outputs/primary_reproduction_report.json

The bootstrap calculation can take considerable time.

## Required inputs
Located in data/final_evaluation/final_test_primary_v1/:

- paired_test_labels.npy
- paired_sequence_probs.npy
- paired_structure_probs.npy
- structure_failure_target.npy
- MA_binary_predictions.npy
- MB_binary_predictions.npy
- A1_predictions.npy
- A2_predictions.npy

All arrays must have shape (4731, 494).

The script verifies SHA256 checksums for all eight required input files,
including A1 and A2 arbitration predictions.

## Expected primary results

| Metric | AP |
| --- | ---: |
| MA | 0.331684732 |
| MB | 0.443298764 |
| MB minus MA | +0.111614032 |

Primary improvement 95% CI: [+0.101829521, +0.121519001].

## Expected arbitration results

| Metric | Micro AP | Macro AP |
| --- | ---: | ---: |
| A0: sequence-only | 0.579878361 | 0.406981793 |
| A1: uncertainty gate | 0.506345929 | 0.301689928 |
| A2: conflict-aware gate | 0.541130147 | 0.347115658 |
| A2 minus A1 | +0.034784218 | +0.045425730 |

Arbitration micro improvement 95% CI:
[+0.030782278, +0.038603761].

Arbitration macro improvement 95% CI:
[+0.035623289, +0.050450715].

A2 improves A1 but does not outperform A0.

## Methods
The structure-failure target is defined as:

    abs(p_structure - y_true) >= 0.5

MA and MB are previously fitted reliability predictors.
A0 is the sequence-only expert.
A1 and A2 are previously fitted arbitration strategies.

Bootstrap resampling samples proteins with replacement while
retaining all 494 GO-MF label entries for each sampled protein.

## Verification status
All primary and arbitration point estimates were independently
reproduced from frozen predictions.

All three reported bootstrap confidence intervals were independently
verified in separate calculations.

The standalone script's full-bootstrap mode has not yet been
completed as one uninterrupted execution.

## Limitations and data availability
This is prediction-level reproduction, not end-to-end training
reproduction.

The data/ directory is excluded from Git. External researchers
will require access to the frozen prediction arrays.

Any release must respect applicable dataset licensing and
institutional requirements.

The script does not verify original dataset construction,
training, calibration fitting, or model selection.
