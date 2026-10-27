# Sequence–Structure Evidence Conflict for Protein Function Prediction

Research code and selected analysis artifacts for:

**Sequence–Structure Evidence Conflict as a Reliability Signal
for Protein Function Prediction**

## Research overview

This project investigates whether disagreement between independently
trained sequence and structure experts provides a useful reliability
signal for Gene Ontology molecular function prediction.

The primary evaluation uses 494 frozen GO-MF labels and 4,731
structure-eligible TEST proteins.

## Repository status

**Pre-publication release candidate — not yet independently validated
as a complete reproducibility package.**

The repository contains selected code, experiment summaries,
manifests, and figure-generation materials.

It does not automatically include large model checkpoints,
raw datasets, or frozen prediction arrays.

## Reproducibility

The original confirmatory analysis and later post-confirmatory
experiments must be distinguished.

The archived primary implementation uses natural logarithms and
probability clipping at 1e-7 for Bernoulli JS divergence.

The execution-time source revision of the original primary TEST
run has not been independently authenticated.

See `docs/RELEASE_AUDIT.md` before using or citing this repository.

## Data and models

Model weights, raw data and large prediction arrays are not
automatically distributed in this release candidate.

Availability, licensing and access instructions must be completed
before publication.

## License and citation

License and citation metadata are pending author approval.
