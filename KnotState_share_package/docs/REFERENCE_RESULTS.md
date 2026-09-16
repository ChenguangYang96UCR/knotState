# Current reference results

These numbers are included only as a consistency check for collaborators. They are not hard-coded into the algorithms.

## Expressivity, order <= 6

| Dataset | Result |
|---|---:|
| SR25 | 105 / 105 pairs separated; 15 / 15 unique signatures |
| SRG(35,16,6,8) | 7,424,585 / 7,424,731 pairs separated (99.9980%) |
| BREC | 259 / 400 pairs separated |
| BREC exact 3-WL-hard subset | 52 / 130 resolved exactly by KnotState |

The interpretation is **non-nested distinguishing power**, not universal KnotState > 3-WL domination.

## BREC limitation

Current GF(2)-nullity KnotState on the tested CFI-3WL-hard family: `0 / 40`.

## Local WLKS reproduction

| Dataset | local WLKS micro-F1 |
|---|---:|
| PPI-BP | 0.6478 |
| HPO-Neuro | 0.6538 |
| HPO-Metab | 0.5745 |
| EM-User | 0.8571 |

## WLKS + KnotState fusion

| Dataset | WLKS | selected KS | lambda | fusion | gain |
|---|---:|---:|---:|---:|---:|
| PPI-BP | 0.6478 | K<=4 | 0.03 | 0.6352 | -0.0126 |
| HPO-Neuro | 0.6538 | K<=4 | 0.01 | 0.6552 | +0.0013 |
| HPO-Metab | 0.5745 | K<=4 | 0.70 | 0.5277 | -0.0468 |
| EM-User | 0.8571 | K<=4 | 0.03 | 0.8980 | +0.0408 |

## EM-User matched permutation validation

- real gain: `+0.0408`
- global shuffled null gain: `0.0058 +/- 0.0246`
- empirical p (global): `0.050`
- matched-K4 null gain: `0.0023 +/- 0.0244`
- empirical p (matched K4): `0.060`

Interpret as suggestive/borderline evidence rather than definitive statistical significance.
