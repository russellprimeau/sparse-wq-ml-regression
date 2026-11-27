# sparse-wq-ml-regression

Minimal inputs, final methods and numerical evidence for *Predicting Sparsely Sampled Water-Quality Parameters from Environmental Records in a Drinking-Water Reservoir*.

Install Python 3.13 and the pinned dependencies, then regenerate the paper's numerical outputs:

```powershell
python -m pip install -r requirements.txt
python scripts/reproduce.py --level 3
```

Level 3 prepares raw inputs and regenerates figures, tables and summaries from supplied forecasts, without fitting models. Outputs go to `data/derived/` and `outputs/`, both ignored by Git. CUDA is not required for reporting. Fitting used PyTorch 2.7.1 with CUDA and XGBoost 3.2.0; platform differences can affect numerical results.

## Reading the generated results

After Level 3, start in `outputs/analysis/`:

| Output | Contents |
|---|---|
| `reported_scores.csv` | Primary no-profiler scores, strongest-reference comparisons and support assessments. |
| `profiler_comparison.json` | Matching-date comparisons with and without profiler inputs, including paired intervals. |
| `gp_control_comparisons.json` | Total-coliform environmental GP versus the two laboratory-history controls. |
| `block_permutation_comparisons.csv` | Original versus rearranged Total-coliform records. |
| `horizon_scores.csv` | Total-coliform scores as the input cutoff moves earlier. |
| `cases/<target>/evaluation.csv` | Dated observations, predictions and reference forecasts in physical units. |
| `figures/` and `tables/` | Numerical publication figures and LaTeX tables. |

`selected_predictors.csv` and its figure describe the predictors retained in the
final fits, including MLR's screening. `results.json` distinguishes `fitted_features`
from configuration-identifying `features`, which can include variables discarded by MLR.
The generated `clipping_sensitivity.csv` and `tables/review_clipping.tex` retain the
small zero-floor sensitivity check omitted from the manuscript appendix.

Total coliforms is the explicitly chosen detailed case study. Default permutation
follow-up uses that case; `--case target:run` permits other specified experiments.
No R-squared cutoff defines practical usefulness.

## Reproduction levels

```powershell
python scripts/reproduce.py --level 3  # report supplied forecasts
python scripts/reproduce.py --level 2  # refit saved configurations and settings
python scripts/reproduce.py --level 1  # repeat full training-only selection
```

Fourteen targets each have two predictor sets: weather/intake measurements, with or without the profiler. Both predict change from the laboratory value available at the predictor-window start. The no-profiler set is the primary screening analysis; added profiler value is assessed on matching later dates. There is no evaluation-based selection between the two sets. Internal directory keys `shared` and `off` identify the two archives, not additional treatments.

All 28 configurations come from complete final-method searches: 240 subset candidates, 250 XGBoost tuning trials, up to 250 GP epochs, six stochastic-finalist seeds. These searches can be expensive. For a smaller scope, add `--target total_coliforms_37c --run noprofiler_differential`. Reduced budgets are development checks, not the reported analysis.

Both sets use identical reference forecasts fitted from eligible laboratory history in the nominal training period. Models and references share a zero floor for nonnegative quantities; pH is unconstrained. Within each fit, transforms and optimization choices use training observations only. Gaps are not interpolated.

## Supplied evidence

- `data/raw/`: laboratory, profiler, intake and weather records.
- `results/models/`: 28 selected configurations, validation forecasts, final settings and seed forecasts.
- `results/bundle/`: primary model identities and horizon scores.
- `results/permutations/`: ten full-search Total-coliform rearrangements and their original-data comparator, including mappings and final fits.
- `results/horizons/`: seven fixed-configuration Total-coliform fits.
- `results/gp_controls/`: prior-only and prior-plus-calendar GP controls.
- `results/targeted_checks/`: fixed-configuration detection-limit sensitivity and scoring-date influence.

Figures 1, 2 and 7 are illustrations and are outside this repository. Search histories, rejected candidates and execution-management records remain in the research archive.

## Follow-up methods

```powershell
python scripts/08_horizons.py --target total_coliforms_37c
python scripts/09_block_permutations.py --case total_coliforms_37c:noprofiler_differential --repetitions 10
python scripts/10_targeted_checks.py --check all
python scripts/11_gp_history_controls.py
```

Horizon fits keep the selected settings and shift both environmental and laboratory-prior cutoffs. GP controls select their optimization duration within training but perform no subset/family search. Detection-limit checks replay saved settings. Block permutations repeat complete selection inside the specified predictor set; they disrupt seasonality and laboratory history as well as environmental alignment. These are conditional exploratory comparisons, not global selection-adjusted significance tests.

The later period informed study development; it is not prospective external validation. See [methods](docs/METHODS.md) and [data provenance and terms](data/README.md). Software is licensed under `LICENSE`.

Utility and laboratory data are supplied with permission to reproduce this study;
other reuse requires permission from their owners. See `data/README.md` for the
scope, contact and separate MET Norway terms. The software GPL licence covers code.
