# Results — Linear Regression from Scratch

## What was run

- **Dataset:** synthetic y = 2.5x − 1 + noise (n=100, seed=42)
- **Method:** batch GD (lr=0.05, epochs=200) vs `numpy.linalg.lstsq`
- **Command:** `python linear_regression_scratch.py`

## Headline metrics

**GD final MSE:** 0.6137  |  **lstsq MSE:** 0.6137  |  GD (w=2.5102, b=-1.0104)

## Plots

- `linear_regression_scratch_plots.png`

## Files

- `metrics.json` — structured metrics from the smoke run
- `JSON.shot` — run snapshot (timestamp, repo, command, metrics) as valid JSON
