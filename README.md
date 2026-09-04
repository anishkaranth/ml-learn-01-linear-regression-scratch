# Week 1 — Linear Regression from Scratch (NumPy)

## Learning goal

Build intuition for **linear regression** by implementing it yourself with NumPy: generate synthetic data, train with **gradient descent**, visualize the fit and loss curve, and compare against a closed-form / least-squares solution.

## What you'll build

- Synthetic 1D regression data with controllable noise
- Mean squared error (MSE) loss and its gradients
- Batch gradient descent that learns weight \(w\) and bias \(b\)
- Plots of data + fitted line, and loss over epochs
- A sanity check against `numpy.linalg.lstsq` (no sklearn required for the core model)

## How to run

```bash
pip install -r requirements.txt
```

**Notebook (interactive):**

```bash
jupyter notebook linear_regression_scratch.ipynb
```

**Script (VS Code / terminal):**

```bash
python linear_regression_scratch.py
```

Both files share the same core logic. The notebook adds markdown intuition and inline plots; the `.py` script is a clean, runnable walkthrough.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `jupyter` (see `requirements.txt`)
