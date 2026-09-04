"""
Week 1 — Linear Regression from Scratch (NumPy)

Educational script: synthetic data, gradient-descent linear regression,
plots, and a closed-form comparison via numpy.linalg.lstsq.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n: int = 100,
    true_w: float = 2.5,
    true_b: float = -1.0,
    noise_std: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = true_w * x + true_b + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    noise = rng.normal(0.0, noise_std, size=n)
    y = true_w * x + true_b + noise
    return x, y


# ---------------------------------------------------------------------------
# 2. Model, loss, gradients
# ---------------------------------------------------------------------------

def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Linear model: y_hat = w * x + b."""
    return w * x + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_pred - y_true) ** 2))


def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    """
    Gradients of MSE w.r.t. w and b.

    For L = (1/n) * sum((w*x + b - y)^2):
      dL/dw = (2/n) * sum((y_hat - y) * x)
      dL/db = (2/n) * sum(y_hat - y)
    """
    y_hat = predict(x, w, b)
    err = y_hat - y
    n = len(y)
    dw = (2.0 / n) * np.sum(err * x)
    db = (2.0 / n) * np.sum(err)
    return float(dw), float(db)


# ---------------------------------------------------------------------------
# 3. Gradient descent
# ---------------------------------------------------------------------------

def train_gd(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    w0: float = 0.0,
    b0: float = 0.0,
) -> tuple[float, float, list[float]]:
    """Batch gradient descent. Returns (w, b, loss_history)."""
    w, b = w0, b0
    history: list[float] = []
    for _ in range(epochs):
        dw, db = gradients(x, y, w, b)
        w -= lr * dw
        b -= lr * db
        history.append(mse(y, predict(x, w, b)))
    return w, b, history


# ---------------------------------------------------------------------------
# 4. Closed-form comparison (numpy lstsq)
# ---------------------------------------------------------------------------

def closed_form_lstsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Solve [x, 1] @ [w, b]^T ≈ y with least squares.
    Equivalent to the normal equations for this 1D affine model.
    """
    A = np.column_stack([x, np.ones_like(x)])
    params, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(params[0]), float(params[1])


# ---------------------------------------------------------------------------
# 5. Main: train, compare, plot
# ---------------------------------------------------------------------------

def main() -> None:
    true_w, true_b = 2.5, -1.0
    x, y = make_synthetic_data(n=100, true_w=true_w, true_b=true_b, noise_std=0.8)

    w_gd, b_gd, history = train_gd(x, y, lr=0.05, epochs=200)
    w_cf, b_cf = closed_form_lstsq(x, y)

    print("=== Linear Regression from Scratch ===")
    print(f"True parameters:     w={true_w:.4f}, b={true_b:.4f}")
    print(f"Gradient descent:    w={w_gd:.4f}, b={b_gd:.4f}, final MSE={history[-1]:.4f}")
    print(f"Closed-form lstsq:   w={w_cf:.4f}, b={b_cf:.4f}, MSE={mse(y, predict(x, w_cf, b_cf)):.4f}")

    # Plot 1: data + fits
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xs = np.linspace(x.min(), x.max(), 200)
    axes[0].scatter(x, y, alpha=0.6, label="data", edgecolors="none")
    axes[0].plot(xs, predict(xs, w_gd, b_gd), color="C1", lw=2, label="GD fit")
    axes[0].plot(xs, predict(xs, w_cf, b_cf), color="C2", lw=2, ls="--", label="lstsq fit")
    axes[0].plot(xs, predict(xs, true_w, true_b), color="C3", lw=1.5, ls=":", label="true line")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Data and fitted lines")
    axes[0].legend()

    # Plot 2: loss curve
    axes[1].plot(history, color="C0")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Training loss (gradient descent)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig("linear_regression_scratch_plots.png", dpi=120, bbox_inches="tight")
    print("Saved plot: linear_regression_scratch_plots.png")
    plt.show()


if __name__ == "__main__":
    main()
