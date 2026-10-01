"""Checking and calibrating the predicted uncertainty.

A heteroscedastic model that trains happily is not necessarily calibrated. The
minimal check is the standardized residual z = (y - mu) / sigma on held-out
data: its standard deviation should be near 1, and about 95% of cases should
fall inside |z| < 1.96. If sigma is systematically off by a constant factor,
one scalar fitted on the validation split fixes it -- the same idea as
temperature scaling for classifiers.
"""


import numpy as np

__all__ = ["fit_sigma_scale", "uncertainty_report", "fit_temperature",
           "classification_report"]


def _as_array(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def fit_sigma_scale(mu, sigma, y) -> float:
    """
    Scalar s minimizing the Gaussian NLL of N(y | mu, (s*sigma)^2).

    The minimum is available in closed form: s^2 = mean(((y - mu) / sigma)^2).
    Fit on the validation split, then multiply every predicted sigma by s.
    """
    mu, sigma, y = _as_array(mu), _as_array(sigma), _as_array(y)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    return float(np.sqrt(np.mean(((y - mu) / sigma) ** 2)))


def uncertainty_report(mu, sigma, y) -> dict[str, float]:
    """
    Summary of accuracy and calibration.

    z_std        : should be ~1.0. Above 1 means sigma is too small
                   (overconfident), below 1 too large.
    coverage_95  : fraction inside |z| < 1.96; target 0.95.
    nll          : mean Gaussian NLL, the quantity the loss optimizes.
    sigma_scale  : the multiplicative correction fit_sigma_scale would apply.
    corr_abs_err : correlation between sigma and |y - mu|. A model whose
                   uncertainty is merely a constant gets ~0 here even when
                   z_std is perfect, so this is what tells you the uncertainty
                   is actually informative per case.
    """
    mu, sigma, y = _as_array(mu), _as_array(sigma), _as_array(y)
    err = y - mu
    z = err / sigma
    abs_err = np.abs(err)
    corr = (float(np.corrcoef(sigma, abs_err)[0, 1])
            if sigma.std() > 0 and abs_err.std() > 0 else float("nan"))
    return {
        "n": float(mu.size),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "z_std": float(z.std()),
        "coverage_95": float((np.abs(z) < 1.96).mean()),
        "nll": float(np.mean(0.5 * (np.log(2 * np.pi * sigma ** 2) + z ** 2))),
        "sigma_scale": fit_sigma_scale(mu, sigma, y),
        "corr_abs_err": corr,
    }


def fit_temperature(logits, labels, max_iter: int = 200) -> float:
    """
    Temperature T minimizing the cross-entropy of softmax(logits / T) on held-
    out data (Guo et al., ICML 2017). One scalar, so the argmax and therefore
    the accuracy are unchanged; only the confidence is corrected. Optimized
    over log T, which keeps T positive.
    """
    import torch

    z = torch.as_tensor(np.asarray(_logits_2d(logits)), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels).ravel(), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach())


def _logits_2d(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(len(x), -1)


def classification_report(probs, labels, n_bins: int = 15) -> dict:
    """
    accuracy, nll, brier and ece (expected calibration error of the top-class
    confidence over `n_bins` equal-width bins), plus the confusion matrix as a
    nested list with rows = truth and columns = prediction.
    """
    p = _logits_2d(probs)
    y = np.asarray(labels).ravel().astype(int)
    k = p.shape[1]
    pred = p.argmax(1)
    conf = p.max(1)
    onehot = np.eye(k)[y]
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return {
        "n": float(len(y)),
        "accuracy": float((pred == y).mean()),
        "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "ece": float(ece),
        "confusion": cm.tolist(),
    }
