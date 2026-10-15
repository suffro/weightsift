"""Linear structure diagnostics: how much of a matrix's values could any linear predictor explain?

Each figure is an upper-bound-style estimate of a coding gain for a coder of Gaussian-like residuals, in bits per weight:
a predictor that explains a fraction f of the variance saves at most 0.5 * log2(1 / (1 - f)). They complement the
nonparametric context measurements: if no linear predictor explains a meaningful fraction, predictive coding of the
values has nothing to work with.

  adjacent      Pearson correlation of horizontally and vertically adjacent weights, of their log-magnitudes too
  low rank      energy captured by the best rank-k approximation (SVD), against the same matrix with its entries
                permuted at random (what rank k captures by chance), and the net gain once the k (m + n) factor
                values are paid at 16 bits each
  paired        the correlation of co-located weights of two matrices (gate and up of one expert)
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def gain_bits(fraction: float) -> float:
    return 0.5 * math.log2(1.0 / max(1e-12, 1.0 - fraction))


def adjacent(w: np.ndarray) -> dict:
    logmag = np.log2(np.abs(w.astype(np.float64)) + 1e-30)
    out = {}
    for name, (a, b) in {"row": (w[:, :-1], w[:, 1:]), "column": (w[:-1, :], w[1:, :])}.items():
        rho = _corr(a, b)
        out[f"{name}_corr"] = rho
        out[f"{name}_gain_bits"] = gain_bits(rho * rho)
    for name, (a, b) in {"row": (logmag[:, :-1], logmag[:, 1:]), "column": (logmag[:-1, :], logmag[1:, :])}.items():
        out[f"{name}_logmag_corr"] = _corr(a, b)
    return out


def low_rank(w: np.ndarray, ranks=(1, 4, 16, 64, 256), seed: int = 0, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(w.astype(np.float32)).to(device)
    x = x - x.mean()
    total = float((x * x).sum())
    sv = torch.linalg.svdvals(x).double().cpu().numpy()
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(x.numel(), generator=g).to(device)
    null = x.reshape(-1)[perm].reshape(x.shape)
    sv_null = torch.linalg.svdvals(null).double().cpu().numpy()
    m, n = w.shape
    out = {"shape": [m, n]}
    for k in ranks:
        if k > min(m, n):
            continue
        f = float((sv[:k] ** 2).sum() / total)
        f_null = float((sv_null[:k] ** 2).sum() / total)
        cost = k * (m + n) * 16 / (m * n)
        out[str(k)] = {"energy": f, "energy_null": f_null, "gross_bits": gain_bits(f),
                       "factor_bits": cost, "net_bits": gain_bits(f) - cost}
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    rho = _corr(a, b)
    rho_mag = _corr(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    return {"corr": rho, "gain_bits": gain_bits(rho * rho), "abs_corr": rho_mag}
