from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

from .config import Config


def resolve_workers(workers: int) -> int:
    if workers == -1:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, workers)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) % (2**32 - 1))


def time_grid(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    edges = np.geomspace(cfg.min_time, cfg.max_time, cfg.time_bins + 1).astype(np.float64)
    mids = np.sqrt(edges[:-1] * edges[1:]).astype(np.float64)
    return edges, mids


def piecewise_eval(times: np.ndarray, breaks: Sequence[float], values: Sequence[float]) -> np.ndarray:
    out = np.full_like(times, float(values[0]), dtype=np.float64)
    for t, v in zip(breaks, values[1:]):
        out[times >= float(t)] = float(v)
    return out


def bin_average_log10_ne(
    edges: np.ndarray,
    breaks: Sequence[float],
    values: Sequence[float],
) -> np.ndarray:
    """Average piecewise-constant log10 Ne over each logarithmic time bin.

    The bins are logarithmic, so the average is taken uniformly in log time.
    Breakpoints inside a bin are included explicitly, which prevents short
    bottlenecks from disappearing just because they miss a bin midpoint.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("time edges must be positive and strictly increasing")

    pairs = sorted(
        (float(b), float(v))
        for b, v in zip(breaks, values[1:])
        if np.isfinite(b) and b > 0 and np.isfinite(v)
    )
    cleaned_breaks = [b for b, _ in pairs]
    cleaned_values = [float(values[0]), *(v for _, v in pairs)]
    out: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        cuts = [float(left)]
        cuts.extend(b for b in cleaned_breaks if left < b < right)
        cuts.append(float(right))

        total = 0.0
        weight_sum = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            probe = float(np.sqrt(a * b))
            ne = float(piecewise_eval(np.array([probe], dtype=np.float64), cleaned_breaks, cleaned_values)[0])
            weight = float(np.log(b) - np.log(a))
            total += weight * float(np.log10(max(ne, 1.0)))
            weight_sum += weight
        out.append(total / weight_sum if weight_sum else float("nan"))
    return np.asarray(out, dtype=np.float32)


def loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
