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


def loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
