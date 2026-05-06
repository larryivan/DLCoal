from __future__ import annotations

import numpy as np

from .config import Config


def noise_params_for_split(rng: np.random.Generator, cfg: Config, split: str, source: str) -> dict:
    if source == "clean_constant_map" and split not in {"val_ood_noise", "test_ood_noise"}:
        return {"genotype_error": 0.0, "missing_rate": 0.0}
    if split in {"val_ood_noise", "test_ood_noise"}:
        ge = rng.uniform(cfg.ood_genotype_error_min, cfg.ood_genotype_error_max)
        mr = rng.uniform(cfg.ood_missing_rate_min, cfg.ood_missing_rate_max)
    else:
        ge = rng.uniform(cfg.train_genotype_error_min, cfg.train_genotype_error_max)
        mr = rng.uniform(cfg.train_missing_rate_min, cfg.train_missing_rate_max)
    return {"genotype_error": float(ge), "missing_rate": float(mr)}


def apply_genotype_noise(G: np.ndarray, rng: np.random.Generator, ge: float, mr: float) -> tuple[np.ndarray, np.ndarray]:
    if G.size == 0:
        return G.astype(np.uint8), np.zeros_like(G, dtype=np.uint8)
    X = (G > 0).astype(np.uint8, copy=True)
    if ge > 0:
        flip = rng.random(X.shape) < ge
        X[flip] = 1 - X[flip]
    missing = np.zeros_like(X, dtype=np.uint8)
    if mr > 0:
        miss = rng.random(X.shape) < mr
        missing[miss] = 1
        X[miss] = 0
    return X, missing


def pack_haplotypes(G: np.ndarray, n_haplotypes: int) -> np.ndarray:
    if G.shape[1] != n_haplotypes:
        raise ValueError(f"Expected H={n_haplotypes}, got {G.shape[1]}")
    return np.packbits(G.astype(np.uint8), axis=1, bitorder="little")
