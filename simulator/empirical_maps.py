from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class PiecewiseRateMap:
    chrom: str
    position: np.ndarray
    rate: np.ndarray
    rate_unit: str = "per_bp_per_generation"
    source_format: str = "unknown"

    @property
    def length(self) -> float:
        return float(self.position[-1]) if len(self.position) else 0.0


@dataclass
class EmpiricalMapStore:
    rec_maps: dict[str, PiecewiseRateMap] = field(default_factory=dict)
    mut_maps: dict[str, PiecewiseRateMap] = field(default_factory=dict)

    def common_chroms(self, min_len: int) -> list[str]:
        chroms = sorted(set(self.rec_maps) & set(self.mut_maps), key=_chrom_sort_key)
        return [c for c in chroms if self.rec_maps[c].length >= min_len and self.mut_maps[c].length >= min_len]

    def is_available(self, min_len: int) -> bool:
        return bool(self.common_chroms(min_len))


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    clean = chrom.lower().removeprefix("chr")
    return (int(clean), clean) if clean.isdigit() else (10_000, clean)


def _canonical_chrom(value) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return f"chr{text}"


def _read_table_maybe_gzip(path: str) -> pd.DataFrame:
    kwargs = dict(sep=r"\s+", comment="#", header=None, dtype=str)
    if path.endswith(".gz"):
        kwargs["compression"] = "gzip"
    return pd.read_csv(path, **kwargs)


ROULETTE_RAW_TO_PER_GEN = 1.015e-7
RECOMB_CM_PER_MB_TO_PER_BP = 1.0e-8


def _scale_rates(rates: np.ndarray, baseline_rate: float | None, map_kind: str) -> tuple[np.ndarray, str]:
    rates = np.nan_to_num(rates, nan=baseline_rate or 1e-8, posinf=baseline_rate or 1e-8, neginf=0.0)
    unit = "per_bp_per_generation"
    if len(rates) and np.nanmean(rates) > 1e-6:
        if map_kind == "recombination":
            rates = rates * RECOMB_CM_PER_MB_TO_PER_BP
            unit = "cM_per_Mb_scaled_to_per_bp_per_generation"
        elif map_kind == "mutation":
            rates = rates * ROULETTE_RAW_TO_PER_GEN
            unit = "roulette_raw_scaled_to_per_bp_per_generation"
    rates = np.clip(rates, 0.0, None)
    if len(rates) and np.nanmax(rates) > 1e-4:
        raise ValueError(
            f"{map_kind} map rates look too large for msprime per-bp per-generation rates; "
            "use Chromosome/Start/End/Rate with Rate already scaled to probability per bp per generation"
        )
    return rates, unit


def _piecewise_from_points(
    chrom: str,
    pos: np.ndarray,
    rate_points: np.ndarray,
    baseline_rate: float | None,
    map_kind: str,
) -> PiecewiseRateMap:
    order = np.argsort(pos)
    pos = pos[order].astype(np.float64, copy=False)
    rate_points = rate_points[order].astype(np.float64, copy=False)
    keep = np.concatenate([[True], np.diff(pos) > 0])
    pos = pos[keep]
    rate_points = rate_points[keep]
    if len(pos) == 0:
        raise ValueError(f"No numeric positions for empirical map chromosome {chrom}")
    if len(pos) > 1:
        left_step = pos[1] - pos[0]
        right_step = pos[-1] - pos[-2]
        boundaries = np.concatenate(
            [
                [max(0.0, pos[0] - left_step / 2.0)],
                (pos[:-1] + pos[1:]) / 2.0,
                [pos[-1] + max(1.0, right_step / 2.0)],
            ]
        )
    else:
        step = 1_000.0
        boundaries = np.array([max(0.0, pos[0] - step / 2.0), pos[0] + step / 2.0], dtype=np.float64)
    rates, unit = _scale_rates(rate_points, baseline_rate, map_kind)
    return PiecewiseRateMap(
        chrom=chrom,
        position=boundaries.astype(np.float64),
        rate=rates.astype(np.float64),
        rate_unit=unit,
        source_format="point_centers",
    )


def _piecewise_from_intervals(
    chrom: str,
    starts: np.ndarray,
    ends: np.ndarray,
    rates: np.ndarray,
    baseline_rate: float | None,
    map_kind: str,
) -> PiecewiseRateMap:
    order = np.argsort(starts)
    starts = starts[order].astype(np.float64, copy=False)
    ends = ends[order].astype(np.float64, copy=False)
    rates = rates[order].astype(np.float64, copy=False)
    keep = ends > starts
    starts = starts[keep]
    ends = ends[keep]
    rates = rates[keep]
    if len(starts) == 0:
        raise ValueError(f"No valid intervals for empirical map chromosome {chrom}")

    rates, unit = _scale_rates(rates, baseline_rate, map_kind)
    filler = baseline_rate if baseline_rate is not None else float(np.nanmedian(rates))
    positions = [0.0]
    out_rates: list[float] = []
    current = 0.0
    for start, end, rate in zip(starts, ends, rates):
        start = float(start)
        end = float(end)
        if end <= current:
            continue
        if start > current:
            out_rates.append(float(filler))
            positions.append(start)
            current = start
        if end > current:
            out_rates.append(float(rate))
            positions.append(end)
            current = end

    positions_arr = np.asarray(positions, dtype=np.float64)
    rates_arr = np.asarray(out_rates, dtype=np.float64)
    keep = np.concatenate([[True], np.diff(positions_arr) > 0])
    positions_arr = positions_arr[keep]
    if len(rates_arr) != len(positions_arr) - 1:
        rates_arr = rates_arr[: len(positions_arr) - 1]
    return PiecewiseRateMap(
        chrom=chrom,
        position=positions_arr,
        rate=rates_arr.astype(np.float64),
        rate_unit=unit,
        source_format="bed_intervals",
    )


def load_empirical_rate_table(
    path: str,
    baseline_rate: float | None = None,
    map_kind: str = "generic",
) -> dict[str, PiecewiseRateMap]:
    """Load BED-like or position-rate maps, preserving chromosome boundaries.

    Canonical input is a BED-like interval table with columns:
    chrom, start, end, rate. Coordinates are 0-based half-open intervals and
    rate is per bp per generation.

    Legacy point-center maps are accepted and converted to interval boundaries
    with midpoint cuts. Missing interval gaps are filled with baseline_rate so
    the returned map can be passed to msprime after slicing.
    """
    df = _read_table_maybe_gzip(path)
    maps: dict[str, PiecewiseRateMap] = {}

    if df.shape[1] >= 4:
        chrom_col = df.iloc[:, 0].map(_canonical_chrom)
        starts = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        ends = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        rates = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        parsed = pd.DataFrame({"chrom": chrom_col, "start": starts, "end": ends, "rate": rates}).dropna()
        for chrom, sub in parsed.groupby("chrom", sort=False):
            maps[str(chrom)] = _piecewise_from_intervals(
                str(chrom),
                sub["start"].to_numpy(),
                sub["end"].to_numpy(),
                sub["rate"].to_numpy(),
                baseline_rate,
                map_kind,
            )
    elif df.shape[1] == 3:
        chrom_col = df.iloc[:, 0].map(_canonical_chrom)
        pos = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        rates = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        parsed = pd.DataFrame({"chrom": chrom_col, "pos": pos, "rate": rates}).dropna()
        for chrom, sub in parsed.groupby("chrom", sort=False):
            maps[str(chrom)] = _piecewise_from_points(
                str(chrom),
                sub["pos"].to_numpy(),
                sub["rate"].to_numpy(),
                baseline_rate,
                map_kind,
            )
    elif df.shape[1] == 2:
        pos = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        rates = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        parsed = pd.DataFrame({"pos": pos, "rate": rates}).dropna()
        maps["__single__"] = _piecewise_from_points(
            "__single__",
            parsed["pos"].to_numpy(),
            parsed["rate"].to_numpy(),
            baseline_rate,
            map_kind,
        )
    else:
        raise ValueError(f"Cannot parse empirical map: {path}")

    if not maps:
        raise ValueError(f"Empirical map has no usable numeric rows: {path}")
    return maps


def load_empirical_maps(cfg: Config) -> EmpiricalMapStore:
    store = EmpiricalMapStore()
    if cfg.recomb_map and Path(cfg.recomb_map).exists():
        print(f"[maps] loading empirical recombination map: {cfg.recomb_map}")
        store.rec_maps = load_empirical_rate_table(cfg.recomb_map, cfg.baseline_rec, map_kind="recombination")
    if cfg.mut_map and Path(cfg.mut_map).exists():
        print(f"[maps] loading empirical mutation map: {cfg.mut_map}")
        store.mut_maps = load_empirical_rate_table(cfg.mut_map, cfg.baseline_mu, map_kind="mutation")
    return store


def slice_piecewise_map(
    rng: np.random.Generator,
    rate_map: PiecewiseRateMap,
    length: int,
    obs_noise_sigma: float,
    baseline_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    global_pos = rate_map.position
    global_rate = rate_map.rate
    max_start = max(1.0, global_pos[-1] - length - 1)
    start = float(rng.uniform(0, max_start))
    end = start + length
    idx_start = max(0, np.searchsorted(global_pos, start, side="right") - 1)
    idx_end = min(len(global_rate), np.searchsorted(global_pos, end, side="right"))
    pos = global_pos[idx_start : idx_end + 1].copy()
    rate = global_rate[idx_start:idx_end].copy()

    if len(rate) == 0:
        pos = np.array([0.0, float(length)], dtype=np.float64)
        rate = np.array([baseline_rate], dtype=np.float64)
    else:
        pos = pos - start
        pos[0] = 0.0
        if pos[-1] < length:
            pos = np.append(pos, float(length))
            rate = np.append(rate, rate[-1])
        else:
            pos[-1] = float(length)
        if len(pos) != len(rate) + 1:
            pos = pos[: len(rate) + 1]
            pos[-1] = float(length)

    true_rate = np.clip(rate, baseline_rate * 1e-4, baseline_rate * 500)
    obs_rate = true_rate * rng.lognormal(0.0, obs_noise_sigma, size=true_rate.shape)
    obs_rate = np.clip(obs_rate, baseline_rate * 1e-5, baseline_rate * 1000)
    meta = {
        "chrom": rate_map.chrom,
        "slice_start_bp": start,
        "slice_end_bp": end,
        "rate_unit": rate_map.rate_unit,
        "source_format": rate_map.source_format,
    }
    return pos.astype(np.float64), true_rate.astype(np.float64), pos.astype(np.float64), obs_rate.astype(np.float64), meta
