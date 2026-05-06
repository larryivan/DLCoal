from __future__ import annotations

from typing import Sequence

import msprime
import numpy as np

from .config import Config
from .utils import bin_average_log10_ne, loguniform, time_grid


def build_demography(breaks: Sequence[float], values: Sequence[float]) -> msprime.Demography:
    dem = msprime.Demography()
    dem.add_population(name="pop", initial_size=float(values[0]))
    for t, n in zip(breaks, values[1:]):
        dem.add_population_parameters_change(time=float(t), initial_size=float(n), population="pop")
    dem.sort_events()
    return dem


def choose_demo_type(rng: np.random.Generator, cfg: Config) -> str:
    choices = [
        "smooth_random_walk",
        "single_bottleneck",
        "recent_bottleneck",
        "expansion",
        "contraction",
        "three_epoch",
        "zigzag",
        "ancient_event",
    ]
    probs = np.array(
        [
            cfg.p_smooth_random_walk,
            cfg.p_single_bottleneck,
            cfg.p_recent_bottleneck,
            cfg.p_expansion,
            cfg.p_contraction,
            cfg.p_three_epoch,
            cfg.p_zigzag,
            cfg.p_ancient_event,
        ],
        dtype=float,
    )
    return str(rng.choice(choices, p=probs / probs.sum()))


def sample_custom_demography(rng: np.random.Generator, cfg: Config) -> tuple[msprime.Demography, np.ndarray, dict]:
    edges, mids = time_grid(cfg)
    t_max = float(edges[-1])
    demo_type = choose_demo_type(rng, cfg)

    if demo_type == "smooth_random_walk":
        logn = rng.uniform(3.6, 5.2)
        vals_log = [logn]
        step_sd = rng.uniform(0.08, 0.22)
        for _ in range(1, cfg.time_bins):
            logn = float(np.clip(logn + rng.normal(0, step_sd), 2.7, 5.7))
            vals_log.append(logn)
        arr = np.array(vals_log)
        win = int(rng.integers(3, 7))
        kernel = np.ones(win) / win
        sm = np.convolve(arr, kernel, mode="same")
        sm[: win // 2] = arr[: win // 2]
        sm[-win // 2 :] = arr[-win // 2 :]
        idx = np.linspace(0, cfg.time_bins - 1, num=min(8, cfg.time_bins), dtype=int)
        breaks = [float(mids[i]) for i in idx[1:]]
        values = [10 ** float(sm[i]) for i in idx]
    elif demo_type == "single_bottleneck":
        n0 = loguniform(rng, 5_000, 150_000)
        start = loguniform(rng, 500, t_max / 3)
        end = min(t_max * 0.95, start * rng.uniform(1.5, 20.0))
        nbot = max(200.0, n0 * loguniform(rng, 0.02, 0.5))
        nanc = n0 * loguniform(rng, 0.5, 3.0)
        breaks = [start, end]
        values = [n0, nbot, nanc]
    elif demo_type == "recent_bottleneck":
        n0 = loguniform(rng, 5_000, 200_000)
        start = loguniform(rng, 50, 5_000)
        end = min(t_max * 0.8, start + loguniform(rng, 50, 3_000))
        nbot = max(100.0, n0 * loguniform(rng, 0.005, 0.2))
        nanc = n0 * loguniform(rng, 0.5, 2.0)
        breaks = [start, max(start + 1.0, end)]
        values = [n0, nbot, nanc]
    elif demo_type == "expansion":
        ncur = loguniform(rng, 20_000, 300_000)
        nanc = ncur / loguniform(rng, 2.0, 50.0)
        breaks = [loguniform(rng, 100, 50_000)]
        values = [ncur, max(200.0, nanc)]
    elif demo_type == "contraction":
        ncur = loguniform(rng, 1_000, 50_000)
        nanc = ncur * loguniform(rng, 2.0, 30.0)
        breaks = [loguniform(rng, 500, 100_000)]
        values = [ncur, min(500_000, nanc)]
    elif demo_type == "three_epoch":
        t1 = loguniform(rng, 100, 20_000)
        t2 = loguniform(rng, max(t1 * 1.5, 1_000), t_max * 0.8)
        n0 = loguniform(rng, 3_000, 200_000)
        n1 = np.clip(n0 * loguniform(rng, 0.1, 8.0), 200, 500_000)
        n2 = np.clip(n1 * loguniform(rng, 0.1, 8.0), 200, 500_000)
        breaks = [t1, t2]
        values = [n0, float(n1), float(n2)]
    elif demo_type == "zigzag":
        k = int(rng.integers(4, 8))
        breaks = sorted(list(np.exp(rng.uniform(np.log(200), np.log(t_max * 0.8), size=k))))
        n = loguniform(rng, 3_000, 120_000)
        values = [n]
        sign = 1
        for _ in breaks:
            factor = loguniform(rng, 1.5, 7.0)
            n = n * factor if sign > 0 else n / factor
            n = float(np.clip(n, 200, 500_000))
            values.append(n)
            sign *= -1
    elif demo_type == "ancient_event":
        n0 = loguniform(rng, 3_000, 150_000)
        breaks = [loguniform(rng, 50_000, t_max * 0.9)]
        values = [n0, float(np.clip(n0 * loguniform(rng, 0.1, 10.0), 200, 500_000))]
    else:
        n0 = loguniform(rng, 2_000, 200_000)
        breaks = []
        values = [n0]
        demo_type = "constant"

    y = bin_average_log10_ne(edges, breaks, values)
    meta = {
        "scenario": demo_type,
        "demography_type": demo_type,
        "demography_breaks": [float(x) for x in breaks],
        "demography_values": [float(x) for x in values],
        "target_scale": "log10_Ne",
        "target_aggregation": "log_time_bin_average",
    }
    return build_demography(breaks, values), y, meta
