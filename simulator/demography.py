from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import msprime
import numpy as np

from .config import Config
from .ne_templates import empirical_ne_templates_available, load_empirical_ne_templates
from .utils import bin_average_log10_ne, loguniform, piecewise_eval, time_grid


SamplerResult = tuple[list[float], list[float], dict]
DemoSampler = Callable[[np.random.Generator, Config, np.ndarray, np.ndarray], SamplerResult]


DEMO_PROB_FIELDS = [
    ("constant", "p_constant"),
    ("near_constant", "p_near_constant"),
    ("smooth_random_walk", "p_smooth_random_walk"),
    ("dense_random_walk", "p_dense_random_walk"),
    ("dense_multiscale_random_walk", "p_dense_multiscale_random_walk"),
    ("dense_recent_wiggle", "p_dense_recent_wiggle"),
    ("classic_sawtooth", "p_classic_sawtooth"),
    ("empirical_human_ne_template", "p_empirical_human_ne_template"),
    ("smooth_random_walk_stress", "p_smooth_random_walk_stress"),
    ("dense_random_walk_stress", "p_dense_random_walk_stress"),
    ("single_bottleneck", "p_single_bottleneck"),
    ("recent_bottleneck", "p_recent_bottleneck"),
    ("recent_bottleneck_extreme", "p_recent_bottleneck_extreme"),
    ("recent_founder_recovery", "p_recent_founder_recovery"),
    ("founder_recovery_extreme", "p_founder_recovery_extreme"),
    ("continuous_exponential_growth", "p_continuous_exponential_growth"),
    ("rapid_recent_growth_extreme", "p_rapid_recent_growth_extreme"),
    ("continuous_exponential_decline", "p_continuous_exponential_decline"),
    ("three_epoch", "p_three_epoch"),
    ("serial_founder", "p_serial_founder"),
    ("serial_founder_extreme", "p_serial_founder_extreme"),
    ("ancient_event", "p_ancient_event"),
    ("ancient_recent_compound", "p_ancient_recent_compound"),
    ("ancient_recent_conflict", "p_ancient_recent_conflict"),
    ("oscillating_mild", "p_oscillating_mild"),
    ("zigzag_strong", "p_zigzag_strong"),
    # Legacy step-change/zigzag stress samplers. Kept at low default weight.
    ("expansion", "p_expansion"),
    ("contraction", "p_contraction"),
    ("zigzag", "p_zigzag"),
]


def build_demography(breaks: Sequence[float], values: Sequence[float]) -> msprime.Demography:
    dem = msprime.Demography()
    dem.add_population(name="pop", initial_size=float(values[0]))
    for t, n in zip(breaks, values[1:]):
        dem.add_population_parameters_change(time=float(t), initial_size=float(n), population="pop")
    dem.sort_events()
    return dem


def _clean_breaks_values(breaks: Sequence[float], values: Sequence[float], t_max: float) -> tuple[list[float], list[float]]:
    if len(values) != len(breaks) + 1:
        raise ValueError("demography values must have exactly one more item than breaks")
    pairs = sorted((float(t), float(v)) for t, v in zip(breaks, values[1:]) if 0 < float(t) < t_max)
    out_breaks: list[float] = []
    out_values = [float(values[0])]
    last = 0.0
    for t, v in pairs:
        if t <= last + 1e-6:
            continue
        out_breaks.append(t)
        out_values.append(float(np.clip(v, 50.0, 2_000_000.0)))
        last = t
    out_values[0] = float(np.clip(out_values[0], 50.0, 2_000_000.0))
    return out_breaks, out_values


def _summary(edges: np.ndarray, breaks: Sequence[float], values: Sequence[float], meta: dict) -> dict:
    t_max = float(edges[-1])
    grid = np.geomspace(float(edges[0]), t_max, 256)
    ne = piecewise_eval(grid, breaks, values)
    recent_grid = np.geomspace(float(edges[0]), min(5_000.0, t_max), 96)
    ancient_left = min(max(50_000.0, float(edges[0])), t_max)
    ancient_grid = np.geomspace(ancient_left, t_max, 96) if ancient_left < t_max else np.array([t_max])
    recent_ne = piecewise_eval(recent_grid, breaks, values)
    ancient_ne = piecewise_eval(ancient_grid, breaks, values)
    values_arr = np.asarray(values, dtype=np.float64)
    min_ne = float(np.min(values_arr))
    max_ne = float(np.max(values_arr))
    event_duration = float(meta.get("event_duration", 0.0))
    severity = float(meta.get("event_severity", min_ne / max(max_ne, 1.0)))
    has_recent = bool(meta.get("has_recent_event", any(float(t) <= 5_000.0 for t in breaks)))
    has_ancient = bool(meta.get("has_ancient_event", any(float(t) >= 50_000.0 for t in breaks)))
    return {
        "n_epochs": int(len(values)),
        "n_change_points": int(len(breaks)),
        "has_recent_event": has_recent,
        "has_ancient_event": has_ancient,
        "min_Ne": min_ne,
        "max_Ne": max_ne,
        "Ne_ratio_max_min": float(max_ne / max(min_ne, 1.0)),
        "recent_min_Ne": float(np.min(recent_ne)),
        "ancient_mean_Ne": float(np.mean(ancient_ne)),
        "event_severity": severity,
        "event_duration": event_duration,
        "time_span_min": float(edges[0]),
        "time_span_max": t_max,
    }


def _piecewise_exponential(
    present_ne: float,
    older_ne: float,
    start: float,
    cfg: Config,
    n_steps: int = 20,
) -> tuple[list[float], list[float]]:
    t0 = max(float(cfg.min_time), 1.0)
    if start <= t0 * 1.05:
        return [start], [present_ne, older_ne]
    breaks = np.geomspace(t0, start, n_steps).tolist()
    values = [present_ne]
    for t in breaks:
        frac = np.log(t / t0) / max(np.log(start / t0), 1e-12)
        log_ne = (1.0 - frac) * np.log(present_ne) + frac * np.log(older_ne)
        values.append(float(np.exp(log_ne)))
    return breaks, values


def _smoothed_array(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    kernel = np.ones(win) / win
    sm = np.convolve(arr, kernel, mode="same")
    half = win // 2
    if half:
        sm[:half] = arr[:half]
        sm[-half:] = arr[-half:]
    return sm


def _bin_aligned_values(
    log10_values: Sequence[float],
    edges: np.ndarray,
    idx: Sequence[int] | None = None,
    lo: float = 50.0,
    hi: float = 2_000_000.0,
) -> tuple[list[float], list[float]]:
    arr = np.asarray(log10_values, dtype=np.float64)
    if idx is None:
        use_idx = np.arange(arr.size, dtype=int)
    else:
        use_idx = np.unique(np.asarray(idx, dtype=int))
        use_idx = use_idx[(0 <= use_idx) & (use_idx < arr.size)]
    if use_idx.size == 0 or use_idx[0] != 0:
        use_idx = np.insert(use_idx, 0, 0)
    breaks = [float(edges[i]) for i in use_idx[1:]]
    values = [float(np.clip(10 ** float(arr[i]), lo, hi)) for i in use_idx]
    return breaks, values


def _constant(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n0 = loguniform(rng, 3_000, 200_000)
    return [], [n0], {
        "control_type": "constant",
        "event_severity": 1.0,
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _near_constant(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n0 = loguniform(rng, 5_000, 150_000)
    k = min(8, cfg.time_bins)
    idx = np.linspace(0, cfg.time_bins - 1, num=k, dtype=int)
    step_sd = rng.uniform(0.02, 0.05)
    logn = np.log(n0)
    vals = [logn]
    for _ in range(1, k):
        logn = float(logn + rng.normal(0.0, step_sd))
        vals.append(logn)
    breaks = [float(mids[i]) for i in idx[1:]]
    values = [float(np.exp(v)) for v in vals]
    return breaks, values, {
        "control_type": "near_constant",
        "near_constant_step_sd": float(step_sd),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _smooth_random_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    logn = rng.uniform(3.6, 5.2)
    vals_log = [logn]
    step_sd = rng.uniform(0.08, 0.22)
    for _ in range(1, cfg.time_bins):
        logn = float(np.clip(logn + rng.normal(0, step_sd), 2.7, 5.7))
        vals_log.append(logn)
    arr = np.array(vals_log)
    win = int(rng.integers(3, 7))
    sm = _smoothed_array(arr, win)
    knot_count = min(cfg.time_bins, max(12, int(round(cfg.time_bins * rng.uniform(0.35, 0.55)))))
    idx = np.linspace(0, cfg.time_bins - 1, num=knot_count, dtype=int)
    breaks, values = _bin_aligned_values(sm, edges, idx)
    return breaks, values, {
        "random_walk_step_sd": float(step_sd),
        "smoothing_window": int(win),
        "knot_count": int(len(values)),
        "bin_aligned_demography": True,
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _smooth_random_walk_stress(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    log_min = np.log10(100.0)
    log_max = np.log10(2_000_000.0)
    logn = rng.uniform(2.0, 6.1)
    vals_log = [logn]
    step_sd = rng.uniform(0.22, 0.55)
    jump_prob = rng.uniform(0.05, 0.12)
    for _ in range(1, cfg.time_bins):
        step = rng.normal(0.0, step_sd)
        if rng.random() < jump_prob:
            step += rng.normal(0.0, step_sd * 2.5)
        logn = float(np.clip(logn + step, log_min, log_max))
        vals_log.append(logn)
    arr = np.array(vals_log)
    win = int(rng.integers(1, 4))
    sm = _smoothed_array(arr, win)
    knot_count = min(cfg.time_bins, max(16, int(round(cfg.time_bins * rng.uniform(0.55, 0.85)))))
    idx = np.linspace(0, cfg.time_bins - 1, num=knot_count, dtype=int)
    breaks, values = _bin_aligned_values(sm, edges, idx, lo=100.0)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "smooth_random_walk_stress",
        "random_walk_step_sd": float(step_sd),
        "random_walk_jump_prob": float(jump_prob),
        "smoothing_window": int(win),
        "knot_count": int(len(values)),
        "bin_aligned_demography": True,
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _dense_random_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    logn = rng.uniform(3.5, 5.25)
    anchor = logn
    vals_log = [logn]
    step_sd = rng.uniform(0.035, 0.09)
    mean_reversion = rng.uniform(0.03, 0.08)
    for _ in range(1, cfg.time_bins):
        step = rng.normal(0.0, step_sd) - mean_reversion * (logn - anchor)
        logn = float(np.clip(logn + step, 2.7, 5.85))
        vals_log.append(logn)
    arr = _smoothed_array(np.asarray(vals_log), int(rng.integers(1, 3)))
    breaks, values = _bin_aligned_values(arr, edges)
    return breaks, values, {
        "dense_demography": True,
        "dense_type": "correlated_random_walk",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "random_walk_step_sd": float(step_sd),
        "mean_reversion": float(mean_reversion),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _dense_multiscale_random_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t = np.arange(cfg.time_bins, dtype=np.float64)
    coarse_k = int(rng.integers(5, 10))
    coarse_x = np.linspace(0, cfg.time_bins - 1, coarse_k)
    trend_sd = rng.uniform(0.12, 0.30)
    coarse = [rng.uniform(3.4, 5.25)]
    for _ in range(1, coarse_k):
        coarse.append(float(np.clip(coarse[-1] + rng.normal(0.0, trend_sd), 2.7, 5.9)))
    trend = np.interp(t, coarse_x, np.asarray(coarse))

    wiggle_sd = rng.uniform(0.015, 0.055)
    phi = rng.uniform(0.65, 0.90)
    wiggle = np.zeros(cfg.time_bins, dtype=np.float64)
    for i in range(1, cfg.time_bins):
        wiggle[i] = phi * wiggle[i - 1] + rng.normal(0.0, wiggle_sd)

    arr = np.clip(trend + wiggle, 2.7, 5.9)
    breaks, values = _bin_aligned_values(arr, edges)
    return breaks, values, {
        "dense_demography": True,
        "dense_type": "multiscale_random_walk",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "coarse_knot_count": int(coarse_k),
        "trend_step_sd": float(trend_sd),
        "wiggle_step_sd": float(wiggle_sd),
        "wiggle_phi": float(phi),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _dense_recent_wiggle(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    logn = rng.uniform(3.5, 5.25)
    vals_log = [logn]
    ancient_sd = rng.uniform(0.018, 0.055)
    recent_boost = rng.uniform(0.08, 0.18)
    recent_scale = rng.uniform(6.0, 16.0)
    for i in range(1, cfg.time_bins):
        step_sd = ancient_sd + recent_boost * np.exp(-i / recent_scale)
        logn = float(np.clip(logn + rng.normal(0.0, step_sd), 2.7, 5.85))
        vals_log.append(logn)
    breaks, values = _bin_aligned_values(vals_log, edges)
    return breaks, values, {
        "dense_demography": True,
        "dense_type": "recent_weighted_wiggle",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "ancient_step_sd": float(ancient_sd),
        "recent_step_boost": float(recent_boost),
        "recent_wiggle_scale_bins": float(recent_scale),
        "has_recent_event": True,
        "has_ancient_event": False,
    }


def _dense_random_walk_stress(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    log_min = np.log10(100.0)
    log_max = np.log10(2_000_000.0)
    logn = rng.uniform(2.0, 6.1)
    vals_log = [logn]
    step_sd = rng.uniform(0.12, 0.32)
    jump_prob = rng.uniform(0.03, 0.08)
    for _ in range(1, cfg.time_bins):
        step = rng.normal(0.0, step_sd)
        if rng.random() < jump_prob:
            step += rng.normal(0.0, step_sd * rng.uniform(2.0, 4.0))
        logn = float(np.clip(logn + step, log_min, log_max))
        vals_log.append(logn)
    breaks, values = _bin_aligned_values(vals_log, edges, lo=100.0)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "dense_random_walk_stress",
        "dense_demography": True,
        "dense_type": "stress_random_walk",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "random_walk_step_sd": float(step_sd),
        "random_walk_jump_prob": float(jump_prob),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _single_bottleneck(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 5_000, 150_000)
    start = loguniform(rng, 500, t_max / 3)
    end = min(t_max * 0.95, start * rng.uniform(1.5, 20.0))
    severity = loguniform(rng, 0.02, 0.5)
    nbot = max(200.0, n0 * severity)
    nanc = n0 * loguniform(rng, 0.5, 3.0)
    return [start, end], [n0, nbot, nanc], {
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_severity": float(severity),
    }


def _classic_sawtooth(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    x = (np.log(mids) - np.log(edges[0])) / max(np.log(edges[-1] / edges[0]), 1e-12)
    cycles = int(rng.integers(2, 5))
    phase = rng.uniform(-0.08, 0.08)
    z = (cycles * x + phase) % 1.0
    tri = 1.0 - 4.0 * np.abs(z - 0.5)

    baseline = rng.uniform(4.05, 4.65)
    amplitude = rng.uniform(0.38, 0.72)
    trend = rng.uniform(-0.16, 0.16)
    jitter_sd = rng.uniform(0.0, 0.025)
    jitter = rng.normal(0.0, jitter_sd, size=cfg.time_bins)
    arr = np.clip(baseline + amplitude * tri + trend * (x - 0.5) + jitter, 3.2, 5.65)
    breaks, values = _bin_aligned_values(arr, edges, lo=1_000.0, hi=1_000_000.0)
    return breaks, values, {
        "classic_demography_model": True,
        "dense_demography": True,
        "dense_type": "classic_sawtooth",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "sawtooth_cycles": int(cycles),
        "sawtooth_phase": float(phase),
        "sawtooth_baseline_log10": float(baseline),
        "sawtooth_amplitude_log10": float(amplitude),
        "sawtooth_trend_log10": float(trend),
        "sawtooth_jitter_sd": float(jitter_sd),
        "has_recent_event": True,
        "has_ancient_event": True,
    }


def _template_log10_bin_average(template, edges: np.ndarray, time_scale: float, n_grid: int = 8) -> np.ndarray:
    t = np.maximum(np.asarray(template.generations_ago, dtype=np.float64), 1e-6)
    log_t = np.log(t)
    log_ne = np.log10(np.asarray(template.ne, dtype=np.float64))
    out: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        grid = np.geomspace(float(left), float(right), n_grid) / max(time_scale, 1e-6)
        vals = np.interp(np.log(np.maximum(grid, 1e-6)), log_t, log_ne, left=log_ne[0], right=log_ne[-1])
        out.append(float(np.mean(vals)))
    return np.asarray(out, dtype=np.float64)


def _empirical_human_ne_template(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    store = load_empirical_ne_templates(cfg)
    template = store.choose(rng, cfg)

    time_scale = float(np.exp(rng.normal(0.0, cfg.empirical_ne_template_time_warp_sd)))
    time_scale = float(np.clip(time_scale, 0.45, 2.25))
    log_shift = float(rng.normal(0.0, cfg.empirical_ne_template_log10_shift_sd))
    contrast = float(np.exp(rng.normal(0.0, cfg.empirical_ne_template_contrast_sd)))
    contrast = float(np.clip(contrast, 0.70, 1.45))
    bin_noise_sd = float(cfg.empirical_ne_template_bin_noise_sd)

    raw = _template_log10_bin_average(template, edges, time_scale=time_scale)
    center = float(np.median(raw))
    arr = center + contrast * (raw - center) + log_shift
    if bin_noise_sd > 0:
        noise = rng.normal(0.0, bin_noise_sd, size=cfg.time_bins)
        noise = _smoothed_array(noise, int(rng.integers(2, 5)))
        arr = arr + noise
    arr = np.clip(arr, np.log10(100.0), np.log10(2_000_000.0))

    breaks, values = _bin_aligned_values(arr, edges, lo=100.0, hi=2_000_000.0)
    recent = arr[mids <= 5_000.0]
    ancient = arr[mids >= 50_000.0]
    return breaks, values, {
        "empirical_ne_template": True,
        "target_quality": "empirical_inference_template_perturbed",
        "template_source_key": template.source_key,
        "template_method": template.method,
        "template_population": template.population,
        "template_path": template.path,
        "template_time_unit": template.time_unit_note,
        "template_generation_time_years": 29.0,
        "template_curve_point_count": int(template.ne.size),
        "template_time_min": float(np.min(template.generations_ago)),
        "template_time_max": float(np.max(template.generations_ago)),
        "template_min_Ne": float(np.min(template.ne)),
        "template_max_Ne": float(np.max(template.ne)),
        "template_time_scale": time_scale,
        "template_log10_shift": log_shift,
        "template_contrast_scale": contrast,
        "template_bin_noise_sd": bin_noise_sd,
        "dense_demography": True,
        "dense_type": "empirical_human_ne_template",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "has_recent_event": bool(recent.size and np.ptp(recent) >= 0.15),
        "has_ancient_event": bool(ancient.size and np.ptp(ancient) >= 0.15),
        "event_severity": float(10 ** (np.min(arr) - np.max(arr))),
    }


def _recent_bottleneck(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 5_000, 200_000)
    start = loguniform(rng, 50, 5_000)
    end = min(t_max * 0.8, start + loguniform(rng, 50, 3_000))
    severity = loguniform(rng, 0.005, 0.2)
    nbot = max(100.0, n0 * severity)
    nanc = n0 * loguniform(rng, 0.5, 2.0)
    end = max(start + 1.0, end)
    return [start, end], [n0, nbot, nanc], {
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_severity": float(severity),
        "has_recent_event": True,
    }


def _recent_bottleneck_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 10_000, 700_000)
    start = loguniform(rng, 50, 3_000)
    duration = loguniform(rng, 10, 1_500)
    end = min(t_max * 0.6, start + duration)
    severity = loguniform(rng, 0.0005, 0.02)
    nbot = max(100.0, n0 * severity)
    nanc = loguniform(rng, 1_000, 2_000_000)
    end = max(start + 1.0, end)
    return [start, end], [n0, nbot, nanc], {
        "stress_scenario": True,
        "stress_type": "recent_bottleneck_extreme",
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_severity": float(severity),
        "has_recent_event": True,
    }


def _recent_founder_recovery(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    founder_time = loguniform(rng, 100, 5_000)
    duration = loguniform(rng, 50, 2_000)
    founder_end = min(t_max * 0.8, founder_time + duration)
    severity = loguniform(rng, 0.005, 0.1)
    recovery_factor = loguniform(rng, 5.0, 100.0)
    founder_ne = loguniform(rng, 200, 5_000)
    present_ne = min(1_000_000.0, founder_ne * recovery_factor)
    ancient_ne = founder_ne / severity * loguniform(rng, 0.5, 2.0)
    return [founder_time, max(founder_time + 1.0, founder_end)], [present_ne, founder_ne, ancient_ne], {
        "event_time": float(founder_time),
        "event_duration": float(founder_end - founder_time),
        "event_severity": float(severity),
        "recovery_factor": float(recovery_factor),
        "has_recent_event": True,
    }


def _founder_recovery_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    founder_time = loguniform(rng, 50, 3_000)
    duration = loguniform(rng, 10, 1_000)
    founder_end = min(t_max * 0.6, founder_time + duration)
    founder_ne = loguniform(rng, 100, 2_000)
    recovery_factor = loguniform(rng, 50.0, 1_000.0)
    present_ne = min(2_000_000.0, founder_ne * recovery_factor)
    ancient_ne = loguniform(rng, 5_000, 2_000_000)
    return [founder_time, max(founder_time + 1.0, founder_end)], [present_ne, founder_ne, ancient_ne], {
        "stress_scenario": True,
        "stress_type": "founder_recovery_extreme",
        "event_time": float(founder_time),
        "event_duration": float(founder_end - founder_time),
        "event_severity": float(founder_ne / max(present_ne, 1.0)),
        "recovery_factor": float(recovery_factor),
        "has_recent_event": True,
    }


def _continuous_exponential_growth(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    start = loguniform(rng, 100, 30_000)
    factor = loguniform(rng, 2.0, 100.0)
    older_ne = loguniform(rng, 2_000, 80_000)
    present_ne = min(1_000_000.0, older_ne * factor)
    breaks, values = _piecewise_exponential(present_ne, older_ne, start, cfg)
    mode = "rapid_recent_growth" if start <= 5_000 and factor >= 10 else "mild_growth"
    return breaks, values, {
        "growth_start": float(start),
        "growth_factor": float(factor),
        "growth_mode": mode,
        "continuous_approximation": "piecewise_exponential",
        "has_recent_event": bool(start <= 5_000),
        "event_time": float(start),
        "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
    }


def _rapid_recent_growth_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    start = loguniform(rng, 50, 8_000)
    factor = loguniform(rng, 50.0, 1_000.0)
    older_ne = loguniform(rng, 500, 20_000)
    present_ne = min(2_000_000.0, older_ne * factor)
    breaks, values = _piecewise_exponential(present_ne, older_ne, start, cfg, n_steps=28)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "rapid_recent_growth_extreme",
        "growth_start": float(start),
        "growth_factor": float(factor),
        "growth_mode": "extreme_rapid_recent_growth",
        "continuous_approximation": "piecewise_exponential",
        "has_recent_event": True,
        "event_time": float(start),
        "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
    }


def _continuous_exponential_decline(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    start = loguniform(rng, 100, 30_000)
    factor = loguniform(rng, 2.0, 50.0)
    older_ne = loguniform(rng, 20_000, 300_000)
    present_ne = max(200.0, older_ne / factor)
    breaks, values = _piecewise_exponential(present_ne, older_ne, start, cfg)
    return breaks, values, {
        "decline_start": float(start),
        "decline_factor": float(factor),
        "continuous_approximation": "piecewise_exponential",
        "has_recent_event": bool(start <= 5_000),
        "event_time": float(start),
        "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
    }


def _three_epoch(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    t1 = loguniform(rng, 100, 20_000)
    t2 = loguniform(rng, max(t1 * 1.5, 1_000), t_max * 0.8)
    n0 = loguniform(rng, 3_000, 200_000)
    n1 = float(np.clip(n0 * loguniform(rng, 0.1, 8.0), 200, 500_000))
    n2 = float(np.clip(n1 * loguniform(rng, 0.1, 8.0), 200, 500_000))
    return [t1, t2], [n0, n1, n2], {"event_time": float(t1), "event_duration": float(t2 - t1)}


def _serial_founder(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n_events = int(rng.integers(2, 5))
    base_times = np.geomspace(100.0, 50_000.0, n_events + 2)[1:-1]
    times = sorted((base_times * rng.lognormal(0.0, 0.25, size=n_events)).tolist())
    n = loguniform(rng, 20_000, 300_000)
    values = [n]
    breaks: list[float] = []
    severities: list[float] = []
    durations: list[float] = []
    for idx, t in enumerate(times):
        severity = loguniform(rng, 0.02, 0.5)
        recovery = loguniform(rng, 1.5, 10.0)
        next_t = times[idx + 1] if idx + 1 < len(times) else 50_000.0
        max_duration = max(50.0, min(2_000.0, (next_t - t) * 0.35))
        duration = loguniform(rng, 50, max_duration)
        founder_ne = max(100.0, n * severity)
        recovered_ne = min(1_000_000.0, founder_ne * recovery)
        breaks.extend([float(t), float(t + duration)])
        values.extend([founder_ne, recovered_ne])
        n = recovered_ne
        severities.append(severity)
        durations.append(duration)
    return breaks, values, {
        "n_founder_events": int(n_events),
        "event_severity": float(min(severities)),
        "event_duration": float(sum(durations)),
        "has_recent_event": bool(min(times) <= 5_000),
    }


def _serial_founder_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n_events = int(rng.integers(3, 6))
    base_times = np.geomspace(50.0, 70_000.0, n_events + 2)[1:-1]
    times = sorted((base_times * rng.lognormal(0.0, 0.35, size=n_events)).tolist())
    n = loguniform(rng, 50_000, 2_000_000)
    values = [n]
    breaks: list[float] = []
    severities: list[float] = []
    durations: list[float] = []
    for idx, t in enumerate(times):
        next_t = times[idx + 1] if idx + 1 < len(times) else 80_000.0
        max_duration = max(20.0, min(1_500.0, (next_t - t) * 0.25))
        duration = loguniform(rng, 20, max_duration)
        severity = loguniform(rng, 0.001, 0.05)
        recovery = loguniform(rng, 5.0, 50.0)
        founder_ne = max(100.0, n * severity)
        recovered_ne = min(2_000_000.0, founder_ne * recovery)
        breaks.extend([float(t), float(t + duration)])
        values.extend([founder_ne, recovered_ne])
        n = recovered_ne
        severities.append(severity)
        durations.append(duration)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "serial_founder_extreme",
        "n_founder_events": int(n_events),
        "event_severity": float(min(severities)),
        "event_duration": float(sum(durations)),
        "has_recent_event": bool(min(times) <= 5_000),
    }


def _ancient_event(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 3_000, 150_000)
    t = loguniform(rng, 50_000, t_max * 0.9)
    n1 = float(np.clip(n0 * loguniform(rng, 0.1, 10.0), 200, 500_000))
    return [t], [n0, n1], {"event_time": float(t), "has_ancient_event": True}


def _ancient_recent_compound(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    recent_t = loguniform(rng, 100, 5_000)
    ancient_t = loguniform(rng, 50_000, min(300_000.0, t_max * 0.9))
    n0 = loguniform(rng, 5_000, 200_000)
    recent_factor = loguniform(rng, 0.02, 0.5) if rng.random() < 0.5 else loguniform(rng, 2.0, 20.0)
    n_recent = float(np.clip(n0 * recent_factor, 100, 1_000_000))
    n_ancient = float(np.clip(n_recent * loguniform(rng, 0.1, 10.0), 200, 1_000_000))
    return [recent_t, ancient_t], [n0, n_recent, n_ancient], {
        "recent_event_time": float(recent_t),
        "ancient_event_time": float(ancient_t),
        "event_severity": float(min(n_recent, n0) / max(n_recent, n0)),
        "has_recent_event": True,
        "has_ancient_event": True,
    }


def _ancient_recent_conflict(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    recent_t = loguniform(rng, 50, 5_000)
    ancient_t = loguniform(rng, 50_000, min(300_000.0, t_max * 0.9))
    n0 = loguniform(rng, 5_000, 700_000)
    if rng.random() < 0.5:
        recent_factor = loguniform(rng, 0.001, 0.05)
        ancient_factor = loguniform(rng, 20.0, 300.0)
        conflict_mode = "recent_collapse_ancient_expansion"
    else:
        recent_factor = loguniform(rng, 20.0, 300.0)
        ancient_factor = loguniform(rng, 0.001, 0.05)
        conflict_mode = "recent_expansion_ancient_collapse"
    n_recent = float(np.clip(n0 * recent_factor, 100, 2_000_000))
    n_ancient = float(np.clip(n_recent * ancient_factor, 100, 2_000_000))
    return [recent_t, ancient_t], [n0, n_recent, n_ancient], {
        "stress_scenario": True,
        "stress_type": "ancient_recent_conflict",
        "conflict_mode": conflict_mode,
        "recent_event_time": float(recent_t),
        "ancient_event_time": float(ancient_t),
        "event_severity": float(min(n_recent, n0) / max(n_recent, n0)),
        "has_recent_event": True,
        "has_ancient_event": True,
    }


def _oscillating(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray, strong: bool) -> SamplerResult:
    t_max = float(edges[-1])
    k = int(rng.integers(8, 15) if strong else rng.integers(10, 19))
    breaks = sorted(list(np.exp(rng.uniform(np.log(200), np.log(t_max * 0.8), size=k))))
    n = loguniform(rng, 3_000, 120_000)
    values = [n]
    sign = 1
    lo, hi = (2.0, 8.0) if strong else (1.2, 2.5)
    for _ in breaks:
        factor = loguniform(rng, lo, hi)
        n = n * factor if sign > 0 else n / factor
        n = float(np.clip(n, 200, 500_000))
        values.append(n)
        sign *= -1
    return breaks, values, {
        "oscillation_strength": "strong" if strong else "mild",
        "knot_count": int(len(values)),
        "event_severity": float(min(values) / max(values)),
    }


def _oscillating_mild(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    return _oscillating(rng, cfg, edges, mids, strong=False)


def _zigzag_strong(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    return _oscillating(rng, cfg, edges, mids, strong=True)


def _zigzag_legacy(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
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
    return breaks, values, {
        "legacy_demography_model": True,
        "oscillation_strength": "legacy_zigzag",
        "event_severity": float(min(values) / max(values)),
    }


def _expansion(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    ncur = loguniform(rng, 20_000, 300_000)
    factor = loguniform(rng, 2.0, 50.0)
    nanc = ncur / factor
    t = loguniform(rng, 100, 50_000)
    return [t], [ncur, max(200.0, nanc)], {"legacy_step_change": True, "event_severity": float(1.0 / factor)}


def _contraction(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    ncur = loguniform(rng, 1_000, 50_000)
    factor = loguniform(rng, 2.0, 30.0)
    nanc = ncur * factor
    t = loguniform(rng, 500, 100_000)
    return [t], [ncur, min(500_000, nanc)], {"legacy_step_change": True, "event_severity": float(1.0 / factor)}


DEMO_SAMPLERS: dict[str, DemoSampler] = {
    "constant": _constant,
    "near_constant": _near_constant,
    "smooth_random_walk": _smooth_random_walk,
    "dense_random_walk": _dense_random_walk,
    "dense_multiscale_random_walk": _dense_multiscale_random_walk,
    "dense_recent_wiggle": _dense_recent_wiggle,
    "classic_sawtooth": _classic_sawtooth,
    "empirical_human_ne_template": _empirical_human_ne_template,
    "smooth_random_walk_stress": _smooth_random_walk_stress,
    "dense_random_walk_stress": _dense_random_walk_stress,
    "single_bottleneck": _single_bottleneck,
    "recent_bottleneck": _recent_bottleneck,
    "recent_bottleneck_extreme": _recent_bottleneck_extreme,
    "recent_founder_recovery": _recent_founder_recovery,
    "founder_recovery_extreme": _founder_recovery_extreme,
    "continuous_exponential_growth": _continuous_exponential_growth,
    "rapid_recent_growth_extreme": _rapid_recent_growth_extreme,
    "continuous_exponential_decline": _continuous_exponential_decline,
    "three_epoch": _three_epoch,
    "serial_founder": _serial_founder,
    "serial_founder_extreme": _serial_founder_extreme,
    "ancient_event": _ancient_event,
    "ancient_recent_compound": _ancient_recent_compound,
    "ancient_recent_conflict": _ancient_recent_conflict,
    "oscillating_mild": _oscillating_mild,
    "zigzag_strong": _zigzag_strong,
    "expansion": _expansion,
    "contraction": _contraction,
    "zigzag": _zigzag_legacy,
}


def choose_demo_type(rng: np.random.Generator, cfg: Config) -> str:
    choices: list[str] = []
    probs: list[float] = []
    for demo_type, field in DEMO_PROB_FIELDS:
        weight = float(getattr(cfg, field, 0.0))
        if demo_type == "empirical_human_ne_template" and weight > 0 and not empirical_ne_templates_available(cfg):
            continue
        if weight > 0:
            choices.append(demo_type)
            probs.append(weight)
    if not choices:
        return "constant"
    p = np.asarray(probs, dtype=np.float64)
    return str(rng.choice(choices, p=p / p.sum()))


def sample_custom_demography(rng: np.random.Generator, cfg: Config) -> tuple[msprime.Demography, np.ndarray, dict]:
    edges, mids = time_grid(cfg)
    t_max = float(edges[-1])
    demo_type = choose_demo_type(rng, cfg)
    sampler = DEMO_SAMPLERS.get(demo_type)
    if sampler is None:
        raise ValueError(f"Unknown demography sampler: {demo_type}")

    breaks, values, meta = sampler(rng, cfg, edges, mids)
    breaks, values = _clean_breaks_values(breaks, values, t_max)
    y = bin_average_log10_ne(edges, breaks, values)
    meta = {
        **meta,
        **_summary(edges, breaks, values, meta),
        "scenario": demo_type,
        "demography_type": demo_type,
        "demography_mixture_version": cfg.demography_mixture_version,
        "demography_breaks": [float(x) for x in breaks],
        "demography_values": [float(x) for x in values],
        "target_scale": "log10_Ne",
        "target_aggregation": "log_time_bin_average",
    }
    return build_demography(breaks, values), y, meta
