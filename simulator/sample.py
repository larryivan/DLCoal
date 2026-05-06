from __future__ import annotations

import msprime
import numpy as np

from .config import Config, DEMO_IDS, SOURCE_IDS
from .demography import sample_custom_demography
from .empirical_maps import EmpiricalMapStore
from .maps import choose_source, constant_map, make_rate_map, sample_maps_for_source
from .noise import apply_genotype_noise, pack_haplotypes
from .stdpopsim_support import try_stdpopsim_sample
from .utils import rng_from_seed


def _ensure_haplotype_count(G: np.ndarray, n_haplotypes: int) -> np.ndarray:
    if G.shape[1] == n_haplotypes:
        return G
    if G.shape[1] > n_haplotypes:
        return G[:, :n_haplotypes]
    pad = np.zeros((G.shape[0], n_haplotypes - G.shape[1]), dtype=G.dtype)
    return np.concatenate([G, pad], axis=1)


def simulate_one(task: tuple[int, dict, dict, int]) -> dict:
    index, cfg_dict, empirical_dict, seed = task
    cfg = Config(**cfg_dict)
    empirical = EmpiricalMapStore(
        rec_maps=empirical_dict.get("rec_maps", {}),
        mut_maps=empirical_dict.get("mut_maps", {}),
    )
    rng = rng_from_seed(seed)
    empirical_available = empirical.is_available(cfg.seq_len)
    sample_id = f"sim_{index:08d}"

    source = choose_source(rng, cfg, empirical_available)

    ts = None
    std_meta: dict = {}
    if source == "stdpopsim_anchor":
        std_res = try_stdpopsim_sample(rng, cfg)
        if std_res is not None:
            ts, y, std_meta, _ = std_res
            source = "stdpopsim_anchor"
        else:
            source = "custom_heterogeneous"

    true_rec_pos = true_rec_rate = obs_rec_pos = obs_rec_rate = None
    true_mut_pos = true_mut_rate = obs_mut_pos = obs_mut_rate = None
    dem_meta: dict = {}
    map_meta: dict = {}

    if ts is None:
        dem, y, dem_meta = sample_custom_demography(rng, cfg)
        maps = sample_maps_for_source(rng, cfg, source, empirical)
        true_rec_pos, true_rec_rate, obs_rec_pos, obs_rec_rate, true_mut_pos, true_mut_rate, obs_mut_pos, obs_mut_rate, map_meta = maps
        rec_map = make_rate_map(true_rec_pos, true_rec_rate)
        mut_map = make_rate_map(true_mut_pos, true_mut_rate)
        ts = msprime.sim_ancestry(
            samples={"pop": cfg.n_haplotypes},
            demography=dem,
            sequence_length=cfg.seq_len,
            recombination_rate=rec_map,
            ploidy=1,
            random_seed=int(rng.integers(1, 2**31 - 2)),
        )
        try:
            mut_model = msprime.BinaryMutationModel()
            ts = msprime.sim_mutations(ts, rate=mut_map, model=mut_model, random_seed=int(rng.integers(1, 2**31 - 2)))
        except Exception:
            ts = msprime.sim_mutations(ts, rate=mut_map, random_seed=int(rng.integers(1, 2**31 - 2)))
    else:
        dem_meta = std_meta
        source = "stdpopsim_anchor"
        true_rec_pos, true_rec_rate = constant_map(int(ts.sequence_length), cfg.baseline_rec)
        true_mut_pos, true_mut_rate = constant_map(int(ts.sequence_length), cfg.baseline_mu)
        obs_rec_pos, obs_rec_rate = true_rec_pos.copy(), true_rec_rate.copy()
        obs_mut_pos, obs_mut_rate = true_mut_pos.copy(), true_mut_rate.copy()
        map_meta = {
            "map_type": "stdpopsim_or_constant_fallback",
            "map_mode": "constant_fallback",
            "recomb_mode": "constant_fallback",
            "mu_mode": "constant_fallback",
        }

    G = ts.genotype_matrix()
    if G.shape[0] > 0:
        G = _ensure_haplotype_count(G, cfg.n_haplotypes)
        positions = np.array([site.position for site in ts.sites()], dtype=np.float64)
    else:
        G = np.zeros((0, cfg.n_haplotypes), dtype=np.uint8)
        positions = np.zeros(0, dtype=np.float64)

    from .noise import noise_params_for_sample

    noise = noise_params_for_sample(rng, cfg, source)
    G_noisy, missing = apply_genotype_noise(G, rng, noise["genotype_error"], noise["missing_rate"])

    packed = pack_haplotypes(G_noisy, cfg.n_haplotypes)
    packed_missing = pack_haplotypes(missing, cfg.n_haplotypes)
    positions_u32 = np.clip(np.rint(positions), 0, np.iinfo(np.uint32).max).astype(np.uint32)

    meta = {
        "sample_id": sample_id,
        "sample_index": int(index),
        "source_type": source,
        "source_id": SOURCE_IDS.get(source, -1),
        "scenario": dem_meta.get("scenario", dem_meta.get("demography_type", "stdpopsim")),
        "noise_profile": noise.get("profile", "unknown"),
        "sequence_length": int(round(ts.sequence_length)),
        "n_haplotypes": cfg.n_haplotypes,
        "n_variants": int(G_noisy.shape[0]),
        "num_trees": int(ts.num_trees),
        "num_sites": int(ts.num_sites),
        "seed": int(seed),
        "noise": noise,
        **dem_meta,
        **map_meta,
    }

    out = {
        "sample_id": sample_id,
        "source_id": SOURCE_IDS.get(source, -1),
        "demo_id": DEMO_IDS.get(meta.get("demography_type", "stdpopsim"), 9),
        "positions": positions_u32,
        "geno_packed": packed.astype(np.uint8),
        "missing_packed": packed_missing.astype(np.uint8),
        "obs_rec_pos": np.asarray(obs_rec_pos, dtype=np.float32),
        "obs_rec_rate": np.asarray(obs_rec_rate, dtype=np.float32),
        "obs_mut_pos": np.asarray(obs_mut_pos, dtype=np.float32),
        "obs_mut_rate": np.asarray(obs_mut_rate, dtype=np.float32),
        "target_log10_ne": np.asarray(y, dtype=np.float32),
        "metadata": meta,
    }
    if cfg.include_truth:
        out.update(
            {
                "true_rec_pos": np.asarray(true_rec_pos, dtype=np.float32),
                "true_rec_rate": np.asarray(true_rec_rate, dtype=np.float32),
                "true_mut_pos": np.asarray(true_mut_pos, dtype=np.float32),
                "true_mut_rate": np.asarray(true_mut_rate, dtype=np.float32),
            }
        )
    return out


def simulate_one_with_retries(task: tuple[int, dict, dict, int]) -> dict:
    cfg = Config(**task[1])
    current = task
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            return simulate_one(current)
        except Exception as exc:
            last_error = exc
            current = (task[0], task[1], task[2], task[3] + 17 * (attempt + 1))
    raise RuntimeError(f"sample[{task[0]}] failed after {cfg.max_retries} attempts: {last_error}") from last_error
