from __future__ import annotations

import dataclasses
import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .empirical_maps import EmpiricalMapStore
from .sample import init_worker, simulate_one_with_retries
from .utils import resolve_workers, safe_json_dumps


def concat_with_offsets(arrs: list[np.ndarray], dtype=None) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    total = 0
    for a in arrs:
        total += len(a)
        offsets.append(total)
    if total == 0:
        data = np.array([], dtype=dtype or np.float32)
    else:
        data = np.concatenate(arrs).astype(dtype or arrs[0].dtype, copy=False)
    return data, np.array(offsets, dtype=np.int64)


def write_shard(samples: list[dict], out_path: Path, cfg: Config) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array([s["sample_id"] for s in samples], dtype="U64")
    source_ids = np.array([s["source_id"] for s in samples], dtype=np.int16)
    demo_ids = np.array([s["demo_id"] for s in samples], dtype=np.int16)
    y = np.stack([s["target_log10_ne"] for s in samples]).astype(np.float32)

    positions, variant_offsets = concat_with_offsets([s["positions"] for s in samples], dtype=np.uint32)
    packed_bytes = int(math.ceil(cfg.n_haplotypes / 8))
    geno_list = [s["geno_packed"].reshape(-1, packed_bytes) for s in samples]
    miss_list = [s["missing_packed"].reshape(-1, packed_bytes) for s in samples]
    geno_n = sum(x.shape[0] for x in geno_list)
    miss_n = sum(x.shape[0] for x in miss_list)
    geno = np.concatenate(geno_list, axis=0).astype(np.uint8) if geno_n else np.zeros((0, packed_bytes), dtype=np.uint8)
    missing = np.concatenate(miss_list, axis=0).astype(np.uint8) if miss_n else np.zeros((0, packed_bytes), dtype=np.uint8)

    obs_rec_pos, obs_rec_pos_offsets = concat_with_offsets([s["obs_rec_pos"] for s in samples], dtype=np.float32)
    obs_rec_rate, obs_rec_rate_offsets = concat_with_offsets([s["obs_rec_rate"] for s in samples], dtype=np.float32)
    obs_mut_pos, obs_mut_pos_offsets = concat_with_offsets([s["obs_mut_pos"] for s in samples], dtype=np.float32)
    obs_mut_rate, obs_mut_rate_offsets = concat_with_offsets([s["obs_mut_rate"] for s in samples], dtype=np.float32)

    save_kwargs = dict(
        sample_id=ids,
        source_id=source_ids,
        demography_id=demo_ids,
        target_log10_ne=y,
        variant_positions_bp=positions,
        variant_offsets=variant_offsets,
        genotype_packed=geno,
        missing_packed=missing,
        packed_hap_bytes=np.array([packed_bytes], dtype=np.int16),
        n_haplotypes=np.array([cfg.n_haplotypes], dtype=np.int16),
        sequence_length=np.array([s["metadata"]["sequence_length"] for s in samples], dtype=np.int64),
        obs_rec_pos=obs_rec_pos,
        obs_rec_pos_offsets=obs_rec_pos_offsets,
        obs_rec_rate=obs_rec_rate,
        obs_rec_rate_offsets=obs_rec_rate_offsets,
        obs_mut_pos=obs_mut_pos,
        obs_mut_pos_offsets=obs_mut_pos_offsets,
        obs_mut_rate=obs_mut_rate,
        obs_mut_rate_offsets=obs_mut_rate_offsets,
    )
    if cfg.include_truth:
        true_rec_pos, true_rec_pos_offsets = concat_with_offsets([s["true_rec_pos"] for s in samples], dtype=np.float32)
        true_rec_rate, true_rec_rate_offsets = concat_with_offsets([s["true_rec_rate"] for s in samples], dtype=np.float32)
        true_mut_pos, true_mut_pos_offsets = concat_with_offsets([s["true_mut_pos"] for s in samples], dtype=np.float32)
        true_mut_rate, true_mut_rate_offsets = concat_with_offsets([s["true_mut_rate"] for s in samples], dtype=np.float32)
        save_kwargs.update(
            true_rec_pos=true_rec_pos,
            true_rec_pos_offsets=true_rec_pos_offsets,
            true_rec_rate=true_rec_rate,
            true_rec_rate_offsets=true_rec_rate_offsets,
            true_mut_pos=true_mut_pos,
            true_mut_pos_offsets=true_mut_pos_offsets,
            true_mut_rate=true_mut_rate,
            true_mut_rate_offsets=true_mut_rate_offsets,
        )
    if cfg.compression:
        np.savez_compressed(out_path, **save_kwargs)
    else:
        np.savez(out_path, **save_kwargs)

    meta_path = out_path.with_suffix(".jsonl.gz")
    with gzip.open(meta_path, "wt", encoding="utf-8") as f:
        for s in samples:
            f.write(safe_json_dumps(s["metadata"]) + "\n")


def _metadata_row(meta: dict) -> dict:
    rec_slice = meta.get("rec_slice", {})
    mut_slice = meta.get("mut_slice", {})
    return {
        "sample_id": meta["sample_id"],
        "sample_index": meta.get("sample_index", 0),
        "source_type": meta.get("source_type", ""),
        "scenario": meta.get("scenario", ""),
        "scenario_key": meta.get("scenario_key", ""),
        "demography_type": meta.get("demography_type", ""),
        "demography_mixture_version": meta.get("demography_mixture_version", ""),
        "n_epochs": meta.get("n_epochs", 0),
        "n_change_points": meta.get("n_change_points", 0),
        "has_recent_event": meta.get("has_recent_event", False),
        "has_ancient_event": meta.get("has_ancient_event", False),
        "min_Ne": meta.get("min_Ne", 0.0),
        "max_Ne": meta.get("max_Ne", 0.0),
        "Ne_ratio_max_min": meta.get("Ne_ratio_max_min", 0.0),
        "recent_min_Ne": meta.get("recent_min_Ne", 0.0),
        "ancient_mean_Ne": meta.get("ancient_mean_Ne", 0.0),
        "event_severity": meta.get("event_severity", 0.0),
        "event_duration": meta.get("event_duration", 0.0),
        "map_type": meta.get("map_type", ""),
        "map_mode": meta.get("map_mode", ""),
        "recomb_mode": meta.get("recomb_mode", ""),
        "mu_mode": meta.get("mu_mode", ""),
        "noise_profile": meta.get("noise_profile", ""),
        "n_variants": meta.get("n_variants", 0),
        "variant_density_per_mb": meta.get("variant_density_per_mb", 0.0),
        "genotype_error": meta.get("genotype_error", 0.0),
        "missing_rate": meta.get("missing_rate", 0.0),
        "phase_switch_pair_count": meta.get("phase_switch_pair_count", 0),
        "phaseable_pair_count": meta.get("phaseable_pair_count", 0),
        "mean_obs_rec_rate": meta.get("mean_obs_rec_rate", 0.0),
        "mean_obs_mut_rate": meta.get("mean_obs_mut_rate", 0.0),
        "std_obs_rec_rate": meta.get("std_obs_rec_rate", 0.0),
        "std_obs_mut_rate": meta.get("std_obs_mut_rate", 0.0),
        "obs_rec_rate_min": meta.get("obs_rec_rate_min", 0.0),
        "obs_rec_rate_max": meta.get("obs_rec_rate_max", 0.0),
        "obs_mut_rate_min": meta.get("obs_mut_rate_min", 0.0),
        "obs_mut_rate_max": meta.get("obs_mut_rate_max", 0.0),
        "num_trees": meta.get("num_trees", 0),
        "num_sites": meta.get("num_sites", 0),
        "sequence_length": meta.get("sequence_length", 0),
        "n_haplotypes": meta.get("n_haplotypes", 0),
        "n_diploid_individuals": meta.get("n_diploid_individuals", 0),
        "sample_ploidy": meta.get("sample_ploidy", 0),
        "seed": meta.get("seed", 0),
        "phase_switch_rate_per_mb": meta.get("noise", {}).get("phase_switch_rate_per_mb", 0.0),
        "phase_switch_count": meta.get("noise", {}).get("phase_switch_count", 0),
        "phase_pairing": meta.get("phase_pairing", ""),
        "recomb_rate_unit_before": rec_slice.get("rate_unit_before", ""),
        "recomb_rate_unit_after": rec_slice.get("rate_unit_after", ""),
        "recomb_scaling_method": rec_slice.get("scaling_method", ""),
        "mu_rate_unit_before": mut_slice.get("rate_unit_before", ""),
        "mu_rate_unit_after": mut_slice.get("rate_unit_after", ""),
        "mu_scaling_method": mut_slice.get("scaling_method", ""),
    }


def _write_metadata(metadata_rows: list[dict], full_metadata: list[dict], out_dir: Path) -> dict:
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metadata_rows)
    df.to_csv(meta_dir / "samples.csv", index=False)
    paths = {
        "summary_csv": "metadata/samples.csv",
        "full_jsonl_gz": "metadata/samples.jsonl.gz",
    }
    try:
        df.to_parquet(meta_dir / "samples.parquet", index=False)
        paths["summary_parquet"] = "metadata/samples.parquet"
    except Exception:
        pass
    with gzip.open(meta_dir / "samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for meta in full_metadata:
            f.write(safe_json_dumps(meta) + "\n")
    return paths


def generate_dataset(cfg: Config, empirical: EmpiricalMapStore) -> dict:
    n = cfg.n_samples
    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    done = samples_dir / "DONE.json"
    if done.exists() and not cfg.force:
        print(f"[skip] samples: DONE exists ({done})")
        return json.loads(done.read_text())
    if cfg.force:
        for directory in [samples_dir, out_dir / "metadata"]:
            if directory.exists():
                for f in directory.glob("*"):
                    if f.is_file():
                        f.unlink()

    print(f"[generate] samples n={n} shard_size={cfg.shard_size}")
    cfg_dict = dataclasses.asdict(cfg)
    seeds = [cfg.seed + i * 7919 for i in range(n)]
    tasks = [(i, seeds[i]) for i in range(n)]

    workers = resolve_workers(cfg.workers)
    samples_buffer: list[dict] = []
    shard_paths: list[str] = []
    metadata_rows: list[dict] = []
    full_metadata: list[dict] = []
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}

    def handle_sample(sample: dict) -> None:
        nonlocal samples_buffer
        samples_buffer.append(sample)
        meta = sample["metadata"]
        source = meta.get("source_type", "unknown")
        demo = meta.get("demography_type", "unknown")
        map_mode = meta.get("map_mode", meta.get("map_type", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        demo_counts[demo] = demo_counts.get(demo, 0) + 1
        map_counts[map_mode] = map_counts.get(map_mode, 0) + 1
        metadata_rows.append(_metadata_row(meta))
        full_metadata.append(meta)
        if len(samples_buffer) >= cfg.shard_size:
            shard_idx = len(shard_paths)
            path = samples_dir / f"shard_{shard_idx:05d}.npz"
            write_shard(samples_buffer, path, cfg)
            shard_paths.append(str(path.relative_to(cfg.out_dir)))
            samples_buffer = []

    if workers > 1:
        pending: dict[int, dict] = {}
        next_to_write = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(cfg_dict,)) as ex:
            futures = {ex.submit(simulate_one_with_retries, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=n, desc="simulate samples"):
                sample = fut.result()
                idx = int(sample["metadata"]["sample_index"])
                pending[idx] = sample
                while next_to_write in pending:
                    handle_sample(pending.pop(next_to_write))
                    next_to_write += 1
        if next_to_write != n:
            raise RuntimeError(f"generated {next_to_write} samples, expected {n}")
    else:
        init_worker(cfg_dict)
        for t in tqdm(tasks, desc="simulate samples"):
            handle_sample(simulate_one_with_retries(t))

    if samples_buffer:
        shard_idx = len(shard_paths)
        path = samples_dir / f"shard_{shard_idx:05d}.npz"
        write_shard(samples_buffer, path, cfg)
        shard_paths.append(str(path.relative_to(cfg.out_dir)))

    if len(metadata_rows) != n:
        raise RuntimeError(f"generated {len(metadata_rows)} samples, expected {n}")

    metadata_paths = _write_metadata(metadata_rows, full_metadata, out_dir)
    manifest = {
        "n_samples": len(metadata_rows),
        "n_shards": len(shard_paths),
        "shards": shard_paths,
        "metadata": metadata_paths,
        "empirical_maps": empirical.metadata,
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "map_counts": map_counts,
    }
    done.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
