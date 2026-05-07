from __future__ import annotations

import dataclasses
import gzip
import json
import math
import time
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
    geno_n = sum(x.shape[0] for x in geno_list)
    geno = np.concatenate(geno_list, axis=0).astype(np.uint8) if geno_n else np.zeros((0, packed_bytes), dtype=np.uint8)
    missing_flat_idx, missing_offsets = concat_with_offsets([s["missing_flat_idx"] for s in samples], dtype=np.uint32)

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
        missing_flat_idx=missing_flat_idx,
        missing_offsets=missing_offsets,
        missing_storage=np.array(["sparse_flat_index_uint32"], dtype="U32"),
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


def _shard_npz_path(samples_dir: Path, shard_idx: int) -> Path:
    return samples_dir / f"shard_{shard_idx:05d}.npz"


def _tmp_npz_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")


def _atomic_write_shard(samples: list[dict], out_path: Path, cfg: Config) -> None:
    tmp_path = _tmp_npz_path(out_path)
    tmp_meta = tmp_path.with_suffix(".jsonl.gz")
    final_meta = out_path.with_suffix(".jsonl.gz")
    for path in [tmp_path, tmp_meta]:
        if path.exists():
            path.unlink()
    write_shard(samples, tmp_path, cfg)
    tmp_path.replace(out_path)
    tmp_meta.replace(final_meta)


def _seed_for_index(cfg: Config, index: int) -> int:
    return cfg.seed + int(index) * 7919


def _shard_tasks(n_samples: int, shard_size: int) -> list[tuple[int, int, int]]:
    tasks: list[tuple[int, int, int]] = []
    for shard_idx, start in enumerate(range(0, n_samples, shard_size)):
        count = min(shard_size, n_samples - start)
        tasks.append((shard_idx, start, count))
    return tasks


def _count_key(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def _shard_file_bytes(path: Path) -> int:
    meta_path = path.with_suffix(".jsonl.gz")
    total = path.stat().st_size
    if meta_path.exists():
        total += meta_path.stat().st_size
    return total


def _generate_shard(task: tuple[int, int, int, dict]) -> dict:
    shard_idx, start, count, cfg_dict = task
    cfg = Config(**cfg_dict)
    samples: list[dict] = []
    metadata_rows: list[dict] = []
    full_metadata: list[dict] = []
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}
    shard_t0 = time.perf_counter()
    sample_time_sum = 0.0
    max_sample_sec = 0.0
    slowest_meta: dict = {}

    for index in range(start, start + count):
        sample_t0 = time.perf_counter()
        sample = simulate_one_with_retries((index, _seed_for_index(cfg, index)))
        sample_sec = time.perf_counter() - sample_t0
        sample_time_sum += sample_sec
        samples.append(sample)
        meta = sample["metadata"]
        if sample_sec > max_sample_sec:
            max_sample_sec = sample_sec
            slowest_meta = meta
        source = meta.get("source_type", "unknown")
        demo = meta.get("demography_type", "unknown")
        map_mode = meta.get("map_mode", meta.get("map_type", "unknown"))
        _count_key(source_counts, source)
        _count_key(demo_counts, demo)
        _count_key(map_counts, map_mode)
        metadata_rows.append(_metadata_row(meta))
        full_metadata.append(meta)

    out_path = _shard_npz_path(Path(cfg.out_dir) / "samples", shard_idx)
    _atomic_write_shard(samples, out_path, cfg)
    shard_sec = time.perf_counter() - shard_t0
    return {
        "shard_idx": int(shard_idx),
        "start": int(start),
        "n_samples": int(count),
        "shard_path": str(Path("samples") / out_path.name),
        "shard_bytes": int(_shard_file_bytes(out_path)),
        "shard_sec": float(shard_sec),
        "mean_sample_sec": float(sample_time_sum / max(1, count)),
        "max_sample_sec": float(max_sample_sec),
        "slowest_demography_type": str(slowest_meta.get("demography_type", "unknown")),
        "slowest_source_type": str(slowest_meta.get("source_type", "unknown")),
        "slowest_n_variants": int(slowest_meta.get("n_variants", 0)),
        "slowest_max_Ne": float(slowest_meta.get("max_Ne", 0.0)),
        "metadata_rows": metadata_rows,
        "full_metadata": full_metadata,
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "map_counts": map_counts,
    }


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
        "n_missing_genotypes": meta.get("n_missing_genotypes", 0),
        "missing_storage": meta.get("missing_storage", ""),
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

    workers = resolve_workers(cfg.workers)
    shard_jobs = _shard_tasks(n, cfg.shard_size)
    print(f"[generate] samples n={n} shard_size={cfg.shard_size} shards={len(shard_jobs)} workers={workers}")
    cfg_dict = dataclasses.asdict(cfg)

    shard_results: dict[int, dict] = {}
    metadata_rows: list[dict] = []
    full_metadata: list[dict] = []
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}
    completed_samples = 0
    completed_bytes = 0

    def handle_shard_result(result: dict, progress: tqdm) -> None:
        nonlocal completed_samples, completed_bytes
        shard_results[int(result["shard_idx"])] = result
        completed_samples += int(result["n_samples"])
        completed_bytes += int(result["shard_bytes"])
        avg_bytes = completed_bytes / max(1, completed_samples)
        est_total_bytes = avg_bytes * n
        progress.update(int(result["n_samples"]))
        progress.set_postfix(
            {
                "disk_GB": f"{completed_bytes / 1e9:.1f}",
                "est_GB": f"{est_total_bytes / 1e9:.1f}",
                "shard_s": f"{float(result.get('shard_sec', 0.0)):.0f}",
                "mean_s": f"{float(result.get('mean_sample_sec', 0.0)):.1f}",
                "max_s": f"{float(result.get('max_sample_sec', 0.0)):.1f}",
                "slow": str(result.get("slowest_demography_type", ""))[:18],
                "vars": int(result.get("slowest_n_variants", 0)),
            }
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(cfg_dict,)) as ex:
            futures = {
                ex.submit(_generate_shard, (shard_idx, start, count, cfg_dict)): shard_idx
                for shard_idx, start, count in shard_jobs
            }
            with tqdm(total=n, desc="simulate samples", unit="sample") as progress:
                for fut in as_completed(futures):
                    handle_shard_result(fut.result(), progress)
    else:
        init_worker(cfg_dict)
        with tqdm(total=n, desc="simulate samples", unit="sample") as progress:
            for shard_idx, start, count in shard_jobs:
                handle_shard_result(_generate_shard((shard_idx, start, count, cfg_dict)), progress)

    shard_paths: list[str] = []
    for shard_idx in sorted(shard_results):
        result = shard_results[shard_idx]
        shard_paths.append(result["shard_path"])
        metadata_rows.extend(result["metadata_rows"])
        full_metadata.extend(result["full_metadata"])
        _merge_counts(source_counts, result["source_counts"])
        _merge_counts(demo_counts, result["demography_counts"])
        _merge_counts(map_counts, result["map_counts"])
    metadata_rows.sort(key=lambda row: int(row.get("sample_index", 0)))
    full_metadata.sort(key=lambda meta: int(meta.get("sample_index", 0)))

    if len(metadata_rows) != n:
        raise RuntimeError(f"generated {len(metadata_rows)} samples, expected {n}")

    metadata_paths = _write_metadata(metadata_rows, full_metadata, out_dir)
    avg_bytes_per_sample = completed_bytes / max(1, len(metadata_rows))
    samples_gb = completed_bytes / 1e9
    avg_mb_per_sample = avg_bytes_per_sample / 1e6
    print(f"[disk] samples={samples_gb:.2f} GB avg={avg_mb_per_sample:.2f} MB/sample")
    manifest = {
        "n_samples": len(metadata_rows),
        "n_shards": len(shard_paths),
        "shards": shard_paths,
        "metadata": metadata_paths,
        "empirical_maps": empirical.metadata,
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "map_counts": map_counts,
        "samples_bytes": int(completed_bytes),
        "samples_gb": float(samples_gb),
        "avg_bytes_per_sample": float(avg_bytes_per_sample),
        "avg_mb_per_sample": float(avg_mb_per_sample),
        "estimated_total_samples_gb": float((avg_bytes_per_sample * len(metadata_rows)) / 1e9),
    }
    done.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
