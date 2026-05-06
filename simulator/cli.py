from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .config import Config, apply_preset, split_names
from .dataset_files import write_dataset_files
from .empirical_maps import load_empirical_maps
from .shards import generate_split


def parse_args() -> Config:
    base = apply_preset(Config())
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for f in dataclasses.fields(Config):
        if f.name in {"force", "compression", "include_truth", "enable_stdpopsim"}:
            continue
        default = getattr(base, f.name)
        parser.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=type(default), default=None)

    parser.add_argument("--force", action="store_true", help="overwrite an existing split directory")
    parser.add_argument("--include-truth", action="store_true", help="store true maps for analysis; not model inputs")
    parser.add_argument("--enable-stdpopsim", action="store_true", help="include optional stdpopsim anchors")
    parser.add_argument("--no-compression", dest="compression", action="store_false", help="write uncompressed npz shards")
    parser.set_defaults(compression=None)

    ns = parser.parse_args()
    preset = ns.preset if ns.preset is not None else "mini"
    cfg = apply_preset(Config(preset=preset))
    for f in dataclasses.fields(Config):
        val = getattr(ns, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)
    if cfg.enable_stdpopsim and cfg.p_stdpopsim_anchor == 0.0:
        cfg.p_stdpopsim_anchor = 0.10
        cfg.p_custom_hetero = max(0.0, cfg.p_custom_hetero - 0.10)
    return cfg


def main() -> None:
    cfg = parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"DLCoalSim generator: {cfg.dataset_name}")
    print(f"out_dir={cfg.out_dir}")
    print(f"preset={cfg.preset} seq_len={cfg.seq_len:,} H={cfg.n_haplotypes} T={cfg.time_bins}")
    print("=" * 80)

    empirical = load_empirical_maps(cfg)
    if (cfg.recomb_map or cfg.mut_map) and not empirical.is_available(cfg.seq_len):
        print("[warn] empirical maps unavailable, incomplete, or shorter than seq_len; empirical source will be disabled")

    split_manifests: dict[str, dict] = {}
    for split in split_names(cfg):
        split_manifests[split] = generate_split(cfg, split, empirical)
        write_dataset_files(cfg, split_manifests)
    write_dataset_files(cfg, split_manifests)
    print("[done] dataset written to", cfg.out_dir)
    print("       manifest:", Path(cfg.out_dir) / "manifest.json")
    print("       loader:  ", Path(cfg.out_dir) / "scripts" / "loader.py")
