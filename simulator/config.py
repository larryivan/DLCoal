from __future__ import annotations

from dataclasses import dataclass


SOURCE_IDS = {
    "clean_constant_map": 0,
    "random_global_mu_r": 1,
    "custom_heterogeneous": 2,
    "empirical_map_slice": 3,
    "stdpopsim_anchor": 4,
}
ID_TO_SOURCE = {v: k for k, v in SOURCE_IDS.items()}

DEMO_IDS = {
    "smooth_random_walk": 0,
    "single_bottleneck": 1,
    "recent_bottleneck": 2,
    "expansion": 3,
    "contraction": 4,
    "three_epoch": 5,
    "zigzag": 6,
    "ancient_event": 7,
    "constant": 8,
    "stdpopsim": 9,
}
ID_TO_DEMO = {v: k for k, v in DEMO_IDS.items()}

SPLITS_DEFAULT = [
    "train",
    "val_id",
    "val_recent",
    "val_ood_demo",
    "val_ood_map",
    "val_ood_noise",
    "test_id",
    "test_ood_demo",
    "test_ood_recent",
    "test_ood_map",
    "test_ood_noise",
]


@dataclass
class Config:
    dataset_name: str = "DLCoalSim-Core-v0.1"
    preset: str = "mini"
    out_dir: str = "./DLCoalSim-Out"
    seed: int = 12345
    force: bool = False

    n_train: int = 2000
    n_val: int = 256
    n_test: int = 512
    shard_size: int = 64
    seq_len: int = 1_000_000
    n_haplotypes: int = 32
    time_bins: int = 48
    min_time: float = 50.0
    max_time: float = 500_000.0

    baseline_mu: float = 1.25e-8
    baseline_rec: float = 1.0e-8

    p_clean_constant: float = 0.25
    p_random_global: float = 0.20
    p_custom_hetero: float = 0.45
    p_empirical_slice: float = 0.10
    p_stdpopsim_anchor: float = 0.00

    p_smooth_random_walk: float = 0.25
    p_single_bottleneck: float = 0.18
    p_recent_bottleneck: float = 0.17
    p_expansion: float = 0.12
    p_contraction: float = 0.08
    p_three_epoch: float = 0.10
    p_zigzag: float = 0.07
    p_ancient_event: float = 0.03

    map_segments_min: int = 128
    map_segments_max: int = 512
    observed_map_noise_train: float = 0.25
    observed_map_noise_ood: float = 0.80
    hotspot_missing_prob_train: float = 0.10
    hotspot_missing_prob_ood: float = 0.35

    recomb_map: str = ""
    mut_map: str = ""

    train_genotype_error_min: float = 0.001
    train_genotype_error_max: float = 0.004
    train_missing_rate_min: float = 0.002
    train_missing_rate_max: float = 0.020

    ood_genotype_error_min: float = 0.006
    ood_genotype_error_max: float = 0.020
    ood_missing_rate_min: float = 0.030
    ood_missing_rate_max: float = 0.150

    enable_stdpopsim: bool = False
    stdpopsim_species: str = "HomSap"
    stdpopsim_models: str = ""
    stdpopsim_contig_length: int = 1_000_000

    workers: int = 1
    compression: bool = True
    include_truth: bool = False
    max_retries: int = 3

    splits: str = ",".join(SPLITS_DEFAULT)


def apply_preset(cfg: Config) -> Config:
    if cfg.preset == "mini":
        cfg.dataset_name = "DLCoalSim-Mini"
        cfg.n_train = 512
        cfg.n_val = 96
        cfg.n_test = 128
        cfg.shard_size = 32
        cfg.seq_len = 1_000_000
        cfg.n_haplotypes = 24
        cfg.time_bins = 32
        cfg.workers = 1
        cfg.enable_stdpopsim = False
        cfg.p_stdpopsim_anchor = 0.0
    elif cfg.preset == "core_v01":
        cfg.dataset_name = "DLCoalSim-Core-v0.1"
        cfg.n_train = 20_000
        cfg.n_val = 2_000
        cfg.n_test = 2_000
        cfg.shard_size = 128
        cfg.seq_len = 10_000_000
        cfg.n_haplotypes = 64
        cfg.time_bins = 48
        cfg.workers = -1
        cfg.p_clean_constant = 0.25
        cfg.p_random_global = 0.20
        cfg.p_custom_hetero = 0.45
        cfg.p_empirical_slice = 0.10
        cfg.p_stdpopsim_anchor = 0.0
    elif cfg.preset == "full_v01":
        cfg.dataset_name = "DLCoalSim-Full-v0.1"
        cfg.n_train = 100_000
        cfg.n_val = 5_000
        cfg.n_test = 5_000
        cfg.shard_size = 128
        cfg.seq_len = 20_000_000
        cfg.n_haplotypes = 64
        cfg.time_bins = 48
        cfg.workers = -1
        cfg.p_clean_constant = 0.10
        cfg.p_random_global = 0.15
        cfg.p_custom_hetero = 0.60
        cfg.p_empirical_slice = 0.15
        cfg.p_stdpopsim_anchor = 0.0
    else:
        raise ValueError(f"Unknown preset: {cfg.preset}")
    return cfg


def split_names(cfg: Config) -> list[str]:
    return [s.strip() for s in cfg.splits.split(",") if s.strip()]
