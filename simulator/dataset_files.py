from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from .config import Config, DEMO_IDS, SOURCE_IDS
from .utils import time_grid


README_TEMPLATE = """# {dataset_name}

DLCoalSim is a minimally processed coalescent simulation dataset for demographic
history inference from haplotype data.

## Core Philosophy

The dataset stores simulated observations, not model-specific feature engineering:

- bit-packed haplotype genotypes
- variant positions
- observed recombination/mutation maps
- target log10 Ne(t), averaged within each logarithmic time bin
- structured, reproducible sample metadata

Handcrafted features such as SFS, LD, pi, and haplotype diversity are intentionally
not precomputed in the core dataset.

## Dataset Layout

- `samples/shard_*.npz`: sample shards with ragged arrays encoded by offsets
- `samples/shard_*.jsonl.gz`: full per-sample metadata aligned to each shard
- `metadata/samples.csv`: flat metadata summary for filtering and benchmark recipes
- `metadata/samples.jsonl.gz`: full metadata for all samples
- `metadata/samples.parquet`: optional summary parquet when the local environment supports it
- `config.json` and `manifest.json`: generation config and dataset manifest

Targets are log-time bin averages, not midpoint samples. This keeps short
events such as recent bottlenecks visible in labels when they overlap a time bin.

## Official Model Inputs

Allowed:

- genotype matrix / haplotypes
- variant positions
- observed recombination map
- observed mutation map
- sequence length and sample size

Not allowed as model input:

- true maps, if included
- demography metadata
- source type
- noise parameters
- tree sequence or TMRCA truth
- target Ne(t)
"""


FEATURES_PY = r'''
"""Baseline feature recipes for DLCoalSim."""
import numpy as np


def unpack_genotypes(genotype_packed, n_haplotypes):
    bits = np.unpackbits(genotype_packed, axis=1, bitorder="little")
    return bits[:, :n_haplotypes].astype(np.uint8)


def window_indices(positions_bp, seq_len, n_windows):
    w = np.floor(positions_bp.astype(float) / (seq_len / n_windows)).astype(int)
    return np.clip(w, 0, n_windows - 1)


def basic_window_stats(positions_bp, G_vh, seq_len, n_windows=512):
    out = np.zeros((n_windows, 4), dtype=np.float32)
    if len(positions_bp) == 0:
        return out
    H = G_vh.shape[1]
    wi = window_indices(positions_bp, seq_len, n_windows)
    for w in range(n_windows):
        idx = np.where(wi == w)[0]
        if len(idx) == 0:
            continue
        X = G_vh[idx]
        ac = X.sum(axis=1)
        p = ac / max(1, H)
        out[w, 0] = len(idx) / max(1.0, seq_len / n_windows)
        out[w, 1] = np.mean(2 * p * (1 - p))
        informative = (ac > 0) & (ac < H)
        out[w, 2] = np.mean(ac[informative] == 1) if informative.any() else 0.0
        out[w, 3] = np.mean(p) if len(p) else 0.0
    return out
'''


LOADER_PY = r'''
"""Tiny loader example for DLCoalSim shards."""
import gzip
import json
from pathlib import Path

import numpy as np


class DLCoalSimShard:
    def __init__(self, shard_path):
        self.shard_path = Path(shard_path)
        self.data = np.load(self.shard_path, allow_pickle=False)
        self.n = len(self.data["sample_id"])
        self.n_haplotypes = int(self.data["n_haplotypes"][0])
        self.packed_hap_bytes = int(self.data["packed_hap_bytes"][0])
        meta_path = self.shard_path.with_suffix(".jsonl.gz")
        self.metadata = []
        if meta_path.exists():
            with gzip.open(meta_path, "rt", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f]

    def __len__(self):
        return self.n

    def _slice_offsets(self, key, offsets_key, i):
        off = self.data[offsets_key]
        return self.data[key][off[i]:off[i + 1]]

    def get(self, i, unpack=True):
        v0, v1 = self.data["variant_offsets"][i], self.data["variant_offsets"][i + 1]
        pos = self.data["variant_positions_bp"][v0:v1]
        packed = self.data["genotype_packed"][v0:v1]
        missing = self.data["missing_packed"][v0:v1]
        if unpack:
            G = np.unpackbits(packed, axis=1, bitorder="little")[:, :self.n_haplotypes].astype(np.uint8)
            M = np.unpackbits(missing, axis=1, bitorder="little")[:, :self.n_haplotypes].astype(np.uint8)
        else:
            G, M = packed, missing
        seq = self.data["sequence_length"]
        seq_len = int(seq[i] if len(seq) > 1 else seq[0])
        sample = {
            "sample_id": str(self.data["sample_id"][i]),
            "sequence_length": seq_len,
            "n_haplotypes": self.n_haplotypes,
            "positions_bp": pos,
            "genotypes": G,
            "missing_mask": M,
            "observed_recombination": {
                "position_bp": self._slice_offsets("obs_rec_pos", "obs_rec_pos_offsets", i),
                "rate": self._slice_offsets("obs_rec_rate", "obs_rec_rate_offsets", i),
            },
            "observed_mutation": {
                "position_bp": self._slice_offsets("obs_mut_pos", "obs_mut_pos_offsets", i),
                "rate": self._slice_offsets("obs_mut_rate", "obs_mut_rate_offsets", i),
            },
            "target_log10_ne": self.data["target_log10_ne"][i],
        }
        if self.metadata:
            sample["metadata"] = self.metadata[i]
        return sample
'''


def write_dataset_files(cfg: Config, sample_manifest: dict) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges, mids = time_grid(cfg)
    config_dict = dataclasses.asdict(cfg)
    config_dict["time_edges"] = edges.tolist()
    config_dict["time_mids"] = mids.tolist()
    config_dict["source_ids"] = SOURCE_IDS
    config_dict["demography_ids"] = DEMO_IDS
    (out / "config.json").write_text(json.dumps(config_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "dataset_name": cfg.dataset_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": sample_manifest,
        "time_edges": edges.tolist(),
        "time_mids": mids.tolist(),
        "source_ids": SOURCE_IDS,
        "demography_ids": DEMO_IDS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(README_TEMPLATE.format(dataset_name=cfg.dataset_name), encoding="utf-8")

    scripts = out / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "features.py").write_text(FEATURES_PY, encoding="utf-8")
    (scripts / "loader.py").write_text(LOADER_PY, encoding="utf-8")
