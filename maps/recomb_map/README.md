# Recombination Map Processing

## Overview

This directory processes sex-specific recombination maps from **Palsson et al. (2025)** into a format compatible with `msprime`.

| Item | Description |
|------|-------------|
| **Source** | Palsson et al. (2025) Nature |
| **Coverage** | Autosomes only (Chr 1-22) |
| **Output** | `palsson_2025_autosomes_hg38.bed` |
| **Format** | Tab-separated BED-like intervals: `Chromosome`, `Start`, `End`, `Rate` |

> **Note:** If using pre-processed data, skip this guide. Only follow if regenerating the map.

---

## Quick Start

```bash
# 1. Install dependencies
pip install tqdm

# 2. Download raw data from Palsson et al. (2025)
#    Place maps.pat.tsv and maps.mat.tsv in this directory

# 3. Run processing
cd script
python process.py
```

---

## Methodology

### Why Sex-Averaging?

Autosomes spend 50% of generations in males and 50% in females. The effective recombination rate is:

$$Rate_{sim} = \frac{Rate_{pat} + Rate_{mat}}{2}$$

### Why Autosomes Only?

Sex chromosomes (X, Y) have different effective population sizes ($N_e$), which would confuse demographic inference models.

### Unit Conversion

| Input | Output | Factor |
|-------|--------|--------|
| cM/Mb | probability/bp | × 10⁻⁸ |

---

## Detailed Usage

### Prerequisites

```bash
pip install tqdm
```

### Step 1: Prepare Input Data

Download the raw map files from Palsson et al. (2025) supplementary data and place them in the `raw/` directory:

```
recomb_map/
├── script/
│   └── process.py
├── raw/                ← Place input files here
│   ├── maps.pat.tsv    (Paternal map)
│   └── maps.mat.tsv    (Maternal map)
├── output/             ← Output files go here
│   └── palsson_2025_autosomes_hg38.bed
└── README.md
```

### Step 2: Run Processing

```bash
cd script
python process.py
```

**With custom paths:**

```bash
python process.py --pat ../raw/maps.pat.tsv --mat ../raw/maps.mat.tsv -o ../output/output.bed
```

**Quiet mode:**

```bash
python process.py -q
```

### Step 3: Expected Output

```
════════════════════════════════════════════════════════════
🧬 Recombination Map Processor (Palsson 2025)
════════════════════════════════════════════════════════════
  Paternal: maps.pat.tsv
  Maternal: maps.mat.tsv
  Output:   palsson_2025_autosomes_hg38.bed

────────────────────────────────────────────────────────────
Step 1: Load maps
────────────────────────────────────────────────────────────
  Paternal: ███████████████████████████████████ 100% [5.2M/5.2M]
  Maternal: ███████████████████████████████████ 100% [5.2M/5.2M]
  Paternal bins: 5,234,567
  Maternal bins: 5,234,567

────────────────────────────────────────────────────────────
Step 2: Merge and calculate sex-averaged rate
────────────────────────────────────────────────────────────
  Common bins: 5,234,567
  Merging: ███████████████████████████████████ 100% [5.2M/5.2M]

────────────────────────────────────────────────────────────
Step 3: Save output
────────────────────────────────────────────────────────────
  ✓ Saved 5,234,567 bins to palsson_2025_autosomes_hg38.bed

════════════════════════════════════════════════════════════
✅ Complete!
════════════════════════════════════════════════════════════
```

---

## Command Line Options

```
usage: process.py [-h] [--pat PAT] [--mat MAT] [-o OUTPUT] [-q]

Options:
  --pat PAT        Paternal map file (default: ../raw/maps.pat.tsv)
  --mat MAT        Maternal map file (default: ../raw/maps.mat.tsv)
  -o, --output     Output file (default: ../output/palsson_2025_autosomes_hg38.bed)
  -q, --quiet      Suppress progress output
```

---

## Output Format

**File:** `palsson_2025_autosomes_hg38.bed`

```
Chromosome	Start	End	Rate
1	0	1000000	1.2345678900e-08
1	1000000	2000000	1.3456789000e-08
...
22	50000000	51000000	9.8765432100e-09
```

| Column | Description |
|--------|-------------|
| Chromosome | Autosome number (1-22) |
| Start | Window start coordinate, 0-based inclusive |
| End | Window end coordinate, 0-based exclusive |
| Rate | Recombination probability per bp per generation |

---

## Usage with msprime

```python
import msprime

from simulator.empirical_maps import load_empirical_rate_table, slice_piecewise_map
from simulator.maps import make_rate_map
from simulator.utils import rng_from_seed

maps = load_empirical_rate_table(
    "palsson_2025_autosomes_hg38.bed",
    baseline_rate=1e-8,
    map_kind="recombination",
)

rng = rng_from_seed(123)
pos, rate, *_ = slice_piecewise_map(rng, maps["chr1"], length=1_000_000, obs_noise_sigma=0.0, baseline_rate=1e-8)
rec_map = make_rate_map(pos, rate)

ts = msprime.sim_ancestry(
    samples=10,
    recombination_rate=rec_map,
    sequence_length=1_000_000,
)
```

---

## Troubleshooting

### "No common bins found"

The chromosome naming format differs between files. Check if one uses `chr1` and the other uses `1`.

### Memory issues

The script uses streaming and should work with ~2GB RAM. If issues persist, process chromosomes separately.

### Missing columns

Ensure input files have columns: `Chr`, `pos`, `cMperMb`

---

## Reference

**Palsson, G., et al. (2025).**
Complete human recombination maps. *Nature*, 639, 700–707.
https://doi.org/10.1038/s41586-024-08450-5
