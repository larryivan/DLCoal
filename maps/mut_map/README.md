# Mutation Rate Map Processing

## Overview

This directory processes **Roulette** mutation rate data from **Seplyarskiy et al. (2023)** for use with `msprime`.

| Item | Description |
|------|-------------|
| **Source** | Seplyarskiy et al. (2023) Nature Genetics |
| **Coverage** | Autosomes (Chr 1-22) |
| **Resolution** | Basepair-level → Window-averaged |
| **Output** | `roulette_2023_autosomes_hg38_{window}bp.bed` |
| **Format** | Tab-separated BED-like intervals: `Chromosome`, `Start`, `End`, `Rate` |
| **Rate unit** | Per bp per generation, directly usable by `msprime` |

### What is Roulette?

Roulette provides mutation rate estimates incorporating:
- **7-mer sequence context** effects
- **Regional mutation rate** variation
- **Epigenetic features** (methylation, histone marks)
- **Replication timing** effects

> **Note:** If using pre-processed data, skip this guide. Only follow if regenerating the map.

---

## Quick Start

```bash
# 1. Install dependencies
pip install tqdm numpy pooch

# 2. Download and process (all chromosomes, 1kb windows)
cd script
python download_and_process.py -o ../

# 3. Or process specific chromosomes with custom window size
python download_and_process.py -o ../ -c 21 22 -w 100
```

---

## Directory Structure

```
mut_map/
├── README.md
├── script/
│   └── download_and_process.py           # Processing script
├── vcf/                                   # Downloaded VCF files (~108GB total)
│   ├── 1_rate_v5.2_TFBS_correction_all.vcf.gz
│   ├── 2_rate_v5.2_TFBS_correction_all.vcf.gz
│   └── ...
└── output/                                # Output files
    ├── per_chrom/                         # Per-chromosome files
    │   ├── chr1_roulette_mu_1000bp.bed
    │   ├── chr2_roulette_mu_1000bp.bed
    │   └── ...
    └── roulette_2023_autosomes_hg38_1000bp.bed  # Merged output
```

---

## Detailed Usage

### Prerequisites

```bash
pip install tqdm numpy pooch
```

### Step 1: Download and Process

**Full processing (all chromosomes):**

```bash
cd script
python download_and_process.py -o ../
```

**Skip download (use existing VCF files):**

```bash
python download_and_process.py -o ../ --skip-download
```

**Specific chromosomes:**

```bash
python download_and_process.py -o ../ -c 21 22
```

**Custom window size:**

```bash
# 100bp windows (higher resolution, larger file)
python download_and_process.py -o ../ -w 100

# 10kb windows (lower resolution, smaller file)
python download_and_process.py -o ../ -w 10000
```

### Step 2: Expected Output

```
════════════════════════════════════════════════════════════
🧬 Roulette Mutation Rate Processor
════════════════════════════════════════════════════════════
  Output:      /path/to/mut_map
  Window:      1000 bp
  Chromosomes: [1, 2, 3, ..., 22]

────────────────────────────────────────────────────────────
Step 1: Download
────────────────────────────────────────────────────────────
  ✓ chr1: exists (8.62 GB)
  ✓ chr2: exists (9.12 GB)
  ...

────────────────────────────────────────────────────────────
Step 2: Process VCF files
────────────────────────────────────────────────────────────

  [1/22] Chromosome 1
    100%|███████████████████████████████████| 675M/675M [18:00<00:00, 624k variants/s]
    ✓ 674,850,450 variants → 226,130 windows

  [2/22] Chromosome 2
    100%|███████████████████████████████████| 712M/712M [18:12<00:00, 652k variants/s]
    ✓ 711,763,368 variants → 237,929 windows

  ...

────────────────────────────────────────────────────────────
Step 3: Merge results
────────────────────────────────────────────────────────────
  Merging 22 files...
  ✓ Written 2,881,033 records

────────────────────────────────────────────────────────────
Step 4: Statistics
────────────────────────────────────────────────────────────

  📊 Statistics (per-generation rates):
     Mean:    6.6414e-09
     Median:  6.2173e-09
     Std:     2.3811e-09
     Range:   [1.2530e-09, 1.0020e-07]
     Windows: 2,881,033
     Coverage: 2.88 Gb
     Unit:    /bp/generation

════════════════════════════════════════════════════════════
✅ Complete!
   Output: roulette_2023_autosomes_hg38_1000bp.bed
════════════════════════════════════════════════════════════
```

---

## Command Line Options

```
usage: download_and_process.py [-h] [-o OUTPUT_DIR] [--vcf-dir DIR]
                               [-w WINDOW_SIZE] [-c CHROM [CHROM ...]]
                               [--skip-download] [--raw-roulette-units]
                               [-q]

Options:
  -o, --output-dir      Output directory (default: ../)
  --vcf-dir             VCF files location (default: output-dir/vcf)
  -w, --window-size     Window size in bp (default: 1000)
  -c, --chromosomes     Specific chromosomes (default: 1-22)
  --skip-download       Use existing VCF files
  --raw-roulette-units  Write raw Roulette relative rates instead of msprime-ready per-generation rates
  -q, --quiet           Suppress progress output
```

---

## Output Format

**File:** `roulette_2023_autosomes_hg38_1000bp.bed`

```
Chromosome	Start	End	Rate
1	0	1000	6.6413581500e-09
1	1000	2000	6.2173017500e-09
1	2000	3000	8.0084718000e-09
...
22	50817000	50818000	5.4321000000e-02
```

| Column | Description |
|--------|-------------|
| Chromosome | Autosome number (1-22) |
| Start | Window start (0-based) |
| End | Window end |
| Rate | Mean mutation probability per bp per generation |

---

## Rate Scaling

Raw Roulette values are **relative**. The canonical output file is already scaled to per-generation mutation rates:

$$\mu_{per\_gen} = \mu_{raw} \times 1.015 \times 10^{-7}$$

**Example:**
- Raw rate: `0.065`
- Per-gen rate: `0.065 × 1.015e-7 ≈ 6.6e-9 /bp/generation`

Use `--raw-roulette-units` only when you explicitly need legacy raw Roulette relative rates.

---

## Usage with msprime

### Basic Usage

```python
import msprime

from simulator.empirical_maps import load_empirical_rate_table, slice_piecewise_map
from simulator.maps import make_rate_map
from simulator.utils import rng_from_seed

maps = load_empirical_rate_table(
    "roulette_2023_autosomes_hg38_1000bp.bed",
    baseline_rate=1.25e-8,
    map_kind="mutation",
    rate_unit="per_bp",
)

rng = rng_from_seed(123)
pos, rate, *_ = slice_piecewise_map(rng, maps["chr1"], length=1_000_000, obs_noise_sigma=0.0, baseline_rate=1.25e-8)
mu_map = make_rate_map(pos, rate)

ts = msprime.sim_ancestry(samples=10, sequence_length=1_000_000)
ts = msprime.sim_mutations(ts, rate=mu_map)
```

### Combined with B-score

```python
# 1. Scale Ne with B-score
demo = apply_b_score(demography, avg_b_score)

# 2. Simulate ancestry
ts = msprime.sim_ancestry(demography=demo, recombination_rate=recomb_map)

# 3. Add mutations with variable rate
mu_map = load_mutation_map(roulette_path, chrom, start, end)
ts = msprime.sim_mutations(ts, rate=mu_map)
```

---

## Relationship with Other Maps

| Map | Biological Effect | Use in Simulation |
|-----|-------------------|-------------------|
| **Recombination** | Crossover rate | `recombination_rate` parameter |
| **B-score** | Background selection → reduces Ne | Scale demography |
| **Roulette μ** | Sequence context → varies mutation rate | `rate` in `sim_mutations` |

All three affect observed diversity: $\pi \propto 4 N_e \mu$

---

## Troubleshooting

### Download fails

VCF files are large (~5-10GB each). If download fails:
1. Check internet connection
2. Try downloading manually from: http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/
3. Use `--skip-download` with existing files

### Memory issues

The script uses streaming and should work with ~2GB RAM. If issues persist:
1. Process one chromosome at a time: `-c 1`
2. Use larger window size: `-w 10000`

### Progress bar shows wrong total

The estimated total is based on file size. Actual variant count may differ slightly. This doesn't affect the output.

### Slow processing

Processing is I/O bound (reading compressed VCF). Expected speed:
- ~600k-800k variants/second
- ~15-20 minutes per large chromosome (chr1, chr2)
- ~5-10 minutes per small chromosome (chr21, chr22)

---

## Data Source

**URL:** http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/

**Files:** `{chrom}_rate_v5.2_TFBS_correction_all.vcf.gz`

**Total size:** ~108 GB (all chromosomes)

---

## Reference

**Seplyarskiy, V.B., et al. (2023).**
A mutation rate model at the basepair resolution identifies the mutagenic effect of polymerase III transcription.
*Nature Genetics*.
https://doi.org/10.1038/s41588-023-01562-0

**GitHub:** https://github.com/vseplyarskiy/Roulette
