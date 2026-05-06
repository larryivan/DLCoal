#!/usr/bin/env python3
"""
Download and preprocess Roulette mutation rate data.

Optimized version using:
- pigz: Parallel gzip decompression (3-5x faster)
- polars: Fast DataFrame operations (10-20x faster than pandas)
- multiprocessing: Parallel chromosome processing

Reference:
    Seplyarskiy, V.B., et al. (2023).
    A mutation rate model at the basepair resolution identifies the
    mutagenic effect of polymerase III transcription.
    Nature Genetics. https://doi.org/10.1038/s41588-023-01562-0

Data source:
    http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/

Requirements:
    pip install pooch tqdm numpy polars pyarrow
    brew install pigz  # or: apt install pigz
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pooch
from tqdm import tqdm

# Try to import polars, fall back to slow method if not available
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("Warning: polars not installed, using slow fallback. Install with: pip install polars")

# Check if pigz is available
HAS_PIGZ = shutil.which("pigz") is not None

# =============================================================================
# Constants
# =============================================================================

ROULETTE_BASE_URL = "http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/"
VCF_PATTERN = "{chrom}_rate_v5.2_TFBS_correction_all.vcf.gz"
ROULETTE_SCALE_FACTOR = 1.015e-7
DEFAULT_WINDOW_SIZE = 1000


# =============================================================================
# Download Functions (using pooch)
# =============================================================================

def create_roulette_registry(vcf_dir, chromosomes):
    """Create a pooch registry for Roulette VCF files."""
    registry = {VCF_PATTERN.format(chrom=c): None for c in chromosomes}
    return pooch.create(
        path=vcf_dir,
        base_url=ROULETTE_BASE_URL,
        registry=registry,
        retry_if_failed=3,
        allow_updates=False,
    )


def download_single_chromosome(pup, chrom, verbose=True):
    """Download a single chromosome's VCF file."""
    filename = VCF_PATTERN.format(chrom=chrom)
    try:
        file_path = pup.fetch(filename, progressbar=verbose)
        size = os.path.getsize(file_path)
        if size < 100_000_000:
            return (chrom, None, False, f"File too small: {size} bytes")
        return (chrom, file_path, True, None)
    except Exception as e:
        return (chrom, None, False, str(e))


def download_roulette_vcfs(vcf_dir, chromosomes, max_workers=2, verbose=True):
    """Download Roulette VCF files in parallel."""
    os.makedirs(vcf_dir, exist_ok=True)
    pup = create_roulette_registry(vcf_dir, chromosomes)

    existing, to_download = {}, []
    for chrom in chromosomes:
        path = os.path.join(vcf_dir, VCF_PATTERN.format(chrom=chrom))
        if os.path.exists(path) and os.path.getsize(path) > 100_000_000:
            existing[chrom] = path
            if verbose:
                print(f"  ✓ chr{chrom}: exists ({os.path.getsize(path)/1e9:.2f} GB)")
        else:
            to_download.append(chrom)

    if not to_download:
        return existing

    if verbose:
        print(f"\n  Downloading {len(to_download)} files...")

    downloaded = dict(existing)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_chromosome, pup, c, verbose): c
                   for c in to_download}
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="  Downloading", disable=not verbose):
            chrom, path, success, error = future.result()
            if success:
                downloaded[chrom] = path
            elif verbose:
                tqdm.write(f"  ✗ chr{chrom}: {error}")

    return downloaded


# =============================================================================
# VCF Processing - Optimized with pigz + polars
# =============================================================================

def decompress_vcf_with_pigz(vcf_path, output_path=None, threads=4):
    """
    Decompress VCF using pigz (parallel gzip).
    Returns path to decompressed file.
    """
    if output_path is None:
        output_path = vcf_path.replace('.gz', '')

    if os.path.exists(output_path):
        return output_path

    # Use pigz for parallel decompression
    cmd = ['pigz', '-dc', '-p', str(threads), vcf_path]
    with open(output_path, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)

    return output_path


def process_vcf_with_polars(vcf_path, window_size=DEFAULT_WINDOW_SIZE,
                            filter_pass_only=False, exclude_filters=None,
                            threads=4, verbose=True):
    """
    Process VCF file using polars for fast DataFrame operations.

    This is ~10-20x faster than line-by-line parsing.
    """
    exclude_filters = exclude_filters or []
    file_size_gb = os.path.getsize(vcf_path) / 1e9

    # Step 1: Decompress with pigz if available
    cleanup_temp = False
    temp_path = None

    if vcf_path.endswith('.gz'):
        if HAS_PIGZ:
            temp_dir = os.path.dirname(vcf_path)
            temp_path = os.path.join(temp_dir, f".temp_{os.path.basename(vcf_path).replace('.gz', '')}")

            try:
                # Decompress with progress bar
                if verbose:
                    print(f"    [1/5] Decompressing ({file_size_gb:.2f} GB compressed)...")

                cmd = ['pigz', '-dc', '-p', str(threads), vcf_path]

                with open(temp_path, 'w') as f_out:
                    proc = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.PIPE)

                    # Show progress based on output file size growth
                    if verbose:
                        estimated_size = file_size_gb * 12 * 1e9  # ~12x compression ratio
                        with tqdm(total=estimated_size, unit='B', unit_scale=True,
                                  desc="          ", bar_format='{desc}{percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                            last_size = 0
                            while proc.poll() is None:
                                time.sleep(0.5)
                                if os.path.exists(temp_path):
                                    current_size = os.path.getsize(temp_path)
                                    pbar.update(current_size - last_size)
                                    last_size = current_size
                            # Final update
                            if os.path.exists(temp_path):
                                current_size = os.path.getsize(temp_path)
                                pbar.update(current_size - last_size)
                                pbar.total = current_size
                                pbar.refresh()
                    else:
                        proc.wait()

                    if proc.returncode != 0:
                        raise RuntimeError(f"pigz failed")

                vcf_path = temp_path
                cleanup_temp = True

            except Exception as e:
                if verbose:
                    print(f"    pigz failed: {e}, using Python gzip")
                cleanup_temp = False
        else:
            if verbose:
                print("    Warning: pigz not found, using slower Python gzip")

    # Step 2: Count header lines
    if verbose:
        print(f"    [2/5] Scanning VCF header...")

    skip_rows = 0
    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                skip_rows += 1
            else:
                break

    # Step 3: Read with polars
    if verbose:
        decompressed_size = os.path.getsize(vcf_path) / 1e9
        print(f"    [3/5] Loading data ({decompressed_size:.2f} GB)...")

    with tqdm(total=100, desc="          ", disable=not verbose,
              bar_format='{desc}{percentage:3.0f}%|{bar:30}| [{elapsed}]') as pbar:
        pbar.update(10)  # Start

        df = pl.read_csv(
            vcf_path,
            separator='\t',
            skip_rows=skip_rows,
            has_header=False,
            new_columns=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'],
            schema_overrides={
                'CHROM': pl.Utf8,
                'POS': pl.Int64,
                'ID': pl.Utf8,
                'REF': pl.Utf8,
                'ALT': pl.Utf8,
                'QUAL': pl.Utf8,
                'FILTER': pl.Utf8,
                'INFO': pl.Utf8,
            },
            n_threads=threads,
            low_memory=False,
        )
        pbar.update(90)

    if verbose:
        print(f"          Loaded {len(df):,} records")

    # Step 4: Extract MR and apply filters
    if verbose:
        print(f"    [4/5] Extracting mutation rates & filtering...")

    with tqdm(total=100, desc="          ", disable=not verbose,
              bar_format='{desc}{percentage:3.0f}%|{bar:30}| [{elapsed}]') as pbar:

        # Extract MR
        df = df.with_columns([
            pl.col('INFO').str.extract(r'MR=([0-9.]+)', 1).cast(pl.Float64).alias('MR')
        ])
        pbar.update(40)

        # Apply filters
        if filter_pass_only:
            df = df.filter(pl.col('FILTER') == 'PASS')
        pbar.update(20)

        if exclude_filters:
            for filt in exclude_filters:
                df = df.filter(~pl.col('FILTER').str.contains(filt))
        pbar.update(20)

        # Remove nulls
        df = df.filter(pl.col('MR').is_not_null())
        pbar.update(20)

    if verbose:
        print(f"          {len(df):,} variants after filtering")

    # Step 5: Compute window averages
    if verbose:
        print(f"    [5/5] Computing {window_size}bp window averages...")

    with tqdm(total=100, desc="          ", disable=not verbose,
              bar_format='{desc}{percentage:3.0f}%|{bar:30}| [{elapsed}]') as pbar:

        df = df.with_columns([
            ((pl.col('POS') // window_size) * window_size).alias('window_start')
        ])
        pbar.update(30)

        result = df.group_by('window_start').agg([
            pl.col('MR').mean().alias('avg_mu'),
            pl.col('POS').n_unique().alias('num_positions'),
            pl.col('MR').count().alias('num_records'),
        ]).sort('window_start')
        pbar.update(50)

        result = result.with_columns([
            (pl.col('window_start') + window_size).alias('window_end')
        ])
        pbar.update(20)

    # Cleanup
    if cleanup_temp and temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    if verbose:
        print(f"    ✅ {len(df):,} variants → {len(result):,} windows")

    return result


def process_vcf_fallback(vcf_path, window_size=DEFAULT_WINDOW_SIZE,
                         filter_pass_only=False, exclude_filters=None,
                         verbose=True):
    """
    Process VCF file using streaming to minimize memory usage.
    """
    import gzip
    from collections import defaultdict

    exclude_filters = exclude_filters or []
    windows = defaultdict(lambda: {"sum": 0.0, "count": 0})

    open_func = gzip.open if vcf_path.endswith(".gz") else open
    mode = "rt" if vcf_path.endswith(".gz") else "r"

    # Estimate total lines from file size (for progress bar)
    file_size = os.path.getsize(vcf_path)
    if vcf_path.endswith(".gz"):
        # Based on actual data: ~12.7 bytes per variant in compressed Roulette VCF
        estimated_lines = int(file_size / 13)
    else:
        estimated_lines = int(file_size / 110)

    valid_count = 0

    with open_func(vcf_path, mode) as f:
        pbar = tqdm(
            total=estimated_lines,
            disable=not verbose,
            unit=" variants",
            unit_scale=True,
            bar_format="    {l_bar}{bar:35}{r_bar}",
            colour="green",
            dynamic_ncols=True,
        )

        for line in f:
            pbar.update(1)

            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            pos = int(parts[1])
            filter_status = parts[6]
            info = parts[7]

            # Apply filters
            if filter_pass_only and filter_status != "PASS":
                continue
            if exclude_filters and any(f in filter_status for f in exclude_filters):
                continue

            # Extract MR
            mu = None
            for field in info.split(";"):
                if field.startswith("MR="):
                    try:
                        mu = float(field[3:])
                    except ValueError:
                        pass
                    break

            if mu is None:
                continue

            window_start = (pos // window_size) * window_size
            windows[window_start]["sum"] += mu
            windows[window_start]["count"] += 1
            valid_count += 1

        pbar.close()

    # Convert to list format
    results = []
    for ws in sorted(windows.keys()):
        data = windows[ws]
        if data["count"] > 0:
            results.append({
                'window_start': ws,
                'window_end': ws + window_size,
                'avg_mu': data["sum"] / data["count"],
                'num_positions': data["count"],
            })

    if verbose:
        print(f"    ✓ {valid_count:,} variants → {len(results):,} windows")

    return results


def write_bed_file(output_path, chrom, result, scale_factor=None):
    """Write results to BED format."""
    with open(output_path, "w") as f:
        if HAS_POLARS and isinstance(result, pl.DataFrame):
            for row in result.iter_rows(named=True):
                mu = row['avg_mu']
                if scale_factor:
                    mu *= scale_factor
                f.write(f"{chrom}\t{row['window_start']}\t{row['window_end']}\t{mu:.10e}\n")
        else:
            for row in result:
                mu = row['avg_mu']
                if scale_factor:
                    mu *= scale_factor
                f.write(f"{chrom}\t{row['window_start']}\t{row['window_end']}\t{mu:.10e}\n")


def process_chromosome(args_tuple):
    """
    Process a single chromosome (designed for multiprocessing).

    Args:
        args_tuple: (chrom, vcf_path, output_dir, window_size, scale_to_per_gen,
                     filter_pass_only, exclude_filters, threads, verbose)
    """
    (chrom, vcf_path, output_dir, window_size, scale_to_per_gen,
     filter_pass_only, exclude_filters, threads, verbose) = args_tuple

    if not os.path.exists(vcf_path):
        return (chrom, None, f"VCF not found: {vcf_path}")

    try:
        # Process VCF - always use streaming fallback to avoid memory issues
        # polars loads entire file into memory (~50-100GB for chr1)
        result = process_vcf_fallback(
            vcf_path, window_size,
            filter_pass_only=filter_pass_only,
            exclude_filters=exclude_filters,
            verbose=verbose
        )

        # Write BED
        scale_factor = ROULETTE_SCALE_FACTOR if scale_to_per_gen else None
        suffix = "" if scale_to_per_gen else ".raw"
        bed_filename = f"chr{chrom}_roulette_mu_{window_size}bp{suffix}.bed"
        bed_path = os.path.join(output_dir, bed_filename)

        write_bed_file(bed_path, str(chrom), result, scale_factor)

        n_windows = len(result)
        return (chrom, bed_path, None, n_windows)

    except Exception as e:
        return (chrom, None, str(e), 0)


def merge_bed_files(bed_files, output_path, verbose=True):
    """Merge multiple BED files into one."""
    if verbose:
        print(f"  Merging {len(bed_files)} files...")

    chrom_order = {str(i): i for i in range(1, 23)}
    all_records = []

    for bed_file in bed_files:
        with open(bed_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    chrom, start, end, rate = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
                    all_records.append((chrom_order.get(chrom, 99), start, chrom, end, rate))

    all_records.sort(key=lambda x: (x[0], x[1]))

    with open(output_path, "w") as f:
        f.write("Chromosome\tStart\tEnd\tRate\n")
        for _, start, chrom, end, rate in all_records:
            f.write(f"{chrom}\t{start}\t{end}\t{rate:.10e}\n")

    if verbose:
        print(f"  ✓ Written {len(all_records):,} records")


def compute_stats(bed_files, scaled_to_per_gen=True, verbose=True):
    """Compute genome-wide statistics."""
    rates, total_bp = [], 0

    for bed_file in bed_files:
        with open(bed_file, "r") as f:
            for line in f:
                if line.startswith("Chromosome"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    rates.append(float(parts[3]))
                    total_bp += int(parts[2]) - int(parts[1])

    if not rates:
        return None

    rates = np.array(rates)

    if verbose:
        label = "per-generation rates" if scaled_to_per_gen else "raw Roulette units"
        print(f"\n  📊 Statistics ({label}):")
        print(f"     Mean:    {np.mean(rates):.4e}")
        print(f"     Median:  {np.median(rates):.4e}")
        print(f"     Std:     {np.std(rates):.4e}")
        print(f"     Range:   [{np.min(rates):.4e}, {np.max(rates):.4e}]")
        print(f"     Windows: {len(rates):,}")
        print(f"     Coverage: {total_bp/1e9:.2f} Gb")
        if scaled_to_per_gen:
            print(f"     Unit:    /bp/generation")
        else:
            print(f"\n  📊 Per-generation rate:")
            print(f"     Mean: {np.mean(rates) * ROULETTE_SCALE_FACTOR:.4e} /bp/gen")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess Roulette mutation rate data (optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process all chromosomes
  python download_and_process.py -o ../

  # Process specific chromosomes with 4 threads
  python download_and_process.py -o ../ -c 21 22 -t 4

  # Use 8 threads for processing, 4 for downloading
  python download_and_process.py -o ../ -t 8 --download-workers 4
        """
    )

    parser.add_argument("-o", "--output-dir", default="../")
    parser.add_argument("--vcf-dir", default=None)
    parser.add_argument("-w", "--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("-c", "--chromosomes", type=int, nargs="+", default=None)
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Threads for decompression/parsing (default: 4)")
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--process-workers", type=int, default=1,
                        help="Parallel chromosome processing (default: 1)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--raw-roulette-units",
        action="store_true",
        help="write raw Roulette relative rates instead of msprime-ready per-generation rates",
    )
    parser.add_argument(
        "--scale-to-per-gen",
        action="store_true",
        help="deprecated; outputs are scaled to per-generation rates by default",
    )
    parser.add_argument("--pass-only", action="store_true")
    parser.add_argument("--exclude-filters", nargs="+", default=None)
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    verbose = not args.quiet
    base_dir = os.path.abspath(args.output_dir)
    vcf_dir = args.vcf_dir or os.path.join(base_dir, "vcf")
    output_dir = os.path.join(base_dir, "output")
    per_chrom_dir = os.path.join(output_dir, "per_chrom")
    chromosomes = args.chromosomes or list(range(1, 23))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(per_chrom_dir, exist_ok=True)
    os.makedirs(vcf_dir, exist_ok=True)

    if verbose:
        print("═" * 60)
        print("🧬 Roulette Mutation Rate Processor")
        print("═" * 60)
        print(f"  Output:      {output_dir}")
        print(f"  Window:      {args.window_size} bp")
        print(f"  Chromosomes: {chromosomes}")
        print()

    # Step 1: Download
    if verbose:
        print("─" * 60)
        print("Step 1: Download")
        print("─" * 60)

    if not args.skip_download:
        vcf_files = download_roulette_vcfs(vcf_dir, chromosomes, args.download_workers, verbose)
    else:
        vcf_files = {c: os.path.join(vcf_dir, VCF_PATTERN.format(chrom=c)) for c in chromosomes}
        if verbose:
            print("  Skipped (using existing files)")

    # Step 2: Process
    if verbose:
        print("\n" + "─" * 60)
        print("Step 2: Process VCF files")
        print("─" * 60)

    bed_files = []

    # Prepare arguments for each chromosome
    process_args = [
        (chrom,
         vcf_files.get(chrom, os.path.join(vcf_dir, VCF_PATTERN.format(chrom=chrom))),
         per_chrom_dir, args.window_size, not args.raw_roulette_units,
         args.pass_only, args.exclude_filters, args.threads, verbose)
        for chrom in chromosomes
    ]

    if args.process_workers > 1:
        # Parallel processing of chromosomes
        with ProcessPoolExecutor(max_workers=args.process_workers) as executor:
            futures = {executor.submit(process_chromosome, arg): arg[0] for arg in process_args}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  Overall",
                bar_format="  {desc}: {bar:35} {n}/{total} chromosomes",
                colour="cyan",
                disable=not verbose
            )
            for future in pbar:
                chrom, bed_path, error, n_windows = future.result()
                if bed_path:
                    bed_files.append(bed_path)
                    if verbose:
                        tqdm.write(f"  ✓ chr{chrom}: {n_windows:,} windows")
                elif verbose:
                    tqdm.write(f"  ✗ chr{chrom}: {error}")
    else:
        # Sequential processing
        for i, arg in enumerate(process_args):
            chrom = arg[0]
            if verbose:
                print(f"\n  [{i+1}/{len(process_args)}] Chromosome {chrom}")

            chrom, bed_path, error, n_windows = process_chromosome(arg)
            if bed_path:
                bed_files.append(bed_path)
            elif verbose:
                print(f"    ✗ Error: {error}")

    # Step 3: Merge
    if verbose:
        print("\n" + "─" * 60)
        print("Step 3: Merge results")
        print("─" * 60)

    suffix = "raw" if args.raw_roulette_units else "per_gen"
    final_output = os.path.join(output_dir, f"roulette_2023_autosomes_hg38_{args.window_size}bp.{suffix}.bed")
    if not args.raw_roulette_units:
        final_output = os.path.join(output_dir, f"roulette_2023_autosomes_hg38_{args.window_size}bp.bed")
    merge_bed_files(bed_files, final_output, verbose)

    # Step 4: Stats
    if verbose:
        print("\n" + "─" * 60)
        print("Step 4: Statistics")
        print("─" * 60)
    compute_stats(bed_files, scaled_to_per_gen=not args.raw_roulette_units, verbose=verbose)

    if verbose:
        print("\n" + "═" * 60)
        print(f"✅ Complete!")
        print(f"   Output: {final_output}")
        print("═" * 60)


if __name__ == "__main__":
    main()
