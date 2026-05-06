#!/usr/bin/env python3
"""
Process Palsson et al. (2025) recombination maps.

Reads paternal and maternal maps, filters for autosomes (Chr 1-22),
calculates sex-averaged recombination rate, and converts to msprime format.

Reference:
    Palsson, S., et al. (2025). Nature.

Usage:
    python process.py
    python process.py --pat maps.pat.tsv --mat maps.mat.tsv -o output.bed
"""

import argparse
import os
from collections import defaultdict

from tqdm import tqdm


def count_lines(filepath):
    """Count lines in a file efficiently."""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def load_map_streaming(filepath, valid_chroms, desc="Loading", verbose=True):
    """Load map file using streaming to minimize memory."""
    data = {}
    total_lines = count_lines(filepath)

    with open(filepath, 'r') as f:
        header = None
        pbar = tqdm(
            total=total_lines,
            desc=f"  {desc}",
            bar_format="  {desc}: {bar:35} {percentage:3.0f}% [{n_fmt}/{total_fmt}]",
            colour="green",
            disable=not verbose,
        )

        for line in f:
            pbar.update(1)
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # Parse header
            if header is None:
                header = {col: i for i, col in enumerate(parts)}
                continue

            # Parse data
            chrom = str(parts[header['Chr']])
            if chrom not in valid_chroms:
                continue

            pos = int(parts[header['pos']])
            rate = float(parts[header['cMperMb']])

            key = (chrom, pos)
            data[key] = rate

        pbar.close()

    return data


def process_autosomes(pat_file, mat_file, output_file, verbose=True):
    """
    Process paternal and maternal maps, merge, and output sex-averaged rates.
    """
    if verbose:
        print("═" * 60)
        print("🧬 Recombination Map Processor (Palsson 2025)")
        print("═" * 60)
        print(f"  Paternal: {pat_file}")
        print(f"  Maternal: {mat_file}")
        print(f"  Output:   {output_file}")
        print()

    # Valid autosomes
    valid_chroms = set([str(i) for i in range(1, 23)] + [f"chr{i}" for i in range(1, 23)])

    # Step 1: Load maps
    if verbose:
        print("─" * 60)
        print("Step 1: Load maps")
        print("─" * 60)

    pat_data = load_map_streaming(pat_file, valid_chroms, "Paternal", verbose=verbose)
    mat_data = load_map_streaming(mat_file, valid_chroms, "Maternal", verbose=verbose)

    if verbose:
        print(f"  Paternal bins: {len(pat_data):,}")
        print(f"  Maternal bins: {len(mat_data):,}")

    # Step 2: Merge and calculate sex-averaged rate
    if verbose:
        print()
        print("─" * 60)
        print("Step 2: Merge and calculate sex-averaged rate")
        print("─" * 60)

    common_keys = set(pat_data.keys()) & set(mat_data.keys())

    if len(common_keys) == 0:
        print("  ✗ Error: No common bins found!")
        return

    if verbose:
        print(f"  Common bins: {len(common_keys):,}")

    # Calculate merged data
    merged = []
    pbar = tqdm(
        sorted(common_keys),
        desc="  Merging",
        bar_format="  {desc}: {bar:35} {percentage:3.0f}% [{n_fmt}/{total_fmt}]",
        colour="cyan",
        disable=not verbose,
    )

    for key in pbar:
        chrom, pos = key
        rate_pat = pat_data[key]
        rate_mat = mat_data[key]

        # Sex-averaged rate (cM/Mb)
        rate_avg = (rate_pat + rate_mat) / 2.0

        # Convert to probability per bp (for msprime)
        rate_per_bp = rate_avg * 1e-8

        # Normalize chromosome name
        chrom_clean = chrom.replace('chr', '')

        merged.append((int(chrom_clean), pos, rate_per_bp))

    pbar.close()

    # Sort by chromosome and position
    merged.sort(key=lambda x: (x[0], x[1]))
    by_chrom = defaultdict(list)
    for chrom, pos, rate in merged:
        by_chrom[chrom].append((pos, rate))

    # Step 3: Save
    if verbose:
        print()
        print("─" * 60)
        print("Step 3: Save output")
        print("─" * 60)

    with open(output_file, 'w') as f:
        f.write("Chromosome\tStart\tEnd\tRate\n")
        for chrom in sorted(by_chrom):
            rows = sorted(by_chrom[chrom])
            positions = [pos for pos, _ in rows]
            rates = [rate for _, rate in rows]
            if len(positions) == 1:
                starts = [max(0, positions[0] - 500_000)]
                ends = [positions[0] + 500_000]
            else:
                starts = [max(0, int(round(positions[0] - (positions[1] - positions[0]) / 2)))]
                ends = []
                for left, right in zip(positions[:-1], positions[1:]):
                    midpoint = int(round((left + right) / 2))
                    ends.append(midpoint)
                    starts.append(midpoint)
                ends.append(int(round(positions[-1] + (positions[-1] - positions[-2]) / 2)))
            for start, end, rate in zip(starts, ends, rates):
                f.write(f"{chrom}\t{start}\t{end}\t{rate:.10e}\n")

    if verbose:
        print(f"  ✓ Saved {len(merged):,} bins to {output_file}")
        print()
        print("═" * 60)
        print("✅ Complete!")
        print("═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Process Palsson recombination maps for msprime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pat", default="../raw/maps.pat.tsv",
                        help="Paternal map file (default: ../raw/maps.pat.tsv)")
    parser.add_argument("--mat", default="../raw/maps.mat.tsv",
                        help="Maternal map file (default: ../raw/maps.mat.tsv)")
    parser.add_argument("-o", "--output", default="../output/palsson_2025_autosomes_hg38.bed",
                        help="Output file (default: ../output/palsson_2025_autosomes_hg38.bed)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.pat):
        print(f"Error: Paternal map not found: {args.pat}")
        return

    if not os.path.exists(args.mat):
        print(f"Error: Maternal map not found: {args.mat}")
        return

    process_autosomes(args.pat, args.mat, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
