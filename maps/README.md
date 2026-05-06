# Maps

Both empirical maps use the same msprime-ready BED-like format:

```text
Chromosome	Start	End	Rate
1	0	1000000	1.0e-08
```

Coordinates are 0-based, half-open intervals: `[Start, End)`.

`Rate` is always a probability per bp per generation. This is the value passed
to `msprime.RateMap(position=..., rate=...)` after selecting one chromosome and
turning interval boundaries into the `position` array.

Current maps:

- `recomb_map/output/palsson_2025_autosomes_hg38.bed`: sex-averaged crossover recombination rate.
- `mut_map/output/roulette_2023_autosomes_hg38_1000bp.bed`: Roulette mutation rate scaled from raw relative units to per-generation rate.

The simulator can still read older point-center or raw Roulette files for
compatibility, but canonical map files should follow the interval format above.
