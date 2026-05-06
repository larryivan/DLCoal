from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config
from .utils import piecewise_eval, time_grid


def try_stdpopsim_sample(
    rng: np.random.Generator,
    cfg: Config,
    split: str,
) -> Optional[tuple[object, np.ndarray, dict, object]]:
    if not cfg.enable_stdpopsim:
        return None
    try:
        import stdpopsim  # type: ignore
    except Exception:
        return None

    try:
        species = stdpopsim.get_species(cfg.stdpopsim_species)
        model_ids = [m.strip() for m in cfg.stdpopsim_models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [m.id for m in species.demographic_models[: min(5, len(species.demographic_models))]]
        model_id = str(rng.choice(model_ids))
        model = species.get_demographic_model(model_id)
        contig = species.get_contig(
            length=cfg.stdpopsim_contig_length,
            mutation_rate=cfg.baseline_mu,
            recombination_rate=cfg.baseline_rec,
        )
        pop_id = model.populations[0].id if getattr(model, "populations", None) else "pop_0"
        n_dip = max(1, cfg.n_haplotypes // 2)
        try:
            samples = model.get_samples(n_dip, *([0] * max(0, len(model.populations) - 1)))
        except Exception:
            samples = {pop_id: n_dip}
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples, seed=int(rng.integers(1, 2**31 - 2)))
        try:
            ts = ts.trim()
        except Exception:
            pass

        _, mids = time_grid(cfg)
        breaks: list[float] = []
        values: list[float] = []
        dem = getattr(model, "model", None)
        if dem is not None:
            try:
                first_pop = dem.populations[0]
                pop_name = first_pop.name
                values = [float(first_pop.initial_size)]
                events = sorted(getattr(dem, "events", []), key=lambda e: getattr(e, "time", 0))
                for ev in events:
                    if ev.__class__.__name__ == "PopulationParametersChange":
                        if getattr(ev, "population", None) in {0, pop_name} and getattr(ev, "initial_size", None) is not None:
                            breaks.append(float(ev.time))
                            values.append(float(ev.initial_size))
            except Exception:
                values = [10_000.0]
        if not values:
            values = [10_000.0]

        y = np.log10(piecewise_eval(mids, breaks, values)).astype(np.float32)
        meta = {
            "source_type": "stdpopsim_anchor",
            "demography_type": "stdpopsim",
            "stdpopsim_species": cfg.stdpopsim_species,
            "stdpopsim_model_id": model_id,
            "stdpopsim_population": pop_id,
            "target_quality": "heuristic_single_population",
            "demography_breaks": [float(x) for x in breaks],
            "demography_values": [float(x) for x in values],
        }
        return ts, y, meta, contig
    except Exception:
        return None
