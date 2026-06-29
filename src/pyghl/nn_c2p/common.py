from __future__ import annotations

import pyghl as ghl


def set_flat_metric() -> tuple[ghl.Metric, ghl.ADMAux]:
    metric = ghl.initialize_metric(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    return metric, ghl.compute_ADM_auxiliaries(metric)
