"""Simulation scenario generators (Study 2)."""
from __future__ import annotations

import numpy as np

from comb import km_event_process, EventProcess


def _event_times(rng, n, hazard):
    if hazard == "weibull_ph":
        k, lam = 1.5, 0.05
        return lam * (-np.log(1 - rng.random(n))) ** (1 / k) * 20
    if hazard == "nonph_lognormal":
        return rng.lognormal(2.5, 0.9, n)
    if hazard == "crossing":
        # mixture: half early-risk, half late-risk -> crossing group hazards
        return np.where(rng.random(n) < 0.5,
                        rng.exponential(8, n), rng.gamma(6, 15, n))
    if hazard == "delayed":
        return np.where(rng.random(n) < 0.6,
                        rng.exponential(40, n), rng.gamma(4, 8, n))
    if hazard == "early":
        return rng.exponential(12, n)
    return rng.exponential(25, n)            # constant baseline


def _censor_times(rng, n, t, rate, mech, admin):
    if rate <= 0:
        return np.full(n, np.inf)
    if mech == "early":
        ct = rng.uniform(0, 0.4 * admin, n)
    elif mech == "late":
        ct = rng.uniform(0.6 * admin, admin, n)
    else:
        ct = rng.uniform(0, admin, n)
    return ct


def _calibrate_censor(rng, ct, t, rate):
    """Keep exactly ~rate fraction of subjects censored, drawn only from
    those whose candidate censor time precedes their event time, so the
    realized censoring proportion matches the requested rate."""
    eligible = ct < t
    m = int(round(rate * len(t)))
    idx = np.flatnonzero(eligible)
    keep = rng.choice(idx, size=min(m, len(idx)), replace=False) \
        if len(idx) else np.array([], int)
    out = np.full(len(t), np.inf)
    out[keep] = ct[keep]
    return out


def simulate(rng, n=100, event_burden="medium", censor_rate=0.3,
             censor_mech="uniform", hazard="weibull_ph", ties=False,
             n_groups=1, group_ratio=None):
    """-> list[EventProcess] of simulated IPD."""
    groups = []
    sizes = ([n] if n_groups == 1 else
             np.round(np.array(group_ratio or [1] * n_groups)
                      / sum(group_ratio or [1] * n_groups) * n).astype(int))
    scale = {"low": 1.6, "medium": 1.0, "high": 0.55}[event_burden]
    for gi, m in enumerate(sizes):
        t = _event_times(rng, int(m), hazard) * scale
        if ties:                       # round to coarse grid -> large d_i, c_i
            t = np.round(t / 2.0) * 2.0
        admin = np.percentile(t, 90)
        ct = _calibrate_censor(
            rng, _censor_times(rng, int(m), t, censor_rate, censor_mech,
                               admin), t, censor_rate)
        # shift hazard between groups when n_groups>1 (PH shift)
        if gi == 1:
            t = t / 0.7
        obs = np.minimum(t, ct)
        ev = (t <= ct).astype(int)
        groups.append(km_event_process(obs, ev, label=f"group {gi + 1}"))
    return groups
