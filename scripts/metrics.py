"""Derived survival quantities and reconstruction-error metrics."""
from __future__ import annotations

import numpy as np

from comb import EventProcess


def median_survival(ep: EventProcess) -> float:
    s = ep.km_survival()
    below = np.flatnonzero(s <= 0.5)
    return float(ep.times[below[0]]) if len(below) else float("nan")


def rmst(ep: EventProcess, tau: float | None = None) -> float:
    if tau is None:
        tau = float(ep.times[-1])
    t = np.concatenate([[0.0], ep.times])
    s = np.concatenate([[1.0], ep.km_survival()])
    # step integral: S(t) is right-continuous value on [t_i, t_{i+1})
    tt = np.append(t, tau)
    ss = np.append(s, s[-1])
    idx = tt <= tau
    tt, ss = tt[idx], ss[idx]
    return float(np.sum(ss[:-1] * np.diff(tt)))


def landmark(ep: EventProcess, t0: float) -> float:
    s = np.concatenate([[1.0], ep.km_survival()])
    t = np.concatenate([[0.0], ep.times])
    idx = np.searchsorted(t, t0, side="right") - 1
    return float(s[max(0, idx)])


def logrank_statistic(g1: EventProcess, g2: EventProcess) -> float:
    """Standard log-rank O-E and its variance; returns z."""
    all_t = np.sort(np.unique(np.concatenate([g1.times, g2.times])))
    O_minus_E = V = 0.0
    n1, n2 = g1.n0, g2.n0
    for t in all_t:
        d1 = g1.d[g1.times == t].sum()
        d2 = g2.d[g2.times == t].sum()
        d = d1 + d2
        if d == 0 or (n1 + n2) == 0:
            n1 -= g1.c[g1.times == t].sum(); n2 -= g2.c[g2.times == t].sum()
            continue
        e = d * n1 / (n1 + n2)
        if n1 + n2 > 1:
            V += d * n1 * n2 * (n1 + n2 - d) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        O_minus_E += d1 - e
        n1 -= d1 + g1.c[g1.times == t].sum()
        n2 -= d2 + g2.c[g2.times == t].sum()
    return float(O_minus_E / np.sqrt(V)) if V > 0 else 0.0


def recovery_report(true: EventProcess, dec: EventProcess,
                    t_tol: float) -> dict:
    """Match decoded slots to true times within t_tol."""
    matched_t = sum(
        np.any(np.abs(dec.times - tt) <= t_tol) for tt in true.times)
    d_err = c_err = n_err = 0
    # sequential unique matching: each decoded slot is used at most once,
    # advancing through the sorted decoded timeline (avoids argmin picking
    # a neighbour when two decoded times are nearly equal)
    ordered = (len(dec.times) == len(true.times)
               and np.allclose(dec.times, true.times, atol=2 * t_tol))
    j_ptr = 0
    for i, tt in enumerate(true.times):
        if ordered:
            j = i        # equal slot counts: pair by order (optimal for
                         # sorted sequences); avoids argmin ambiguity
        else:
            j = -1
            best = np.inf
            for j2 in range(j_ptr, len(dec.times)):
                if dec.times[j2] - tt > t_tol:
                    break
                if abs(dec.times[j2] - tt) < best:
                    best = abs(dec.times[j2] - tt)
                    j = j2
            if j >= 0:
                j_ptr = j + 1
        if j < 0 or abs(dec.times[j] - tt) > t_tol:
            d_err += true.d[i]; c_err += true.c[i]
            n_err += abs(int(true.n[i]))
        else:
            d_err += abs(int(dec.d[j]) - int(true.d[i]))
            c_err += abs(int(dec.c[j]) - int(true.c[i]))
            n_err += abs(int(dec.n[j]) - int(true.n[i]))
    # exact: same slot count, times match pairwise within tol, and the
    # multiset of (d_i, c_i) pairs matches per-slot — a d<->c swap at any
    # slot is an error, not an exact recovery
    pair_err = d_err + c_err
    exact = (len(dec.times) == len(true.times)
             and np.allclose(np.sort(dec.times), true.times, atol=t_tol)
             and pair_err == 0)
    return {"time_recall": matched_t / max(1, len(true.times)),
            "d_err": int(d_err), "c_err": int(c_err), "n_err": int(n_err),
            "exact": bool(exact)}
