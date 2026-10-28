"""Direct comparator benchmark (revision sec.9).

Pipeline A: Survival Comb figure (L2) -> render -> decode -> EventProcess.
Pipeline B: conventional KM figure -> vector digitization -> Guyot-style
            reconstruction -> EventProcess.

Both pipelines start from the SAME source datasets and produce the same
output object, so every downstream metric is computed identically.
Measured: runtime (encode/render, decode/reconstruct, total), event-process
fidelity (exact, time recall, d/c/n errors), and downstream analysis
fidelity (KM L-inf error, median, RMST, log-rank, Cox logHR with CI).
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from comb import (decode_survival_comb_svg, render_survival_comb,     # noqa: E402
                  render_comparator, to_pseudo_ipd)
from guyot import (km_steps_from_svg, step_drops,                    # noqa: E402
                   event_process_from_steps)
from metrics import (median_survival, rmst, logrank_statistic,       # noqa: E402
                     recovery_report)                               # noqa: E402
from scenarios import simulate                                      # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
os.makedirs(OUT, exist_ok=True)

NS = [50, 100, 250, 500, 1000]
SEEDS = [0, 1, 2]


def km_linf(true_ep, est_ep):
    """L-inf distance between the two KM curves over the true time grid."""
    tt = np.concatenate([[0.0], true_ep.times])
    st = np.concatenate([[1.0], true_ep.km_survival()])
    if len(est_ep.times) == 0:
        return 1.0
    se = np.concatenate([[1.0], est_ep.km_survival()])
    te = np.concatenate([[0.0], est_ep.times])
    s_e = np.interp(tt, te, se, left=1.0, right=se[-1])
    return float(np.max(np.abs(st - s_e)))


def cox_loghr(g1, g2):
    """Cox PH log-hazard-ratio (group 2 vs 1) with 95% CI, via lifelines."""
    try:
        from lifelines import CoxPHFitter
        df = pd.DataFrame(to_pseudo_ipd([g1, g2]))
        df["arm"] = (df["group"] != df["group"].iloc[0]).astype(int)
        fit = CoxPHFitter().fit(df[["time", "status", "arm"]],
                                "time", "status")
        s = fit.summary.loc["arm"]
        return float(s["coef"]), float(s["coef lower 95%"]), \
            float(s["coef upper 95%"]), float(s["p"])
    except Exception:
        return (np.nan,) * 4


def cox_ipd(df):
    from lifelines import CoxPHFitter
    fit = CoxPHFitter().fit(df[["time", "status", "arm"]], "time", "status")
    s = fit.summary.loc["arm"]
    return float(s["coef"]), float(s["coef lower 95%"]), \
        float(s["coef upper 95%"]), float(s["p"])


def pseudo_ipd_df(groups):
    return pd.DataFrame(to_pseudo_ipd(groups)).rename(
        columns={"group": "arm"})


def logrank_p(groups):
    from lifelines.statistics import multivariate_logrank_test
    df = pseudo_ipd_df(groups)
    r = multivariate_logrank_test(df["time"], df["arm"], df["status"])
    return float(r.test_statistic), float(r.p_value)


def run_one(groups, tmpdir, label):
    rows = []
    tol = max(0.007 * float(groups[0].times[-1]), 0.5)
    n_groups = len(groups)

    # ---------- Pipeline A: Survival Comb ----------
    tracemalloc.start()
    t0 = time.perf_counter()
    comb_svg = os.path.join(tmpdir, f"a_{label}.svg")
    info = render_survival_comb(groups, comb_svg, level=2)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    dec = decode_survival_comb_svg(comb_svg)
    t_dec = time.perf_counter() - t0
    _, peak_a = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    eps_a = dec.groups
    rows.append(dict(pipeline="comb_L2", n_groups=n_groups,
                     t_encode=t_enc, t_decode=t_dec,
                     t_total=t_enc + t_dec,
                     dropped_slots=info["dropped_slots"],
                     capacity_utilization=info["capacity_utilization"],
                     status=dec.status, mem_mb=peak_a / 1e6))
    return eps_a, rows


def run_one_guyot(groups, tmpdir, label):
    tol = max(0.007 * float(groups[0].times[-1]), 0.5)
    tracemalloc.start()
    t0 = time.perf_counter()
    km_svg = os.path.join(tmpdir, f"b_{label}.svg")
    render_comparator(groups, km_svg, "km")
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    pls, _ = km_steps_from_svg(km_svg)
    # the KM curves are the longest polylines; legend swatches and axis
    # strokes are much shorter
    pls = sorted(pls, key=lambda p: -len(p[0]))[:len(groups)]
    eps_b = []
    for gi, g in enumerate(groups):
        if gi < len(pls):
            t, s = pls[gi]
            tt, ss = step_drops(t, s)
        else:
            tt, ss = np.zeros(0), np.zeros(0)
        eps_b.append(event_process_from_steps(tt, ss, g.n0))
    if len(groups) > 1:
        # reconstructed arms are unlabeled: pair them to the true groups
        # by slot count, descending
        eps_sorted = sorted(eps_b, key=lambda e: -len(e.times))
        grp_sorted = sorted(range(len(groups)),
                            key=lambda gi: -len(groups[gi].times))
        paired = [None] * len(groups)
        for e, gi in zip(eps_sorted, grp_sorted):
            paired[gi] = e
        eps_b = paired
    t_dec = time.perf_counter() - t0
    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return eps_b, dict(pipeline="guyot", n_groups=len(groups),
                       t_encode=t_enc, t_decode=t_dec,
                       t_total=t_enc + t_dec, dropped_slots=np.nan,
                       capacity_utilization=np.nan, status="n/a",
                       mem_mb=peak_b / 1e6)


def main():
    import tempfile
    rows = []
    for n in NS:
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + n + seed)
            for n_groups in (1, 2):
                groups = simulate(rng, n=n, censor_rate=0.3,
                                  n_groups=n_groups)
                label = f"n{n}_g{n_groups}_s{seed}"
                with tempfile.TemporaryDirectory() as td:
                    eps_a, ra = run_one(groups, td, label)
                    eps_b, rb = run_one_guyot(groups, td, label)
                tol = max(0.007 * float(groups[0].times[-1]), 0.5)
                for pipe, eps, base in (("comb_L2", eps_a, ra[0]),
                                        ("guyot", eps_b, rb)):
                    d_err = c_err = n_err = 0
                    exact = trec = 0.0
                    km_err = med_err = rmst_err = 0.0
                    for gi, tg in enumerate(groups):
                        rep = recovery_report(tg, eps[gi], tol)
                        d_err += rep["d_err"]
                        c_err += rep["c_err"]
                        n_err += rep["n_err"]
                        trec += rep["time_recall"]
                        exact += rep["exact"]
                        km_err += km_linf(tg, eps[gi])
                        med_err += abs(median_survival(tg)
                                       - median_survival(eps[gi]))
                        rmst_err += abs(rmst(tg) - rmst(eps[gi]))
                    r = dict(base)
                    r.update(n=n, seed=seed,
                             exact_groups=int(exact),
                             n_groups_dec=n_groups,
                             exact_group_fraction=exact / n_groups,
                             exact_run=int(exact == n_groups),
                             time_recall=trec / n_groups,
                             d_err=d_err, c_err=c_err, n_err=n_err,
                             km_linf=km_err / n_groups,
                             median_err=med_err / n_groups,
                             rmst_err=rmst_err / n_groups)
                    # two-group analyses
                    if n_groups == 2 and all(len(g.times) for g in eps):
                        lr_t = logrank_statistic(groups[0], groups[1])
                        lr_e = logrank_statistic(eps[0], eps[1])
                        r["logrank_stat"] = lr_e
                        r["logrank_err"] = abs(lr_e - lr_t)
                        hr, lo, hi, p = cox_loghr(eps[0], eps[1])
                        hr_t, lo_t, hi_t, p_t = cox_loghr(*groups)
                        r["cox_loghr"] = hr
                        r["cox_loghr_err"] = abs(hr - hr_t)
                        r["cox_p"] = p
                        r["cox_p_true"] = p_t
                        try:
                            st_t, p_t2 = logrank_p(groups)
                            st_e, p_e2 = logrank_p(eps)
                            r["logrank_p"] = p_e2
                            r["logrank_p_true"] = p_t2
                        except Exception:
                            pass
                    rows.append(r)
                    print(f"{label} {pipe}: exact={r['exact_groups']}/"
                          f"{n_groups} trec={r['time_recall']:.3f} "
                          f"d_err={d_err} km_err={r['km_linf']:.3f} "
                          f"T={r['t_total']:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "comparator_benchmark.csv")
    df.to_csv(path, index=False)
    FR = os.path.join(os.path.dirname(__file__), "..", "FINAL_RESULTS")
    os.makedirs(FR, exist_ok=True)
    df.to_csv(os.path.join(FR, "existing_method_comparison.csv"),
              index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
