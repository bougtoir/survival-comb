"""Study 3: reconstruction benchmark.

For each scenario cell, R datasets are generated. The Survival Comb figure is
rendered at 300 dpi and decoded; comparators are assessed by their
*information ceiling* (what a perfect vector digitizer could recover):

  km           -> (event times, S(t)) only. (n_i,d_i,c_i) not identifiable.
  km_censor    -> + censoring times (marks). counts still unidentifiable.
  km_risktable -> + n at table grid points only.
  comb         -> full (t_i,n_i,d_i,c_i) modulo graphical-resolution merges.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from comb import (decode_survival_comb, decode_survival_comb_svg,  # noqa: E402
                  render_survival_comb)
from metrics import (median_survival, rmst, landmark,         # noqa: E402
                     logrank_statistic, recovery_report)
from scenarios import simulate                                 # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures", "_sim")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

REPS = int(os.environ.get("SC_REPS", "5"))
DPI = int(os.environ.get("SC_DPI", "300"))

NS = [50, 250, 1000, 2500, 5000]
CENSOR = [0.0, 0.3, 0.5, 0.7]
MECHS = ["early", "uniform", "late"]
HAZARDS = ["weibull_ph", "nonph_lognormal", "crossing", "delayed"]
EXTRA = ["ties", "two_group_unequal"]


def comparator_ceiling(kind: str) -> dict:
    """Information-theoretic ceiling of each comparator (perfect digitizer)."""
    return {
        "km":           {"t": 1, "d": 0, "c": 0, "n": 0},
        "km_censor":    {"t": 1, "d": 0, "c": 0, "n": 0},
        "km_risktable": {"t": 1, "d": 0, "c": 0, "n": 0},   # n only at grid
        "comb":         {"t": 1, "d": 1, "c": 1, "n": 1},
    }[kind]


def main():
    rows = []
    detail = []
    scen_id = 0
    for n, cr, mech, hz in itertools.product(NS, CENSOR, MECHS, HAZARDS):
        if cr == 0.0 and mech != "early":
            continue                        # censor mech irrelevant at 0%
        extras = ([None] + EXTRA) if (mech == "uniform"
                                      and hz in ("weibull_ph", "crossing")) \
            else [None]
        for extra in extras:
            scen_id += 1
            name = f"N{n}_c{int(cr*100)}_{mech}_{hz}" + (f"_{extra}" if extra else "")
            for rep in range(REPS):
                rng = np.random.default_rng(100000 * scen_id + rep)
                groups = simulate(rng, n=n, censor_rate=cr, censor_mech=mech,
                                  hazard=hz,
                                  ties=(extra == "ties"),
                                  n_groups=(2 if extra == "two_group_unequal" else 1),
                                  group_ratio=[3, 1])
                # raster decode for N<=1000; vector (structure-only)
                # path beyond that -- the operating envelope of each
                # path is part of the reported result
                use_svg = n > 1000
                ext = "svg" if use_svg else "png"
                path = os.path.join(FIG, f"s{scen_id}_r{rep}.{ext}")
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = (render_survival_comb(groups, path, dpi=DPI)
                            if not use_svg
                            else render_survival_comb(groups, path))
                dec = (decode_survival_comb_svg(path) if use_svg
                       else decode_survival_comb(path,
                                                 n_groups=len(groups)))
                rep_row = {"scenario": name, "rep": rep,
                           "path": "svg" if use_svg else "raster",
                           "decode_status": dec.status,
                           "dropped_slots": info["dropped_slots"],
                           "capacity_utilization":
                               info["capacity_utilization"],
                           "stratum": ("beyond" if info["dropped_slots"] > 0
                                       else "near" if
                                       info["capacity_utilization"] > 0.8
                                       else "within"),
                           "collisions": dec.collisions,
                           "warnings": ";".join(dec.warnings)}
                ok_all = True
                for gi, (tg, dg) in enumerate(zip(groups, dec.groups)):
                    tol = max(0.007 * float(tg.times[-1]), 0.5)
                    rr = recovery_report(tg, dg, tol)
                    ok_all &= rr["exact"]
                    rep_row.update({f"g{gi}_time_recall": rr["time_recall"],
                                    f"g{gi}_d_err": rr["d_err"],
                                    f"g{gi}_c_err": rr["c_err"],
                                    f"g{gi}_n_err": rr["n_err"],
                                    f"g{gi}_exact": rr["exact"]})
                    for q, f in (("median", median_survival),
                                 ("rmst", rmst),
                                 ("landmark_25pct",
                                  lambda e: landmark(e, 0.25 * e.times[-1]))):
                        rep_row[f"g{gi}_{q}_err"] = abs(f(dg) - f(tg))
                if len(groups) == 2 and len(dec.groups) == 2:
                    rep_row["logrank_z_err"] = abs(
                        logrank_statistic(dec.groups[0], dec.groups[1])
                        - logrank_statistic(groups[0], groups[1]))
                rep_row["exact_all"] = ok_all
                detail.append(rep_row)
            # comparator ceilings are scenario-independent; record once
            for kind in ("km", "km_censor", "km_risktable"):
                c = comparator_ceiling(kind)
                rows.append({"scenario": name, "method": kind,
                             "event_process_identifiable": 0})
            rows.append({"scenario": name, "method": "comb",
                         "event_process_identifiable": 1})
    det = pd.DataFrame(detail)
    det.to_csv(os.path.join(OUT, "reconstruction_detail.csv"), index=False)
    summ = (det.groupby("scenario")
              .agg(exact_rate=("exact_all", "mean"),
                   time_recall=("g0_time_recall", "mean"),
                   d_err=("g0_d_err", "mean"), c_err=("g0_c_err", "mean"),
                   n_err=("g0_n_err", "mean"),
                   collisions=("collisions", "mean"),
                   median_err=("g0_median_err", "mean"),
                   rmst_err=("g0_rmst_err", "mean"),
                   dropped=("dropped_slots", "mean"),
                   cap_util=("capacity_utilization", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(OUT, "simulation_summary.csv"), index=False)
    print(summ.to_string(max_rows=40))
    print(f"\noverall exact-recovery rate: {det.exact_all.mean():.3f} "
          f"({det.exact_all.sum()}/{len(det)})")


if __name__ == "__main__":
    main()
