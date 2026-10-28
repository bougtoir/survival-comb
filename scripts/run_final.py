"""FINAL authoritative results pipeline.

One script produces every number the final manuscript, tables, QC
report and cover letter cite. Outputs land in FINAL_RESULTS/.

Metrics contract (final prompt sections 2-3):
  exact_group_fraction = (# exactly recovered groups) / (# groups) in [0,1]
  exact_run            = 1 iff ALL groups in the run are exact
  acceptance           = P(status == OK)
  conditional_exact    = P(exact_run | OK)
  unconditional_exact  = P(exact_run)
  false_ok_rate        = P(inexact | OK) = 1 - conditional_exact
  partial_rate         = P(status == PARTIAL)

Grids:
  practical: N in {50,100,250,500} (ordinary densities, 1-2 groups)
  stress:    N in {1000,2500,5000} (svg; raster only N=1000 spot)
Raster benchmark lives on the practical grid + a small N=1000 stress
probe; vector benchmark covers both grids.
"""
import os, sys, time, tracemalloc, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from comb import (render_survival_comb, decode_survival_comb,       # noqa: E402
                  decode_survival_comb_svg)
from scenarios import simulate                                      # noqa: E402
from metrics import recovery_report, median_survival, rmst          # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "FINAL_RESULTS")
FIG = os.path.join(OUT, "_figs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

REPS_PRACTICAL = int(os.environ.get("FR_REPS", "10"))
REPS_STRESS = int(os.environ.get("FR_STRESS_REPS", "3"))
DPI = 300

# catastrophic undetected-error thresholds (prespecified per section 12)
CATA_N_ERR_FRAC = 0.25     # mean |n error| > 25% of N0
CATA_RMST_FRAC = 0.10      # |RMST err| > 10% of tau


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan,) * 3
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, cen - half), min(1.0, cen + half)


def gate_metrics(df):
    """Aggregate per-run decode rows into the reporting metrics."""
    n = len(df)
    ok = int((df.status == "OK").sum())
    ex = int(df.exact_run.sum())
    ok_ex = int((df.exact_run[df.status == "OK"]).sum())
    res = dict(
        n=n, acceptance=wilson(ok, n)[0],
        acceptance_lo=wilson(ok, n)[1], acceptance_hi=wilson(ok, n)[2],
        cond_exact=wilson(ok_ex, ok)[0],
        cond_exact_lo=wilson(ok_ex, ok)[1],
        cond_exact_hi=wilson(ok_ex, ok)[2],
        uncond_exact=wilson(ex, n)[0],
        uncond_exact_lo=wilson(ex, n)[1],
        uncond_exact_hi=wilson(ex, n)[2],
        partial_rate=wilson(n - ok, n)[0],
        false_ok=(1 - wilson(ok_ex, ok)[0]) if ok else np.nan,
    )
    return res


def bench_run(groups, path_fn, dec_fn, tol):
    """Encode+decode one dataset; returns per-run metrics row."""
    t0 = time.perf_counter()
    tracemalloc.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {"dpi": 300} if path_fn.endswith(".png") else {}
        info = render_survival_comb(groups, path_fn, **kw)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    dec = dec_fn(path_fn)
    t_dec = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g_exact = []
    d_err = c_err = n_err = 0.0
    med_err = rmst_err = np.nan
    for gi, (tg, dg) in enumerate(zip(groups, dec.groups)):
        rr = recovery_report(tg, dg, tol)
        g_exact.append(rr["exact"])
        d_err += rr["d_err"]; c_err += rr["c_err"]; n_err += rr["n_err"]
        if gi == 0:
            try:
                med_err = abs(median_survival(dg) - median_survival(tg))
                rmst_err = abs(rmst(dg) - rmst(tg))
            except Exception:
                pass
    n_slots = sum(len(g.times) for g in groups)
    mean_n_err = n_err / max(n_slots, 1)
    cat = (dec.status == "OK"
           and (mean_n_err > CATA_N_ERR_FRAC * groups[0].n0
                or (np.isfinite(rmst_err)
                    and rmst_err > CATA_RMST_FRAC * groups[0].times[-1])))
    return dict(
        status=dec.status,
        exact_group_fraction=float(np.mean(g_exact)),
        exact_run=int(all(g_exact) and len(g_exact) == len(groups)),
        exact_groups=int(sum(g_exact)),
        n_groups=len(groups),
        n_slots=n_slots,
        d_err=d_err, c_err=c_err, n_err=n_err,
        median_err=med_err, rmst_err=rmst_err,
        catastrophic_false_ok=int(cat),
        t_encode=t_enc, t_decode=t_dec, t_total=t_enc + t_dec,
        mem_mb=peak / 1e6,
        dropped_slots=info.get("dropped_slots", 0),
        aggregated_slots=info.get("aggregated_slots", 0),
        capacity_utilization=info.get("capacity_utilization", np.nan),
        stratum=("beyond" if info.get("dropped_slots", 0) > 0
                 else "near" if info.get("capacity_utilization", 0) > 0.8
                 else "within"),
        warnings=";".join(dec.warnings),
    )


def main():
    rows = []
    grids = [("practical", [50, 100, 250, 500], REPS_PRACTICAL),
             ("stress", [1000, 2500, 5000], REPS_STRESS)]
    for family, ns, reps in grids:
        for n in ns:
            for rep in range(reps):
                rng = np.random.default_rng(91000 + n * 31 + rep)
                groups = simulate(rng, n=n, censor_rate=0.3, n_groups=2,
                                  group_ratio=[2, 1])
                tol = max(0.007 * groups[0].times[-1], 0.5)
                svg = os.path.join(FIG, f"{family}_n{n}_r{rep}.svg")
                r = bench_run(groups, svg,
                              lambda p: decode_survival_comb_svg(p), tol)
                r.update(family=family, N=n, rep=rep, path="svg")
                rows.append(r)
                # raster: full practical grid + N=1000 spot in stress
                if family == "practical" or n == 1000:
                    png = os.path.join(FIG, f"{family}_n{n}_r{rep}.png")
                    r = bench_run(
                        groups, png,
                        lambda p: decode_survival_comb(p, n_groups=2),
                        tol)
                    r.update(family=family, N=n, rep=rep, path="raster")
                    rows.append(r)
                print(f"{family} n{n} rep{rep} done", flush=True)
    det = pd.DataFrame(rows)
    det.to_csv(os.path.join(OUT, "benchmark_runs.csv"), index=False)

    # per-path benchmark tables
    for path, fn in [("svg", "benchmark_vector.csv"),
                     ("raster", "benchmark_raster.csv")]:
        sub = det[det.path == path]
        agg = (sub.groupby(["family", "N"])
                  .apply(lambda g: pd.Series(gate_metrics(g)),
                         include_groups=False)
                  .reset_index())
        agg["median_d_err"] = sub.groupby(
            ["family", "N"]).d_err.median().values
        agg["t_decode_med"] = sub.groupby(
            ["family", "N"]).t_decode.median().values
        agg["t_decode_p95"] = sub.groupby(
            ["family", "N"]).t_decode.quantile(0.95).values
        agg["cap_util_med"] = sub.groupby(
            ["family", "N"]).capacity_utilization.median().values
        agg["catastrophic_false_ok"] = \
            sub.groupby(["family", "N"]).catastrophic_false_ok.mean().values
        agg.to_csv(os.path.join(OUT, fn), index=False)
        print(agg.round(3).to_string())

    # capacity results: family x stratum x path panels (section 8-9)
    cap = (det.groupby(["family", "path", "stratum"])
              .apply(lambda g: pd.Series(
                  {**gate_metrics(g),
                   "cap_util_med": g.capacity_utilization.median()}),
              include_groups=False)
              .reset_index())
    cap.to_csv(os.path.join(OUT, "capacity_results.csv"), index=False)

    # encode/decode scaling (timing tables 3-4)
    t = (det.groupby(["path", "family", "N"])
            .agg(t_encode=("t_encode", "median"),
                 t_decode=("t_decode", "median"),
                 t_total=("t_total", "median"))
            .reset_index())
    t.to_csv(os.path.join(OUT, "benchmark_timing.csv"), index=False)
    # section-19 file set: encode/decode benchmark tables
    t[["path", "family", "N", "t_encode"]].to_csv(
        os.path.join(OUT, "benchmark_encode.csv"), index=False)
    t[["path", "family", "N", "t_decode"]].to_csv(
        os.path.join(OUT, "benchmark_decode.csv"), index=False)


if __name__ == "__main__":
    main()
