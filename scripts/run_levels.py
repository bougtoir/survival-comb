"""Level benchmark: L0-L2 vs comparators — reconstructibility, visual
complexity, and encode/decode runtimes.

Outputs (analysis/):
  level_benchmark.csv   one row per render/variant (reconstruction + timing)
  complexity.csv        visual-complexity metrics per variant
  timing_scaling.csv    encode/decode runtimes across N grid
  master_table.csv      prompt section-18 master benchmark table
"""
import os, sys, time, tracemalloc, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from comb import (EventProcess, render_survival_comb, render_comparator,
                  decode_survival_comb, decode_survival_comb_svg,
                  decode_level1, to_pseudo_ipd)
from scenarios import simulate
from metrics import (recovery_report, median_survival, rmst, landmark,
                     logrank_statistic)
from visual_metrics import image_metrics, svg_metrics, ssim

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures", "_levels")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

REPS = int(os.environ.get("LEVEL_REPS", "8"))
DPI = 300


def decode_variant(kind, level, png, svg, n_groups):
    """Return decoded list[EventProcess] or None. kind describes path."""
    try:
        if kind == "comb_svg":
            return decode_survival_comb_svg(svg).groups
        if kind == "comb_raster":
            return decode_survival_comb(png, n_groups=n_groups).groups
        if kind == "comb_l1":
            rows = decode_level1(svg)
            out = []
            for gi, rs in enumerate(rows):
                if not rs:
                    continue
                ts = np.array([t for t, _ in rs])
                ns = np.array([n for _, n in rs])
                # coarse: d/c unrecoverable; emit slots with d/c unknown=0
                out.append(EventProcess(ts, ns.astype(int),
                                        np.zeros(len(ts), int),
                                        np.zeros(len(ts), int),
                                        int(rs[0][1] * 10 / 10 if rs else 0)))
            return out
    except Exception:
        return None
    return None


def analytic_errors(true_g, dec_g, t_max):
    """Secondary-endpoint error dict (|.| unless noted)."""
    if dec_g is None or len(dec_g.times) == 0:
        return dict(median_err=np.nan, rmst_err=np.nan,
                    landmark_err=np.nan)
    tm, dm = median_survival(true_g), median_survival(dec_g)
    return dict(
        median_err=abs(dm - tm) if np.isfinite(tm) and np.isfinite(dm)
        else np.nan,
        rmst_err=abs(rmst(true_g, t_max) - rmst(dec_g, t_max)),
        landmark_err=abs(landmark(true_g, t_max / 2)
                         - landmark(dec_g, t_max / 2)),
    )


def main():
    t0 = time.time()
    rows, cx_rows = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(1000 + rep)
        groups = simulate(rng, n=100, censor_rate=0.3, n_groups=2,
                          group_ratio=[2, 1])
        variants = [
            ("km", None, "raster"),
            ("km_censor", None, "raster"),
            ("km_risktable", None, "raster"),
            ("comb", 1, "svg"),
            ("comb", 2, "svg"),
            ("comb", 2, "raster"),
        ]
        base_png = os.path.join(FIG, f"r{rep}_km_raster.png")
        for kind, level, src in variants:
            tag = f"L{level}" if level is not None else kind
            png = os.path.join(FIG, f"r{rep}_{tag}_{src}.png")
            svg = png[:-4] + ".svg"
            t_e0 = time.perf_counter()
            tracemalloc.start()
            if kind == "comb":
                render_survival_comb(groups, svg, level=level)
                render_survival_comb(groups, png, level=level, dpi=DPI)
            else:
                render_comparator(groups, png, kind, dpi=DPI)
                render_comparator(groups, svg, kind)
            t_enc = time.perf_counter() - t_e0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            t_d0 = time.perf_counter()
            dec = decode_variant(
                "comb_svg" if src == "svg" else
                "comb_raster" if kind == "comb" and level == 2 else
                "comb_l1" if level == 1 else "none",
                level, png, svg, len(groups))
            t_dec = time.perf_counter() - t_d0

            ex = d_err = c_err = np.nan
            errs = dict(median_err=np.nan, rmst_err=np.nan,
                        landmark_err=np.nan)
            if dec:
                tol = max(0.007 * groups[0].times[-1], 0.5)
                reps_g = [recovery_report(groups[gi], dec[gi], tol)
                          for gi in range(min(len(groups), len(dec)))]
                ex = float(np.mean([r["exact"] for r in reps_g]))
                d_err = float(np.sum([r["d_err"] for r in reps_g]))
                c_err = float(np.sum([r["c_err"] for r in reps_g]))
                errs = analytic_errors(groups[0], dec[0],
                                       groups[0].times[-1])
            rows.append(dict(rep=rep, variant=tag, source=src, level=level,
                             t_encode_s=t_enc, t_decode_s=t_dec,
                             mem_peak_mb=peak / 1e6,
                             exact_rate=ex, d_err=d_err, c_err=c_err,
                             **errs))
            # complexity
            cx = image_metrics(png)
            cx.update(svg_metrics(svg))
            cx["ssim_vs_km"] = ssim(png, base_png)
            cx.update(rep=rep, variant=tag)
            cx_rows.append(cx)
        print(f"rep {rep} done ({time.time() - t0:.0f}s)", flush=True)

    bench = pd.DataFrame(rows)
    bench.to_csv(os.path.join(OUT, "level_benchmark.csv"), index=False)
    cxdf = pd.DataFrame(cx_rows)
    cxdf.to_csv(os.path.join(OUT, "complexity.csv"), index=False)

    # ---- timing scaling ------------------------------------------------
    trows = []
    N_GRID = [50, 100, 250, 500, 1000, 5000]
    for n in N_GRID:
        rng = np.random.default_rng(7)
        groups = simulate(rng, n=n, censor_rate=0.3, n_groups=2)
        for kind, level in [("km", None), ("km_risktable", None),
                            ("comb", 1), ("comb", 2)]:
            tag = f"L{level}" if level is not None else kind
            svg = os.path.join(FIG, "tmp_t.svg")
            png = os.path.join(FIG, "tmp_t.png")
            t_e0 = time.perf_counter()
            render_survival_comb(groups, svg, level=level) if kind == "comb" \
                else render_comparator(groups, svg, kind)
            t_enc = time.perf_counter() - t_e0
            render_survival_comb(groups, png, level=level, dpi=DPI) \
                if kind == "comb" else render_comparator(groups, png, kind,
                                                         dpi=DPI)
            t_dec = np.nan
            if kind == "comb" and level == 2:
                t_d0 = time.perf_counter()
                decode_survival_comb_svg(svg)
                t_dec = time.perf_counter() - t_d0
                t_d0 = time.perf_counter()
                decode_survival_comb(png, n_groups=2)
                t_dec_r = time.perf_counter() - t_d0
            elif kind == "comb" and level == 1:
                t_d0 = time.perf_counter()
                decode_level1(svg)
                t_dec = time.perf_counter() - t_d0
                t_dec_r = np.nan
            else:
                t_dec_r = np.nan
            trows.append(dict(N=n, variant=tag, t_encode_s=t_enc,
                              t_decode_svg_s=t_dec,
                              t_decode_raster_s=t_dec_r))
        print(f"N={n} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(trows).to_csv(os.path.join(OUT, "timing_scaling.csv"),
                               index=False)

    # ---- master table (section 18) -------------------------------------
    agg = bench.groupby("variant").agg(
        exact_rate=("exact_rate", "mean"),
        t_encode_s=("t_encode_s", "median"),
        t_decode_s=("t_decode_s", "median"),
        median_err=("median_err", "mean"),
        rmst_err=("rmst_err", "mean"),
        landmark_err=("landmark_err", "mean")).reset_index()
    cx = cxdf.groupby("variant").agg(
        ink_ratio=("ink_ratio", "mean"), edge_density=("edge_density", "mean"),
        entropy=("entropy", "mean"), components=("components", "mean"),
        png_bytes=("png_bytes", "mean"), svg_bytes=("svg_bytes", "mean"),
        ssim_vs_km=("ssim_vs_km", "mean")).reset_index()
    master = agg.merge(cx, on="variant", how="outer")
    order = ["km", "km_censor", "km_risktable", "L1", "L2"]
    master["ord"] = master["variant"].map(
        {v: i for i, v in enumerate(order)}).fillna(9)
    master.sort_values("ord").drop(columns="ord").to_csv(
        os.path.join(OUT, "master_table.csv"), index=False)
    print(master.to_string(index=False))


if __name__ == "__main__":
    main()
