"""Study 4: reconstructibility under transmission."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from comb import decode_survival_comb, render_survival_comb   # noqa: E402
from metrics import recovery_report                          # noqa: E402
from scenarios import simulate                                # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
TMP = os.path.join(OUT, "_deg")
os.makedirs(TMP, exist_ok=True)

REPS = int(os.environ.get("DEG_REPS", "8"))


def degrade(img: Image.Image, kind: str) -> Image.Image:
    w, h = img.size
    if kind == "none":
        return img
    if kind == "jpeg_q90":
        import io
        b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=90)
        b.seek(0); return Image.open(b).convert("L")
    if kind == "jpeg_q70":
        import io
        b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=70)
        b.seek(0); return Image.open(b).convert("L")
    if kind == "grayscale":
        return img.convert("L").convert("RGB").convert("L")
    if kind == "resize_05":
        return img.resize((int(w * 0.5), int(h * 0.5)), Image.LANCZOS)
    if kind == "resize_025":   # heavy downscale
        return img.resize((int(w / 4), int(h / 4)), Image.LANCZOS)
    raise ValueError(kind)


# revision-spec validation grid: dpi x {png, grayscale, jpeg, scaled}
KINDS = ["none", "grayscale", "jpeg_q90", "jpeg_q70",
         "resize_05", "resize_025"]
DPIS = [96, 150, 300, 600, 1200]


def main():
    rows = []
    for dpi, kind, rep in __import__("itertools").product(DPIS, KINDS,
                                                        range(REPS)):
        rng = np.random.default_rng(7000 + 1000 * DPIS.index(dpi) + rep)
        groups = simulate(rng, n=int(os.environ.get("DEG_N", "100")), censor_rate=0.3, hazard="weibull_ph")
        base = os.path.join(TMP, f"d{dpi}_{kind}_{rep}.png")
        render_survival_comb(groups, base, dpi=dpi)
        img = Image.open(base)
        im2 = degrade(img, kind)
        im2.save(base)
        import time
        t0 = time.perf_counter()
        dec = decode_survival_comb(base, n_groups=1)
        t_dec = time.perf_counter() - t0
        tol = max(0.007 * float(groups[0].times[-1]), 0.5)
        if dec.groups:
            rr = recovery_report(groups[0], dec.groups[0], tol)
        else:
            rr = {"time_recall": 0.0, "d_err": -1, "c_err": -1,
                  "n_err": -1, "exact": False}
        rows.append({"dpi": dpi, "degradation": kind, "rep": rep,
                     "status": dec.status, "t_decode": t_dec,
                     "exact": rr["exact"], "time_recall": rr["time_recall"],
                     "d_err": rr["d_err"], "c_err": rr["c_err"],
                     "collisions": dec.collisions,
                     "warn": ";".join(dec.warnings)})
    # stratify: resolution-invariance is claimed only within the
    # raster path's measured capacity envelope (N ~<= 100 at 300 dpi)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "degradation_detail.csv"), index=False)
    FR = os.path.join(os.path.dirname(__file__), "..", "FINAL_RESULTS")
    os.makedirs(FR, exist_ok=True)
    df.to_csv(os.path.join(FR, "degradation_runs.csv"), index=False)
    from run_final import wilson
    def cell(g):
        n = len(g); ok = int((g.status == "OK").sum())
        ex = int(g["exact"].sum())
        ok_ex = int(g["exact"][g.status == "OK"].sum())
        a = wilson(ok, n); c = wilson(ok_ex, ok); u = wilson(ex, n)
        return pd.Series(dict(
            n=n, P_OK=a[0], P_OK_lo=a[1], P_OK_hi=a[2],
            P_exact_given_OK=c[0], P_exact_given_OK_lo=c[1],
            P_exact_given_OK_hi=c[2],
            P_exact=u[0], P_exact_lo=u[1], P_exact_hi=u[2],
            d_err_med=g.d_err.median(),
            d_err_q25=g.d_err.quantile(.25),
            d_err_q75=g.d_err.quantile(.75),
            t_decode_med=g.t_decode.median(),
            t_decode_p95=g.t_decode.quantile(.95)))
    piv = (df.groupby(["dpi", "degradation"])
             .apply(cell, include_groups=False).reset_index())
    piv.to_csv(os.path.join(OUT, "degradation_summary.csv"), index=False)
    piv.to_csv(os.path.join(FR, "degradation.csv"), index=False)
    print(piv.pivot(index="degradation", columns="dpi",
                    values="P_exact").round(3).to_string())


if __name__ == "__main__":
    main()
