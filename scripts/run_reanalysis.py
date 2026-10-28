"""Pseudo-IPD reanalysis validation (Capability C).

Decode the comb figure's pseudo-IPD and repeat standard survival
analyses (median, RMST, log-rank, Cox PH) on it; compare against the
same analyses on the true event process. Quantities reported WITH
95% CIs / p-values are kept — the validation asks whether the decoded
pseudo-IPD reproduces them, including interval agreement.

Writes analysis/reanalysis_validation.csv and a summary printout.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from comb import render_survival_comb, decode_survival_comb_svg  # noqa: E402
from scenarios import simulate                                   # noqa: E402
from metrics import median_survival, rmst                        # noqa: E402
from run_comparator import cox_ipd, pseudo_ipd_df, logrank_p     # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "analysis")
os.makedirs(OUT, exist_ok=True)

NS = [50, 100, 250, 500]
REPS = int(os.environ.get("REA_REPS", "8"))


def analyses(groups):
    """truth-side analysis bundle for a 2-group process."""
    df = pseudo_ipd_df(groups)
    df["arm"] = (df["arm"] != df["arm"].iloc[0]).astype(int)
    hr, lo, hi, p_cox = cox_ipd(df)
    z2, p_lr = logrank_p(groups)
    out = dict(logrank_chi2=z2, logrank_p=p_lr,
               cox_loghr=hr, cox_lo=lo, cox_hi=hi, cox_p=p_cox)
    for gi, g in enumerate(groups):
        out[f"g{gi}_median"] = median_survival(g)
        out[f"g{gi}_rmst"] = rmst(g, float(g.times[-1]))
    return out


def main():
    rows = []
    for n in NS:
        for rep in range(REPS):
            rng = np.random.default_rng(31000 + n * 100 + rep)
            groups = simulate(rng, n=n, censor_rate=0.3, n_groups=2,
                              group_ratio=[2, 1])
            svg = os.path.join(OUT, "_rea.svg")
            render_survival_comb(groups, svg)
            dec = decode_survival_comb_svg(svg)
            try:
                a_t = analyses(groups)
                a_d = analyses(dec.groups)
            except Exception as e:
                print(f"n{n} rep{rep}: analysis failed: {e}")
                continue
            row = dict(N=n, rep=rep, status=dec.status,
                       dlogrank_p=abs(a_t["logrank_p"] - a_d["logrank_p"]),
                       dcox_loghr=abs(a_t["cox_loghr"] - a_d["cox_loghr"]),
                       # 95% CI agreement: decoded estimate inside the
                       # true CI is the headline consistency check
                       loghr_in_true_ci=float(
                           a_t["cox_lo"] <= a_d["cox_loghr"]
                           <= a_t["cox_hi"]),
                       ci_width_err=abs(
                           (a_t["cox_hi"] - a_t["cox_lo"])
                           - (a_d["cox_hi"] - a_d["cox_lo"])),
                       dcox_p=abs(a_t["cox_p"] - a_d["cox_p"]),
                       p_flip=int((a_t["logrank_p"] < 0.05)
                                  != (a_d["logrank_p"] < 0.05)))
            for gi in range(2):
                row[f"g{gi}_median_err"] = abs(
                    a_t[f"g{gi}_median"] - a_d[f"g{gi}_median"])
                row[f"g{gi}_rmst_err"] = abs(
                    a_t[f"g{gi}_rmst"] - a_d[f"g{gi}_rmst"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "reanalysis_validation.csv"),
              index=False)
    FR = os.path.join(os.path.dirname(__file__), "..", "FINAL_RESULTS")
    os.makedirs(FR, exist_ok=True)
    df.to_csv(os.path.join(FR, "pseudo_ipd_validation.csv"),
              index=False)
    summ = (df.groupby("N")
              .agg(status_partial=("status", lambda s:
                                   float(np.mean(s != "OK"))),
                   cox_loghr_err=("dcox_loghr", "mean"),
                   loghr_in_true_ci=("loghr_in_true_ci", "mean"),
                   logrank_p_err=("dlogrank_p", "mean"),
                   cox_p_err=("dcox_p", "mean"),
                   p_flip_rate=("p_flip", "mean"),
                   median_err=("g0_median_err", "mean"),
                   rmst_err=("g0_rmst_err", "mean"))
              .round(4))
    print(summ.to_string())


if __name__ == "__main__":
    main()
