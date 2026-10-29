"""Generate manuscript figures from analysis outputs."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from comb import render_survival_comb, render_comparator       # noqa: E402
from scenarios import simulate                                  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
ANA = os.path.join(ROOT, "analysis")
FR = os.path.join(ROOT, "FINAL_RESULTS")


def fig1_schematic():
    """Anatomy of the Survival Comb (annotated example)."""
    rng = np.random.default_rng(42)
    groups = simulate(rng, n=40, censor_rate=0.3)
    render_survival_comb(groups, os.path.join(FIG, "FIGURE_1_comb.png"),
                         dpi=300, title="Survival Comb (schematic)")


def fig2_comparators():
    rng = np.random.default_rng(7)
    groups = simulate(rng, n=60, censor_rate=0.4)
    for kind, name in [("km", "a_km"), ("km_censor", "b_censor"),
                       ("km_risktable", "c_risktable"), ("comb", "d_comb")]:
        render_comparator(groups, os.path.join(FIG, f"FIGURE_2_{name}.png"),
                          kind, dpi=300)


def fig3_simulation_heatmap():
    p = os.path.join(ANA, "simulation_summary.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df["hazard"] = df.scenario.str.extract(r"_([a-z_]+)$", expand=False)
    piv = df.assign(
        N=df.scenario.str.extract(r"N(\d+)").astype(int),
        c=df.scenario.str.extract(r"_c(\d+)_").astype(int))
    g = piv.pivot_table(index="c", columns="N", values="exact_rate",
                        aggfunc="mean")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(g.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(g.columns)), g.columns)
    ax.set_yticks(range(len(g.index)), g.index)
    ax.set_xlabel("N"); ax.set_ylabel("censoring %")
    ax.set_title("Exact event-process recovery rate")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_3_recovery_heatmap.png"), dpi=300)
    plt.close(fig)


def fig4_degradation():
    """Raster channel: P(OK), P(exact|OK), P(exact) separately."""
    p = os.path.join(FR, "degradation.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    kinds = [k for k in df.degradation.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for metric, ax, lab in [("P_OK", axes[0], "P(OK) acceptance"),
                            ("P_exact_given_OK", axes[1],
                             "P(exact | OK)"),
                            ("P_exact", axes[2], "P(exact)")]:
        for dpi, sub in df.groupby("dpi"):
            sub = sub.set_index("degradation").reindex(kinds)
            ax.plot(range(len(kinds)), sub[metric], "o-",
                    label=f"{dpi} dpi")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(lab)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_7_transmission.png"), dpi=300)
    plt.close(fig)


def fig5_levels():
    """Same dataset at L0/L1/L2 at journal size + magnified inset."""
    rng = np.random.default_rng(11)
    groups = simulate(rng, n=80, censor_rate=0.3)
    paths = []
    labels = ["L0 standard KM", "L1 risk-set ticks", "L2 full comb"]
    for lv, name in [(None, "a_L0"), (1, "b_L1"), (2, "c_L2")]:
        p = os.path.join(FIG, f"FIGURE_9_{name}.png")
        if lv is None:
            render_comparator(groups, p, "km", dpi=300)
        else:
            render_survival_comb(groups, p, dpi=300, level=lv)
        paths.append(p)
    # panel with zoomed comb-band inset for L1/L2
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, p, lab in zip(axes, paths, labels):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
        if "L0" not in lab:
            h, w = img.shape[:2]
            ins = ax.inset_axes([0.15, 0.02, 0.7, 0.35])
            ins.imshow(img[int(h * 0.10):int(h * 0.30),
                           int(w * 0.1):int(w * 0.6)])
            ins.axis("off")
            ins.set_title("comb band ×4", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_9_levels.png"), dpi=300)
    plt.close(fig)


def fig6_timing():
    p = os.path.join(ANA, "timing_scaling.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for tag, sub in df.groupby("variant"):
        sub = sub.sort_values("N")
        axes[0].plot(sub.N, sub.t_encode_s, "o-", label=tag)
        axes[1].plot(sub.N, sub.t_decode_svg_s, "o-", label=tag)
        r = sub.dropna(subset=["t_decode_raster_s"])
        if len(r):
            axes[1].plot(r.N, r.t_decode_raster_s, "s--",
                         label=f"{tag} raster")
    for ax, t in zip(axes, ["encode time (s)", "decode time (s)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(t)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_5_timing.png"), dpi=300)
    plt.close(fig)


def fig7_tradeoff():
    p = os.path.join(ANA, "master_table.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.scatter(df.edge_density, df.exact_rate.fillna(0),
               s=60)
    for _, r in df.iterrows():
        ax.annotate(r.variant, (r.edge_density,
                                (r.exact_rate if np.isfinite(r.exact_rate)
                                 else 0.0) + 0.03),
                    fontsize=8, ha="center")
    ax.set_xlabel("edge density (visual complexity surrogate)")
    ax.set_ylabel("exact event-process recovery")
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_8_tradeoff.png"), dpi=300)
    plt.close(fig)


def rev_fig3_capacity():
    """Exact-run recovery vs capacity utilization (both paths)."""
    p = os.path.join(FR, "benchmark_runs.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for strata, sub, mk in [("within", df[df.stratum == "within"], "o"),
                            ("near", df[df.stratum == "near"], "s"),
                            ("beyond", df[df.stratum == "beyond"], "^")]:
        if len(sub):
            ax.scatter(sub.capacity_utilization,
                       sub.exact_run.astype(float), s=8, alpha=0.4,
                       marker=mk, label=strata)
    ax.set_xlabel("packet-strip capacity utilization")
    ax.set_ylabel("exact run (all groups)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="stratum")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_3_capacity.png"), dpi=300)
    plt.close(fig)


def rev_fig6_comparator():
    """Survival Comb vs digitize-then-reconstruct: fidelity vs runtime."""
    p = os.path.join(FR, "existing_method_comparison.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for pipe, sub in df.groupby("pipeline"):
        sub = sub.sort_values("n")
        axes[0].plot(sub.n, sub.t_total, "o-", label=pipe)
        axes[1].plot(sub.n, sub.km_linf, "o-", label=pipe)
    axes[0].set_ylabel("encode+decode runtime (s)")
    axes[1].set_ylabel("KM curve sup-norm error")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_6_comparator.png"), dpi=300)
    plt.close(fig)


def rev_fig9_reanalysis():
    """Source vs figure-reconstructed pseudo-IPD reanalysis."""
    p = os.path.join(FR, "pseudo_ipd_validation.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].scatter(df.N, df.dcox_loghr, s=14)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("|Cox log-HR error|")
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[1].scatter(df.N, df.dlogrank_p.clip(lower=1e-16), s=14)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("|log-rank p error|")
    axes[1].set_yscale("log")
    fig.suptitle("True vs figure-decoded reanalysis (within capacity)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_10_reanalysis.png"), dpi=300)
    plt.close(fig)


def rev_fig10_envelope():
    """Operating envelope: unconditional exact rate vs N, by path."""
    p = os.path.join(FR, "benchmark_runs.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for (fam, path), sub in df.groupby(["family", "path"]):
        g = sub.groupby("N").exact_run.mean()
        ax.plot(g.index, g.values, "o-",
                label=f"{fam} / {path}")
    ax.set_xscale("log"); ax.set_xlabel("N")
    ax.set_ylabel("P(exact)"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_4_envelope.png"), dpi=300)
    plt.close(fig)


def rev_fig2_anatomy():
    """Magnified micro-encoding: comb band + packet strip, annotated."""
    rng = np.random.default_rng(3)
    groups = simulate(rng, n=300, censor_rate=0.35)
    p = os.path.join(FIG, "_anatomy.png")
    render_survival_comb(groups, p, dpi=300)
    img = plt.imread(p)
    h, w = img.shape[:2]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6),
                             gridspec_kw={"height_ratios": [3, 1.4]})
    axes[0].imshow(img); axes[0].axis("off")
    axes[0].set_title("Survival Comb at journal size", fontsize=9)
    axes[1].imshow(img[int(h * 0.055):int(h * 0.30), :])
    axes[1].axis("off")
    axes[1].set_title("comb band: teeth (upper = events, lower = censors) "
                      "and packet strip (sub-resolution bursts + header)",
                      fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_2_anatomy.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    rev_fig2_anatomy()
    fig1_schematic()
    fig2_comparators()
    fig3_simulation_heatmap()
    fig4_degradation()
    fig5_levels()
    fig6_timing()
    fig7_tradeoff()
    rev_fig3_capacity()
    rev_fig6_comparator()
    rev_fig9_reanalysis()
    rev_fig10_envelope()
    print("figures written to", FIG)
