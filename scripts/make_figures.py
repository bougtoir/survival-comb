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

# reader-facing typography: serif (Liberation Serif = metric-compatible
# Times New Roman substitute on this build host), grayscale-robust
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.titlesize": 9,
})
MS = ["o", "s", "^", "D", "v"]          # marker cycle (grayscale safe)
LS = ["-", "--", "-.", ":"]

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
ANA = os.path.join(ROOT, "analysis")
FR = os.path.join(ROOT, "FINAL_RESULTS")


def _side_by_side(paths, labels, out, dpi=300, titlesize=10):
    """Composite same-size renders into one labelled multi-panel figure."""
    imgs = [plt.imread(p) for p in paths]
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.4, 4.4))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(lab, fontsize=titlesize, loc="left", fontweight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def fig1_reader():
    """Reader view: (A) conventional KM vs (B) Survival Comb, same data.

    Both panels are rendered by the comb path (level 0 vs 2) so frame,
    axes, ticks, curve coordinates, margins and fonts are identical -
    only the reconstruction layer differs. No technical annotations."""
    rng = np.random.default_rng(7)
    groups = simulate(rng, n=60, censor_rate=0.35)
    p_a = os.path.join(FIG, "_fig1_a.png")
    p_b = os.path.join(FIG, "_fig1_b.png")
    render_survival_comb(groups, p_a, dpi=300, level=0)
    render_survival_comb(groups, p_b, dpi=300, level=2)
    _side_by_side([p_a, p_b],
                  ["A  Conventional Kaplan-Meier figure",
                   "B  Survival Comb (same data)"],
                  os.path.join(FIG, "FIGURE_1.png"))


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
    for metric, ax, lab, letter in [("P_OK", axes[0],
                                     "P(OK) acceptance", "A"),
                                    ("P_exact_given_OK", axes[1],
                                     "P(exact | OK)", "B"),
                                    ("P_exact", axes[2],
                                     "P(exact) unconditional", "C")]:
        for j, (dpi, sub) in enumerate(df.groupby("dpi")):
            sub = sub.set_index("degradation").reindex(kinds)
            ax.plot(range(len(kinds)), sub[metric],
                    marker=MS[j % len(MS)], ls=LS[j % len(LS)],
                    color=str(0.15 * (j % 4)), ms=4, lw=1.0,
                    label=f"{dpi} dpi")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(lab)
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_7_transmission.png"), dpi=300)
    plt.close(fig)


def fig5_levels():
    """Same dataset at L0/L1/L2: identical geometry, layers differ."""
    rng = np.random.default_rng(11)
    groups = simulate(rng, n=80, censor_rate=0.3)
    paths = []
    for lv, name in [(0, "a_L0"), (1, "b_L1"), (2, "c_L2")]:
        p = os.path.join(FIG, f"FIGURE_9_{name}.png")
        # all levels render through the comb path, so box, axes, ticks,
        # curve coordinates, margins and fonts are pixel-identical
        render_survival_comb(groups, p, dpi=300, level=lv)
        paths.append(p)
    _side_by_side(paths,
                  ["A  L0: conventional display",
                   "B  L1: risk-set ticks",
                   "C  L2: event process"],
                  os.path.join(FIG, "FIGURE_9_levels.png"),
                  titlesize=9)


def fig6_timing():
    p = os.path.join(ANA, "timing_scaling.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for j, (tag, sub) in enumerate(df.groupby("variant")):
        sub = sub.sort_values("N")
        col = str(0.15 * (j % 4))
        axes[0].plot(sub.N, sub.t_encode_s, marker=MS[j % len(MS)],
                     ls=LS[j % len(LS)], color=col, ms=4, lw=1.0,
                     label=tag)
        axes[1].plot(sub.N, sub.t_decode_svg_s, marker=MS[j % len(MS)],
                     ls=LS[j % len(LS)], color=col, ms=4, lw=1.0,
                     label=f"{tag} (vector)")
        r = sub.dropna(subset=["t_decode_raster_s"])
        if len(r):
            axes[1].plot(r.N, r.t_decode_raster_s, marker="s", ms=4,
                         ls=":", color=col, lw=1.0,
                         label=f"{tag} (raster)")
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
    for j, (pipe, sub) in enumerate(df.groupby("pipeline")):
        sub = sub.sort_values("n")
        col = str(0.15 * (j % 4))
        for ax, ycol in zip(axes, ["t_total", "km_linf"]):
            ax.plot(sub.n, sub[ycol], marker=MS[j % len(MS)],
                    ls=LS[j % len(LS)], color=col, ms=4, lw=1.0,
                    label=pipe)
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
    for j, ((fam, path), sub) in enumerate(df.groupby(["family",
                                                      "path"])):
        g = sub.groupby("N").exact_run.mean()
        ax.plot(g.index, g.values, marker=MS[j % len(MS)],
                ls=LS[j % len(LS)], color=str(0.15 * (j % 4)),
                ms=4, lw=1.0, label=f"{fam} / {path}")
    ax.set_xscale("log"); ax.set_xlabel("N")
    ax.set_ylabel("P(exact)"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_4_envelope.png"), dpi=300)
    plt.close(fig)


def rev_fig2_anatomy():
    """Decoder view: magnified comb band with sparse callouts.

    The band sits directly beneath the x-axis; a decoded slot is
    annotated as (t_i, n_i, d_i, c_i) so the mapping is readable in
    seconds."""
    from comb import (decode_survival_comb, COMB_BAND_TOP_MM,
                      COMB_ROWS_BOT_MM, COMB_STRIP_TOP_MM,
                      COMB_BAND_BOT_MM, FIG_H_MM, HEADER_MM)
    rng = np.random.default_rng(3)
    groups = simulate(rng, n=300, censor_rate=0.35)
    p = os.path.join(FIG, "_anatomy.png")
    # single calibration bar lives only in this explanatory figure
    render_survival_comb(groups, p, dpi=300, calib_count=10)
    img = plt.imread(p)
    h, w = img.shape[:2]
    dec = decode_survival_comb(p)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2),
                             gridspec_kw={"height_ratios": [3, 1.9]})
    axes[0].imshow(img); axes[0].axis("off")
    axes[0].set_title("A  Survival Comb at journal size", loc="left",
                      fontsize=10, fontweight="bold")
    # crop: comb band (mm 86..120 of 120 mm figure height)
    y0, y1 = int(h * 86 / FIG_H_MM), int(h * 120 / FIG_H_MM)
    crop = img[y0:y1, :]
    axes[1].imshow(crop)
    axes[1].axis("off")
    axes[1].set_title("B  comb band magnified (beneath the time axis "
                      "and labels)", loc="left", fontsize=10,
                      fontweight="bold")
    ch = crop.shape[0]
    # crop covers fig-mm 86..120; rows zone 90.5..104.5 (centre ~97.5),
    # strip 105..117, calibration bar at far left (x < 0). All callout
    # text sits below the crop; arrows point up into the band so no
    # label ink touches the encoding.
    row_c = (97.5 - 86) / 34          # teeth centre line fraction
    lab_y = 1.30 * ch                 # label baseline below the crop
    axes[1].annotate("calibration: height = count x pitch",
                     xy=(0.022 * w, 0.80 * ch), xytext=(0.06 * w, lab_y),
                     fontsize=8, ha="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", lw=0.7,
                                     connectionstyle="arc3,rad=0.15"))
    axes[1].annotate("header: t_max, N0",
                     xy=(0.10 * w, 0.15 * ch), xytext=(0.28 * w, lab_y),
                     fontsize=8, ha="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", lw=0.7,
                                     connectionstyle="arc3,rad=-0.15"))
    axes[1].annotate("event mark (up: d_i)",
                     xy=(0.42 * w, (row_c - 0.05) * ch),
                     xytext=(0.48 * w, lab_y), fontsize=8, ha="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", lw=0.7,
                                     connectionstyle="arc3,rad=-0.12"))
    axes[1].annotate("censor mark (down: c_i)",
                     xy=(0.58 * w, (row_c + 0.05) * ch),
                     xytext=(0.66 * w, lab_y), fontsize=8, ha="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", lw=0.7,
                                     connectionstyle="arc3,rad=-0.12"))
    axes[1].annotate("packet strip (sub-resolution bursts)",
                     xy=(0.80 * w, 0.75 * ch),
                     xytext=(0.86 * w, lab_y), fontsize=8, ha="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", lw=0.7))
    if dec.groups and len(dec.groups[0].times):
        g = dec.groups[0]
        i = len(g.times) // 3
        axes[1].text(0.98 * w, lab_y + 0.16 * ch, "decoded slot: "
                     f"(t={g.times[i]:.1f}, n={g.n[i]}, "
                     f"d={g.d[i]}, c={g.c[i]})",
                     fontsize=8, ha="right", clip_on=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FIGURE_2_anatomy.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    rev_fig2_anatomy()
    fig1_reader()
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
