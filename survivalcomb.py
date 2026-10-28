"""Survival Comb public API (canonical entry points).

Thin stable wrappers over scripts/comb.py + the benchmark runners.
Importable offline; all heavy work delegates to the implementation in
scripts/ so there is exactly one code path.

    encode(groups, path, ...)            -> info dict
    decode(path, n_groups=1)             -> DecodeResult (raster)
    decode_svg(path)                     -> DecodeResult (vector)
    decode_raster(path, n_groups)        -> DecodeResult (alias of decode)
    strip_metadata(svg_path)             -> None (no metadata is written;
                                             returns the path unchanged)
    capacity_check(groups)               -> feasibility + capacity report
    validate_event_process(ep)           -> list of consistency violations
    reconstruct_pseudo_ipd(groups)       -> analysis-equivalent rows
    benchmark_encode(...)                -> seconds
    benchmark_decode(...)                -> seconds
    benchmark_existing_methods(...)      -> comparator table (CSV path)
    benchmark_degradation(...)           -> degradation grid (CSV path)
    generate_validation_report(out_md)   -> writes a markdown summary
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))

from comb import (render_survival_comb, decode_survival_comb,      # noqa: E402,F401
                  decode_survival_comb_svg, to_pseudo_ipd,
                  STRIP_LANES, COMB_X1_MM, COMB_X0_MM, HEADER_MM,
                  STRIP_BAR_PITCH_MM, consistency_violations)


def encode(groups, path, level=2, dpi=300):
    """Render the Survival Comb figure; returns the encoder info dict
    (capacity utilization, packed/dropped/aggregated slots, pitch)."""
    return render_survival_comb(groups, path, level=level, dpi=dpi)


def decode(path, n_groups=1):
    """Decode a raster comb figure -> DecodeResult with .status
    (OK | PARTIAL), .warnings and .consistency_violations."""
    return decode_survival_comb(path, n_groups=n_groups)


def decode_raster(path, n_groups=1):
    return decode_survival_comb(path, n_groups=n_groups)


def decode_svg(path):
    """Decode the SVG/vector path purely from geometry (Path A/B —
    no metadata is ever read or written)."""
    return decode_survival_comb_svg(path)


def strip_metadata(svg_path):
    """Compatibility hook: the encoder never writes metadata into the
    figure, so stripping is a no-op returning the same path."""
    return svg_path


def capacity_check(groups, pitch=STRIP_BAR_PITCH_MM):
    """Dry-run capacity report for the packet strip at `pitch` mm:
    usable lanes, bar capacity, estimated sub-resolution slot demand.
    Mirrors the encoder's layout model (docs/LEVELS.md capacity)."""
    axis_mm = COMB_X1_MM - COMB_X0_MM
    lane_w = [axis_mm - 1.0] + [axis_mm + HEADER_MM - 3.0] * (STRIP_LANES - 1)
    cap_bars = int(sum(lane_w) / pitch)
    # same collision rule the encoder uses: slots whose neighbors are
    # closer than one pitch unit become packet-strip entries
    demand = 0
    for g in groups:
        for i in range(len(g.times)):
            l = g.times[i - 1] if i else -1e18
            r = g.times[i + 1] if i + 1 < len(g.times) else 1e18
            if min(g.times[i] - l, r - g.times[i]) < 0.9 * (
                    g.times[-1] / axis_mm):
                demand += 1
    return {"pitch_mm": pitch, "lanes": STRIP_LANES,
            "bar_capacity": cap_bars, "slot_demand": demand,
            "fits_with_margin": cap_bars > 4 * demand}


def validate_event_process(ep):
    """Internal-consistency violations for a decoded event process
    (empty list = consistent). See DecodeResult.finalize()."""
    return consistency_violations([ep])


def reconstruct_pseudo_ipd(groups):
    """Expand decoded event processes to analysis-equivalent
    de-identified pseudo-IPD rows (synthetic ids only)."""
    return to_pseudo_ipd(groups)


def benchmark_encode(groups, path="/tmp/_sc_bench.svg", level=2):
    t0 = time.perf_counter()
    render_survival_comb(groups, path, level=level)
    return time.perf_counter() - t0


def benchmark_decode(path, n_groups=1, svg=False):
    t0 = time.perf_counter()
    if svg:
        decode_survival_comb_svg(path)
    else:
        decode_survival_comb(path, n_groups=n_groups)
    return time.perf_counter() - t0


def benchmark_existing_methods():
    """Run the direct comparator pipeline (Comb vs Guyot-style
    digitize->reconstruct) and return the CSV path."""
    import subprocess
    subprocess.run([sys.executable,
                    os.path.join(_HERE, "scripts", "run_comparator.py")],
                   check=True)
    return os.path.join(_HERE, "analysis", "comparator_benchmark.csv")


def benchmark_degradation():
    """Run the DPI/degradation grid; returns the summary CSV path."""
    import subprocess
    subprocess.run([sys.executable,
                    os.path.join(_HERE, "scripts", "run_degradation.py")],
                   check=True)
    return os.path.join(_HERE, "analysis", "degradation_summary.csv")


def generate_validation_report(out_md):
    """Assemble the headline QC numbers from the analysis CSVs into a
    markdown report."""
    import pandas as pd
    lines = ["# Survival Comb validation report", ""]
    det = os.path.join(_HERE, "analysis", "reconstruction_detail.csv")
    if os.path.exists(det):
        d = pd.read_csv(det)
        if "stratum" in d.columns:
            tab = (d.groupby(["path", "stratum"])
                     .agg(reps=("exact_all", "count"),
                          exact_rate=("exact_all", "mean"))
                     .round(3).reset_index())
            lines += ["## Event-process recovery by capacity stratum",
                      "", tab.to_markdown(index=False), ""]
    comp = os.path.join(_HERE, "analysis", "comparator_benchmark.csv")
    if os.path.exists(comp):
        c = pd.read_csv(comp)
        lines += ["## Direct comparator (Guyot-style) summary", "",
                  c.describe().round(3).to_markdown(), ""]
    rea = os.path.join(_HERE, "analysis", "reanalysis_validation.csv")
    if os.path.exists(rea):
        r = pd.read_csv(rea)
        lines += ["## Pseudo-IPD reanalysis validation", "",
                  r.describe().round(4).to_markdown(), ""]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    return out_md
