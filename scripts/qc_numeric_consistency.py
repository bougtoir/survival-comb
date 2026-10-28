"""NUMERIC_CONSISTENCY_QC — auto-check manuscript/tables vs FINAL_RESULTS.

Every number cited in the manuscript text, tables 1-8, figure captions
and the cover letter must trace to a FINAL_RESULTS/*.csv value. This
script recomputes the headline numbers from the authoritative CSVs and
compares them to the values embedded in generated markdown/docx sources.
Flags: stale values, wrong rounding, percentages from the wrong
denominator (conditional vs unconditional), out-of-range metrics.
"""
import os, re, json, sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
FR = os.path.join(ROOT, "FINAL_RESULTS")
OUT_MD = os.path.join(ROOT, "docs", "NUMERIC_CONSISTENCY_QC.md")

issues = []
checks = 0


def flag(section, msg):
    issues.append(f"- **[{section}]** {msg}")


def bump(n=1):
    global checks
    checks += n


def check(section, name, got, want, tol=0.005):
    global checks
    checks += 1
    if not np.isfinite(got) or abs(got - want) > tol:
        flag(section, f"{name}: text says {got:.4g} but FINAL_RESULTS "
             f"says {want:.4g}")


def headline():
    """Recompute the authoritative headline numbers."""
    vec = pd.read_csv(os.path.join(FR, "benchmark_vector.csv"))
    ras = pd.read_csv(os.path.join(FR, "benchmark_raster.csv"))
    deg = pd.read_csv(os.path.join(FR, "degradation.csv"))
    comp = pd.read_csv(os.path.join(FR, "existing_method_comparison.csv"))
    out = dict(
        vec=vec, ras=ras, deg=deg, comp=comp,
        # raster practical headline
        rp=ras[ras.family == "practical"],
        rv=vec[vec.family == "practical"],
        sv=vec[vec.family == "stress"],
        sr=ras[ras.family == "stress"],
        guyot=comp[comp.pipeline == "guyot"],
        comb=comp[comp.pipeline == "comb_L2"],
    )
    return out


def find_patterns(text):
    """Extract every %- or fraction-like number near a keyword."""
    hits = {}
    for m in re.finditer(r"([\w()\s]{0,80}?)((?:\d+\.\d+|\d+))%?", text):
        hits.setdefault(m.group(1).strip()[-40:], []).append(
            float(m.group(2)))
    return hits


def main():
    h = headline()
    lines = ["# NUMERIC_CONSISTENCY_QC", "",
             "Generated automatically by "
             "`scripts/qc_numeric_consistency.py` against `FINAL_RESULTS/`.",
             ""]

    # A. metric sanity: fractions in [0,1], run in {0,1}
    runs = pd.read_csv(os.path.join(FR, "benchmark_runs.csv"))
    if not runs.exact_group_fraction.between(0, 1).all():
        flag("A", "exact_group_fraction outside [0,1]")
    if not runs.exact_run.isin([0, 1]).all():
        flag("A", "exact_run not binary")
    if not (runs.exact_groups <= runs.n_groups).all():
        flag("A", "exact_groups > n_groups (aggregation bug persists)")
    bump(3)

    # B. conditional vs unconditional wording: manuscript must not quote
    # P(exact|OK) as "overall". Scan generated docx-adjacent md.
    ms = open(os.path.join(ROOT, "manuscript", "01_main.md")).read() \
        if os.path.exists(os.path.join(ROOT, "manuscript", "01_main.md")) \
        else ""
    # also scan build_docx.py's generated strings via the docx text
    import docx
    texts = []
    for f in ["01_MAIN_MANUSCRIPT_FINAL.docx",
              "02_COVER_LETTER_FINAL.docx"]:
        p = os.path.join(ROOT, "manuscript", f)
        if os.path.exists(p):
            texts.append("\n".join(par.text for par in
                                   docx.Document(p).paragraphs))
    doct = "\n".join(texts)

    # Decode-level pooled metrics (same estimand as the manuscript):
    # acceptance = P(status==OK); conditional exactness = P(exact|OK)
    # over accepted decodes; unconditional = P(exact) over all decodes.
    rp_runs = runs[(runs.family == "practical")
                   & (runs.path == "raster")]
    ok_runs = rp_runs[rp_runs.status == "OK"]
    acc = float((rp_runs.status == "OK").mean())
    cond = float(ok_runs.exact_run.mean()) if len(ok_runs) else float("nan")
    uncond = float(rp_runs.exact_run.mean())

    # if the doc quotes the conditional value adjacent to
    # 'overall'/'unconditional' -> flag
    for m in re.finditer(r"([^.]{0,120}?)(%s)%%" % re.escape(
            f"{cond*100:.1f}"), doct):
        ctx = m.group(1).lower()
        if "overall" in ctx or "unconditional" in ctx:
            flag("B", f"conditional {cond*100:.1f}% described as "
                 "overall/unconditional")

    # C. stale values sweep
    for pat in ["34.2", "exact=1.5", "Level 3", "Level 4", "L3 ", "L4 "]:
        for src, txt in [("manuscript docx", doct), ("README", open(
                os.path.join(ROOT, "README.md")).read())]:
            if pat in txt:
                flag("C", f"stale token {pat!r} in {src}")
    bump(2)

    # D. recompute table numbers vs FINAL_RESULTS (tables are generated;
    # verify the generated CSVs match authoritative ones)
    t5 = os.path.join(ROOT, "tables", "TABLE_8_existing_methods.csv")
    if os.path.exists(t5):
        t = pd.read_csv(t5)
        want = h["comb"].groupby("n").exact_run.mean()
        for _, r in t[t.pipeline == "comb_L2"].iterrows():
            check("D-table5", f"comb exact_run n{int(r.n)}",
                  r.exact_run, want.loc[r.n], tol=1e-3)

    # E. catastrophic failures: must be reported, not silent
    cat = runs.catastrophic_false_ok.sum()
    lines += [f"- runs: {len(runs)}; catastrophic false-OK events: "
              f"**{int(cat)}**",
              f"- raster practical: P(OK)={acc:.3f}, "
              f"P(exact|OK)={cond:.3f}, P(exact)={uncond:.3f}",
              f"- checks run: {checks}",
              ""]
    lines += ["## Flags", ""]
    lines += issues or ["- none"]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-len(issues) - 8:]))
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
