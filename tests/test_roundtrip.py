"""Round-trip tests: encode -> render -> raster decode -> exact recovery."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from comb import (km_event_process, render_survival_comb,      # noqa: E402
                  decode_survival_comb, decode_survival_comb_svg,
                  decode_level1, to_pseudo_ipd, MAX_COUNT)
from scenarios import simulate                                  # noqa: E402
from metrics import recovery_report                            # noqa: E402


def spaced_times(rng, n, lo=0.0, hi=100.0, min_gap_units=None):
    """Distinct times guaranteed resolvable at canonical tooth pitch."""
    # grid + jitter: deterministically resolvable spacing
    step = (hi - lo) / (n + 1)
    jit = rng.uniform(-0.3 * step, 0.3 * step, n)
    return np.arange(1, n + 1) * step + jit + lo


def _check(groups, path, n_groups=1, dpi=300):
    render_survival_comb(groups, path, dpi=dpi)
    res = decode_survival_comb(path, n_groups=n_groups)
    assert not res.warnings, res.warnings
    assert len(res.groups) == n_groups
    for tg, dg in zip(groups, res.groups):
        assert len(dg.times) == len(tg.times)
        assert np.array_equal(dg.d, tg.d)
        assert np.array_equal(dg.c, tg.c)
        assert np.array_equal(dg.n, tg.n)
        tol = max(0.007 * float(tg.times[-1]), 0.5)
        assert np.abs(dg.times - tg.times).max() <= tol


def _group(rng, n, censor_rate=0.3, t_range=(0, 100), ties=None):
    t = spaced_times(rng, n, *t_range)
    if ties:
        t = np.round(t / ties) * ties
        t = np.unique(t)          # collisions become tied events
        t = t[t > 0]
        n = len(t)
    ev = np.ones(n, int)
    ev[rng.random(n) < censor_rate] = 0
    return [km_event_process(t, ev, "g")]


def test_sparse_exact(tmp_path):
    rng = np.random.default_rng(11)
    _check(_group(rng, 25), str(tmp_path / "a.png"))


def test_ties_exact(tmp_path):
    rng = np.random.default_rng(12)
    _check(_group(rng, 300, censor_rate=0.5, ties=3.0, t_range=(0, 600)),
           str(tmp_path / "b.png"))


def test_zero_censoring(tmp_path):
    rng = np.random.default_rng(13)
    _check(_group(rng, 30, censor_rate=0.0), str(tmp_path / "c.png"))


def test_two_groups(tmp_path):
    rng = np.random.default_rng(14)
    g1 = km_event_process(spaced_times(rng, 60, 0, 120),
                          (rng.random(60) > 0.25).astype(int), "a")
    g2 = km_event_process(spaced_times(rng, 40, 0, 110),
                          (rng.random(40) > 0.25).astype(int), "b")
    render_survival_comb([g1, g2], str(tmp_path / "g.png"), dpi=300)
    res = decode_survival_comb(str(tmp_path / "g.png"), n_groups=2)
    assert len(res.groups) == 2
    for tg, dg in zip([g1, g2], res.groups):
        assert len(dg.times) == len(tg.times)
        assert np.array_equal(dg.d, tg.d) and np.array_equal(dg.c, tg.c)


def test_header_n0_and_tmax(tmp_path):
    rng = np.random.default_rng(15)
    _check(_group(rng, 100), str(tmp_path / "h.png"))


@pytest.mark.parametrize("dpi", [600, 300])
def test_dpi_sweep(tmp_path, dpi):
    rng = np.random.default_rng(16)
    _check(_group(rng, 25, censor_rate=0.2), str(tmp_path / f"d{dpi}.png"),
           dpi=dpi)


def test_low_dpi_degrades_gracefully(tmp_path):
    """150 dpi: full exact recovery not guaranteed, but the decode must
    still locate most slots without warnings."""
    rng = np.random.default_rng(17)
    groups = _group(rng, 25, censor_rate=0.2)
    render_survival_comb(groups, str(tmp_path / "low.png"), dpi=150)
    res = decode_survival_comb(str(tmp_path / "low.png"), n_groups=1)
    assert res.groups and len(res.groups[0].times) >= 20


def test_svg_decode_exact(tmp_path):
    """Vector decode (Path A/B: structure only, no metadata) must be exact
    at N=50 single group and two-group."""
    for n_groups in (1, 2):
        rng = np.random.default_rng(30 + n_groups)
        groups = simulate(rng, n=50, censor_rate=0.3, n_groups=n_groups)
        p = str(tmp_path / f"svg_{n_groups}.svg")
        render_survival_comb(groups, p, level=2)
        dec = decode_survival_comb_svg(p).groups
        assert len(dec) == n_groups
        for gi in range(n_groups):
            rep = recovery_report(
                groups[gi], dec[gi],
                max(0.007 * float(groups[gi].times[-1]), 0.5))
            assert rep["exact"], rep


def test_level1_risk_trajectory(tmp_path):
    """L1 decode returns (t_i, n_i quantized to 10%) — times exact where
    ticks resolve, n within 5% of N0."""
    rng = np.random.default_rng(40)
    groups = simulate(rng, n=50, censor_rate=0.3)
    g = groups[0]
    p = str(tmp_path / "l1.svg")
    render_survival_comb(groups, p, level=1)
    rows = decode_level1(p)[0]
    assert len(rows) >= len(g.times) * 0.8
    for t, n in rows:
        i = np.argmin(np.abs(g.times - t))
        assert abs(g.times[i] - t) < 0.01 * float(g.times[-1])
        assert abs(n - g.n[i]) <= 0.05 * g.n0


def test_dpi_invariance(tmp_path):
    """300/600/1200-dpi renders of the same data must decode to the same
    slot count and identical d/c counts (resolution-invariant within the
    raster path's supported envelope)."""
    rng = np.random.default_rng(0)
    groups = simulate(rng, n=50, censor_rate=0.3)
    tg = groups[0]
    outs = []
    for dpi in (300, 600, 1200):
        p = str(tmp_path / f"d{dpi}.png")
        render_survival_comb(groups, p, dpi=dpi, level=2)
        dec = decode_survival_comb(p, n_groups=1).groups[0]
        outs.append(dec)
        # slot count and total d/c must be right at every resolution
        assert len(dec.times) == len(tg.times)
        assert int(dec.d.sum()) == int(tg.d.sum())
        assert int(dec.c.sum()) == int(tg.c.sum())
    # same seed, same figure content: decodes must agree across dpi
    for a, b in ((outs[0], outs[1]), (outs[0], outs[2])):
        assert np.array_equal(a.d, b.d)
        assert np.array_equal(a.c, b.c)


def test_pseudo_ipd(tmp_path):
    """Capability A: pseudo-IPD row counts equal sum(d)+sum(c)."""
    rng = np.random.default_rng(50)
    groups = simulate(rng, n=40, censor_rate=0.3)
    g = groups[0]
    rows = to_pseudo_ipd(groups)
    assert len(rows) == int(g.d.sum() + g.c.sum())
    assert {r["status"] for r in rows} <= {0, 1}
    assert sum(r["status"] for r in rows) == int(g.d.sum())


def test_exact_metrics_bounds():
    """exact_group_fraction in [0,1]; exact_run binary (final QC s.2)."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        "exact_group_fraction": [0.0, 0.5, 1.0],
        "exact_run": [0, 0, 1],
        "exact_groups": [0, 1, 2],
        "n_groups": [2, 2, 2],
    })
    assert df.exact_group_fraction.between(0, 1).all()
    assert set(df.exact_run) <= {0, 1}
    assert (df.exact_groups <= df.n_groups).all()
    # run is exact iff every group is exact
    assert (df.exact_run ==
            (df.exact_groups == df.n_groups).astype(int)).all()


def _eq(a, b):
    return (len(a.times) == len(b.times) and np.array_equal(a.d, b.d)
            and np.array_equal(a.c, b.c))


def test_calibration_marker_ignored(tmp_path):
    """The Figure-2 calibration bar must not alter decoding (regression
    per revision prompt s.17-18): identical event process with and
    without the marker, raster at 150/300/600 dpi and vector SVG."""
    rng = np.random.default_rng(21)
    groups = simulate(rng, n=80, censor_rate=0.3)
    for dpi in (150, 300, 600):
        p0 = tmp_path / f"no_cal_{dpi}.png"
        p1 = tmp_path / f"cal_{dpi}.png"
        render_survival_comb(groups, str(p0), dpi=dpi)
        render_survival_comb(groups, str(p1), dpi=dpi, calib_count=10)
        r0 = decode_survival_comb(str(p0), n_groups=1)
        r1 = decode_survival_comb(str(p1), n_groups=1)
        assert r0.status == r1.status
        assert len(r0.groups) == len(r1.groups)
        for a, b in zip(r0.groups, r1.groups):
            assert _eq(a, b)
    s0 = tmp_path / "no_cal.svg"
    s1 = tmp_path / "cal.svg"
    render_survival_comb(groups, str(s0), dpi=300)
    render_survival_comb(groups, str(s1), dpi=300, calib_count=10)
    v0 = decode_survival_comb_svg(str(s0))
    v1 = decode_survival_comb_svg(str(s1))
    assert v0.status == v1.status
    for a, b in zip(v0.groups, v1.groups):
        assert _eq(a, b)
