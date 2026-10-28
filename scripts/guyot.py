"""Guyot-style KM-to-pseudo-IPD reconstruction (Pipeline B comparator).

Implements a faithful simplified version of the published reconstruction
idea (Guyot et al. 2012, iterative risk-set recovery): digitized KM step
coordinates -> estimate events d_i from conditional survival drops ->
allocate censored observations c_i proportional to at-risk exposure in
each inter-event gap -> iterate. No external at-risk table is assumed
(the figure-only setting, matching Survival Comb's input contract).

Digitization is done from the vector SVG of the SAME conventional KM
figure rendered for the benchmark, i.e. the comparator gets perfect
vector digitization (best case).
"""
from __future__ import annotations

import re

import numpy as np

_PT_PER_MM = 72.0 / 25.4


def km_steps_from_svg(path):
    """Extract calibrated KM step coordinates (t_i, S_i) from a conventional
    KM SVG. Calibration comes from the axis tick labels themselves, exactly
    as a real curve digitizer would read them.

    Returns list of (times, surv) arrays, one per step polyline, plus the
    fitted axis maps.
    """
    s = open(path).read()
    # ---- x calibration: xtick blocks carry the label in a comment and the
    # tick mark position in the first <use x=".."> of the block
    xt = []
    for m in re.finditer(r'id="xtick_\d+".*?<!--\s*([-\d.eE]+)\s*-->', s, re.S):
        pass
    blocks = re.split(r'(?=id="xtick_)', s)
    for b in blocks:
        mv = re.search(r'<!--\s*([-\d.eE]+)\s*-->', b)
        mx = re.search(r'<use[^>]*x="([\d.]+)"', b)
        if mv and mx:
            xt.append((float(mx.group(1)), float(mv.group(1))))
    yt = []
    blocks = re.split(r'(?=id="ytick_)', s)
    for b in blocks:
        mv = re.search(r'<!--\s*([-\d.eE]+)\s*-->', b)
        my = re.search(r'<use[^>]*y="([\d.]+)"', b)
        if mv and my:
            yt.append((float(my.group(1)), float(mv.group(1))))
    if len(xt) < 2 or len(yt) < 2:
        raise ValueError("axis tick calibration failed")
    xp, xv = np.array([p for p, _ in xt]), np.array([v for _, v in xt])
    yp, yv = np.array([p for p, _ in yt]), np.array([v for _, v in yt])
    xa, xb = np.polyfit(xp, xv, 1)
    ya, yb = np.polyfit(yp, yv, 1)
    # ---- step polylines: long M/L vertex chains = the KM curves
    out = []
    for m in re.finditer(r'd="([^"]+)"', s):
        dm = m.group(1)
        verts = re.findall(r'([ML])\s*([\d.eE+-]+)\s+([\d.eE+-]+)', dm)
        if len(verts) >= 8 and all(v[0] in "ML" for v in verts):
            pts = np.array([[float(a), float(b)] for _, a, b in verts])
            t = xa * pts[:, 0] + xb
            sv = ya * pts[:, 1] + yb
            out.append((t, sv))
    return out, (xa, xb, ya, yb)


def step_drops(t, s):
    """Turn a calibrated step polyline into (event times, S just after).
    Post-step convention: drops are vertical segments.
    """
    tt, ss = [], []
    for i in range(1, len(t)):
        if s[i] < s[i - 1] - 1e-9:
            tt.append(t[i])
            ss.append(s[i])
    return np.asarray(tt), np.asarray(ss)


def guyot_reconstruct(times, surv, n0, iters=60):
    """Guyot-style iterative reconstruction.

    times : event (drop) times of the KM step, sorted
    surv  : S(t_i) just after the drop at t_i
    n0    : initial cohort size (from the figure's risk axis/header, or
            assumed known -- both pipelines are given the same N0)

    Returns (times, d, c, n) integer arrays.
    """
    times = np.asarray(times, float)
    surv = np.asarray(surv, float)
    K = len(times)
    if K == 0:
        return times, np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    S_prev = np.concatenate([[1.0], surv[:-1]])
    gaps = np.diff(np.concatenate([[0.0], times]))
    gaps = np.maximum(gaps, 1e-9)
    c = np.zeros(K, float)
    d = np.zeros(K, int)
    n = np.zeros(K, float)
    for _ in range(iters):
        # forward pass: events from conditional survival drops
        ni = float(n0)
        for i in range(K):
            sp = S_prev[i]
            di = int(round(ni * (sp - surv[i]) / sp)) if surv[i] < sp else 0
            di = max(di, 1) if surv[i] < sp else 0
            d[i] = di
            n[i] = ni
            ni = max(ni - di - c[i], 0.0)
        # at-risk remaining after the last event (conditional-probability
        # estimate): at-risk after event K ~ d_K * S_K / (S_{K-1} - S_K)
        if surv[-2] > surv[-1] if K > 1 else True:
            sp_last = S_prev[-1]
            drop = sp_last - surv[-1]
            n_end = (d[-1] * surv[-1] / drop) if drop > 0 else 0.0
        else:
            n_end = n[-1] - d[-1] - c[-1]
        # censoring budget: initial size minus events minus end-at-risk
        B = max(float(n0) - d.sum() - n_end, 0.0)
        weights = gaps * np.maximum(n - d, 0.0)
        if weights.sum() > 0:
            c = B * weights / weights.sum()
        else:
            c = c * 0.0
    # final forward pass with converged c
    ni = float(n0)
    cc = np.zeros(K, int)
    for i in range(K):
        sp = S_prev[i]
        di = int(round(ni * (sp - surv[i]) / sp)) if surv[i] < sp else 0
        di = max(di, 1) if surv[i] < sp else 0
        di = min(di, int(ni))               # events cannot exceed at-risk
        ci = int(round(c[i]))
        ci = min(ci, int(ni - di))          # at-risk cannot go negative
        d[i] = di
        cc[i] = ci
        n[i] = ni
        ni = max(ni - di - ci, 0.0)
    return times, d.astype(int), cc, n.astype(int)


def event_process_from_steps(times, surv, n0):
    """Wrap guyot_reconstruct into an EventProcess.

    Guyot-style: the c_i censored subjects of interval i are spread
    uniformly across the gap (t_{i-1}, t_i], giving them their own
    times rather than lumping them at event times.
    """
    from comb import EventProcess
    t, d, c, n = guyot_reconstruct(times, surv, n0)
    slots = []
    prev = 0.0
    for i in range(len(t)):
        if c[i] > 0:
            # uniform censoring within the preceding gap (Guyot assumption)
            for j in range(1, int(c[i]) + 1):
                slots.append((prev + (t[i] - prev) * j / (int(c[i]) + 1),
                              0, 1))
        slots.append((t[i], int(d[i]), 0))
        prev = t[i]
    slots.sort()
    tt = np.array([s[0] for s in slots])
    dd = np.array([s[1] for s in slots])
    cc = np.array([s[2] for s in slots])
    ni = n0
    nn = np.empty(len(tt), int)
    for i in range(len(tt)):
        nn[i] = ni
        ni = max(ni - dd[i] - cc[i], 0)
    return EventProcess(times=tt, n=nn, d=dd, c=cc, n0=int(n0))
