"""Survival Comb: information-preserving Kaplan-Meier visualization.

Encoding spec (canonical geometry, all lengths in mm on a 180 x 120 mm figure):

  Main axes : x in [0, t_max] (shared), y in [0, 1.05]  -> conventional KM curve
  Comb band : same x axis, band immediately beneath the x-axis, hugging it,
              one row per group.  Each group row is split into an upper half
              (event teeth) and a lower half (censoring teeth).

  Header    : reserved x<0 margin of the comb band (left of the x=0 divider).
              Contains, per figure:
                - 4 significand digit-bars + 1 exponent bar encoding t_max
                - 4 digit-bars per group encoding N0_g
              Digit bars have height = (digit + 0.5) * LEVEL_MM, so a bar of
              height h decodes to round(h / LEVEL_MM - 0.5) in 0..9.
              The exponent bar encodes e + 4 in the same scale (e in -4..5),
              giving t_max ~= (sig / 1000) * 10**e with sig in [1000, 9999].

  Slots     : at each distinct event/censoring time t_i of group g a tooth is
              drawn centred at x = t_i:
                * upper bar from the row centre upward, height d_i * PITCH_MM
                * lower bar from the row centre downward, height c_i * PITCH_MM
              If a count exceeds CAP (fits in half-row), the tooth becomes a
              sentinel bar (full half-row height) followed by exactly
              DIGIT_K digit-bars encoding the count in base 10.

  Recovery  : n_{i+1} = n_i - d_i - c_i with n_1 = N0 (risk set at the first
              time uses n_1 = N0 - (events/censors strictly before t_1) = N0
              under the usual convention that c_i at t_i is removed after the
              risk set at t_i is formed).

  Resolution limits (predefined graphical resolution):
    * two distinct times closer than TOOTH_W_MM merge into one cluster and are
      reported as a collision (not silently misdecoded);
    * counts > 10**DIGIT_K - 1 are out of spec;
    * t_i is recovered with positional tolerance ~= pixel/mm scale.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from itertools import zip_longest


def _digits_to_int(digits):
    v = 0
    for d in digits:
        v = v * 10 + int(d)
    return v


def _risk_sets(n0, d, c):
    n = np.zeros(len(d), int)
    risk = int(n0)
    for i in range(len(d)):
        n[i] = risk
        risk -= int(d[i]) + int(c[i])
    return n

import numpy as np

# ---- canonical geometry (mm on a FIG_W x FIG_H mm figure) -------------------
FIG_W_MM = 180.0
FIG_H_MM = 120.0
COMB_X0_MM = 28.0          # left edge of the x=0 position of comb/main axes
COMB_X1_MM = 175.0         # right edge (x = t_max)
MAIN_TOP_MM = 10.0
MAIN_BOT_MM = 80.5
# comb band sits below the x-axis line, tick labels and "Time" title:
# vertical order KM plot -> axis line -> tick labels -> Time -> gap -> comb
COMB_BAND_TOP_MM = 90.5    # measured from top of figure
COMB_ROWS_BOT_MM = 104.5   # group rows live in TOP..ROWS_BOT
COMB_STRIP_TOP_MM = 105.0  # overflow packet strip
COMB_BAND_BOT_MM = 117.0
HEADER_MM = 26.0           # reserved header margin left of x=0 (5 + 5G bars, G<=4)

PITCH_MM = 0.45            # height unit of a unary count tooth
LEVEL_MM = 0.6             # height unit of a digit level
TOOTH_W_MM = 0.25          # tooth width
EPS_COLLIDE_MM = 0.4       # time gap below which a slot is packed to the strip
TICK_MM = 0.05             # time granularity inside a packet
# absolute-time quantum for packed slots (3 digits). 0.15 mm -> 980
# quanta cover the 147 mm axis in a 3-digit code; sub-resolution anyway.
PACKET_POS_MM = 0.15
PACKET_TICK_MM = 0.05      # intra-burst offset unit (1 digit -> 0.45 mm)
STRIP_LEVEL_MM = 0.20      # digit level height inside the packet strip
STRIP_BAR_PITCH_MM = 0.55  # bar spacing inside the packet strip
STRIP_BAR_W_MM = 0.25
STRIP_LANES = 5            # stacked bar lanes inside the packet strip
STRIP_LANE_PITCH_MM = 2.2
RISK_LEVEL_MM = 0.25   # L1 risk-tick height unit (10 levels -> 2.5 mm)
PACKET_MAX_SLOTS = 9       # slots per packet (cnt field is 1 digit)
# packet layout: [group_id, cnt, mode, base_pos(3)] + cnt * [off(1) + count(s)]
#                + [check(1)]   where check = -(sum of preceding digits) mod 10
# base_pos = absolute position of the first slot, quantized to
# PACKET_POS_MM (0.15 mm) over the axis (<= 980 quanta); packed slots in a
# burst all lie within EPS_COLLIDE_MM of a drawn tooth, so each slot's
# offset from base is a single digit in PACKET_TICK_MM (0.05 mm) units.
# mode 0: d,c <= 2 -> one joint digit 1+3d+c; mode 1: d,c <= 9 (1 digit each);
# mode 2: d,c <= 99 (2 digits each). Slot bars = 1 + mode_digits.
DIGIT_K = 4                # digit bars after a sentinel (counts <= 9999)
MAX_COUNT = 10 ** DIGIT_K - 1
STRIP_H_MM = COMB_BAND_BOT_MM - COMB_STRIP_TOP_MM          # 9 mm
STRIP_LANE_TOPS = tuple(STRIP_H_MM - 0.3 - k * STRIP_LANE_PITCH_MM
                        for k in range(STRIP_LANES))  # data-y tops


def _mm_x(fig, mm):
    return mm / FIG_W_MM * fig.get_size_inches()[0] * fig.dpi


@dataclass
class EventProcess:
    """E = {(t_i, n_i, d_i, c_i)} for one group."""
    times: np.ndarray      # t_i (distinct, sorted)
    n: np.ndarray          # risk set just before t_i
    d: np.ndarray          # events at t_i
    c: np.ndarray          # censored at t_i (after risk set formed)
    n0: int                # initial risk set
    label: str = ""

    def km_survival(self) -> np.ndarray:
        s = np.ones(len(self.times))
        cur = 1.0
        for i in range(len(self.times)):
            if self.d[i] > 0 and self.n[i] > 0:
                cur *= 1.0 - self.d[i] / self.n[i]
            s[i] = cur
        return s


def km_event_process(times: np.ndarray, events: np.ndarray,
                     label: str = "") -> EventProcess:
    """Build E from individual-level (time, event) data, right-censored."""
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    order = np.argsort(times, kind="stable")
    times, events = times[order], events[order]
    uniq = np.unique(times)
    n0 = len(times)
    n = np.zeros(len(uniq), int)
    d = np.zeros(len(uniq), int)
    c = np.zeros(len(uniq), int)
    risk = n0
    for i, t in enumerate(uniq):
        # events/censorings strictly before t have already left the risk set
        n[i] = risk
        mask = times == t
        d[i] = int(events[mask].sum())
        c[i] = int(mask.sum() - d[i])
        risk -= d[i] + c[i]
    return EventProcess(uniq, n, d, c, n0, label)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _enc_digits(value: int, k: int):
    if value > 10 ** k - 1 or value < 0:
        raise ValueError(f"value {value} does not fit in {k} digit bars")
    return [(value // 10 ** (k - 1 - j)) % 10 for j in range(k)]


def _encode_tmax(t_max: float):
    """-> (4 significand digits, exponent e) with t_max ~ sig/1000 * 10**e."""
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    e = math.floor(math.log10(t_max)) - 3
    sig = int(round(t_max / 10 ** e))
    if sig > 9999:                     # rounding overflow
        sig //= 10
        e += 1
    if not (-4 <= e <= 5):
        raise ValueError(f"t_max {t_max} out of exponent range")
    return _enc_digits(sig, 4), e


def render_survival_comb(groups: list[EventProcess], path: str,
                         dpi: int = 150, title: str | None = None,
                         show_axis_labels: bool = True, level: int = 2,
                         calib_count: int | None = None):
    """Render the Survival Comb figure to `path` (raster or .svg).

    level 0: conventional KM (comparator; no comb band)
    level 1: comb band with risk-set ticks only — tick height encodes
             n_i/N0 in RISK_LEVELS levels of RISK_LEVEL_MM (coarse n(t))
    level 2: full comb — per-slot d_i/c_i teeth + packet strip (CORE)

    Downstream reconstruction capabilities (docs/LEVELS.md) are not
    graphical levels: Capability A = pseudo-IPD expansion
    (to_pseudo_ipd), B = multi-arm comparison, C = reanalysis of derived
    endpoints. They add no ink beyond level 2.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # level 0 keeps the comb-path frame/axes so L0/L1/L2 panels have
    # identical geometry; only the reconstruction layers are absent
    risk_only = (level == 1)
    bare_band = (level == 0)

    t_max = max(float(g.times[-1]) for g in groups)
    G = len(groups)
    row_mm = (COMB_ROWS_BOT_MM - COMB_BAND_TOP_MM) / G
    half_mm = row_mm / 2.0
    cap = int(half_mm / PITCH_MM) - 1          # unary capacity per half-row

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=dpi)

    def mm_ax(x0, y0, w, h):
        return fig.add_axes([x0 / FIG_W_MM, 1 - (y0 + h) / FIG_H_MM,
                             w / FIG_W_MM, h / FIG_H_MM])

    ax_main = mm_ax(COMB_X0_MM, MAIN_TOP_MM,
                    COMB_X1_MM - COMB_X0_MM, MAIN_BOT_MM - MAIN_TOP_MM)
    ax_comb = mm_ax(COMB_X0_MM - HEADER_MM, COMB_BAND_TOP_MM,
                    COMB_X1_MM - COMB_X0_MM + HEADER_MM,
                    COMB_BAND_BOT_MM - COMB_BAND_TOP_MM)

    ax_main.set_xlim(0, t_max)
    ax_main.set_ylim(0, 1.05)
    if show_axis_labels:
        ax_main.set_ylabel("Survival probability")
    if title:
        ax_main.set_title(title)
    ax_main.tick_params(labelsize=7)
    # x tick labels live below the comb band (band hugs the axis); the
    # axes-level labels are suppressed and re-drawn under the band so no
    # text ink lands inside the decoded region
    for tl in ax_main.get_xticklabels():
        tl.set_visible(False)
    ax_main.set_xlabel("")

    # x<0 header margin in comb axes
    hdr_data = HEADER_MM / (COMB_X1_MM - COMB_X0_MM) * t_max
    ax_comb.set_xlim(-hdr_data, t_max)
    ax_comb.set_ylim(0, COMB_BAND_BOT_MM - COMB_BAND_TOP_MM)
    ax_comb.axis("off")
    ax_comb.axvline(0, color="0.65", lw=0.8)         # x=0 divider (below binarize threshold)
    ax_comb.axvline(t_max, color="0.65", lw=0.8)     # right edge marker

    # y in comb axes is in mm-from-top data units
    def ydata(mm_from_top):
        return mm_from_top

    def bar(x, y_base_mm, height_mm, up, color="k"):
        y0 = y_base_mm if up else y_base_mm - height_mm
        w_data = TOOTH_W_MM / (COMB_X1_MM - COMB_X0_MM) * t_max
        ax_comb.add_patch(plt.Rectangle(
            (x - w_data / 2, ydata(y0)), w_data, abs(height_mm),
            facecolor=color, edgecolor="none", zorder=3))

    def digit_bar(x, y_base_mm, digit, up):
        bar(x, y_base_mm, (digit + 0.5) * LEVEL_MM, up)

    # ---- header --------------------------------------------------------
    step_data = 1.0 / (COMB_X1_MM - COMB_X0_MM) * t_max   # 1 mm in data units
    sig, e = _encode_tmax(t_max)
    hx = -hdr_data + 0.5 * step_data
    # header layout is anchored at x=0 going left: last bars nearest divider
    # header digits sit in the rows zone, just above the packet strip:
    # strip occupies the lower STRIP_H_MM of the band (data-y 0..STRIP_H)
    yb = (COMB_ROWS_BOT_MM - COMB_BAND_TOP_MM) + 0.5  # bottom of digit bars
    y_hdr = yb + 12.0 * LEVEL_MM
    if not bare_band:
        for j, dg in enumerate(sig):
            digit_bar(hx + j * step_data, y_hdr, dg, up=False)
        digit_bar(hx + 4 * step_data, y_hdr, e + 4, up=False)
        hx += 5 * step_data
        for g in groups:
            for j, dg in enumerate(_enc_digits(g.n0, 4)):
                digit_bar(hx + j * step_data, y_hdr, dg, up=False)
            hx += 5 * step_data        # 4 digits + gap

    # ---- teeth ---------------------------------------------------------
    band_h_mm = COMB_BAND_BOT_MM - COMB_BAND_TOP_MM
    packets = {gi: [] for gi in range(len(groups))}
    for gi, g in enumerate(groups):
        centre = band_h_mm - (gi * row_mm + half_mm)   # data-y = from bottom
        ax_comb.axhline(centre, color="0.75", lw=0.4, zorder=1)
        eps_data = EPS_COLLIDE_MM / (COMB_X1_MM - COMB_X0_MM) * t_max
        # teeth are never drawn flush against the frame divider (x0 line
        # would absorb them); t~0 slots are nudged right by half a tooth
        t_floor = (TOOTH_W_MM + 0.2) / (COMB_X1_MM - COMB_X0_MM) * t_max
        last_drawn = -1.0
        # guard: a drawn tooth must clear the previous packed slot by more
        # than the decode noise band (~0.2 mm); a nearer event is packed
        # instead so packet-vs-tooth ordering stays recoverable
        guard_data = 0.25 / (COMB_X1_MM - COMB_X0_MM) * t_max
        prev_x = -1.0      # position of the previous event slot
        prev_packed = False
        for i, t in enumerate(g.times):
            td = max(float(t), t_floor)
            if bare_band:
                continue
            if td - last_drawn < eps_data or (prev_packed
                                              and td - prev_x < guard_data):
                prev_packed = True
                prev_x = td
                if risk_only:
                    # L1: no packet strip; draw the risk tick anyway
                    # (merged is acceptable at a coarse level)
                    lev = int(round(10 * g.n[i] / g.n0))
                    if lev > 0:
                        bar(float(t), centre, lev * RISK_LEVEL_MM, True)
                    continue
                # absolute position in mm (burst-delta coded at write time)
                x_mm = float(t) / t_max * (COMB_X1_MM - COMB_X0_MM)
                packets[gi].append((x_mm, int(g.d[i]), int(g.c[i])))
                continue
            last_drawn = td
            t = td
            prev_packed = False
            prev_x = td
            if risk_only:
                lev = int(round(10 * g.n[i] / g.n0))
                if lev > 0:
                    bar(float(t), centre, lev * RISK_LEVEL_MM, True)
                continue
            for val, up in ((int(g.d[i]), True), (int(g.c[i]), False)):
                if val == 0:
                    continue
                if val <= cap:
                    bar(float(t), centre, val * PITCH_MM, up)
                else:
                    # sentinel + DIGIT_K digit bars
                    bar(float(t), centre, half_mm, up)
                    for k, dg in enumerate(_enc_digits(val, DIGIT_K)):
                        xk = float(t) + (k + 1) * 0.8 * step_data
                        if up:
                            y0 = centre
                            h = (dg + 0.5) * LEVEL_MM
                        else:
                            y0 = centre - (dg + 0.5) * LEVEL_MM
                            h = (dg + 0.5) * LEVEL_MM
                        w_data = TOOTH_W_MM / (COMB_X1_MM - COMB_X0_MM) * t_max
                        ax_comb.add_patch(plt.Rectangle(
                            (xk - w_data / 2, ydata(y0)), w_data, h,
                            facecolor="k", edgecolor="none", zorder=3))

        # macro KM curve
        s = g.km_survival()
        tt = np.concatenate([[0.0], g.times])
        ss = np.concatenate([[1.0], s])
        # grayscale-robust series: shade + linestyle, never colour alone
        gs_col = ["k", "0.45", "0.65", "0.3"][gi % 4]
        gs_ls = ["-", "--", "-.", ":"][gi % 4]
        ax_main.step(tt, ss, where="post", lw=1.4, color=gs_col, ls=gs_ls,
                     label=g.label or f"group {gi + 1}")
    # ---- overflow packet strip --------------------------------------
    axis_mm = COMB_X1_MM - COMB_X0_MM
    lane = 0
    dropped = 0
    # lanes 1+ extend into the left header margin (header digits sit higher)
    lane_start = [1.5] + [3.0 - HEADER_MM] * (STRIP_LANES - 1)
    # ---- build packet digit sequences (one per burst chunk) ----
    items = []                       # (group_idx, digit_seq, n_slots)
    for gi in range(len(groups)):
        slots = packets[gi]
        # chunk by burst: packed slots within a burst lie in one
        # ~EPS_COLLIDE_MM window; each chunk's base is the 0.15-mm floor
        # of its first slot, so offsets fit one digit while the slot's
        # position stays within 0.45 mm of base; split otherwise
        chunks = []
        cur = []
        base_x = 0.0
        for sl in slots:
            if cur and (sl[0] - base_x > 9 * PACKET_TICK_MM
                        or len(cur) >= PACKET_MAX_SLOTS):
                chunks.append(cur)
                cur = []
            if not cur:
                base_x = int(sl[0] / PACKET_POS_MM) * PACKET_POS_MM
            cur.append(sl)
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            if all(dv <= 2 and cv <= 2 for _, dv, cv in chunk):
                mode = 0
            elif all(dv <= 9 and cv <= 9 for _, dv, cv in chunk):
                mode = 1
            else:
                mode = 2
            base_pq = int(chunk[0][0] / PACKET_POS_MM)   # floor, mm quanta
            seq = [gi + 1, len(chunk), mode] + _enc_digits(base_pq, 3)
            for x_mm, dv, cv in chunk:
                off = int(round((x_mm - base_pq * PACKET_POS_MM)
                                / PACKET_TICK_MM))
                if mode == 0:
                    seq += [min(max(off, 0), 9),
                            1 + 3 * min(dv, 2) + min(cv, 2)]
                else:
                    seq += ([min(max(off, 0), 9)]
                            + _enc_digits(min(dv, 99), mode)
                            + _enc_digits(min(cv, 99), mode))
            chk = (-sum(seq)) % 100
            seq += [chk // 10, chk % 10]   # 2-digit checksum (graceful failure)
            items.append((gi, seq, len(chunk), chunk))

    # ---- overflow aggregation: slots that cannot be laid out are not
    # silently lost; one aggregate packet per group stores their summed
    # event/censor counts at the median dropped position (mode 3), so
    # totals and at-risk trajectories stay recoverable. Times within
    # the aggregate are flagged PARTIAL by the decoder. ----

    # adaptive bar pitch: pick the largest pitch that still fits every
    # packet; smaller pitch only buys capacity, never more ink than needed
    def _layout(pitch, reserve=0.0):
        """Dry run: return the index set of items that cannot be laid."""
        right = axis_mm - 1.0 - reserve
        lane_p = 0
        xc = lane_start[0]
        drop_idx = set()
        for ii, (_, seq, nsl, _c) in enumerate(items):
            if lane_p >= STRIP_LANES:
                drop_idx.add(ii)
                continue
            w_need = len(seq) * pitch
            if xc + w_need > right:
                lane_p += 1
                if lane_p >= STRIP_LANES:
                    drop_idx.add(ii)
                    continue
                xc = lane_start[lane_p]
            xc += w_need + pitch
        return drop_idx

    pitch = STRIP_BAR_PITCH_MM
    drop_idx = _layout(pitch)
    for cand in (STRIP_BAR_PITCH_MM, 0.45, 0.38):
        di = _layout(cand)
        pitch = cand
        if not di:
            drop_idx = di
            break
        if len(di) < len(drop_idx):
            drop_idx = di

    # fold every dropped chunk into one aggregate overflow packet
    # per group (mode 3): [gid+1, 9, 3, base(3), d(4), c(4), chk(2)]
    def _agg_packets(gi, xs):
        """Up to 3 quantile-binned aggregate packets for one group."""
        xs = sorted(xs)
        nb = min(3, max(1, int(np.ceil(len(xs) / 400))))
        bins = np.array_split(xs, nb)
        out = []
        for bin_ in bins:
            if len(bin_) == 0:
                continue
            d_agg = sum(dv for _, dv, _c in bin_)
            c_agg = sum(cv for _, _d, cv in bin_)
            pos = sorted(x for x, _d, _c in bin_)[len(bin_) // 2]
            base_pq = int(pos / PACKET_POS_MM)
            seq = ([gi + 1, 9, 3] + _enc_digits(base_pq, 3)
                   + _enc_digits(min(d_agg, 9999), 4)
                   + _enc_digits(min(c_agg, 9999), 4))
            chk = (-sum(seq)) % 100
            seq += [chk // 10, chk % 10]
            out.append((gi, seq, 0, []))
        return out

    overflow = []
    for gi in range(len(groups)):
        xs = [sl for ii, (g2, _s, _n, chunk) in enumerate(items)
              if ii in drop_idx and g2 == gi for sl in chunk]
        overflow += _agg_packets(gi, xs)
    if overflow:
        # reserve tail width so the aggregate packets always fit, then
        # recompute which data packets are displaced and fold THOSE into
        # the aggregate as well (one pass; aggregate size is unchanged)
        ovf_w = len(overflow) * 17 * pitch
        drop_idx = _layout(pitch, reserve=ovf_w)
        overflow = []
        for gi in range(len(groups)):
            xs = [sl for ii, (g2, _s, _n, chunk) in enumerate(items)
                  if ii in drop_idx and g2 == gi for sl in chunk]
            overflow += _agg_packets(gi, xs)
    aggregated_slots = sum(items[ii][2] for ii in drop_idx)
    # overflow packets go LAST: their median-of-dropped position is late
    # in the stream, and the decoder requires non-decreasing positions
    items = ([it for ii, it in enumerate(items) if ii not in drop_idx]
             + overflow)

    ovf_right = len(overflow) * 17 * pitch if overflow else 0.0
    bars_used = 0
    drawn = []
    xc_mm = lane_start[0]
    n_data = len(items) - len(overflow)
    for ii, (gi, seq, nsl, chunk) in enumerate(items):
        # data packets share lanes bounded by the reserved tail width;
        # overflow packets (mode 3) are drawn into that reserved tail
        is_ovf = ii >= n_data
        right = axis_mm - 1.0 if is_ovf else axis_mm - 1.0 - ovf_right
        if lane >= STRIP_LANES:
            dropped += nsl         # strip full; documented capacity limit
            continue
        w_need = len(seq) * pitch
        if xc_mm + w_need > right:
            lane += 1
            if lane >= STRIP_LANES:
                dropped += nsl
                continue
            xc_mm = lane_start[lane] if not is_ovf else xc_mm
        ytop = STRIP_LANE_TOPS[lane]
        for j, dg in enumerate(seq):
            x_mm = COMB_X0_MM + xc_mm + j * pitch
            x_data = (x_mm - COMB_X0_MM) / axis_mm * t_max
            h = (dg + 1.0) * STRIP_LEVEL_MM   # min 0.2mm: 0-digits must survive rasterization
            w_data = STRIP_BAR_W_MM / axis_mm * t_max
            ax_comb.add_patch(plt.Rectangle(
                (x_data - w_data / 2, ytop - h), w_data, h,
                facecolor="k", edgecolor="none", zorder=3))
        xc_mm += w_need + pitch
        bars_used += len(seq)
        drawn.append(seq)
    overflow_dropped = sum(1 for (_g, seq, _n, _c) in items
                           if len(seq) > 6 and seq[2] == 3
                           and seq not in drawn)
    if dropped or aggregated_slots:
        import warnings
        warnings.warn(
            f"packet strip capacity exceeded: {aggregated_slots} "
            f"sub-resolution slot(s) folded into aggregate packet(s)"
            + (f", {dropped} lost entirely" if dropped else "")
            + (", aggregate packet(s) unencoded" if overflow_dropped
               else ""))

    for lt in STRIP_LANE_TOPS:
        ax_comb.axhline(lt, color="0.75", lw=0.3, zorder=1)

    # ---- optional calibration bar (Figure 2 / anatomy only) ----------
    # One reference bar far left of x = 0, standing on the comb baseline
    # (band bottom edge, data-y 0) with its count printed below. Drawn
    # in 0.65 gray -- the same shade as the axis divider -- so it is a
    # visible reference yet invisible to both decoders (below the raster
    # binarize threshold and outside the SVG #000000 bar filter). Its
    # height uses the encoder's own tooth mapping: count * PITCH_MM.
    if calib_count is not None and not bare_band:
        cx = -hdr_data + 1.5 * step_data   # far left of x = 0, off all lanes
        ax_comb.plot([cx - 1.0 * step_data, cx + 1.0 * step_data],
                     [0.0, 0.0], color="0.65", lw=0.8, zorder=2)
        ax_comb.add_patch(plt.Rectangle(
            (cx - 0.5 * step_data, 0.0), step_data,
            calib_count * PITCH_MM,
            facecolor="0.65", edgecolor="none", zorder=3))
        ax_comb.text(cx, -1.0, f"{calib_count:g}", ha="center", va="top",
                     fontsize=6, color="0.35", clip_on=False, zorder=4)

    if G > 1:
        ax_main.legend(fontsize=7, frameon=False)

    # x tick labels + axis title between the KM axes and the comb band
    # (outside decoded geometry); order: axis -> ticks -> Time -> comb
    for tv in ax_main.get_xticks():
        if 0 <= tv <= t_max:
            fig.text(COMB_X0_MM / FIG_W_MM
                     + tv / t_max * (COMB_X1_MM - COMB_X0_MM) / FIG_W_MM,
                     1 - (MAIN_BOT_MM + 1.8) / FIG_H_MM,
                     f"{tv:g}", ha="center", va="top", fontsize=7)
    fig.text((COMB_X0_MM + COMB_X1_MM) / 2 / FIG_W_MM,
             1 - (MAIN_BOT_MM + 5.5) / FIG_H_MM,
             "Time", ha="center", va="top", fontsize=8)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    # capacity utilization: packet-strip bars actually used vs the strip's
    # total single-lane-equivalent capacity (lanes x usable width / pitch)
    n_packed = sum(len(p) for p in packets.values())
    lane_w = [axis_mm - 1.0] + [axis_mm + HEADER_MM - 3.0] * (STRIP_LANES - 1)
    cap_bars = int(sum(lane_w) / pitch)
    return {"t_max": t_max, "cap": cap,
            "packed_slots": n_packed,
            "dropped_slots": dropped,
            "aggregated_slots": aggregated_slots,
            "aggregate_packets": len(overflow) - overflow_dropped,
            "aggregate_packets_lost": overflow_dropped,
            "strip_bars_used": bars_used,
            "strip_bar_pitch_mm": pitch,
            "strip_bar_capacity": cap_bars,
            "capacity_utilization": (bars_used / cap_bars
                                     if cap_bars else 0.0)}


# ---------------------------------------------------------------------------
# comparators (renderers used in the benchmark)
# ---------------------------------------------------------------------------

def _km_axes(groups, path, dpi, censor_marks=False, risk_table=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_max = max(float(g.times[-1]) for g in groups)
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=dpi)
    ax = fig.add_axes([COMB_X0_MM / FIG_W_MM, 0.28 if risk_table else 0.12,
                       (COMB_X1_MM - COMB_X0_MM) / FIG_W_MM,
                       0.6 if risk_table else 0.8])
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.05)
    for gi, g in enumerate(groups):
        s = g.km_survival()
        tt = np.concatenate([[0.0], g.times])
        ss = np.concatenate([[1.0], s])
        gs_col = ["k", "0.45", "0.65", "0.3"][gi % 4]
        gs_ls = ["-", "--", "-.", ":"][gi % 4]
        line, = ax.step(tt, ss, where="post", lw=1.4, color=gs_col, ls=gs_ls,
                        label=g.label or f"group {gi + 1}")
        if censor_marks:
            ct = g.times[g.c > 0]
            cs = np.interp(ct, tt, ss)
            ax.plot(ct, cs, "|", color=line.get_color(), ms=5)
    if len(groups) > 1:
        ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    if risk_table:
        ticks = ax.get_xticks()
        ticks = ticks[(ticks >= 0) & (ticks <= t_max)]
        tbl = []
        for g in groups:
            row = []
            for tv in ticks:
                idx = np.searchsorted(g.times, tv, side="right") - 1
                # at-risk AT the tick = risk set after the last past event
                row.append(int(g.n[idx] - g.d[idx] - g.c[idx])
                           if idx >= 0 else g.n0)
            tbl.append(row)
        ax.table(cellText=tbl, cellLoc="center", loc="bottom",
                 bbox=[0, -0.45, 1, 0.35])
        fig.texts  # noqa
        for gi, g in enumerate(groups):
            fig.text(0.02, 0.13 - gi * 0.035,
                     g.label or f"group {gi + 1}", fontsize=6)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def render_comparator(groups, path, kind, dpi=150):
    if kind == "km":
        _km_axes(groups, path, dpi)
    elif kind == "km_censor":
        _km_axes(groups, path, dpi, censor_marks=True)
    elif kind == "km_risktable":
        _km_axes(groups, path, dpi, risk_table=True)
    elif kind == "comb":
        render_survival_comb(groups, path, dpi=dpi)
    else:
        raise ValueError(kind)


# ---------------------------------------------------------------------------
# decoder (raster -> EventProcess)
# ---------------------------------------------------------------------------

@dataclass
class DecodeResult:
    groups: list[EventProcess]
    collisions: int = 0          # merged clusters (times closer than resolution)
    warnings: list[str] = field(default_factory=list)
    status: str = "OK"           # OK | PARTIAL (capacity/desync losses)

    def finalize(self) -> "DecodeResult":
        """Set decode status and run internal consistency checks."""
        if self.warnings or self.collisions:
            self.status = "PARTIAL"
        for v in consistency_violations(self.groups):
            self.warnings.append(f"consistency: {v}")
            self.status = "PARTIAL"
        return self


def consistency_violations(groups) -> list[str]:
    """Deterministic checks on a decoded event process:
    n non-increasing, n_{i+1} = n_i - d_i - c_i, d_i,c_i >= 0, n_i >= d_i."""
    out = []
    for gi, g in enumerate(groups):
        n = np.asarray(g.n); d = np.asarray(g.d); c = np.asarray(g.c)
        if np.any(d < 0) or np.any(c < 0):
            out.append(f"group {gi}: negative d/c")
        if np.any(n < d):
            out.append(f"group {gi}: n_i < d_i")
        if len(n) > 1:
            if np.any(np.diff(n) > 0):
                out.append(f"group {gi}: n not non-increasing")
            bad = np.nonzero(n[1:] != n[:-1] - d[:-1] - c[:-1])[0]
            if len(bad):
                out.append(
                    f"group {gi}: n transition violated at "
                    f"{len(bad)} slot(s) (first idx {int(bad[0])})")
    return out


def _binarize(img):
    from PIL import Image
    import numpy as np
    im = Image.open(img)
    # all geometry is calibrated from the measured frame, so the decoder is
    # resolution-invariant; thin bars degrade only below ~1 px/level
    g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    # Otsu threshold
    hist, edges = np.histogram(g, bins=256, range=(0, 1))
    w = hist / hist.sum()
    mu = np.cumsum(w * np.arange(256))
    mu_t = mu[-1]
    var = (mu_t * np.cumsum(w) - mu) ** 2 / (np.cumsum(w) * (1 - np.cumsum(w)) + 1e-12)
    thr = edges[np.argmax(var)] / 1.0
    thr = min(max(thr, 0.25), 0.6)     # keep dark ink, drop light gray guides
    return g < thr, 1.0 - g            # binary ink + ink darkness [0,1]


def _clusters(mask_cols):
    """indices -> list of (start, end) contiguous runs."""
    idx = np.flatnonzero(mask_cols)
    if len(idx) == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    ends = np.concatenate([idx[cuts], [idx[-1]]])
    return list(zip(starts, ends))


def decode_survival_comb(img_path, n_groups: int = 1) -> DecodeResult:
    """Decode a Survival Comb raster back to event processes."""
    ink, ink_dark = _binarize(img_path)
    H, W = ink.shape
    res = DecodeResult(groups=[])

    # --- locate main-axes frame: strongest long horizontal & vertical lines
    row_ink = ink.sum(axis=1)
    col_ink = ink.sum(axis=0)
    hlines = np.where(row_ink > 0.5 * W)[0]
    vlines = np.where(col_ink > 0.25 * H)[0]
    if len(hlines) == 0 or len(vlines) == 0:
        res.warnings.append("frame not found")
        return res.finalize()
    # main-axes top spine: the hline followed ~0.59*H below by the bottom
    # spine (MAIN_BOT_MM - MAIN_TOP_MM of FIG_H_MM). Dense strip bars can
    # merge into spurious long lines above it, so verify the pairing.
    main_h = (MAIN_BOT_MM - MAIN_TOP_MM) / FIG_H_MM
    y_main_top = int(hlines[0])
    y_main_bot = 0
    for i in range(len(hlines)):
        later = hlines[hlines > hlines[i]]
        pair = np.abs(later - hlines[i] - main_h * H) < 0.04 * H
        if len(later) and np.any(pair):
            y_main_top = int(hlines[i])
            y_main_bot = int(later[np.flatnonzero(pair)[0]])
            break
    x0 = int(vlines[0]); x1 = int(vlines[-1])
    ppmm = (x1 - x0) / (COMB_X1_MM - COMB_X0_MM)          # px per mm (x)
    # calibrate y from BOTH main-axes spines: anchoring on the top spine
    # alone lets a ~1 px detection offset grow into ~1 mm of band drift at
    # the strip, 100 mm below; two-point calibration is stable there
    if y_main_bot > y_main_top:
        ppmm_y = (y_main_bot - y_main_top) / (MAIN_BOT_MM - MAIN_TOP_MM)
    else:
        ppmm_y = y_main_top / MAIN_TOP_MM                # px per mm (y)
    if ppmm <= 0 or ppmm_y <= 0:
        res.warnings.append("bad calibration")
        return res.finalize()

    # band extents from canonical geometry (robust to anisotropic scaling)
    band_top = y_main_top - int(round((MAIN_TOP_MM - COMB_BAND_TOP_MM) * ppmm_y))
    band_bot = y_main_top - int(round((MAIN_TOP_MM - COMB_BAND_BOT_MM) * ppmm_y))
    rows_bot = y_main_top - int(round((MAIN_TOP_MM - COMB_ROWS_BOT_MM) * ppmm_y))
    strip_top = y_main_top - int(round((MAIN_TOP_MM - COMB_STRIP_TOP_MM) * ppmm_y))
    band_h_mm = COMB_ROWS_BOT_MM - COMB_BAND_TOP_MM

    # --- header: clusters left of x0 inside the band
    # header digits occupy the left margin above the packet strip
    hdr = ink[band_top:strip_top + 1, max(0, x0 - int(HEADER_MM * ppmm) - 2):x0 - 1]
    hcols = hdr.sum(axis=0)
    hcl = _clusters(hcols > 0)
    heights = []
    for a, b in hcl:
        colrun = hdr[:, a:b + 1].any(axis=1)
        ys = np.flatnonzero(colrun)
        heights.append((ys.max() - ys.min() + 1) / ppmm_y if len(ys) else 0.0)
    digits = [int(round(h / LEVEL_MM - 0.5)) for h in heights]
    if len(digits) < 5 + 4 * n_groups:
        res.warnings.append(f"header underrun: {len(digits)} bars")
        return res.finalize()
    sig = digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]
    e = digits[4] - 4
    t_max = sig * 10.0 ** e
    n0s = []
    p = 5
    for g in range(n_groups):
        n0s.append(digits[p] * 1000 + digits[p + 1] * 100 +
                   digits[p + 2] * 10 + digits[p + 3])
        p += 4

    # --- group rows: equal division of the band (layout is deterministic)
    band_h_px = rows_bot - band_top
    row_ranges = [(int(round(gi * band_h_px / n_groups)),
                   int(round((gi + 1) * band_h_px / n_groups)) - 1)
                  for gi in range(n_groups)]

    half_px = band_h_mm / n_groups / 2.0 * ppmm_y
    # encoder nudges t~0 teeth right of the divider; skip nearer ink
    t_floor_dec = (TOOTH_W_MM + 0.2) * ppmm / (x1 - x0) * t_max * 0.5

    row_slots = {}
    ink2 = ink.copy()
    ink2[:, max(0, x0 - 1):x0 + 2] = False   # residual divider ink guard
    ink2[:, max(0, x1 - 1):x1 + 2] = False
    xm = 0                            # slice starts at x0 absolute
    for gi, (ra, rb) in enumerate(row_ranges):
        centre = (ra + rb) / 2.0
        c_abs = band_top + centre
        # tooth bases are anchored on the row-centre guide line. Measure
        # the baseline directly: the centre guide is drawn at 0.75 gray,
        # below the dark-ink binarize threshold, so it survives as a
        # detectable horizontal line across the row. d_i teeth extend
        # above it, c_i teeth below; ink above/below B gives each side's
        # height. Raster straddle of ~2 px is absorbed by excluding the
        # baseline row itself and requiring >= 0.5*PITCH_MM of ink.
        from PIL import Image
        g = np.asarray(Image.open(img_path).convert("L"), dtype=float) / 255.0
        gx0, gx1 = x0, x1 + 1
        guide = (g > 0.68) & (g < 0.85)
        gy = np.arange(ra, rb + 1)
        gscore = guide[band_top + ra: band_top + rb + 1, gx0:gx1].sum(axis=1)
        cand = gy[gscore > 0.4 * (gx1 - gx0)]
        if len(cand):
            gcl = _clusters(np.abs(cand - centre) <= 2.5)
            gcl = [cc for cc in gcl if cc[0] < len(cand)]
            if gcl:
                cc = gcl[0]
                B = float(cand[cc[0]:cc[1] + 1].max())
            else:
                B = float(centre)
        else:
            B = float(centre)
        half_mm = band_h_mm / n_groups / 2.0
        row_ink = ink2[band_top + ra: band_top + rb + 1, x0:x1 + 1]
        d_map = {}
        c_map = {}
        cl = _clusters(row_ink.sum(axis=0) > 0)
        tooth_w = TOOTH_W_MM * ppmm
        floor_mm = 0.5 * PITCH_MM
        i = 0
        while i < len(cl):
            a, b = cl[i]
            width = b - a + 1
            if width > 1.45 * tooth_w:
                res.collisions += 1
            colrun = row_ink[:, a:b + 1].any(axis=1)
            runs = _clusters(colrun)
            run = min(runs, key=lambda rc: min(
                abs(rc[0] - B), abs(rc[1] - B)))
            Brel = int(round(B)) - ra
            u_px = int(np.clip(Brel - run[0], 0, None))
            d_px = int(np.clip(run[1] - Brel, 0, None))
            hu = u_px / ppmm_y
            hd = d_px / ppmm_y
            t = (xm + (a + b) / 2.0) / (x1 - x0) * t_max
            if t < t_floor_dec:
                i += 1
                continue          # encoder nudges t~0 teeth right
            sides = []
            if hu >= floor_mm:
                sides.append((hu, d_map))
            if hd >= floor_mm:
                sides.append((hd, c_map))
            for side_h, store in sides:
                if side_h > 0.85 * half_mm:
                    # sentinel: next DIGIT_K clusters are digit bars
                    digs = []
                    for k in range(1, DIGIT_K + 1):
                        if i + k < len(cl):
                            a2, b2 = cl[i + k]
                            cr = row_ink[:, a2:b2 + 1].any(axis=1)
                            r2 = min(_clusters(cr), key=lambda rc: min(
                                abs(rc[0] - Brel), abs(rc[1] - Brel)))
                            hh = (r2[1] - r2[0] + 1) / ppmm_y
                            digs.append(int(round(hh / LEVEL_MM - 0.5)))
                    val = 0
                    for dg in digs:
                        val = val * 10 + max(0, min(9, dg))
                    i += DIGIT_K
                else:
                    val = max(1, int(round(side_h / PITCH_MM)))
                store.setdefault(round(t, 6), 0)
                store[round(t, 6)] += val
            i += 1
        all_t = sorted(set(d_map) | set(c_map))
        row_slots[gi] = {t: (d_map.get(t, 0), c_map.get(t, 0))
                        for t in all_t}
        times = np.array(all_t)
        d = np.array([d_map.get(t, 0) for t in all_t], int)
        c = np.array([c_map.get(t, 0) for t in all_t], int)
        n = np.zeros(len(all_t), int)
        risk = n0s[gi]
        for i in range(len(all_t)):
            n[i] = risk
            risk -= d[i] + c[i]
        res.groups.append(EventProcess(times, n, d, c, n0s[gi]))

    # ---- overflow strip: packed (sub-resolution) slots ----------------
    # lane k top from-top = STRIP_TOP + 0.3 + k*PITCH; bars hang <= 2.375mm
    bars = []
    lane_rows = []   # (row_top_px, row_bot_px, x_start_px)
    for k in range(STRIP_LANES):
        top_mm = COMB_STRIP_TOP_MM + 0.3 + k * STRIP_LANE_PITCH_MM - 0.15
        # exclusive window: bars hang <= (9.5 * STRIP_LEVEL_MM) from the
        # lane top; keep the row inside the lane pitch so the next lane's
        # bars never bleed in
        # the window reaches to just above the next lane top: enough room
        # for a full-height bar (2.0 mm below a lane top rendered with a
        # ~0.2 mm rasterization offset), without admitting next-lane ink
        bot_mm = (COMB_STRIP_TOP_MM + 0.3 + k * STRIP_LANE_PITCH_MM
                  + STRIP_LANE_PITCH_MM - 0.02)
        r0 = y_main_top - int(round((MAIN_TOP_MM - top_mm) * ppmm_y))
        r1 = y_main_top - int(round((MAIN_TOP_MM - bot_mm) * ppmm_y))
        # lanes 1+ start in the left header margin (see encoder lane_start)
        xs = x0 if k == 0 else x0 - int(round((HEADER_MM - 3.0) * ppmm))
        lane_rows.append((r0, r1, max(0, xs)))
    for r0, r1, xs in lane_rows:
        strip = ink2[r0:r1, xs:x1 + 1]
        dark = ink_dark[r0:r1, xs:x1 + 1]
        scl = _clusters(strip.sum(axis=0) > 0)
        for a, b in scl:
            row_mask = strip[:, a:b + 1].any(axis=1)
            if not row_mask.any():
                continue
            # the bar is the first contiguous ink run hanging from the lane
            # top; a rasterization offset at the lane top must not truncate
            # tall bars at the window bottom, so measure the run itself
            runs = _clusters(row_mask)
            # keep the run whose top is nearest the lane top; a run whose
            # top sits deeper than ~0.6 mm is the next lane's bar bleeding
            # into the bottom margin of this window, not a bar of this lane
            top_i = min(runs, key=lambda rc: rc[0])
            if top_i[0] / ppmm_y > 1.4:
                continue
            ink_run = strip[top_i[0]:top_i[1] + 1, a:b + 1]
            dark_run = dark[top_i[0]:top_i[1] + 1, a:b + 1]
            hh = float(dark_run.sum()) / (b - a + 1) / ppmm_y
            # slivers of the previous lane's tall bars can bleed ~0.1 mm
            # into the top of this window; a real bar is >= 0.2 mm tall
            if hh < 0.8 * STRIP_LEVEL_MM:
                continue
            bars.append(int(round(hh / STRIP_LEVEL_MM - 1.0)))
    pk_by_group = {gi: [] for gi in range(n_groups)}
    p = 0
    npk = 0
    resyncs = 0
    while p + 3 <= len(bars):
        gi_raw, cnt, mode = bars[p], bars[p + 1], bars[p + 2]
        if (not (1 <= gi_raw <= n_groups)
                or not (1 <= cnt <= PACKET_MAX_SLOTS)
                or mode not in (0, 1, 2, 3)):
            # desynced: skip one bar and retry (a fused/dropped bar shifts the
            # stream); give up only after many fruitless skips
            if resyncs > len(bars):
                break
            p += 1
            resyncs += 1
            continue
        if p + 6 > len(bars):
            break
        base = bars[p + 3] * 100 + bars[p + 4] * 10 + bars[p + 5]
        slot_len = 1 + {0: 1, 1: 2, 2: 4}.get(mode, 0)
        need = 8 if mode == 3 else slot_len * cnt
        if p + 6 + need + 1 >= len(bars):
            break
        seg = bars[p + 6:p + 6 + need]
        # 2-digit checksum: drop corrupt parses instead of emitting bad slots
        chk = 10 * bars[p + 6 + need] + bars[p + 6 + need + 1]
        if (sum(bars[p:p + 6 + need]) + chk) % 100 != 0:
            if resyncs > len(bars):
                break
            p += 1
            resyncs += 1
            continue
        cand = []
        if mode == 3:
            # aggregate overflow packet: 4-digit d, 4-digit c, no offset
            pq = base * (PACKET_POS_MM / PACKET_TICK_MM)
            dv = int("".join(map(str, seg[:4])))
            cv = int("".join(map(str, seg[4:8])))
            cand.append((pq, dv, cv))
            res.warnings.append("aggregate overflow packet decoded")
        for k in range(cnt if mode != 3 else 0):
            b = slot_len * k
            # composite position: base (0.15 mm quanta) + off (0.05 mm)
            pq = base * (PACKET_POS_MM / PACKET_TICK_MM) + seg[b]
            if mode == 0:
                v = seg[b + 1] - 1
                dv, cv = v // 3, v % 3
            elif mode == 1:
                dv, cv = seg[b + 1], seg[b + 2]
            else:
                dv = seg[b + 1] * 10 + seg[b + 2]
                cv = seg[b + 3] * 10 + seg[b + 4]
            cand.append((pq, dv, cv))
        # slot positions are non-decreasing within a group's stream (equal
        # quanta allowed); a decrease means a spurious header — resync
        prev = pk_by_group[gi_raw - 1]
        if (any(cand[k][0] < cand[k - 1][0] for k in range(1, len(cand)))
                or (prev and cand[0][0] < prev[-1][0] - 9)):
            if resyncs > len(bars):
                break
            p += 1
            resyncs += 1
            continue
        if prev and cand and cand[0][0] < prev[-1][0]:
            cand[0] = (prev[-1][0], cand[0][1], cand[0][2])
        pk_by_group[gi_raw - 1].extend(cand)
        p += 6 + need + 2
        npk += 1
    if resyncs:
        res.warnings.append(f"packet desync x{resyncs}")
    if npk:
        pos_data = PACKET_TICK_MM * ppmm / (x1 - x0) * t_max
        new_groups = []
        # collision window in data-time units: packed slots follow their
        # burst tooth within ~0.4 mm; decode noise is ~0.15 mm (tooth
        # centre) + 0.03 mm (packet quant), so search a bounded window
        lo_w = 0.45 * pos_data / PACKET_TICK_MM
        hi_w = 0.20 * pos_data / PACKET_TICK_MM
        for gi in range(n_groups):
            slots = dict(row_slots.get(gi, {}))
            pk = deque(pk_by_group[gi])
            merged = []
            rowq = deque(sorted(slots))
            row_times = list(rowq)
            while rowq or pk:
                t_r = rowq[0] if rowq else np.inf
                t_p = (pk[0][0] * pos_data) if pk else np.inf
                if pk and t_p <= t_r:
                    pq, dv, cv = pk.popleft()
                    t = round(pq * pos_data, 6)
                    # a packed slot chronologically follows its burst
                    # tooth; quantization may place it slightly off — snap
                    # to just after the nearest drawn tooth in the window
                    j0 = bisect.bisect_left(row_times, t - lo_w)
                    j1 = bisect.bisect_right(row_times, t + hi_w)
                    below = [u for u in row_times[j0:j1] if u <= t]
                    if not below and j1 > j0:
                        # no drawn tooth below within the collision
                        # window: the burst tooth decoded late — snap the
                        # packet to just after the nearest tooth above
                        t_b = min(row_times[j0:j1], key=lambda u: u - t)
                        t = round(t_b + 1e-4, 6)
                    merged.append((t, dv, cv))
                else:
                    t = round(rowq.popleft(), 6)
                    dv, cv = slots[t]
                    merged.append((t, dv, cv))
            merged.sort()
            # spread slots sharing the same decoded time quantum by a
            # sub-quantum step (see decode_survival_comb_svg)
            jit = 0.4 * pos_data   # sub-tick jitter in data units
            for j in range(1, len(merged)):
                if merged[j][0] <= merged[j - 1][0] + 1e-9:
                    merged[j] = (merged[j - 1][0] + jit,
                                 merged[j][1], merged[j][2])
            all_t = [t for t, _, _ in merged]
            times = np.array(all_t)
            d = np.array([dv for _, dv, _ in merged], int)
            c = np.array([cv for _, _, cv in merged], int)
            n = np.zeros(len(all_t), int)
            risk = n0s[gi]
            for i in range(len(all_t)):
                n[i] = risk
                risk -= d[i] + c[i]
            new_groups.append(EventProcess(times, n, d, c, n0s[gi]))
        res.groups = new_groups
    return res.finalize()


# ---------------------------------------------------------------------------
# pseudo-IPD expansion (Capability A output)
# ---------------------------------------------------------------------------

def to_pseudo_ipd(groups):
    """Expand an event-process table into analysis-equivalent de-identified
    pseudo-IPD rows: one row per (slot, event multiplicity). Ordering within
    a time slot is arbitrary; synthetic_id carries no original identity."""
    rows = []
    sid = 0
    for gi, g in enumerate(groups):
        for t, d, c in zip(g.times, g.d, g.c):
            for _ in range(int(d)):
                sid += 1
                rows.append({"synthetic_id": sid, "group": g.label or gi + 1,
                             "time": float(t), "status": 1})
            for _ in range(int(c)):
                sid += 1
                rows.append({"synthetic_id": sid, "group": g.label or gi + 1,
                             "time": float(t), "status": 0})
    return rows


# ---------------------------------------------------------------------------
# vector (SVG) decode — Path A/B: structure-only, no metadata involved
# ---------------------------------------------------------------------------

_PT_PER_MM = 72.0 / 25.4


def _svg_filled_rects(path):
    """Return filled axis-aligned rects in mm-from-top-left of figure.
    Handles both direct filled <path> rects and matplotlib's deduplicated
    <use xlink:href="#id" .../> glyph references."""
    import re
    s = open(path).read()
    rects = []

    def _bbox(d):
        nums = [float(v) for v in re.findall(r'-?[\d.]+(?:e-?\d+)?', d)]
        if len(nums) < 4:
            return None
        xs, ys = nums[0::2], nums[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    # referenced glyph paths
    glyphs = {}
    for m in re.finditer(r'<path id="([^"]+)" d="([^"]+)"[^>]*/?>', s):
        bb = _bbox(m.group(2))
        if bb:
            glyphs[m.group(1)] = bb
    # <use> instances (x/y attrs or translate())
    for m in re.finditer(r'<use xlink:href="#([^"]+)"([^>]*)/>', s):
        ref = m.group(1)
        if ref not in glyphs or ref.startswith("DejaVu"):
            continue
        rest = m.group(2)
        if 'style="fill:' not in rest and 'fill' not in rest:
            st = re.search(r'style="([^"]*)"', rest)
            if st and "stroke" in st.group(1) and "fill" not in st.group(1):
                continue
        attrs = dict(re.findall(r'([\w:]+)="([^"]*)"', rest))
        tx = float(attrs.get("x", 0.0))
        ty = float(attrs.get("y", 0.0))
        tm = re.search(r'translate\(([\d.\-e]+)[ ,]+([\d.\-e]+)', rest)
        if tm:
            tx, ty = float(tm.group(1)), float(tm.group(2))
        x0, y0, x1, y1 = glyphs[ref]
        w = (x1 - x0) / _PT_PER_MM
        h = (y1 - y0) / _PT_PER_MM
        if w > 0 and h > 0:
            rects.append(dict(x=(x0 + tx) / _PT_PER_MM,
                              y=(y0 + ty) / _PT_PER_MM, w=w, h=h,
                              fill=attrs.get("style", "")))

    # direct paths: matplotlib emits the bars as <path ... clip-path=...>
    # with default (black) fill and no id/style
    for m in re.finditer(r'<path\s+([^>]*?)/?>', s):
        tag = m.group(1)
        if 'id=' in tag or 'fill: none' in tag:
            continue
        dm = re.search(r'd="([^"]+)"', tag)
        if not dm:
            continue
        fm = re.search(r'fill: (#[0-9a-fA-F]+)', tag)
        bb = _bbox(dm.group(1))
        if not bb:
            continue
        x0, y0, x1, y1 = bb
        w = (x1 - x0) / _PT_PER_MM
        h = (y1 - y0) / _PT_PER_MM
        if w > 0 and h > 0:
            rects.append(dict(x=x0 / _PT_PER_MM, y=y0 / _PT_PER_MM,
                              w=w, h=h,
                              fill=fm.group(1) if fm else "#000000"))
    return rects


def decode_survival_comb_svg(path):
    """Decode a Survival Comb SVG purely from vector geometry (no metadata).
    Returns DecodeResult. Compatible with metadata-stripped SVG (Path B):
    the figure never writes metadata, so this is identical."""
    axis_mm = COMB_X1_MM - COMB_X0_MM
    band_h = COMB_BAND_BOT_MM - COMB_BAND_TOP_MM
    rects = _svg_filled_rects(path)
    # keep ink bars: thin, dark, inside comb band x-range
    def is_bar(r, wlo, whi):
        return (wlo <= r["w"] <= whi and 0.05 <= r["h"] <= band_h + 2
                and r["fill"].lower() in ("#000000", "#000"))
    bars = [r for r in rects if is_bar(r, 0.1, 0.7)]
    hdr = [r for r in bars if r["x"] + r["w"] < COMB_X0_MM - 0.5
           and r["y"] < COMB_STRIP_TOP_MM]
    body = [r for r in bars if r["x"] >= COMB_X0_MM - 0.3]
    strip = [r for r in bars if COMB_STRIP_TOP_MM <= r["y"] <= COMB_BAND_BOT_MM + 1]

    # ---- header -> t_max, N0s --------------------------------------------
    hdr.sort(key=lambda r: r["x"])
    x_start = COMB_X0_MM - HEADER_MM + 0.5      # hx anchor (mm)
    def _digits_at(seq, base_x, n):
        out = []
        for j in range(n):
            x = base_x + j * 1.0
            cand = [r for r in seq if abs(r["x"] - x) < 0.45]
            if not cand:
                raise ValueError("svg header digit missing")
            r = min(cand, key=lambda r: abs(r["x"] - x))
            out.append(int(round(r["h"] / LEVEL_MM - 0.5)))
        return out
    try:
        sig = _digits_at(hdr, x_start, 4)
        exp = _digits_at(hdr, x_start + 4.0, 1)[0] - 4
        t_max = _digits_to_int(sig) * 10.0 ** exp
        rest = hdr[len([r for r in hdr if r["x"] < x_start + 5.0]):]
        # group N0s follow at stride 5mm
        n0s = []
        bx = x_start + 5.0
        while any(abs(r["x"] - bx) < 0.45 for r in hdr):
            try:
                d4 = _digits_at(hdr, bx, 4)
            except ValueError:
                break
            n0s.append(_digits_to_int(d4))
            bx += 5.0
    except Exception as e:
        raise ValueError(f"svg header decode failed: {e}")
    G = len(n0s)
    row_mm = (COMB_ROWS_BOT_MM - COMB_BAND_TOP_MM) / G
    half_mm = row_mm / 2.0
    cap = int(half_mm / PITCH_MM) - 1

    # ---- row teeth -------------------------------------------------------
    groups_out = []
    strip_used = [False] * len(strip)
    t_floor = (TOOTH_W_MM + 0.2) / axis_mm * t_max
    for gi in range(G):
        centre = COMB_BAND_TOP_MM + gi * row_mm + half_mm
        teeth = [r for r in body
                 if COMB_BAND_TOP_MM + gi * row_mm <= r["y"] + 0.2
                 <= COMB_BAND_TOP_MM + (gi + 1) * row_mm
                 or (r["y"] <= centre <= r["y"] + r["h"])]
        up = sorted([r for r in teeth if abs(r["y"] + r["h"] - centre) < 0.3],
                    key=lambda r: r["x"])
        dn = sorted([r for r in teeth if abs(r["y"] - centre) < 0.3],
                    key=lambda r: r["x"])
        def read_row(row_bars):
            vals = []
            i = 0
            while i < len(row_bars):
                r = row_bars[i]
                t = (r["x"] + r["w"] / 2 - COMB_X0_MM) / axis_mm * t_max
                if t < t_floor * 0.5:
                    i += 1
                    continue
                n_un = int(round(r["h"] / PITCH_MM))
                if n_un <= cap:
                    vals.append((t, n_un))
                else:
                    # sentinel + DIGIT_K digit bars at +0.8mm (consumed, not
                    # read as teeth)
                    digs = []
                    for k in range(DIGIT_K):
                        xk = r["x"] + (k + 1) * 0.8
                        cand = [b for b in row_bars
                                if abs(b["x"] - xk) < 0.4]
                        if cand:
                            digs.append(int(round(cand[0]["h"] / LEVEL_MM - 0.5)))
                    vals.append((t, _digits_to_int(digs) if digs else n_un))
                    i += DIGIT_K      # skip the digit bars
                i += 1
            return vals
        d_rows = read_row(up)
        c_rows = read_row(dn)
        # pair by x-time proximity (counts may differ between halves)
        times, dd, cc = [], [], []
        used = set()
        for t, dv in d_rows:
            best = None
            for ci, (tc, cv) in enumerate(c_rows):
                if ci in used:
                    continue
                if abs(tc - t) < 0.05 / axis_mm * t_max and (best is None
                                            or abs(tc - t) < abs(c_rows[best][0] - t)):
                    best = ci
            times.append(t)
            dd.append(dv)
            if best is None:
                cc.append(0)
            else:
                used.add(best)
                cc.append(c_rows[best][1])
        for ci, (tc, cv) in enumerate(c_rows):
            if ci not in used:
                times.append(tc)
                dd.append(0)
                cc.append(cv)
        order = np.argsort(times)
        times = np.asarray(times)[order]
        dd = np.asarray(dd)[order]
        cc = np.asarray(cc)[order]
        groups_out.append(EventProcess(np.asarray(times),
                                       _risk_sets(n0s[gi], dd, cc),
                                       np.asarray(dd), np.asarray(cc),
                                       n0s[gi]))

    # ---- packet strip ----------------------------------------------------
    for r in strip:
        # rendered lane tops, mm from figure top (bars hang down from the
        # lane line; lane order matches encoder emission order)
        lane = min(range(STRIP_LANES),
                   key=lambda k: abs(r["y"]
                                     - (COMB_STRIP_TOP_MM + 0.3
                                        + k * STRIP_LANE_PITCH_MM)))
        r["lane"] = lane
        r["digit"] = int(round(r["h"] / STRIP_LEVEL_MM - 1.0))
    strip.sort(key=lambda r: (r["lane"], r["x"]))
    lanes = [[r for r in strip if r["lane"] == k] for k in range(STRIP_LANES)]
    stream = []
    for k, lane_bars in enumerate(lanes):
        # ordering within lane by x; lanes 1+ start in the header margin
        stream += [(r["x"], r["digit"]) for r in lane_bars]
    groups_out2 = groups_out
    # drawn-tooth times per group (snapshot before packets are appended)
    drawn_times = {gi: list(groups_out2[gi].times) for gi in range(G)}
    p = 0
    prev = {gi: -1 for gi in range(G)}
    agg_overflow = False
    while p < len(stream):
        try:
            gid, cnt, mode = stream[p][1], stream[p + 1][1], stream[p + 2][1]
        except IndexError:
            break
        if not (1 <= gid <= G and 1 <= cnt <= PACKET_MAX_SLOTS and 0 <= mode <= 3):
            p += 1
            continue
        if p + 6 > len(stream):
            break
        base = (100 * stream[p + 3][1] + 10 * stream[p + 4][1]
                + stream[p + 5][1])
        slot_len = 1 + {0: 1, 1: 2, 2: 4}.get(mode, 0)
        need = 8 if mode == 3 else slot_len * cnt
        if p + 6 + need + 2 > len(stream):
            break
        # 2-digit checksum: drop corrupt parses instead of emitting bad slots
        chk = 10 * stream[p + 6 + need][1] + stream[p + 6 + need + 1][1]
        if (sum(b[1] for b in stream[p:p + 6 + need]) + chk) % 100 != 0:
            p += 1
            continue
        p_start = p
        p += 6
        ok = True
        buf = []
        prev_buf = -1
        if mode == 3:
            # aggregate overflow packet: no per-slot offsets; one
            # (pq, d_agg, c_agg) record at the packet base position
            pq = base * (PACKET_POS_MM / PACKET_TICK_MM)
            dv = sum(stream[p + k][1] * 10 ** (3 - k) for k in range(4))
            cv = sum(stream[p + 4 + k][1] * 10 ** (3 - k)
                     for k in range(4))
            buf.append((pq, dv, cv))
            agg_overflow = True
            p += 8
        for _s in range(cnt if mode != 3 else 0):
            try:
                pq = (base * (PACKET_POS_MM / PACKET_TICK_MM)
                      + stream[p][1])
                if mode == 0:
                    j = stream[p + 1][1]
                    dv, cv = divmod(max(j - 1, 0), 3)
                    p += 2
                elif mode == 1:
                    dv, cv = stream[p + 1][1], stream[p + 2][1]
                    p += 3
                else:
                    dv = 10 * stream[p + 1][1] + stream[p + 2][1]
                    cv = 10 * stream[p + 3][1] + stream[p + 4][1]
                    p += 5
            except IndexError:
                ok = False
                break
            if pq < prev_buf:
                ok = False          # within-packet decrease = corruption
                break
            if pq < prev[gid - 1]:
                if pq < prev[gid - 1] - 9:
                    ok = False      # beyond a burst window = spurious
                    break
                pq = prev[gid - 1]  # burst straddles a base cell: clamp
            prev_buf = pq
            buf.append((pq, dv, cv))
        if not ok:
            import os
            if os.environ.get('SCDBG'):
                print('RESYNC at', p_start, 'cnt', cnt, 'mode', mode, 'buf', len(buf))
            p = p_start + 1     # spurious header: resync one bar forward
            continue
        p += 2                 # skip the 2 check digits
        for pq, dv, cv in buf:
            prev[gid - 1] = pq
        g = groups_out2[gid - 1]
        drawn = drawn_times[gid - 1]
        lo_w = 0.45 / axis_mm * t_max       # collision window in data time
        hi_w = 0.20 / axis_mm * t_max
        for pq, dv, cv in buf:
            # a packed slot chronologically follows its burst tooth;
            # quantization may place it slightly off — snap to just after
            # the nearest drawn tooth in the window (pq is in
            # PACKET_TICK_MM units here)
            t = (pq * PACKET_TICK_MM) / axis_mm * t_max
            j0 = bisect.bisect_left(drawn, t - lo_w)
            j1 = bisect.bisect_right(drawn, t + hi_w)
            if j1 > j0:
                t_b = min(drawn[j0:j1],
                          key=lambda u: (t - u) if u <= t
                          else 1.6 * (u - t))
                if t_b > t:
                    t = t_b + 1e-4
            g.times = np.append(g.times, t)
            g.d = np.append(g.d, dv)
            g.c = np.append(g.c, cv)
    for g in groups_out2:
        order = np.argsort(g.times, kind="stable")
        times = np.asarray(g.times, dtype=float)[order]
        dd = np.asarray(g.d, dtype=int)[order]
        cc = np.asarray(g.c, dtype=int)[order]
        # spread slots sharing the same decoded time quantum by a
        # sub-quantum step: order is preserved, each keeps its (d, c),
        # and the jitter stays far inside the position tolerance
        jitter = 0.02 * t_max / axis_mm
        for j in range(1, len(times)):
            if times[j] <= times[j - 1] + 1e-9:
                times[j] = times[j - 1] + jitter
        g.times = times
        g.d = dd
        g.c = cc
        g.n = _risk_sets(g.n0, g.d, g.c)
    res = DecodeResult(groups=groups_out2)
    if agg_overflow:
        res.warnings.append("aggregate overflow packet decoded")
    return res.finalize()


def decode_level1(img_or_svg):
    """Level 1 decode: risk-set ticks only -> (t_i, n_i approx) per group.
    Accepts SVG path (vector) or raster path (300 dpi recommended)."""
    if str(img_or_svg).endswith(".svg"):
        rects = _svg_filled_rects(img_or_svg)
        bars = [r for r in rects
                if 0.1 <= r["w"] <= 0.7 and 0.15 <= r["h"] <= 12
                and r["fill"].lower() in ("#000000", "#000")]
        hdr = [r for r in bars if r["x"] + r["w"] < COMB_X0_MM - 0.5]
        hdr.sort(key=lambda r: r["x"])
        x_start = COMB_X0_MM - HEADER_MM + 0.5
        def _dg(base_x, j):
            x = base_x + j
            cand = [r for r in hdr if abs(r["x"] - x) < 0.45]
            return int(round(cand[0]["h"] / LEVEL_MM - 0.5))
        sig = [_dg(x_start, j) for j in range(4)]
        exp = _dg(x_start, 4) - 4
        t_max = _digits_to_int(sig) * 10.0 ** exp
        n0s = []
        bx = x_start + 5.0
        while True:
            ds = []
            for j in range(4):
                x = bx + j
                cand = [r for r in hdr if abs(r["x"] - x) < 0.45]
                if not cand:
                    break
                ds.append(int(round(cand[0]["h"] / LEVEL_MM - 0.5)))
            if len(ds) != 4:
                break
            n0s.append(_digits_to_int(ds))
            bx += 5.0
        G = len(n0s)
        row_mm = (COMB_ROWS_BOT_MM - COMB_BAND_TOP_MM) / G
        half = row_mm / 2
        out = []
        for gi in range(G):
            centre = COMB_BAND_TOP_MM + gi * row_mm + half
            ticks = [r for r in bars
                     if abs(r["y"] + r["h"] - centre) < 0.3]
            rows = []
            for r in sorted(ticks, key=lambda r: r["x"]):
                t = (r["x"] + r["w"] / 2 - COMB_X0_MM) / (COMB_X1_MM - COMB_X0_MM) * t_max
                lev = int(round(r["h"] / RISK_LEVEL_MM))
                rows.append((t, lev / 10.0 * n0s[gi]))
            out.append(rows)
        return out
    raise ValueError("raster L1 decode: use decode_survival_comb path (L2 engine)")
