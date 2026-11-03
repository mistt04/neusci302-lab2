import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------- settings
FS_FILTER = dict(sg_window=21, sg_order=3, median_window=21, boxcar_window=20)  # samples at 2 kHz

# time windows in seconds from the start of each sweep
WINDOWS = {
    # pulses: light on at 0.5 s, off at 1.5 s
    "pulse_on_ref": (0.10, 0.45), "pulse_on_search": (0.50, 0.80),
    "pulse_off_ref": (1.30, 1.50), "pulse_off_search": (1.50, 1.80),
    "pulse_plateau": (1.30, 1.50),
    # ramps: rising ramp starts at 1.0 s, falling ramp starts at 5.0 s
    "rise_ref": (0.20, 0.90), "fall_ref": (4.70, 5.00),
}
RISE_START, FALL_START = 1.0, 5.0
L1_COLOR, L2_COLOR, PR_COLOR = "#B85C38", "#4F8A83", "#2C5F8D"   # terracotta, muted teal, steel blue  # same colours as the annotated diagram

# ---------------------------------------------------------------- plotting style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 11, "mathtext.fontset": "custom", "mathtext.default": "regular", "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "0.2", "axes.labelcolor": "0.1", "xtick.color": "0.2", "ytick.color": "0.2", "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 4, "ytick.major.size": 4, "axes.labelpad": 6, "text.color": "0.1", "figure.dpi": 100, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- loading
def load_labchart_text(path):
    """Return a list of blocks. Each block is an array with columns time, ch1, ch2, ch3...
    One block is one sweep (one 'subpage')."""
    blocks, rows = [], None
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("Interval="):
                if rows:
                    blocks.append(np.array(rows, dtype=float))
                rows = []
            elif rows is not None and line[:1] in "-0123456789":
                parts = line.rstrip("\n").split("\t")
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass
    if rows:
        blocks.append(np.array(rows, dtype=float))
    return blocks


# ---------------------------------------------------------------- filters (numpy only)
def boxcar(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def median(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(np.pad(x, w // 2, mode="edge"), w), axis=1)


def savgol(x, w, order):
    m = w // 2
    A = np.vander(np.arange(-m, m + 1), order + 1, increasing=True)
    c = np.linalg.pinv(A)[0]
    return np.convolve(np.pad(x, m, mode="edge"), c[::-1], mode="valid")


def make_filter(name):
    if name == "sg":
        return lambda x: savgol(x, FS_FILTER["sg_window"], FS_FILTER["sg_order"])
    if name == "median":
        return lambda x: median(x, FS_FILTER["median_window"])
    return lambda x: boxcar(x, FS_FILTER["boxcar_window"])


# ---------------------------------------------------------------- helpers
def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def mean_sem(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def bootstrap_slope(x, y, n=5000, seed=0):
    """Slope of y on x with a 95 percent bootstrap CI (resampling sweeps)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    slope = np.polyfit(x, y, 1)[0]
    s = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        s.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(s, [2.5, 97.5])
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    r2 = 1 - resid.var() / y.var()
    return slope, lo, hi, r2


def light_to_dark(n, lo=0.35, hi=0.95):
    return [plt.cm.Blues(v) for v in np.linspace(lo, hi, n)]


def panel_letter(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-58, 8), textcoords="offset points",
                fontsize=16, fontweight="bold", va="bottom", annotation_clip=False)


# ---------------------------------------------------------------- pulses
def analyse_pulses(blocks, filt):
    sweeps = []
    for b in blocks:
        t, led, erg = b[:, 0], b[:, 1], b[:, 2]
        amp = round(float(np.median(led[win(t, 0.7, 1.3)])), 2)        # command voltage during the pulse
        f = filt(erg)
        w = WINDOWS
        base = erg[win(t, *w["pulse_on_ref"])].mean()
        on = f[win(t, *w["pulse_on_search"])].max() - base               # L1 ON peak, above baseline
        pre = erg[win(t, *w["pulse_off_ref"])].mean()
        off = f[win(t, *w["pulse_off_search"])].min() - pre              # L2 OFF trough, below pre-off level
        plateau = pre - base                                             # photoreceptor plateau depth
        sweeps.append(dict(amp=amp, base=base, on=on, pre_off=pre, off=off, plateau=plateau,
                           t=t, erg=erg - base, led=led, raw=erg, trec=t))
    return sweeps


def smooth_display(y, n=41):
    """Light 20 ms moving average used ONLY to make averaged traces easier to read in a figure."""
    pad = n // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(n) / n, mode="valid")


def _panel_depol(axB, sweeps, amps, by):
    """Photoreceptor depolarization vs log command voltage (shared by the pulse figures)."""
    mag = lambda S: [-s["plateau"] for s in S]
    for a in amps:
        axB.plot([a] * len(by[a]), mag(by[a]), ".", ms=3.5, color=PR_COLOR, alpha=0.25, zorder=1)
    m = [np.mean(mag(by[a])) for a in amps]
    e = [mean_sem(mag(by[a]))[1] for a in amps]
    axB.errorbar(amps, m, e, fmt="o", ms=5, color=PR_COLOR, capsize=2.5, zorder=3)
    allv = np.array([s["amp"] for s in sweeps])
    ally = np.array([-s["plateau"] for s in sweeps])
    slope, lo, hi, r2 = bootstrap_slope(np.log10(allv), ally)
    xs = np.linspace(np.log10(allv).min(), np.log10(allv).max(), 50)
    axB.plot(10 ** xs, np.polyval(np.polyfit(np.log10(allv), ally, 1), xs), "k--", lw=1)
    axB.text(0.04, 0.95, f"R² = {r2:.2f}", transform=axB.transAxes, va="top", fontsize=10.5)
    axB.set_xscale("log")
    axB.set_xlim(0.08, 10)
    axB.set_xticks([0.1, 1, 10])
    axB.set_xticklabels(["0.1", "1", "10"])
    axB.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    axB.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axB.tick_params(axis="x", which="major", length=6)
    axB.tick_params(axis="x", which="minor", length=3)
    axB.grid(axis="x", which="major", color="0.88", lw=0.8, zorder=0)
    axB.set_ylim(0, 0.7)
    axB.set_xlabel("Command voltage (V, log scale)")
    axB.set_ylabel("Photoreceptor depolarization (V)")


def figure_pulses(sweeps, out, filt):
    """Pulse figure. A: average ERG at a subset of amplitudes. B: photoreceptor depolarization vs command voltage (log axis)."""
    amps = sorted(set(s["amp"] for s in sweeps))
    by = {a: [s for s in sweeps if s["amp"] == a] for a in amps}
    show = [a for a in amps if a in (0.1, 0.2, 0.4, 0.7, 1.25, 2, 3, 4, 6)]   # 9 of the 16 voltages, roughly even on a log scale
    tr_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("tr", matplotlib.cm.Blues(np.linspace(0.30, 1.0, 256)))
    v_norm = matplotlib.colors.LogNorm(vmin=min(amps), vmax=max(amps))
    cols = [tr_cmap(v_norm(a)) for a in show]
    fig = plt.figure(figsize=(7.4, 7.9))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.35, 1.0], hspace=0.50,
                     left=0.13, right=0.87, top=0.95, bottom=0.08)
    inner = outer[0].subgridspec(2, 1, height_ratios=[0.60, 1.0], hspace=0.10)

    # A. average trace for each of the 16 amplitudes, LED command above; window centred on the 1-s pulse
    axL = fig.add_subplot(inner[0])
    axA = fig.add_subplot(inner[1], sharex=axL)
    for a, c in zip(show, cols):
        S = by[a]
        t = S[0]["t"] - 0.5
        axL.plot(t, np.mean([s["led"] for s in S], axis=0), color=c, lw=1.3)
        axA.plot(t, smooth_display(np.mean([s["erg"] for s in S], axis=0)), color=c, lw=1.3, label=f"{a:g}")
    axL.set_ylabel("Command voltage (V)")
    axL.set_yscale("symlog", linthresh=0.1, linscale=0.35)   # log spacing so the 16 levels spread apart
    axL.set_ylim(0, 7.5)
    axL.set_yticks([0, 0.1, 0.4, 1.25, 3, 6])
    axL.set_yticklabels(["0", "0.1", "0.4", "1.25", "3", "6"])
    axL.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plt.setp(axL.get_xticklabels(), visible=False)
    axL.tick_params(axis="x", length=0)
    axA.axhline(0, color="0.35", lw=0.6, zorder=0)
    axA.set_xlim(-0.5, 1.5)                                       # pulse (0 to 1 s) sits in the middle of the axis
    axA.set_ylim(-0.92, 0.45)
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("ERG response (V)")
    # direct labels at the right end of five traces, spread apart with short leader lines (no colour bar)
    lab = []
    for a, c in zip(show, cols):
        if a in (0.1, 0.4, 1.25, 3, 6):
            S = by[a]; tt = S[0]["t"] - 0.5
            yy = smooth_display(np.mean([s["erg"] for s in S], axis=0))
            lab.append([a, c, float(yy[np.argmin(abs(tt - 1.5))])])
    gap = 0.16
    ys = [l[2] for l in lab]                       # lightest (top) to darkest (bottom)
    mid = np.mean(ys); n = len(lab)
    ypos = [mid + (n - 1) / 2 * gap - i * gap for i in range(n)]
    for (a, c, y0), y1 in zip(lab, ypos):
        axA.annotate(f"{a:g} V", xy=(1.5, y0), xytext=(1.63, y1), textcoords="data", color=c, fontsize=10,
                     va="center", ha="left", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=1, shrinkB=3))
    panel_letter(axL, "A")

    axB = fig.add_subplot(outer[1])
    _panel_depol(axB, sweeps, amps, by)
    panel_letter(axB, "B")
    fig.savefig(os.path.join(out, "fig_pulse_series.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_pulse_series.pdf"))
    plt.close(fig)


def figure_pulses_boxed(sweeps, out, filt):
    """Boxed pulse figure. A: one box per command voltage (all 16), mean ERG over faint single trials, with the
    1-s light pulse drawn at the top of every box. B: photoreceptor depolarization vs command voltage."""
    amps = sorted(set(s["amp"] for s in sweeps))
    by = {a: [s for s in sweeps if s["amp"] == a] for a in amps}
    tr_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("tr", matplotlib.cm.Blues(np.linspace(0.30, 1.0, 256)))
    v_norm = matplotlib.colors.LogNorm(vmin=min(amps), vmax=max(amps))
    fig = plt.figure(figsize=(7.4, 9.2))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1.0], hspace=0.30,
                     left=0.13, right=0.97, top=0.95, bottom=0.07)
    grid = outer[0].subgridspec(4, 4, hspace=0.12, wspace=0.10)
    first = None
    for k, a in enumerate(amps):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(grid[r, c], sharex=first, sharey=first)
        if first is None:
            first = ax
        col = tr_cmap(v_norm(a))
        S = by[a]
        t = S[0]["t"] - 0.5
        for s_ in S:
            ax.plot(t, smooth_display(s_["erg"]), color=col, lw=0.5, alpha=0.25)
        ax.plot(t, smooth_display(np.mean([s_["erg"] for s_ in S], axis=0)), color=col, lw=1.4)
        ax.hlines(0.36, 0, 1, color="0.15", lw=2.2)                       # the 1-s light pulse
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        ax.text(0.04, 0.05, f"{a:g} V", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color="0.1")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.92, 0.45)
        ax.set_xticks([0, 1]); ax.set_yticks([-0.8, -0.4, 0, 0.4])
        ax.tick_params(labelsize=8.5, length=2.5)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("0.45")
        if c != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        if r != 3:
            plt.setp(ax.get_xticklabels(), visible=False)
        if k == 0:
            panel_letter(ax, "A")
    fig.text(0.55, 0.463, "Time (s)", ha="center", va="top", fontsize=10.5)
    fig.text(0.055, 0.72, "ERG response (V)", rotation=90, ha="center", va="center", fontsize=10.5)
    axB = fig.add_subplot(outer[1])
    _panel_depol(axB, sweeps, amps, by)
    panel_letter(axB, "B")
    fig.savefig(os.path.join(out, "fig_pulse_series_boxed.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_pulse_series_boxed.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- ramps
def analyse_ramps(blocks, filt):
    sweeps = []
    for b in blocks:
        t, led, erg = b[:, 0], b[:, 1], b[:, 2]
        f = filt(erg)
        if led[win(t, 1.03, 1.06)].mean() > 5.3:  # LED jumped to 5.5 V right at 1 s, so this is a FALLING-ramp sweep (a rising ramp is still climbing then)
            kind = "fall"
            ref = erg[win(t, *WINDOWS["fall_ref"])].mean()
            tt = t[(t > FALL_START)]
            ll = led[(t > FALL_START)]
            t_hi = tt[np.argmax(ll < 5.2)]                      # first time below 5.2 V
            t_lo = tt[np.argmax(ll < 2.8)]                      # first time below 2.8 V
            rate = 2.4 / (t_lo - t_hi)                           # V/s
            end = t_lo + 0.1
            m = win(t, FALL_START, end + 1.0)
            amp = f[m].min() - ref                               # L2 trough, below pre-fall level
            t0 = FALL_START
            search = (FALL_START, end + 1.0)
        else:
            kind = "rise"
            ref = erg[win(t, *WINDOWS["rise_ref"])].mean()
            tt = t[(t > RISE_START)]
            ll = led[(t > RISE_START)]
            t_lo = tt[np.argmax(ll > 2.8)]
            t_hi = tt[np.argmax(ll > 5.2)]
            rate = 2.4 / (t_hi - t_lo)
            end = t_hi + 0.1
            m = win(t, RISE_START, end + 0.5)
            amp = f[m].max() - ref                               # L1 peak, above baseline
            t0 = RISE_START
            search = (RISE_START, end + 0.5)
        sweeps.append(dict(kind=kind, rate=rate, group=int(round(np.log2(rate))), amp=amp, ref=ref,
                           t=t - t0, erg=erg - ref, led=led, raw=erg, trec=t, search=search))
    return sweeps


def figure_ramps(sweeps, out, filt):
    """Ramp figure. A, B: average ERG for rising and falling ramps with the measured point marked.
    C, D: L1 peak and L2 trough against ramp rate (log axis), each with a line of best fit."""
    fig = plt.figure(figsize=(8.4, 9.4))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.8], hspace=0.30,
                     left=0.12, right=0.97, top=0.96, bottom=0.07)
    top = outer[0].subgridspec(2, 2, height_ratios=[0.25, 1.0], hspace=0.10, wspace=0.40)
    setup = (("rise", L1_COLOR, "Rising ramps", (-0.6, 3.8), (-1.0, 0.95), "ERG response (V)"),
             ("fall", L2_COLOR, "Falling ramps", (-0.6, 5.0), (-0.85, 0.7), "ERG response (V)"))
    for col, (kind, colr, name, xlim, ylim, ylab) in enumerate(setup):
        S = [s for s in sweeps if s["kind"] == kind]
        groups = sorted(set(s["group"] for s in S))
        cols = light_to_dark(len(groups), lo=0.30, hi=1.0)
        axL = fig.add_subplot(top[0, col])
        axT = fig.add_subplot(top[1, col], sharex=axL)
        axI = None
        if kind == "rise":
            axI = axT.inset_axes([0.43, 0.63, 0.54, 0.29])
        for g, c in zip(groups, cols):
            G = [s for s in S if s["group"] == g]
            t = G[0]["t"]
            r = np.mean([s["rate"] for s in G])
            y = filt(np.mean([s["erg"] for s in G], axis=0))
            axL.plot(t, np.mean([s["led"] for s in G], axis=0), color=c, lw=1.3)
            axT.plot(t, y, color=c, lw=1.4, label=f"{r:.0f}")
            # mark the measured extreme of the average trace
            dur = 3.0 / r
            if kind == "rise":
                m = (t >= 0) & (t <= dur + 0.6)
                k = np.argmax(np.where(m, y, -np.inf))
            else:
                m = (t >= 0) & (t <= dur + 1.1)
                k = np.argmin(np.where(m, y, np.inf))
            axT.plot(t[k], y[k], "o", ms=5, color=c, mec="white", mew=0.8, zorder=5)
            if axI is not None:
                axI.plot(t, y, color=c, lw=1.4)
                axI.plot(t[k], y[k], "o", ms=5, color=c, mec="white", mew=0.8, zorder=5)
        axL.set_ylabel("Command\nvoltage (V)")
        axL.set_title(name, fontsize=12.5, fontweight="semibold", pad=8)
        plt.setp(axL.get_xticklabels(), visible=False)
        axL.tick_params(axis="x", length=0)
        axT.axhline(0, color="0.35", lw=0.6, zorder=0)
        axT.set_xlim(*xlim)
        axT.set_ylim(*ylim)
        axT.set_xlabel("Time from ramp start (s)")
        axT.set_ylabel(ylab)
        axT.legend(frameon=False, title="Ramp rate (V/s)", title_fontsize=9.5, ncol=5, columnspacing=0.8,
                   handlelength=1.1, handletextpad=0.4, fontsize=9.5, loc="lower right" if kind == "fall" else "lower center")
        panel_letter(axL, "AB"[col])
        if axI is not None:
            axI.set_xlim(-0.05, 0.45)
            axI.set_ylim(-0.1, 0.7)
            axI.axhline(0, color="0.35", lw=0.6, zorder=0)
            axI.tick_params(labelsize=9)
            axI.set_title("first 0.45 s", fontsize=9.5, pad=2)
            _rect, _conns = axT.indicate_inset_zoom(axI, edgecolor="0.6", lw=0.8)
            for _c in _conns:
                _c.set_visible(False)
            for _sp in axI.spines.values():
                _sp.set_edgecolor("0.5")

    # C, D. response against rate, one panel per cell type. Same blues as A and B (one colour per rate).
    bot = outer[1].subgridspec(1, 2, wspace=0.40)
    blues = light_to_dark(5, lo=0.32, hi=1.0)
    cfg = (("rise", "o", "L1 peak (V)", (-0.05, 0.9), "C", (0.04, 0.95), "left", "top"),
           ("fall", "s", "L2 trough (V)", (-0.9, 0.05), "D", (0.04, 0.95), "left", "top"))
    for col, (kind, mk, ylab, ylim, letter, tpos, ha, va) in enumerate(cfg):
        ax = fig.add_subplot(bot[0, col])
        S = [s for s in sweeps if s["kind"] == kind]
        x = np.array([s["rate"] for s in S])
        y = np.array([s["amp"] for s in S])
        groups = sorted(set(s["group"] for s in S))
        for g, c in zip(groups, blues):
            G = [s for s in S if s["group"] == g]
            gx = np.mean([s["rate"] for s in G])
            ax.plot([s["rate"] for s in G], [s["amp"] for s in G], ".", ms=4, color=c, alpha=0.35, zorder=1)
            m, e = mean_sem([s["amp"] for s in G])
            ax.errorbar([gx], [m], [e], fmt=mk, ms=7, color=c, mec="0.15", mew=0.8, ecolor="0.15",
                        capsize=2.5, zorder=4)
        slope, lo, hi, r2 = bootstrap_slope(np.log10(x), y)
        xs = np.linspace(np.log10(x).min(), np.log10(x).max(), 50)
        yfit = np.polyval(np.polyfit(np.log10(x), y, 1), xs)
        ax.plot(10 ** xs, yfit, "--", color="0.15", lw=1.3, zorder=3)
        ax.text(*tpos, f"R² = {r2:.2f}", transform=ax.transAxes, color="0.15", fontsize=10.5, ha=ha, va=va)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 20)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(*ylim)
        ax.set_xlabel("Ramp rate (V/s, log scale)")
        ax.set_ylabel(ylab)
        panel_letter(ax, letter)
    fig.savefig(os.path.join(out, "fig_ramp_series.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_ramp_series.pdf"))
    plt.close(fig)


def figure_pipeline(P, R, out, filt):
    """Raw data to refined measurement, for one pulse, one rising ramp and one falling ramp.
    Row 1: LED. Row 2: all 15 raw sweeps. Row 3: one sweep, raw and smoothed, with the measurement windows.
    Row 4: average of the 15 sweeps (the traces used in the other figures)."""
    def pick(S, key):
        med = np.median([s[key] for s in S])
        return min(S, key=lambda s: abs(s[key] - med))                  # the most typical sweep, not a hand-picked one
    PULSE_EXAMPLE_V = 2.0                                               # a clean page (low noise), chosen for readability
    pul = [s for s in P if s["amp"] == PULSE_EXAMPLE_V] or [s for s in P if s["amp"] == P[-1]["amp"]]
    rise = [s for s in R if s["kind"] == "rise" and s["group"] == 2]
    fall = [s for s in R if s["kind"] == "fall" and s["group"] == 2]
    cols = [(f"Pulse, {PULSE_EXAMPLE_V:g} V", pul, "on", (0.0, 3.0)),
            ("Rising ramp, 4 V/s", rise, "amp", (0.0, 4.0)),
            ("Falling ramp, 4 V/s", fall, "amp", (4.0, 8.0))]
    fig = plt.figure(figsize=(10.5, 10.0))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.3, 1, 1, 1], hspace=0.14, wspace=0.30,
                  left=0.10, right=0.985, top=0.94, bottom=0.06)
    RAWC, SMC = "#9DB4CE", "#0B3D91"
    for c, (title, S, key, xl) in enumerate(cols):
        ex = pick(S, key)
        t = ex["trec"]
        led = ex["led"]
        raw = ex["raw"]
        f = filt(raw)
        axL = fig.add_subplot(gs[0, c])
        ax2 = fig.add_subplot(gs[1, c], sharex=axL)
        ax3 = fig.add_subplot(gs[2, c], sharex=axL)
        ax4 = fig.add_subplot(gs[3, c], sharex=axL)
        axL.plot(t, led, color="0.2", lw=1.3)
        axL.set_title(title, fontsize=12.5, fontweight="semibold", pad=8)
        axL.set_xlim(*xl)
        # row 2: all 15 raw sweeps as recorded
        for s in S:
            ax2.plot(s["trec"], s["raw"], color=RAWC, lw=0.6, alpha=0.7)
        ax2.plot(t, raw, color="0.25", lw=0.7)
        # row 3: the highlighted sweep, raw and smoothed, with the windows used to read it
        ax3.plot(t, raw, color=RAWC, lw=0.7, zorder=2)
        ax3.plot(t, f, color=SMC, lw=1.5, zorder=3)
        if c == 0:
            w = WINDOWS
            lvl_b = raw[win(t, *w["pulse_on_ref"])].mean()
            lvl_p = raw[win(t, *w["pulse_off_ref"])].mean()
            ax3.axvspan(*w["pulse_on_ref"], color="0.88", zorder=0)
            ax3.axvspan(*w["pulse_on_search"], color="#F6D9CB", zorder=0)
            ax3.axvspan(*w["pulse_off_ref"], color="0.88", zorder=0)
            ax3.axvspan(*w["pulse_off_search"], color="#E3D5EE", zorder=0)
            m1 = win(t, *w["pulse_on_search"]); k1 = np.where(m1)[0][np.argmax(f[m1])]
            m2 = win(t, *w["pulse_off_search"]); k2 = np.where(m2)[0][np.argmin(f[m2])]
            ax3.plot([0.1, 1.5], [lvl_b, lvl_b], "--", color="0.35", lw=0.9, zorder=1)
            ax3.plot([1.0, 1.8], [lvl_p, lvl_p], "--", color="0.35", lw=0.9, zorder=1)
            for (xk, y0, y1, colr, lab, dx) in ((t[k1], lvl_b, f[k1], L1_COLOR, "L1 peak", 0.06),
                                                (t[k2], lvl_p, f[k2], L2_COLOR, "L2 trough", 0.06),
                                                (1.0, lvl_b, lvl_p, PR_COLOR, "photoreceptor\ndepolarization", 0.05)):
                ax3.annotate("", xy=(xk, y1), xytext=(xk, y0), arrowprops=dict(arrowstyle="<->", color=colr, lw=1.4,
                                                                         shrinkA=0, shrinkB=0), zorder=5)
                ax3.text(xk + dx, (y0 + y1) / 2, lab, color=colr, fontsize=9.5, va="center", ha="left", zorder=7,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax3.plot([t[k1], t[k2]], [f[k1], f[k2]], "o", ms=5.5, color="k", mfc="white", zorder=6)
        else:
            rk = "rise_ref" if key == "amp" and ex["kind"] == "rise" else "fall_ref"
            ref_lo, ref_hi = WINDOWS[rk]
            lvl = raw[win(t, ref_lo, ref_hi)].mean()
            sh, colr, lab = ((("#F6D9CB", L1_COLOR, "L1 peak") if ex["kind"] == "rise" else ("#E3D5EE", L2_COLOR, "L2 trough")))
            ax3.axvspan(ref_lo, ref_hi, color="0.88", zorder=0)
            ax3.axvspan(*ex["search"], color=sh, zorder=0)
            m = win(t, *ex["search"])
            idx = np.where(m)[0]
            k = idx[np.argmax(f[m])] if ex["kind"] == "rise" else idx[np.argmin(f[m])]
            ax3.plot([ref_lo, ex["search"][1]], [lvl, lvl], "--", color="0.35", lw=0.9, zorder=1)
            ax3.annotate("", xy=(t[k], f[k]), xytext=(t[k], lvl), arrowprops=dict(arrowstyle="<->", color=colr, lw=1.4,
                                                                          shrinkA=0, shrinkB=0), zorder=5)
            ax3.text(t[k] + 0.08, (lvl + f[k]) / 2, lab, color=colr, fontsize=9.5, va="center", ha="left", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax3.plot([t[k]], [f[k]], "o", ms=5.5, color="k", mfc="white", zorder=6)
        # row 4: average of the 15 sweeps, each re-referenced to its own baseline
        M = np.mean([s["erg"] for s in S], axis=0)
        ax4.plot(S[0]["trec"], M, color=PR_COLOR, lw=1.3)
        ax4.axhline(0, color="0.35", lw=0.6, zorder=0)
        vis = win(t, *xl)
        allraw = np.concatenate([s["raw"][vis] for s in S])
        lo2, hi2 = allraw.min(), allraw.max(); pad = 0.04 * (hi2 - lo2)
        ax2.set_ylim(lo2 - pad, hi2 + pad)
        ax3.set_ylim(f[vis].min() - 0.18, f[vis].max() + 0.22)
        ax4.set_ylim(M[vis].min() - 0.08, M[vis].max() + 0.08)
        ax4.set_xlabel("Time in sweep (s)")
        for a in (axL, ax2, ax3):
            plt.setp(a.get_xticklabels(), visible=False)
            a.tick_params(axis="x", length=0)
        if c == 0:
            axL.set_ylabel("LED (V)")
            ax2.set_ylabel("1. Raw sweeps (V)")
            ax3.set_ylabel("2. One sweep, read (V)")
            ax4.set_ylabel("3. Mean of 15,\nre. baseline (V)")
        axL.annotate("ABC"[c], xy=(0, 1), xycoords="axes fraction", xytext=(-52 if c == 0 else -34, 14), textcoords="offset points",
                     fontsize=16, fontweight="bold", va="bottom", annotation_clip=False)
    fig.savefig(os.path.join(out, "fig_raw_to_refined.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_raw_to_refined.pdf"))
    plt.close(fig)


def figure_annotated_trace(P, out, volts=2.5):
    """One unsmoothed pulse sweep with the three response regions labelled (single-trace style used in fly ERG papers)."""
    S = [s for s in P if s["amp"] == volts] or P[:15]
    def noise(s):                                           # high-frequency roughness of the unsmoothed trace
        d = np.diff(s["raw"]); return np.median(np.abs(d - np.median(d)))
    ex = min(S, key=noise)                                  # the cleanest sweep on that page
    t = ex["trec"]                                          # recording time, light on at 0.5 s
    y = ex["raw"] - ex["base"]                              # unsmoothed, baseline subtracted
    fig = plt.figure(figsize=(7.6, 4.3))
    ax = fig.add_axes([0.10, 0.15, 0.875, 0.60])
    axS = fig.add_axes([0.10, 0.90, 0.875, 0.05], sharex=ax)
    axS.hlines(0, 0.5, 1.5, color="0.15", lw=7, capstyle="butt")
    axS.set_ylim(-1, 1)
    axS.axis("off")
    bands = ((0.5, 0.60, "#F6D9CB", L1_COLOR, "L1\nresponse", 0.54),
             (0.60, 1.5, "#DCE8F7", PR_COLOR, "photoreceptor\nresponse", 1.10),
             (1.5, 1.65, "#E3D5EE", L2_COLOR, "L2\nresponse", 1.63))
    for lo, hi, shade, colr, lab, xc in bands:
        ax.axvspan(lo, hi, color=colr, alpha=0.26, zorder=0, lw=0)
        ax.text(xc, 1.03, lab, color=colr, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=10.5, fontweight="semibold", linespacing=1.15)
    ax.plot(t, y, color="0.1", lw=0.8, zorder=3)
    ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, 3.0)
    lo, hi = y[win(t, 0, 3.0)].min(), y[win(t, 0, 3.0)].max()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERG")
    fig.savefig(os.path.join(out, "fig_annotated_trace.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_annotated_trace.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- csv + summary
def write_csv(path, sweeps, keys):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep_index"] + keys)
        for i, s in enumerate(sweeps, 1):
            w.writerow([i] + [round(float(s[k]), 5) if not isinstance(s[k], str) else s[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulses")
    ap.add_argument("--ramps")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--filter", default="sg", choices=["sg", "median", "boxcar"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    filt = make_filter(a.filter)
    if a.pulses:
        P = analyse_pulses(load_labchart_text(a.pulses), filt)
        figure_pulses(P, a.out, filt)
        figure_pulses_boxed(P, a.out, filt)
        figure_annotated_trace(P, a.out)
        pipe_P = P
        write_csv(os.path.join(a.out, "pulse_sweeps.csv"), P, ["amp", "base", "on", "pre_off", "off", "plateau"])
        print("pulses: %d sweeps, %d amplitudes" % (len(P), len(set(s["amp"] for s in P))))
        for amp in sorted(set(s["amp"] for s in P)):
            S = [s for s in P if s["amp"] == amp]
            print(f"  {amp:>5} V  n={len(S)}  L1 {np.mean([s['on'] for s in S]):+.3f}  L2 {np.mean([s['off'] for s in S]):+.3f}  plateau {np.mean([s['plateau'] for s in S]):+.3f}")
    if a.ramps:
        R = analyse_ramps(load_labchart_text(a.ramps), filt)
        figure_ramps(R, a.out, filt)
        if a.pulses:
            figure_pipeline(pipe_P, R, a.out, filt)
        write_csv(os.path.join(a.out, "ramp_sweeps.csv"), R, ["kind", "rate", "amp", "ref"])
        print("ramps: %d sweeps" % len(R))
        for kind in ("rise", "fall"):
            for g in sorted(set(s["group"] for s in R if s["kind"] == kind)):
                S = [s for s in R if s["kind"] == kind and s["group"] == g]
                print(f"  {kind} rate {np.mean([s['rate'] for s in S]):>5.2f} V/s  n={len(S)}  amp {np.mean([s['amp'] for s in S]):+.3f}")


if __name__ == "__main__":
    main()
