"""Figures of the quench paper (main.tex) from the dU scan and the B-resolved crossover scan.

    python3 plot_figs.py      -> fig_convergence.pdf, fig_dmft.pdf, fig_crossover.pdf, fig_strong.pdf, fig_ng.pdf (+ png)
    python3 plot_figs.py fig_ng   -> only the named figures

Data: ../dU_scan (adaptive B <= 15 quenches 0 -> U to t = 12 with n(eps, t), GPU code gpu/gpu_ed.py adapt;
B <= 11, 13 to t = 10; thermal gGA; digitized EKW DMFT), ../crossover_scan (B <= 1, 3, ..., 13 for U = 0.6 ... 2). Units: half-bandwidth W = 1, hbar = 1
(EKW: U = U_EKW/2, t = 2 t_EKW).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [f"{ROOT}/dU_scan", f"{ROOT}/crossover_scan"]
import analyze_scan as sc          # noqa: E402
import analyze_crossover as cr     # noqa: E402

# Okabe-Ito color-blind-safe palette (its yellow is too faint on white and is not used)
BLUE, VERM, AQUA, ORANGE, MAGENTA, SKY, INK = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
                                               "#56B4E9", "#000000")
MUTED, GRID = "#52514e", "#e4e3df"
COLS = [BLUE, VERM, AQUA, ORANGE, MAGENTA, SKY]   # U-indexed series; ink is reserved for DMFT
NAVY = "#00395f"                                  # B <= 15, darker than the B <= 13 blue
CB = {1: "#8a8987", 3: MAGENTA, 5: ORANGE, 7: AQUA, 9: SKY, 11: VERM, 13: BLUE, 15: NAVY}   # B <= 1 ... 15
BMAX, TMAX = 15, 10.0                             # td-gGA runs of Figs. 1, 2 (B <= 15 runs reach t = 12)
B17_U = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)   # B <= 17 (low-rank GPU solver) to t = 10
LR = f"{ROOT}/gpu/lowrank"


def b17(u):
    """B <= 17 at U = u: dense B <= 15 up to the 15 -> 17 insertion, then the low-rank B = 17 stage
    (gpu/lowrank/B17nk_U<U>: rerun with n(eps), or B17_U<U>). Returns d (t, d), Delta n (t, jump) or None,
    and t_conv: B <= 17 departs from B <= 15 by 1e-3, plus the spacing of the last two insertions."""
    if u not in B17_U:
        return None
    for name in (f"B17nk_U{u:g}", f"B17_U{u:g}"):
        d17 = f"{LR}/{name}/stage_B17"
        if os.path.exists(f"{d17}/observables.dat"):
            break
    else:
        return None
    a15 = sc.load(u, 15)
    o = np.loadtxt(f"{d17}/observables.dat")
    t_ins = o[0, 0]
    d = np.vstack([a15[a15[:, 0] < t_ins - 1e-9, :2], o[:, :2]])
    j = None
    if os.path.exists(f"{d17}/nk.dat"):
        j15, j17 = jump_nk(f"{sc.run_dir(u, 15)}/nk_td.dat"), jump_nk(f"{d17}/nk.dat")
        j = np.vstack([j15[j15[:, 0] < t_ins - 1e-9], j17])
    # t_conv: B <= 17 departs from B <= 15 by 1e-3, plus one pair spacing (the paper's rule for B <= 13)
    m = {round(r[0], 4): r[1] for r in a15}
    off = [r[0] for r in o if round(r[0], 4) in m and abs(r[1] - m[round(r[0], 4)]) > 1e-3]
    t15 = sc.insert_times(u, 15)[-1]
    tc = (off[0] if off else o[-1, 0]) + (t_ins - t15)
    return d, j, tc
E0 = 4/(3*np.pi)
C2 = 0.820                                        # second-order strong-coupling c of omega = U - c/U (strong_c/)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "axes.labelcolor": INK, "legend.frameon": False, "mathtext.fontset": "cm"})


def finish(axes):
    for ax in np.ravel(axes):
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def tag(ax, s):
    ax.text(-0.02, 1.0, f"({s})", transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5, color=INK)


EXTRA_GRAY = "#8a8987"                             # the extra strong-coupling U (no DMFT)
DMFT_EXTRA_U = 8.0                                   # U shown in Fig. 2 beyond the DMFT set


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{HERE}/{name}.{ext}", dpi=250)


def split(t, y, tc):
    """solid up to the B-convergence time tc, dotted after"""
    m = t <= (tc if tc else t[-1]) + 1e-9
    return (t[m], y[m]), (t[~m | np.r_[m[1:] & ~m[1:], False]], y[~m | np.r_[m[1:] & ~m[1:], False]])


DMFT_MK = dict(ls="none", marker="o", ms=3.4, mfc="none", mec=INK, mew=0.9, zorder=1)   # Fig. 1: DMFT, open circles behind the td-gGA lines


def thin(t, y, dt=0.2):
    """points of a (smooth, densely sampled) DMFT curve spaced by about dt in t, for markers"""
    keep, last = [], -np.inf
    for k, x in enumerate(t):
        if x - last >= dt - 1e-9:
            keep.append(k)
            last = x
    return np.asarray(t)[keep], np.asarray(y)[keep]


def fig_convergence():
    """d(t) and Delta n(t) for B <= 1, ..., 13 at U = 1.5 (prethermal side) and U = 3 (collapse-revival)
    vs DMFT"""
    dm, _ = sc.ekw_d()
    dn = sc.digitized(f"{ROOT}/dU_scan/ekw_digitized/ekw_fig2c_dn_smooth.dat")
    dn.update(sc.digitized(f"{ROOT}/dU_scan/ekw_digitized/ekw_fig2d_dn_smooth.dat"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.9), sharex=True)
    bs = [b for b in cr.BS if b < BMAX] + [BMAX]     # B <= BMAX from the dU scan (cr.BS also has 15 for Fig. 3)
    for i, u in enumerate((1.5, 3.0)):
        dinf = cr.thermal_d(u)[1]
        for b in bs:
            a = cr.load_d(u, b) if b < BMAX else sc.load(u, BMAX)
            lw = 1.4 if b == BMAX else 0.9
            axes[0, i].plot(a[:, 0], a[:, 1], color=CB[b], lw=lw, zorder=3 if b == BMAX else 2)
            j = cr.jump(u, b) if b < BMAX else jump_dU(u)
            if j is not None:
                axes[1, i].plot(j[:, 0], j[:, 1], color=CB[b], lw=lw, zorder=3 if b == BMAX else 2)
        axes[0, i].plot(*thin(*dm[u]), **DMFT_MK)
        axes[1, i].plot(*thin(*dn[u]), **DMFT_MK)
        axes[0, i].axhline(dinf, color=MUTED, lw=0.7, ls=":")
        axes[0, i].text(0.98, 0.04, f"$U = {u:g}$", transform=axes[0, i].transAxes, ha="right", va="bottom", fontsize=9)
        axes[1, i].set_xlabel("$t$")
        axes[1, i].set_ylim(-0.03, 1.03)
        axes[1, i].set_xlim(0, TMAX)
        axes[0, i].set_ylim(0.0, 0.26)
    hs = [plt.Line2D([], [], color=CB[b], lw=1.4 if b == BMAX else 0.9) for b in bs]
    hs.append(plt.Line2D([], [], **DMFT_MK))
    hs.append(plt.Line2D([], [], color=MUTED, lw=0.7, ls=":"))
    fig.legend(hs, [f"$B\\leq{b}$" for b in bs] + ["DMFT", r"thermal $d_\infty$"], loc="upper center",
               ncol=10, fontsize=7, handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.53, 1.0))
    axes[0, 0].set_ylabel(r"$d(t)$")
    axes[1, 0].set_ylabel(r"$\Delta n(t)$")
    for k, ax in enumerate(axes.ravel()):
        tag(ax, "abcd"[k])
    finish(axes)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.11, wspace=0.18, hspace=0.16)
    save(fig, "fig_convergence")


def fig_dmft(extra=8.0, name="fig_dmft"):
    """d(t) and Delta n(t) at all U of EKW Fig. 2: td-gGA (B <= 15, solid to t_conv) vs DMFT (dashed),
    thermal gGA d_inf (ticks at the right edge) vs DMFT d_th; plus U = extra in (b, d) (no DMFT)"""
    dm, dth = sc.ekw_d()
    dn = sc.digitized(f"{ROOT}/dU_scan/ekw_digitized/ekw_fig2c_dn_smooth.dat")
    dn.update(sc.digitized(f"{ROOT}/dU_scan/ekw_digitized/ekw_fig2d_dn_smooth.dat"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    strong = sc.STRONG + [extra]                     # no DMFT data
    scols = COLS[:5] + [EXTRA_GRAY]                  # not ink: ink marks DMFT in this figure
    for i, (us, cs) in enumerate(((sc.WEAK, COLS), (strong, scols))):
        for u, c in zip(us, cs):
            a = sc.load(u, BMAX) if not fine(u) else np.loadtxt(f"{fine(u)}/observables.dat")
            tc = sc.t_converged_trig(u, BMAX)
            j = jump_dU(u)
            r17 = None if fine(u) else b17(u)        # U = 8: fine-mesh B <= 15, equal to B <= 17 to 6e-5 for t <= 10
            if r17 is not None:
                a, tc = r17[0], r17[2]
                j = r17[1] if r17[1] is not None else j
            lw = 0.8 if u == extra else 1.1
            (t1, y1), (t2, y2) = split(a[:, 0], a[:, 1], tc)
            axes[0, i].plot(t1, y1, color=c, lw=lw)
            axes[0, i].plot(t2, y2, color=c, lw=0.8, ls=":")
            if u in dm:
                axes[0, i].plot(*dm[u], color=INK, lw=0.8, ls=(0, (3, 2)))
            dinf = sc.thermal_d(u)[1]
            axes[0, i].plot([TMAX + 0.15, TMAX + 0.6], [dinf]*2, color=c, lw=1.6, clip_on=False)
            if u in dth:
                axes[0, i].plot([TMAX + 0.65, TMAX + 1.1], [dth[u]]*2, color=INK, lw=1.6, clip_on=False)
            if j is not None:
                (t1, y1), (t2, y2) = split(j[:, 0], j[:, 1], tc)
                axes[1, i].plot(t1, y1, color=c, lw=lw)
                axes[1, i].plot(t2, y2, color=c, lw=0.8, ls=":")
            if u in dn:
                axes[1, i].plot(*dn[u], color=INK, lw=0.8, ls=(0, (3, 2)))
        axes[1, i].set_xlabel("$t$")
        axes[1, i].set_xlim(0, TMAX)
        axes[1, i].set_ylim(-0.02, 1.02)
    # direct labels
    for u, c in zip(sc.WEAK, COLS):
        a = sc.load(u, BMAX)
        y = np.interp(3.0, a[:, 0], a[:, 1])
        axes[0, 0].text(3.0, y - 0.004, f"$U={u:g}$", color=c, fontsize=7, ha="left", va="top")
        j = jump_dU(u)
        if j is not None and u < 1.5:
            axes[1, 0].text(TMAX - 0.1, np.interp(TMAX - 0.1, j[:, 0], j[:, 1]) + 0.035, f"$U={u:g}$", color=c, fontsize=7,
                            ha="right", va="bottom")
    axes[1, 0].text(2.6, 0.08, "$U=1.5$", color=COLS[5], fontsize=7, ha="left")
    # left-aligned at t = 8, just above each curve (U = 1.65 below, the U = 2 curve is close above)
    pos = {1.65: 0.0885, 2.0: 0.1145, 2.5: 0.1405, 3.0: 0.176, 4.0: 0.2, extra: 0.228}
    for u, c in zip(strong, scols):
        axes[0, 1].text(8.0, pos[u], f"$U={u:g}$", color=c, fontsize=7, ha="left", va="bottom")
    # (d): the curves cross too often for direct labels; legend in the empty upper right
    handles = [plt.Line2D([], [], color=c, lw=0.8 if u == extra else 1.1) for u, c in zip(strong, scols)]
    axes[1, 1].legend(handles, [f"$U={u:g}$" for u in strong], loc="upper right", fontsize=7, ncol=2,
                      frameon=False, handlelength=1.6, columnspacing=1.2)
    axes[0, 1].text(TMAX + 0.35, 0.255, r"$d_\infty$ ", fontsize=7, color=MUTED, ha="center")
    axes[0, 1].text(TMAX + 0.9, 0.255, r"$d_{\rm th}$", fontsize=7, color=MUTED, ha="center")
    axes[0, 0].set_ylabel(r"$d(t)$")
    axes[1, 0].set_ylabel(r"$\Delta n(t)$")
    axes[0, 0].set_ylim(0.085, 0.255)
    axes[0, 1].set_ylim(0.085, 0.255)
    for k, ax in enumerate(axes.ravel()):
        tag(ax, "abcd"[k])
    finish(axes)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.09, wspace=0.27, hspace=0.12)
    save(fig, name)


def jump_nk(f):
    a = np.loadtxt(f)
    out = []
    for t in np.unique(a[:, 0]):
        x = a[a[:, 0] == t]
        e, n = x[:, 1], x[:, 2]
        out.append((t, n[e < 0][np.argmax(e[e < 0])] - n[e > 0][np.argmin(e[e > 0])]))
    return np.array(out)


FINE = {8.0: f"{ROOT}/dU_scan/U0_dU8_B15_fine"}   # runs with output and n(eps) every 0.02 (fast oscillation)


def fine(u):
    d = FINE.get(u)
    return d if d and os.path.exists(f"{d}/DONE") else None


def jump_dU(u, bmax=BMAX):
    """Delta n(t) from n(eps, t): the fine-mesh run if any, the B <= bmax run (nk_td.dat), else the B <= 13
    analysis (nk_td_dU<U>.dat)"""
    f = f"{fine(u) or sc.run_dir(u, bmax)}/nk_td.dat"
    if not os.path.exists(f):
        f = f"{ROOT}/dU_scan/nk_td_dU{u:g}.dat"
    if not os.path.exists(f):
        return None
    a = np.loadtxt(f)
    out = [(0.0, 1.0)]
    for t in np.unique(a[:, 0]):
        x = a[a[:, 0] == t]
        e, n = x[:, 1], x[:, 2]
        out.append((t, n[e < 0][np.argmax(e[e < 0])] - n[e > 0][np.argmin(e[e > 0])]))
    out = np.array(sorted(set(out)))
    return out[np.r_[True, np.diff(out[:, 0]) > 1e-6]]


def crossover_data():
    res = {}
    for u in cr.us():
        dinf = cr.thermal_d(u)[1]
        for b in cr.BS:
            a = cr.load_d(u, b)
            if a is None:
                continue
            t, d = a[:, 0], a[:, 1]
            m = (t >= cr.WIN[0]) & (t <= cr.WIN[1])
            j = cr.jump(u, b)
            stat, rev = cr.dn_features(j) if j is not None else (None, None)
            res[(u, b)] = dict(dist=np.sqrt(np.mean((d[m] - dinf)**2)), amp=0.5*np.ptp(d[m]), stat=stat, rev=rev)
    return res


def first_min(t, d, t0=0.3):
    """first stationary point of d(t) after t0: the first local minimum of |dd/dt| (a minimum of d where dd/dt
    changes sign, a shoulder of d where it only comes close to zero), refined by a parabola through the three
    neighbouring points; returns (t1, d(t1)) or (nan, nan)"""
    v = np.abs(np.gradient(d, t))
    for k in range(1, len(d) - 1):
        if t[k] > t0 and v[k] < v[k-1] and v[k] <= v[k+1]:
            a, b, c = v[k-1], v[k], v[k+1]
            den = a - 2*b + c
            x = min(max(0.5*(a - c)/den if den > 0 else 0.0, -0.5), 0.5)
            tt = t[k] + x*(t[k+1] - t[k])
            return tt, np.interp(tt, t, d)
    return np.nan, np.nan


def b1_dense():
    """td-GA (B = 1) on a dense U grid around U_c^dyn, to t = 20 (crossover_scan/B1_dense)"""
    base = f"{ROOT}/crossover_scan/B1_dense"
    uu = sorted(float(n[1:]) for n in os.listdir(base) if n.startswith("U") and os.path.isdir(f"{base}/{n}"))
    return np.array([(u,) + first_min(*np.loadtxt(f"{base}/U{u:g}/observables.dat")[:, :2].T) for u in uu])


def fig_crossover():
    """(a) time t1 and (b) value d(t1) of the first stationary point of d(t) (first minimum; a shoulder for B <= 3 near the crossover) vs U: the diverging half-period and the complete
    doublon depletion of td-GA at U_c^dyn, removed by the ghost orbitals (B = 1 dense, crossover_scan/B1_dense;
    B <= 3 ... 15 from the crossover scan; DMFT from the smooth digitized EKW curves); (c) rms of d - d_inf over
    4 <= t <= 7 for all B. DMFT crossover bracket 1.5-1.65 shaded"""
    res = crossover_data()
    U = cr.us()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    g = b1_dense()
    for ax, col in zip(axes[:2], (1, 2)):
        ax.plot(g[:, 0], g[:, col], "-", color=CB[1], lw=0.9, zorder=2)
    for b in cr.BS[1:]:
        r = np.array([(u,) + first_min(*cr.load_d(u, b)[:, :2].T) for u in U if cr.load_d(u, b) is not None])
        for ax, col in zip(axes[:2], (1, 2)):
            ax.plot(r[:, 0], r[:, col], "-o", color=CB[b], lw=1.3 if b == 15 else 0.9, ms=2,
                    zorder=3 if b == 15 else 2)
    dm = sc.ekw_d()[0]
    e = np.array([(u,) + first_min(*dm[u]) for u in sorted(dm) if 0.5 <= u <= 2.05])
    for ax, col in zip(axes[:2], (1, 2)):
        ax.plot(e[:, 0], e[:, col], **dict(DMFT_MK, zorder=4))
    for b in cr.BS:
        uu = [x for x in U if (x, b) in res]
        axes[2].plot(uu, [res[(x, b)]["dist"] for x in uu], "-" if b == 1 else "-o", color=CB[b],
                     lw=1.3 if b == 15 else 0.9, ms=2, zorder=3 if b == 15 else 2, label=f"$B\\leq{b}$")
    # (a): the curves visible in (a) and (b), where B <= 5 ... 15 coincide, and DMFT; (c): all B
    hs = [plt.Line2D([], [], color=CB[1], lw=0.9), plt.Line2D([], [], color=CB[3], lw=0.9, marker="o", ms=2),
          plt.Line2D([], [], color=CB[15], lw=1.3, marker="o", ms=2), plt.Line2D([], [], **DMFT_MK)]
    axes[0].legend(hs, ["$B\\leq 1$", "$B\\leq 3$", "$B\\leq 5$ to $15$", "DMFT"], loc="upper left", fontsize=6,
                   frameon=False, handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
    axes[0].set_ylim(1.4, 6.3)
    axes[0].set_ylabel("$t_1$")
    axes[1].set_ylim(-0.005, 0.19)
    axes[1].set_ylabel("$d(t_1)$")
    axes[2].set_yscale("log")
    axes[2].set_ylim(9e-5, 0.3)
    axes[2].set_ylabel(r"$\eta_d$")
    axes[2].legend(loc="lower left", ncol=2, fontsize=6, frameon=False, handlelength=1.6, columnspacing=1.0,
                   labelspacing=0.25)
    for k, ax in enumerate(axes):
        ax.set_xlim(0.55, 2.05)
        ax.set_xlabel("$U$")
        ax.axvspan(1.5, 1.65, color=GRID, alpha=0.6, lw=0, zorder=0)    # DMFT bracket of the crossover
        ax.axvline(1.6, color=INK, lw=0.7, ls=":")
        ax.axvline(cr.UC_GA, color=CB[1], lw=0.7, ls="--")
        tag(ax, "abc"[k])
    axes[1].text(1.585, 0.172, r"$U_c^{\rm DMFT}$", color=INK, fontsize=7, ha="right")
    axes[1].text(1.715, 0.172, r"$U_c^{\rm dyn}$", color=MUTED, fontsize=7, ha="left")
    finish(axes)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.18, wspace=0.38)
    save(fig, "fig_crossover")


def fig_strong():
    """(a) scaled d vs U t (four first-order periods) for U = 3, 5, 10, with the first-order first period, d_stat and
    d_th^(1); (c) U (U - omega) vs U (fit with the constrained even envelope; inset: d(t) and the fit at U = 5); (d) late d vs U: td-gGA, DMFT, d_stat, d_inf, DMFT d_th;
    (b) oscillation envelope (each extremum of the scaled d measured from the line through its neighbours, 1 at first order) for U = 3, 4, 5, 6, 10 vs the
    short-time Gaussian exp(-sigma^2 t^2/2), sigma^2 = 3/4 (strong_c/sigma2.py).
    Converged runs (B <= 17 at all U) via dU_scan/strong_B15.py; the fit of (c) uses t = 0 ... 10"""
    import strong_B15 as st
    dm, dth = sc.ekw_d()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    panels = list(axes)                                  # positional order (a)-(d)
    axes = [panels[0], panels[2], panels[1], panels[3]]  # axes[1]: U(U - omega) in (c); axes[2]: envelope in (b)
    XMAX = 8*np.pi                                     # four periods of the first-order result
    x0 = np.linspace(0, 2*np.pi, 200)
    axes[0].plot(x0, 0.5*(1 - np.cos(x0)), color=MUTED, lw=0.7, ls=":")      # first order, first period
    for k, (u, c) in enumerate(zip((3.0, 5.0, 10.0), (BLUE, AQUA, VERM))):
        t, d, tc, _ = st.series(u)
        m = (t <= TMAX) & (u*t <= XMAX + 1)
        # output every 0.05 is only ~13 points per period at U = 10: draw a cubic spline through them
        # (error ~4e-4 on this scale, estimated from a spline through every other point: 6e-3 at U = 10)
        from scipy.interpolate import CubicSpline
        tf = np.linspace(0, XMAX/u, 1200)
        axes[0].plot(u*tf, CubicSpline(t[m], (0.25 - d[m])*u/E0)(tf), color=c, lw=0.8, label=f"$U={u:g}$")
    axes[0].axhline(0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    axes[0].axhline(1.0, color=MUTED, lw=0.7, ls=":")
    axes[0].text(0.985*XMAX, 0.45, r"$d_{\rm stat}$", fontsize=6.5, color=INK, ha="right", va="top")
    axes[0].text(0.985*XMAX, 1.02, r"$d_{\rm th}^{(1)}$", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=6, frameon=False, handlelength=1.6,
                   labelspacing=0.25, borderaxespad=0.2)
    axes[0].set_xlim(0, XMAX)
    axes[0].set_xticks(np.arange(0, XMAX + 0.1, 2*np.pi))
    axes[0].set_xticklabels(["0", r"$2\pi$", r"$4\pi$", r"$6\pi$", r"$8\pi$"])
    axes[0].set_ylim(-0.05, 1.12)
    axes[0].set_xlabel("$Ut$")
    axes[0].set_ylabel(r"$[1/4 - d(t)]\,U/|E_0|$")
    # frequencies from the damped-cosine fits
    rs = [r for r in map(st.analyze, st.STRONG_U) if r is not None]
    uu = np.array([r["u"] for r in rs])
    # constrained even envelope (strong_B15.fit_even), fit window to t = 10
    axes[1].plot(uu, [r["u"]*(r["u"] - r["omega"]) for r in rs], "-o", color=BLUE, lw=1.1, ms=3)
    # inset: d(t) and the fit at a representative U
    r5 = next(r for r in rs if r["u"] == 5.0)
    ins = axes[1].inset_axes([0.40, 0.50, 0.57, 0.45])
    m5 = (r5["t"] >= r5["tmin1"] - 0.3) & (r5["t"] <= st.TMAX)
    tf = np.linspace(r5["t"][m5][0], st.TMAX, 1500)
    ins.plot(r5["t"], r5["d"], color=BLUE, lw=0.9)
    ins.plot(tf, st.damped_even(tf, *r5["fitpar"]), color=VERM, lw=0.6, ls=(0, (2, 1.5)))
    ins.set_xlim(0, st.TMAX)
    ins.set_ylim(0.165, 0.252)
    ins.set_xticks([0, 5, 10])
    ins.set_yticks([0.18, 0.22])
    ins.tick_params(labelsize=5, length=2, pad=1)
    ins.set_xlabel("$t$", fontsize=5.5, labelpad=0)
    ins.set_ylabel("$d$", fontsize=5.5, labelpad=0)
    ins.text(0.97, 0.93, "$U=5$", transform=ins.transAxes, fontsize=5.5, ha="right", va="top")
    axes[1].axhline(C2, color=INK, lw=0.7, ls=":")
    axes[1].text(10.0, C2 - 0.03, r"second order, $c=0.820$", color=INK, fontsize=6.5, ha="right", va="top")
    axes[1].set_xlabel("$U$")
    axes[1].set_ylabel(r"$U\,(U - \omega)$")
    axes[1].set_ylim(0.7, 1.65)
    # envelope: height of each extremum of the scaled d(t) above/below the straight line through its two neighbouring
    # extrema of opposite type (cancels a linearly drifting center), at the extremum; t = 0 (x = 0) counts as a minimum
    for k, (u, c) in enumerate(zip((3.0, 4.0, 5.0, 6.0, 10.0), (BLUE, SKY, AQUA, ORANGE, VERM))):
        t, d, _, _ = st.series(u)
        x = (0.25 - d)*u/E0
        ex = [("min", 0.0, 0.0)] + [e for e in sc.extrema(t, x) if e[1] <= TMAX]
        a = np.array([(ex[i][1], abs(ex[i][2] - np.interp(ex[i][1], [ex[i-1][1], ex[i+1][1]], [ex[i-1][2], ex[i+1][2]])))
                      for i in range(1, len(ex) - 1)])
        axes[2].plot(np.r_[0, a[:, 0]], np.r_[1, a[:, 1]], "-o", color=c, lw=0.8, ms=1.6, zorder=3 if u == 10 else 2,
                     label=f"$U={u:g}$")
    tg = np.linspace(0, 4, 200)
    axes[2].plot(tg, np.exp(-0.375*tg**2), color=INK, lw=0.8, ls=(0, (3, 2)), label=r"$e^{-\sigma^2t^2/2}$")
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="upper right", fontsize=6, frameon=False, handlelength=1.6, labelspacing=0.25,
                   borderaxespad=0.2)
    axes[2].set_xlabel("$t$")
    axes[2].set_ylabel("envelope")
    # levels
    rows = []
    for u in [1.65, 2.0] + st.STRONG_U:
        lv = st.late_level(u)
        if lv is not None and lv[1] is not None:
            rows.append((u, *lv))
    rows = np.array(rows)
    ug = np.linspace(1.65, 10.2, 200)
    axes[3].plot(rows[:, 0], rows[:, 1], "o-", color=BLUE, lw=1.1, ms=3, label="td-gGA, late")
    axes[3].plot(ug, 0.25 - E0/(2*ug), color=INK, lw=0.8, ls=(0, (3, 2)), label=r"$d_{\rm stat}$")
    axes[3].plot(rows[:, 0], rows[:, 2], "s-", color=AQUA, lw=1.1, ms=3, label=r"thermal gGA $d_\infty$")
    ud = [x for x in sorted(dth) if x >= 1.65]
    axes[3].plot(ud, [dth[x] for x in ud], "x", color=INK, ms=5, mew=1.2, label=r"DMFT $d_{\rm th}$")
    axes[3].legend(loc="lower right", fontsize=5.5, frameon=False, handlelength=1.2, handletextpad=0.4,
                   labelspacing=0.2, borderaxespad=0.1)
    axes[3].set_xlabel("$U$")
    axes[3].set_xlim(1.4, 10.6)
    axes[3].set_ylabel(r"$d$")
    axes[3].set_yticks([0.10, 0.15, 0.20])
    for k, ax in enumerate(panels):
        tag(ax, "abcd"[k])
    finish(panels)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.18, wspace=0.45)
    save(fig, "fig_strong")


NG = f"{ROOT}/insert_test/ng_short_time"          # first ghost pair at t = 0, B <= 3, seeds +X and -X (negseed/)
C_NG = 1/256 - 1/(36*np.pi**2)                    # Eq. (ng): n_g = C U^2 t^4


def fig_ng():
    """Appendix A, for the pair inserted at t = 0 into the Fermi sea, averaged over runs with seed +X and -X:
    (b) r = n_g/(C U^2 t^4) vs t (log axis); the average cancels the term linear in the seed amplitude; points with
    C U^2 t^4 < 1e-12, comparable to the seed occupation 2.5e-13, are left out.
    (a) weight moved off the original orbital, (1 - |R_1|^2)/(U^2 t^2/4) (R.dat, last column pair), with the atomic
    limit sin^2(Ut/2)/(Ut/2)^2 for U = 16."""
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True)
    for u, c in zip((0.1, 1.0, 2.0, 4.0, 8.0, 16.0), COLS):
        rs, ws = [], []
        for d in (NG, f"{NG}/negseed"):
            e = np.loadtxt(f"{d}/U{u:g}/stage_B3/delta_eig.dat")
            t = e[:, 0]
            rs.append((e[:, 1] - e[0, 1])/(C_NG*u**2*np.maximum(t, 1e-12)**4))
            R = np.loadtxt(f"{d}/U{u:g}/stage_B3/R.dat")
            ws.append((1 - R[:, 5]**2 - R[:, 6]**2)/(u**2*np.maximum(R[:, 0], 1e-12)**2/4))
        r, w = 0.5*(rs[0] + rs[1]), 0.5*(ws[0] + ws[1])
        m = (t >= 2e-3) & (C_NG*u**2*t**4 >= 1e-12)
        tm, rm = t[m], r[m]
        out = np.nonzero((rm > 1.3) | (rm < 0.85))[0]  # cut at the first exit from the y range (no re-entering branch)
        if out.size:
            tm, rm = tm[:out[0] + 1], rm[:out[0] + 1]
        axes[1].plot(tm, rm, color=c, lw=1.1)
        m = R[:, 0] >= 2e-3
        axes[0].plot(R[m, 0], w[m], color=c, lw=1.1, label=f"$U={u:g}$")
    tt = np.logspace(np.log10(2e-3), np.log10(0.6), 200)
    axes[0].plot(tt, np.sin(8*tt)**2/(8*tt)**2, color=INK, lw=0.8, ls=":")
    axes[0].plot([0.012, 0.0178], [0.612, 0.612], color=INK, lw=0.8, ls=":")   # line sample, left of the label
    axes[0].text(0.0197, 0.55, r"$\sin^2(Ut/2)/(Ut/2)^2$" "\n" r"$U=16$", fontsize=6.5, color=INK, ha="left",
                 multialignment="right")
    for ax in axes:
        ax.axhline(1.0, color=INK, lw=0.7, ls=(0, (3, 2)))
    axes[0].set_xscale("log")
    axes[0].set_xlim(2e-3, 0.6)
    axes[1].set_ylim(0.85, 1.3)
    axes[1].set_yticks([0.9, 1.0, 1.1, 1.2])
    axes[0].set_ylim(0.4, 1.08)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel(r"$n_{\rm g}/(C\,U^2t^4)$")
    axes[0].set_ylabel(r"$(1-|\mathcal{R}_1|^2)/(U^2t^2/4)$")
    axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=6.5, ncol=2, loc="upper left")
    for k, ax in enumerate(axes):
        tag(ax, "ab"[k])
    finish(axes)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.11, hspace=0.12)
    save(fig, "fig_ng")

if __name__ == "__main__":
    if sys.argv[1:]:                                 # e.g. plot_figs.py fig_ng: only the named figures
        for f in sys.argv[1:]:
            globals()[f]()
        sys.exit()
    fig_convergence()
    for u in (6.0, 8.0):                             # candidates for the extra strong-coupling U
        fig_dmft(u, f"fig_dmft_U{u:g}")
    fig_dmft(DMFT_EXTRA_U)
    fig_crossover()
    fig_strong()
    fig_ng()
