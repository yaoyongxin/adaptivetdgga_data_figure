"""B-resolved crossover scan: d(t), Delta n(t) and crossover diagnostics vs U for B <= 1, 3, ..., 15 (B <= 15: links to dU_scan/U0_dU<U>_B15).

    python3 analyze_crossover.py   -> crossover_summary.txt, crossover_dt.png, crossover_diag.png (+ pdf)

Per (U, Bmax), from U<U>/B<Bmax>/observables.dat and U<U>/nk_B<Bmax>.dat (Delta n = n(0-) - n(0+) at
the Gauss nodes closest to eps = 0, +-0.0021):
- Delta n_1: the first revival maximum, only after a collapse (first local minimum < 0.05): the
  first local maximum after it (t <= 7, height above the minimum > 0.005);
- Delta n_stat (no revival): the prethermal plateau, Delta n where |d Delta n/dt| is smallest for
  1 <= t <= 7 up to the first local minimum (only if Delta n there is > 0.01 and |slope| < 0.02);
- dist_d: rms of d(t) - d_inf over WIN = [4, 7] (d_inf from thermal gGA, B = 3);
- amp_d: half the peak-to-peak range of d(t) over WIN (oscillation amplitude).
DMFT reference: EKW Fig. 3(a) (dU_scan/ekw_digitized/ekw_fig3a.dat; U = U_EKW/2).
"""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E0 = 4/(3*np.pi)
BS = [1, 3, 5, 7, 9, 11, 13, 15]
WIN = (4.0, 7.0)
UC_GA = 16/(3*np.pi)          # td-GA dynamical transition U_c^BR/2 (Brinkman-Rice U_c^BR = 8|E0| = 3.395)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CB = {1: MUTED, 3: "#e87ba4", 5: "#eda100", 7: "#1baf7a", 9: "#eb6834", 11: "#2a78d6", 13: INK, 15: "#000080"}


def us():
    vals = (float(os.path.basename(d)[1:]) for d in glob.glob(f"{HERE}/U*") if os.path.isdir(d))
    return sorted(u for u in vals if u <= 2.0)            # U3/ is for Fig. 1 of paper_quench only


def load_d(u, b):
    f = f"{HERE}/U{u:g}/B{b}/observables.dat"
    return np.loadtxt(f) if os.path.exists(f) else None


def jump(u, b):
    f = f"{HERE}/U{u:g}/nk_B{b}.dat"
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


def thermal_d(u, interp=True):
    """(T_eff, d_inf) from thermal gGA (B = 3) at E(T) = -E0; for a U without thermal runs (U = 1.52 ... 1.58)
    a cubic spline in U through the U that have them (leave-one-out error ~1e-5 near U = 1.5-1.6; checked
    against GEM at U = 1.52 ... 1.58, which now have their own runs: <= 1e-5)."""
    fs = (glob.glob(f"{HERE}/thermal/U{u:g}_B3*.dat") + glob.glob(f"{ROOT}/dU_scan/thermal/U{u:g}_B3*.dat")
          + glob.glob(f"{ROOT}/finite_T/U{u}_B3*.dat"))
    if not fs:
        if not interp:
            return None, None
        from scipy.interpolate import CubicSpline
        pts = [(x, *thermal_d(x, False)) for x in us()]
        pts = np.array([q for q in pts if q[2] is not None])
        return float(CubicSpline(pts[:, 0], pts[:, 1])(u)), float(CubicSpline(pts[:, 0], pts[:, 2])(u))
    a = np.vstack([np.atleast_2d(np.loadtxt(f)) for f in fs])
    a = a[np.argsort(a[:, 0])]
    T, d, E = a[:, 0], a[:, 2], a[:, 3]
    k = np.where(np.diff(np.sign(E + E0)))[0]
    if not len(k):
        return None, None
    k = k[0]
    w = (-E0 - E[k])/(E[k+1] - E[k])
    return T[k] + w*(T[k+1] - T[k]), d[k] + w*(d[k+1] - d[k])


def t_conv(u):
    a, b = load_d(u, 13), load_d(u, 11)
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    off = np.where(np.abs(a[:n, 1] - b[:n, 1]) > 1e-3)[0]
    return a[off[0], 0] if len(off) else a[n-1, 0]


def dn_features(j):
    """(Delta n_stat, t), (Delta n_1, t) or None. Revival only after a collapse (first minimum < 0.05,
    as in DMFT for U > U_c^dyn); otherwise the plateau: smallest |slope| for 1 <= t <= 7 up to the
    first minimum (B = 1 at weak U: the first minimum of its undamped oscillation)"""
    t, y = j[:, 0], j[:, 1]
    m = t <= 7.05
    t, y = t[m], y[m]
    mins = [k for k in range(1, len(y) - 1) if y[k] <= y[k-1] and y[k] < y[k+1]]
    rev = None
    if mins and y[mins[0]] < 0.05:
        k0 = mins[0]
        maxs = [k for k in range(k0 + 1, len(y) - 1) if y[k] >= y[k-1] and y[k] > y[k+1]
                and y[k] - y[k0] > 0.005]
        if maxs:
            rev = (y[maxs[0]], t[maxs[0]])
    stat = None
    s = np.abs(np.gradient(y, t))
    w = (t >= 1) & (t <= 7) & (y > 0.01)
    if mins:
        w &= t <= t[mins[0]] + 1e-9
    if rev is None and w.any():
        k = np.where(w)[0][np.argmin(s[w])]
        if s[k] < 0.02:
            stat = (y[k], t[k])
    return stat, rev


def ekw3a():
    rows = [l.split() for l in open(f"{ROOT}/dU_scan/ekw_digitized/ekw_fig3a.dat") if l[0] != "#"]
    return {k: np.array([[float(r[2]), float(r[3])] for r in rows if r[0] == k]) for k in ("dn_stat", "dn_1")}


def main():
    U = us()
    res = {}
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"B-resolved crossover scan; dist_d, amp_d over t in {WIN}; Delta n_stat (t), Delta n_1 (t)")
    for u in U:
        Te, dinf = thermal_d(u)
        tc = t_conv(u)
        say(f"U = {u:g}: T_eff = {Te if Te else np.nan:.3f}, d_inf = {dinf if dinf else np.nan:.4f}, "
            f"B<=13 vs B<=11 within 1e-3 to t = {tc if tc else np.nan:.2f}")
        for b in BS:
            a = load_d(u, b)
            if a is None:
                continue
            t, d = a[:, 0], a[:, 1]
            m = (t >= WIN[0]) & (t <= WIN[1])
            dist = np.sqrt(np.mean((d[m] - dinf)**2)) if dinf else np.nan
            amp = 0.5*(d[m].max() - d[m].min())
            j = jump(u, b)
            stat, rev = dn_features(j) if j is not None else (None, None)
            res[(u, b)] = dict(t=t, d=d, j=j, dist=dist, amp=amp, stat=stat, rev=rev, dinf=dinf)
            say(f"   B<={b:2d}: dist_d {dist:.4f}  amp_d {amp:.4f}  "
                f"dn_stat {('%.3f (t=%.1f)' % stat) if stat else '   -        '}  "
                f"dn_1 {('%.3f (t=%.2f)' % rev) if rev else '-'}")
    open(f"{HERE}/crossover_summary.txt", "w").write("\n".join(lines) + "\n")
    plot_dt(res, U)
    plot_diag(res, U)


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8})


def finish(axes):
    for ax in np.ravel(axes):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def plot_dt(res, U):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    show = [u for u in (0.85, 1.25, 1.6, 2.0) if any((u, b) in res for b in BS)]
    fig, axes = plt.subplots(2, len(show), figsize=(3.0*len(show), 5.4), dpi=200, sharex=True)
    axes = np.atleast_2d(axes).reshape(2, len(show))
    for i, u in enumerate(show):
        for b in BS:
            r = res.get((u, b))
            if r is None:
                continue
            lw = 1.3 if b == 13 else 0.9
            axes[0, i].plot(r["t"], r["d"], color=CB[b], lw=lw, label=f"$B\\leq{b}$")
            if r["j"] is not None:
                axes[1, i].plot(r["j"][:, 0], r["j"][:, 1], color=CB[b], lw=lw)
        if res.get((u, 13)) and res[(u, 13)]["dinf"]:
            axes[0, i].axhline(res[(u, 13)]["dinf"], color=INK, lw=0.7, ls=(0, (4, 3)))
        axes[0, i].set_title(f"$U = {u:g}$", fontsize=9)
        axes[1, i].set_xlabel("$t$")
        axes[1, i].set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("$d(t)$")
    axes[1, 0].set_ylabel(r"$\Delta n(t)$")
    axes[0, -1].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    fig.suptitle("td-gGA, adaptive $B \\leq 1, \\ldots, 13$ (dashed: thermal $d_\\infty$)", fontsize=9)
    finish(axes)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.09, wspace=0.25, hspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(f"{HERE}/crossover_dt.{ext}")


def plot_diag(res, U):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    ek = ekw3a()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=200)
    for b in BS:
        uu = [u for u in U if (u, b) in res]
        st = [(u, res[(u, b)]["stat"][0]) for u in uu if res[(u, b)]["stat"]]
        rv = [(u, res[(u, b)]["rev"][0]) for u in uu if res[(u, b)]["rev"]]
        lw = 1.3 if b == 13 else 0.9
        if st:
            axes[0].plot(*np.array(st).T, "-o", color=CB[b], lw=lw, ms=2.5)
        if rv:
            axes[0].plot(*np.array(rv).T, "--s", color=CB[b], lw=lw, ms=2.5)
        axes[1].plot(uu, [res[(u, b)]["dist"] for u in uu], "-o", color=CB[b], lw=lw, ms=2.5,
                     label=f"$B\\leq{b}$")
        axes[2].plot(uu, [res[(u, b)]["amp"] for u in uu], "-o", color=CB[b], lw=lw, ms=2.5)
    axes[0].plot(*ek["dn_stat"].T, "x", color=INK, ms=5, mew=1.2)
    axes[0].plot(*ek["dn_1"].T, "+", color=INK, ms=6, mew=1.2)
    axes[0].set_xlim(0.55, 2.05)
    axes[0].set_ylim(-0.02, 1.0)
    axes[0].set_title(r"$\Delta n_{\rm stat}$ (solid), $\Delta n_1$ (dashed); DMFT: x, +", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_title(r"rms $d(t) - d_\infty$, $4 \leq t \leq 7$", fontsize=8)
    axes[2].set_yscale("log")
    axes[2].set_title(r"oscillation amplitude of $d$, $4 \leq t \leq 7$", fontsize=8)
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    for ax in axes:
        ax.set_xlim(0.55, 2.05)
        ax.set_xlabel("$U$")
        ax.axvline(1.6, color=MUTED, lw=0.7, ls=":")
        ax.axvline(UC_GA, color=MUTED, lw=0.7, ls="--")
    for x, lab in ((1.6, "DMFT $U_c^{\\rm dyn}$"), (UC_GA, "td-GA $U_c$")):
        axes[0].text(x + (-0.015 if x < 1.65 else 0.015), 0.6, lab, transform=axes[0].get_xaxis_transform(),
                     fontsize=6.5, color=MUTED, ha="right" if x < 1.65 else "left")
    finish(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.14, wspace=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(f"{HERE}/crossover_diag.{ext}")


if __name__ == "__main__":
    main()
