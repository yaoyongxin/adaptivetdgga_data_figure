"""dU scan of td-gGA quenches 0 -> dU vs DMFT: d(t), oscillation period, damping and relaxation vs U.

    python3 analyze_scan.py      (from anywhere; writes dU_scan/scan_summary.txt, scan_d.png/pdf,
                                  scan_levels.png/pdf)

Inputs: adaptive B <= 13 runs (dU_scan/U0_dU<U>_B13), B <= 11 runs (dU_scan/, or insert_test/adaptive
for dU = 1.5, 2.5): convergence time = where B <= 13 departs from B <= 11 by 1e-3, plus one pair
spacing; thermal gGA (GEM, B = 3; finite_T/ for 1.5, 2.5, dU_scan/thermal/ otherwise) for d_inf;
DMFT only from the digitized Fig. 2(a), (b) of Eckstein, Kollar, Werner, PRL 103, 056403 (2009)
(ekw_digitized/*_smooth.dat: robust splines through the digitized points with the exact short-time
expansion, smooth_ekw.py; tick-mark axis calibration, rms 0.0006 / 0.0003 against the independent manual
digitizations paper/dt_u{15,25}_tdmft.dat;
units converted: U = U_EKW/2, t = 2 t_EKW).
Strong-coupling references (E0 = -4/(3 pi)): first order in the hopping,
    d(t) = 1/4 - |E0|/(2U) (1 - cos Ut),
i.e. period 2 pi/U and centre d_stat = 1/4 - |E0|/(2U) (EKW Eq. 5), and the high-T thermal value
d_th = 1/4 - |E0|/U (half of the relaxation happens in the prethermal stage).
"""
import glob
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E0 = 4/(3*np.pi)
DUS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.65, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]
WEAK = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]          # EKW Fig. 2(a), (c)
STRONG = [1.65, 2.0, 2.5, 3.0, 4.0]               # EKW Fig. 2(b), (d)
BEYOND = [3.5, 5.0, 8.0]                          # no DMFT
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b"]


def run_dir(u, bmax):
    d = f"{HERE}/U0_dU{u:g}_B{bmax}"
    if not os.path.exists(d) and u in (1.5, 2.5):
        d = (f"{ROOT}/B15/U{u:g}_B15" if bmax == 15 else f"{ROOT}/insert_test/adaptive/U0_dU{u:g}_B{bmax}")
    return d


def load(u, bmax):
    f = f"{run_dir(u, bmax)}/observables.dat"
    return np.loadtxt(f) if os.path.exists(f) else None


def insert_times(u, bmax=13):
    f = f"{run_dir(u, bmax)}/adaptive.log"
    return [float(l.split("t=")[1].split()[0]) for l in open(f) if "insert at" in l]


def t_converged(u, bmax=13):
    """first time d(B <= bmax) departs from d(B <= bmax - 2) by 1e-3, plus one pair spacing (the
    last two insertion times of the B <= bmax run); the end of the shorter run if they never depart"""
    a, b = load(u, bmax), load(u, bmax - 2)
    if a is None or b is None:
        return None
    m = {round(r[0], 4): r[1] for r in b}
    t = np.array([r[0] for r in a if round(r[0], 4) in m])
    dev = np.array([abs(r[1] - m[round(r[0], 4)]) for r in a if round(r[0], 4) in m])
    off = np.where(dev > 1e-3)[0]
    ti = insert_times(u, bmax)
    spacing = ti[-1] - ti[-2] if len(ti) >= 2 else 0
    return (t[off[0]] if len(off) else t[-1]) + spacing



def trigger(u, bmax, thres=1e-5):
    """time at which the outermost pair of the last stage (B = bmax) reaches thres, i.e. when a run
    with B <= bmax + 2 would insert its next pair (gpu_ed.py adapt accuracy estimate); None if never"""
    f = f"{run_dir(u, bmax)}/stage_B{bmax}/delta_eig.dat"
    if not os.path.exists(f):
        return None
    e = np.atleast_2d(np.loadtxt(f))
    dmin = np.minimum(e[:, 1:].min(1), 1 - e[:, 1:].max(1))
    k = np.nonzero(dmin >= thres)[0]
    return e[k[0], 0] if len(k) else None


def t_converged_trig(u, bmax=15, lag_min=2.15):
    """t_conv of B <= bmax without a B <= bmax + 2 run: trigger(u, bmax) plus the lag after which
    B <= bmax - 2 and bmax - 4 depart by 1e-3 at the same U (both pairs give the same lag to ~0.25);
    if they never depart, the lag is at least the rest of their window (and lag_min). None: no limit
    in the window (the next pair is never needed)."""
    tt = trigger(u, bmax)
    if tt is None:
        return None
    a, b, t_prev = load(u, bmax - 2), load(u, bmax - 4), trigger(u, bmax - 4)
    lag = lag_min
    if a is not None and b is not None and t_prev is not None:
        m = {round(r[0], 4): r[1] for r in b}
        rows = [(r[0], abs(r[1] - m[round(r[0], 4)])) for r in a if round(r[0], 4) in m]
        off = [t for t, x in rows if x > 1e-3]
        lag = (off[0] - t_prev) if off else max(lag_min, rows[-1][0] - t_prev)
    return tt + lag


def thermal_d(u):
    """d_inf from thermal gGA (B = 3): linear interpolation at E(T) = -E0."""
    fs = glob.glob(f"{ROOT}/finite_T/U{u}_B3*.dat") + glob.glob(f"{HERE}/thermal/U{u:g}_B3*.dat")
    if not fs:
        return None, None
    a = np.vstack([np.atleast_2d(np.loadtxt(f)) for f in fs])
    a = a[np.argsort(a[:, 0])]
    T, d, E = a[:, 0], a[:, 2], a[:, 3]
    k = np.where(np.diff(np.sign(E + E0)))[0]
    if not len(k):
        return None, None
    k = k[0]
    w = (-E0 - E[k])/(E[k+1] - E[k])
    return T[k] + w*(T[k+1] - T[k]), d[k] + w*(d[k+1] - d[k])


def extrema(t, d, min_sep=0.0):
    """parabolic-refined extrema; pairs closer than min_sep (digitization noise) are dropped"""
    out = []
    for k in range(1, len(t) - 1):
        if (d[k] - d[k-1])*(d[k+1] - d[k]) < 0:
            y0, y1, y2 = d[k-1:k+2]
            h = t[1] - t[0]
            den = y0 - 2*y1 + y2
            x = 0.5*(y0 - y2)/den
            out.append(("max" if den < 0 else "min", t[k] + x*h, y1 - 0.25*(y0 - y2)*x))
    k = 0
    while k < len(out) - 1:
        if out[k+1][1] - out[k][1] < min_sep:
            del out[k:k+2]
        else:
            k += 1
    return out


def damped(t, dbar, A, g, w, ph, s):
    return dbar + s*(t - 5) + A*np.exp(-g*t)*np.cos(w*t + ph)


def digitized(fn):
    """smooth EKW curves (ekw_digitized/*_smooth.dat, robust splines through the digitized points,
    see smooth_ekw.py) -> {U_ours: (t, y)} in our units"""
    out = {}
    for l in open(fn):
        if l[0] != "#":
            s, t, y = l.split()
            out.setdefault(float(s[1:])/2, []).append((2*float(t), float(y)))
    return {u: tuple(np.array(v).T) for u, v in out.items()}


def ekw_d():
    dm = digitized(f"{HERE}/ekw_digitized/ekw_fig2a_d_smooth.dat")
    dm.update(digitized(f"{HERE}/ekw_digitized/ekw_fig2b_d_smooth.dat"))
    th = {}
    for f in ("ekw_fig2a_dth_arrows.dat", "ekw_fig2b_dth_arrows.dat"):
        th.update({r[1]: r[2] for r in np.loadtxt(f"{HERE}/ekw_digitized/{f}")})
    return dm, th


def main():
    dm, dmth = ekw_d()
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("dU scan, td-gGA adaptive B <= 13, U_i = 0 -> U = dU (half-bandwidth 1); DMFT = digitized EKW Fig. 2")
    say(f"{'U':>4} {'B_end':>5} {'t_conv':>6} {'T_eff':>6} {'d_inf':>7} {'d_th DMFT':>9} {'d_stat sc':>9} "
        f"{'d_med':>7} {'d_med DMFT':>10} {'late':>7} {'d(10)':>7}")
    res = {}
    for u in DUS:
        a = load(u, 13)
        if a is None:
            continue
        t, d = a[:, 0], a[:, 1]
        tc = t_converged(u)
        Te, dth = thermal_d(u)
        ex = extrema(t, d)
        mx = [e for e in ex if e[0] == "max"]
        mn = [e for e in ex if e[0] == "min"]
        dmed = 0.5*(mx[0][2] + mn[1][2]) if len(mx) >= 1 and len(mn) >= 2 and u >= 2.5 else np.nan
        dmed_dm, exd = np.nan, []
        if u in dm:
            exd = [e for e in extrema(*dm[u]) if e[1] > 0.3]
            mxd = [e for e in exd if e[0] == "max"]
            mnd = [e for e in exd if e[0] == "min"]
            if mxd and len(mnd) >= 2 and u >= 2.5:
                dmed_dm = 0.5*(mxd[0][2] + mnd[1][2])
        # late level: mean over the last two periods 2 pi/U (U >= 2.5) or the last 2 time units
        # before t_conv (or t = 10)
        te = min(tc or 10, 10)
        m = (t > te - (2*2*np.pi/u if u >= 2.5 else 2)) & (t <= te)
        late = d[m].mean()
        res[u] = dict(t=t, d=d, tc=tc, Te=Te, dth=dth, ex=ex, exd=exd, late=late, dmed=dmed,
                      dmed_dm=dmed_dm, B=int(a[-1, 2]) if a.shape[1] > 2 else 13)
        say(f"{u:4g} {res[u]['B']:5d} {tc if tc else np.nan:6.2f} {Te if Te else np.nan:6.3f} "
            f"{dth if dth else np.nan:7.4f} {dmth.get(u, np.nan):9.4f} {0.25 - E0/(2*u):9.4f} "
            f"{dmed:7.4f} {dmed_dm:10.4f} {late:7.4f} {d[-1]:7.4f}")
    say()
    say("td-gGA - DMFT d(t) on the common, B-converged window (t >= 0.2): mean, rms, max |.|;")
    say("late: means over the last 1.5 time units of the DMFT data (td-gGA, DMFT)")
    for u, r in res.items():
        if u not in dm:
            continue
        td, yd = dm[u]
        m = (td >= 0.2) & (td <= min(r["tc"] or 10, r["t"][-1]))
        dev = np.interp(td[m], r["t"], r["d"]) - yd[m]
        ml = m & (td >= td[m][-1] - 1.5)
        r["dev"] = dev
        r["late_dm"] = (np.interp(td[ml], r["t"], r["d"]).mean(), yd[ml].mean(), td[m][-1])
        say(f"  U = {u:4g}: t <= {td[m][-1]:5.2f}: mean {dev.mean():+.4f}  rms {np.sqrt((dev**2).mean()):.4f}  "
            f"max {np.abs(dev).max():.4f};  late {r['late_dm'][0]:.4f} vs {r['late_dm'][1]:.4f}")
    say()
    say("extrema of d(t) (t, d); half-periods vs pi/U; DMFT extrema below")
    for u, r in res.items():
        ex = r["ex"]
        ts = [e[1] for e in ex]
        hp = np.diff(ts)
        say(f"U = {u:g}: pi/U = {np.pi/u:.3f}, 2 pi/U = {2*np.pi/u:.3f}; t_conv ~ {r['tc']:.1f}" if r["tc"]
            else f"U = {u:g}: pi/U = {np.pi/u:.3f}")
        say("   " + "  ".join(f"{e[0]} {e[1]:.3f} ({e[2]:.4f})" for e in ex))
        say("   half-periods: " + " ".join(f"{h:.3f}" for h in hp))
        if r["exd"]:
            say("   DMFT: " + "  ".join(f"{e[0]} {e[1]:.3f} ({e[2]:.4f})" for e in r["exd"]))
        # damped-cosine fit on [t_first_min, t_conv]
        if u >= 2 and len(ex) >= 3 and r["tc"]:
            m = (r["t"] >= ex[0][1] - 0.3) & (r["t"] <= min(r["tc"], 10))
            try:
                p, _ = curve_fit(damped, r["t"][m], r["d"][m],
                                 p0=[r["d"][m].mean(), 0.05, 0.4, u*0.8, 0, -0.001], maxfev=40000)
                res_ = np.abs(damped(r["t"][m], *p) - r["d"][m]).max()
                r["omega"] = p[3]
                say(f"   fit on [{r['t'][m][0]:.2f}, {r['t'][m][-1]:.2f}]: omega = {p[3]:.3f} "
                    f"(omega/U = {p[3]/u:.3f}, period {2*np.pi/p[3]:.3f}), gamma = {p[2]:.3f}, "
                    f"drift {p[5]:+.4f}/time, max residual {res_:.1e}")
            except RuntimeError:
                say("   fit failed")
    open(f"{HERE}/scan_summary.txt", "w").write("\n".join(lines) + "\n")
    plot_d(res, dm)
    plot_levels(res, dmth)


def style(plt):
    INK, MUTED = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8})
    return INK, MUTED, "#e4e3df"


def finish(axes, GRID):
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def spread(ys, gap):
    """label positions: keep the order of ys, at least gap apart"""
    i = np.argsort(ys)
    y = np.array(ys, float)[i]
    for k in range(1, len(y)):
        y[k] = max(y[k], y[k-1] + gap)
    y -= (y.mean() - np.mean(ys))*0.5
    out = np.empty_like(y)
    out[i] = y
    return out


def plot_d(res, dm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=200)
    groups = [(WEAK, "EKW Fig. 2(a): $U \\leq 1.5$", 0.0025),
              (STRONG, "EKW Fig. 2(b): $U \\geq 1.65$", 0.004),
              (BEYOND, "beyond the PRL (no DMFT)", 0.004)]
    for ax, (us, title, gap) in zip(axes, groups):
        us = [u for u in us if u in res]
        ylab = []
        for u, c in zip(us, COLS):
            r = res[u]
            tc = min(r["tc"] or 10, 10.01)
            ok = r["t"] <= tc
            ax.plot(r["t"][ok], r["d"][ok], color=c, lw=1.2)
            ax.plot(r["t"][~ok], r["d"][~ok], color=c, lw=1.0, ls=(0, (2, 1.5)))
            if u in dm:
                td, yd = dm[u]
                k = np.arange(0, len(td), 15)
                ax.plot(td[k], yd[k], "o", color=c, ms=2.2, mfc="none", mew=0.7)
            ylab.append(r["d"][-1])
        for u, c, y in zip(us, COLS, spread(ylab, gap)):
            ax.text(10.2, y, f"$U={u:g}$", color=c, fontsize=7, va="center")
        ax.set_xlim(0, 11.6)
        ax.set_xlabel("$t$")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("$d(t)$")
    fig.suptitle("td-gGA (lines; dashed: past B convergence) vs DMFT (circles, digitized)", fontsize=8.5)
    finish(axes, GRID)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.14, wspace=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(f"{HERE}/scan_d.{ext}")


def plot_levels(res, dmth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200)
    # half-periods (U >= 2.5, collapse and revival)
    ax = axes[0]
    for u, r in res.items():
        if u < 2.5:
            continue
        ts = [e[1] for e in r["ex"]]
        hp = np.diff(ts)
        mids = 0.5*(np.array(ts[1:]) + np.array(ts[:-1]))
        ok = mids <= (r["tc"] or 10)
        ax.plot([u], [ts[0]*u/np.pi], "s", color=COLS[0], ms=4)
        ax.plot(np.full(ok.sum(), u), hp[ok]*u/np.pi, "o", color=COLS[1], ms=3.2, mfc="none", mew=0.9)
        if r["exd"]:
            tsd = [e[1] for e in r["exd"]]
            ax.plot([u + 0.12], [tsd[0]*u/np.pi], "x", color=INK, ms=4)
            if len(tsd) > 1:
                ax.plot(np.full(len(tsd) - 1, u + 0.12), np.diff(tsd)*u/np.pi, "+", color=INK, ms=4)
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.text(6.6, 1.006, r"$\pi/U$", ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(4.6, 1.33, "td-gGA: first minimum (squares),\nlater half-periods (circles);\n"
            "DMFT: x, +", fontsize=6.5, color=MUTED, va="top")
    ax.set_xlabel("$U$")
    ax.set_ylabel(r"half-period $\times\, U/\pi$")
    ax.set_title("collapse-and-revival period", fontsize=8)
    # levels vs U
    ax = axes[1]
    U = np.linspace(1.5, 8.5, 200)
    ax.plot(U, 0.25 - E0/(2*U), color=MUTED, lw=0.9, ls="--")
    ax.plot(U, 0.25 - E0/U, color=MUTED, lw=0.9, ls=":")
    ax.text(2.6, 0.236, r"$d_{\rm stat}$ (strong coupl.)", fontsize=6.5, color=MUTED)
    ax.text(5.0, 0.25 - E0/5.0 - 0.011, r"$d_{\rm th}$ (strong coupl.)", fontsize=6.5, color=MUTED)
    us = list(res)
    ax.plot(us, [res[u]["dth"] for u in us], "-o", color=COLS[0], ms=3, lw=1, label="thermal gGA")
    ax.plot(list(dmth), list(dmth.values()), "x", color=INK, ms=4.5, label="thermal DMFT")
    ax.plot(us, [res[u]["late"] for u in us], "-s", color=COLS[1], ms=3, lw=1,
            label="td-gGA, late level")
    ul = [u for u in us if "late_dm" in res[u]]
    ax.plot(ul, [res[u]["late_dm"][1] for u in ul], "D", color=COLS[2], ms=3.2, mfc="none", mew=0.9,
            label="DMFT, end of its data")
    ax.plot(us, [res[u]["dmed"] for u in us], "^", color=COLS[3], ms=3.5, mfc="none",
            label=r"td-gGA, $d_{\rm med}$")
    ax.set_xlabel("$U$")
    ax.set_ylabel("$d$")
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0.05, 0.25)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("prethermal vs thermal level", fontsize=8)
    finish(axes, GRID)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.15, wspace=0.28)
    for ext in ("png", "pdf"):
        fig.savefig(f"{HERE}/scan_levels.{ext}")


if __name__ == "__main__":
    main()
