"""Strong-coupling analysis of the quenches 0 -> U >= 2.5 with the converged runs of the paper (Fig. 4, Sec. III.D):
B <= 17 at all U (paper_quench/plot_figs.b17), cut at min(t_conv, 10); the damped-cosine fit uses t = 0 ... 10.
Per U: t_conv, first minimum of d(t) vs pi/U and vs the first-order value 1/4 - |E0|/U, rms deviation from the
first-order result 1/4 - |E0|/(2U) (1 - cos Ut) over the first period, damped-cosine fit with the constrained even
envelope of fit_even (omega, its spread over window ends t = 8.5 ... 10, weight w and rate gamma of the tail, drift),
algebraic decay exponent of the envelope for t > 2, late level (mean over the last two periods 2 pi/U), d_stat, d_inf.
Writes strong_B15.txt; strong_level(u) etc. are used by plot_figs.fig_strong."""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper_quench"))
import analyze_scan as sc          # noqa: E402

E0 = 4/(3*np.pi)
TMAX = 10.0
STRONG_U = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
SIGMA2 = 0.75                       # variance of the doublon-holon band (Appendix B; ../strong_c/sigma2.py)
TENDS = (8.5, 9.0, 9.5, 10.0)       # ends of the fit window for the spread of omega


def envelope(t, a, w, g):
    """even envelope a [(1 - w) exp(-b t^2) + w sech(g t)] with exponential tail ~ 2 w e^{-g t}; b is fixed by the
    analytic initial curvature, (1 - w) b + w g^2/2 = sigma^2/2, so A(t)/A(0) = 1 - sigma^2 t^2/2 + O(t^4)"""
    b = (SIGMA2/2 - w*g*g/2)/(1 - w)
    return a*((1 - w)*np.exp(-b*t*t) + w/np.cosh(g*t))


def damped_even(t, dbar, a, w, g, om, ph, s):
    """d_fit(t) = dbar + s (t - 5) + A(t) cos(om t + ph), paper Eq. (fit) with its dbar = dbar - 5 s (same function)"""
    return dbar + s*(t - 5) + envelope(t, a, w, g)*np.cos(om*t + ph)


def fit_even(t, d, t0, tend, u):
    """fit of damped_even to d(t) on t0 <= t <= tend; starting values from the single-exponential fit"""
    m = (t >= t0) & (t <= tend)
    p, _ = curve_fit(sc.damped, t[m], d[m], p0=[d[m].mean(), 0.05, 0.4, u*0.8, 0, -0.001], maxfev=40000)
    q, _ = curve_fit(damped_even, t[m], d[m], p0=[p[0], abs(p[1]), 0.3, 0.15, p[3], p[4], p[5]],
                     bounds=([0, 0, 0.01, 0.01, 0, -10, -1], [1, 1, 0.9, 1.0, 2*u, 10, 1]), maxfev=200000)
    return q, m


def series(u):
    """(t, d, t_conv, B) of the converged run at U = u, or None"""
    import plot_figs as pf
    r = pf.b17(u)
    if r is not None:
        d, _, tc = r
        return d[:, 0], d[:, 1], tc, 17
    a = sc.load(u, 15)
    if a is None:
        return None
    return a[:, 0], a[:, 1], sc.t_converged_trig(u), 15


def analyze(u):
    s = series(u)
    if s is None:
        return None
    t, d, tc, B = s
    te = min(tc or TMAX, TMAX)
    ex = sc.extrema(t, d)
    out = dict(u=u, B=B, tc=tc, te=te, t=t, d=d)
    mn = [e for e in ex if e[0] == "min"]
    out["tmin1"], out["dmin1"] = mn[0][1], mn[0][2]
    out["xmin1"] = (0.25 - mn[0][2])*u/E0                        # scaled first minimum, 1 at first order
    p1 = (t <= 2*np.pi/u)
    d1 = 0.25 - E0/(2*u)*(1 - np.cos(u*t[p1]))
    out["rms1"] = np.sqrt(np.mean((d[p1] - d1)**2))
    t0 = 0.0                      # fit window t = 0 ... 10: the envelope has the analytic short-time form (2026-11-23)
    q, m = fit_even(t, d, t0, TMAX, u)
    oms = [fit_even(t, d, t0, te_, u)[0][4] for te_ in TENDS]
    out.update(omega=q[4], omega_lo=min(oms), omega_hi=max(oms), w=q[2], gamma=q[3], drift=q[6], fitpar=q,
               fitres=np.abs(damped_even(t[m], *q) - d[m]).max(),
               fitrms=np.sqrt(np.mean((damped_even(t[m], *q) - d[m])**2)))
    # envelope: half the difference of successive extrema, at their mean time, for t > 2
    ee = [e for e in ex if e[1] <= te]
    amp = np.array([(0.5*(a[1] + b[1]), 0.5*abs(a[2] - b[2])) for a, b in zip(ee, ee[1:])])
    k = amp[:, 0] > 2
    out["alpha"] = -np.polyfit(np.log(amp[k, 0]), np.log(amp[k, 1]), 1)[0] if k.sum() >= 3 else np.nan
    ml = (t > TMAX - 2*2*np.pi/u) & (t <= TMAX)                   # late level: last two periods before t = 10
    out["late"] = d[ml].mean()
    out["dstat"] = 0.25 - E0/(2*u)
    out["Teff"], out["dinf"] = sc.thermal_d(u)
    return out


def late_level(u):
    """(late level, d_inf) at U = u: mean over the last two periods 2 pi/U (U >= 2.5) or the last 2 time units
    before t = 10"""
    s = series(u)
    if s is None:
        return None
    t, d, tc, _ = s
    ml = (t > TMAX - (2*2*np.pi/u if u >= 2.5 else 2)) & (t <= TMAX)
    return d[ml].mean(), sc.thermal_d(u)[1]


def main():
    rows = [r for r in map(analyze, STRONG_U) if r is not None]
    hdr = (f"{'U':>4} {'B':>3} {'t_conv':>6} {'t_min1':>6} {'pi/U':>6} {'1-x_min1':>8} {'rms1':>7} {'omega/U':>7} "
           f"{'U(U-w)':>7} {'spread':>6} {'fitrms':>7} {'w':>5} {'gamma':>6} {'drift':>8} {'alpha':>6} {'late':>7} {'d_stat':>7} {'d_inf':>7} {'T_eff':>6}")
    lines = ["Strong coupling, converged td-gGA (B <= 17), fit t = 0 ... 10, other columns t <= min(t_conv, 10)",
             hdr]
    for r in rows:
        f = lambda x, fmt: (format(x, fmt) if x is not None and np.isfinite(x) else "nan").rjust(len(format(1.0, fmt)))
        lines.append(f"{r['u']:4g} {r['B']:3d} {r['tc'] if r['tc'] else np.nan:6.2f} {r['tmin1']:6.3f} "
                     f"{np.pi/r['u']:6.3f} {1 - r['xmin1']:8.4f} {r['rms1']:7.1e} {r['omega']/r['u']:7.3f} "
                     f"{r['u']*(r['u'] - r['omega']):7.3f} {r['u']*(r['omega_hi'] - r['omega_lo']):6.3f} {r['fitrms']:7.1e} {r['w']:5.3f} {r['gamma']:6.3f} {r['drift']:+8.1e} {r['alpha']:6.2f} "
                     f"{r['late']:7.4f} {r['dstat']:7.4f} {r['dinf'] if r['dinf'] else np.nan:7.4f} "
                     f"{r['Teff'] if r['Teff'] else np.nan:6.3f}")
    open(os.path.join(HERE, "strong_B15.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
