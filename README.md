# Data and scripts for the figures of the adaptive td-gGA quench paper

Raw data and Python scripts that produce every figure of the paper (half-filled Hubbard model on the
Bethe lattice, quench from U = 0, time-dependent ghost Gutzwiller with adaptive ghost insertion).
The folder layout mirrors the working repository, so the scripts run unchanged.

## Reproduce the figures

    cd paper_quench
    python plot_figs.py              # Figs. 1-6 except the schematic (also fig_dmft_U6/U8, candidate variants)
    python plot_figs.py fig_strong   # a single figure
    python adaptive_td_gga.py        # schematic of the adaptive scheme

Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11. Output: `fig_*.pdf`/`.png` in `paper_quench/`.

| Figure | file | function |
|---|---|---|
| schematic | `adaptive_td_gga.pdf` | `adaptive_td_gga.py` |
| convergence in B, U = 1.5 and 3 | `fig_convergence.pdf` | `plot_figs.fig_convergence` |
| td-gGA vs DMFT at all U | `fig_dmft.pdf` | `plot_figs.fig_dmft` |
| td-GA transition to crossover | `fig_crossover.pdf` | `plot_figs.fig_crossover` |
| strong coupling | `fig_strong.pdf` | `plot_figs.fig_strong` (analysis in `dU_scan/strong_B15.py`) |
| short-time ghost occupation | `fig_ng.pdf` | `plot_figs.fig_ng` |

## Data

Column formats: `observables.dat` (t, double occupancy d), `nk_td.dat` / `nk.dat` (momentum distribution
n(eps, t)), `delta_eig.dat` (t, eigenvalues of the embedding density matrix), `R.dat` (t, Re/Im R),
`adaptive.log` (ghost-pair insertion times), `DONE` (marks a finished run; read by the scripts).

- `dU_scan/U0_dU<U>_B<B>/`: adaptive td-gGA, quench 0 -> U, B <= B (11, 13, 15; B <= 15 to t = 12);
  `U0_dU8_B15_fine`: output every 0.02.
- `gpu/lowrank/B17*_U<U>/`: B <= 17 (low-rank GPU solver) to t = 10 for 2.5 <= U <= 10, continuing the
  B <= 15 runs from the 15 -> 17 insertion (`stage_B17`).
- `crossover_scan/U<U>/B<B>/`: B-resolved quenches (B <= 1 ... 15) through the dynamical crossover;
  `B1_dense`: td-GA on a fine U grid.
- `dU_scan/thermal/`, `crossover_scan/thermal/`, `finite_T/`: thermal gGA (B = 3) scans of E(T), d(T)
  at the final U, for T_eff and d_inf.
- `dU_scan/ekw_digitized/`: DMFT curves digitized from M. Eckstein, M. Kollar, P. Werner,
  Phys. Rev. Lett. 103, 056403 (2009), Fig. 2: d(t), Delta n(t) and the thermal values d_th (arrows).
- `insert_test/ng_short_time/`: B = 3 runs with the first ghost pair inserted at t = 0
  (`negseed/`: opposite sign of the seed); `insert_test/adaptive/`: B <= 11 references.
