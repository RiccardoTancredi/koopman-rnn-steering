"""Draft figures from results/analysis/linear (figures 1-3, flip-flop sweep) and
results/analysis/kae (figure 4) and results/analysis/interventions (figure 5).

    uv run python scripts/05_figures.py
Writes PNG files to figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIN = ROOT / "results" / "analysis" / "linear"
KAE = ROOT / "results" / "analysis" / "kae"
INT = ROOT / "results" / "analysis" / "interventions"
FIG = ROOT / "figures"

# Reference categorical palette (fixed order) + neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA = "#eda100", "#e87ba4"
INK, MUTED, GRID = "#2b2b2b", "#8a8a85", "#d9d8d3"
TASKS = [("flipflop", "auto", "3-bit flip-flop"), ("sine", "omega0.30", "sine (ω = 0.30)"),
         ("dms", "auto", "DMS delay")]
CELLS = [("gru", "GRU"), ("tanh", "tanh RNN")]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "figure.dpi": 150,
})


def load(run: str, cond: str):
    p = LIN / run / f"{cond}.npz"
    return np.load(p) if p.exists() else None


def runs_for(task: str, cell: str):
    summary = json.loads((LIN / "summary.json").read_text())
    return [r for r, v in summary.items()
            if r.startswith(f"{task}_{cell}_") and not v.get("excluded")]


def fig1_trajectories():
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for i, (cell, cname) in enumerate(CELLS):
        for j, (task, cond, title) in enumerate(TASKS):
            ax = axes[i, j]
            d = load(f"{task}_{cell}_h64_s0", cond)
            if d is None:
                ax.set_axis_off()
                continue
            mu, B = d["pca_mean"], d["pca_basis"][:, :2]
            for tr in d["traj_h"]:
                z = (tr - mu) @ B
                ax.plot(z[:, 0], z[:, 1], color=MUTED, lw=0.6, alpha=0.6)
            if len(d["fp_h"]):
                z = (d["fp_h"] - mu) @ B
                st = d["fp_stable"]
                ax.scatter(z[st, 0], z[st, 1], s=36, color=BLUE, zorder=4, edgecolor="white",
                           linewidth=1, label="stable fixed point")
                ax.scatter(z[~st, 0], z[~st, 1], s=36, facecolor="white", edgecolor=ORANGE,
                           linewidth=1.5, zorder=3, label="unstable fixed point")
            ax.set_title(f"{cname} · {title}")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Task trajectories (grey) and fixed points of the autonomous map, seed 0")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG / "fig1_trajectories.png")
    plt.close(fig)


def fig2_spectra():
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    th = np.linspace(0, 2 * np.pi, 200)
    for i, (cell, cname) in enumerate(CELLS):
        for j, (task, cond, title) in enumerate(TASKS):
            ax = axes[i, j]
            d = load(f"{task}_{cell}_h64_s0", cond)
            if d is None:
                ax.set_axis_off()
                continue
            ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.8)
            if len(d["fp_h"]):
                # Stable points for the memory tasks; all points for the oscillator.
                sel = d["fp_stable"] if d["fp_stable"].any() else np.ones(len(d["fp_h"]), bool)
                lam = d["fp_eigvals"][sel].ravel()
                which = "stable fixed points" if d["fp_stable"].any() else "fixed points"
                ax.scatter(lam.real, lam.imag, s=14, marker="x", color=ORANGE, linewidth=1,
                           label=f"Jacobian at {which}")
            lam = d["dmd_eigvals"]
            ax.scatter(lam.real, lam.imag, s=16, facecolor="none", edgecolor=BLUE, linewidth=1,
                       label="global DMD (σ = 0.5)")
            if task == "sine":
                w = 0.30
                ax.plot([0, 1.15 * np.cos(w)], [0, 1.15 * np.sin(w)], color=INK, lw=0.6, ls=":")
                ax.text(1.17 * np.cos(w), 1.17 * np.sin(w), "ω", fontsize=8)
            ax.set_aspect("equal")
            lim = max(1.15, np.abs(lam).max() * 1.05)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{cname} · {title}")
            ax.set_xlabel("Re λ")
            ax.set_ylabel("Im λ")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab.split(" at ")[0] not in [x.split(" at ")[0] for x in labels]:
                handles.append(h)
                labels.append(lab.replace("stable fixed points", "fixed points"))
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Spectra: global DMD of the hidden states vs Jacobians at fixed points, seed 0")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG / "fig2_spectra.png")
    plt.close(fig)


def _files(task, cell):
    """All analysis files (one per run and condition) of the solved runs."""
    return [np.load(p) for run in runs_for(task, cell) for p in sorted((LIN / run).glob("*.npz"))]


def _mean_over(task, cell, key):
    """Stack a fixed-shape array across files: (n_files, ...)."""
    return np.array([d[key] for d in _files(task, cell)])


def fig3_prediction_and_local():
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    methods = [("dmd_full", "affine DMD, full rank", BLUE), ("dmd_r10", "affine DMD, rank 10", AQUA),
               ("persistence", "persistence", MUTED)]
    for j, (task, _, title) in enumerate(TASKS):
        ax = axes[0, j]
        for key, label, color in methods:
            for cell, ls in [("gru", "-"), ("tanh", "--")]:
                c = _mean_over(task, cell, f"curve_hidden_{key}")
                if len(c):
                    ax.plot(np.arange(1, c.shape[1]), c.mean(0)[1:], color=color, ls=ls, lw=1.5,
                            label=f"{label} ({cell})")
        ax.set_yscale("log")
        ax.set_title(title.split(" (")[0])
        ax.set_xlabel("prediction horizon (steps)")
        ax.set_ylabel("hidden-state NMSE")

        ax = axes[1, j]
        for cell, ls, color in [("gru", "-", BLUE), ("tanh", "--", ORANGE)]:
            files = _files(task, cell)
            loc = [np.nanmean(d["local_dist"], 0) for d in files if len(d["local_dist"])]
            if not loc:
                continue
            sig = files[0]["local_sigmas"]
            ax.plot(sig, np.mean(loc, 0), color=color, ls=ls, marker="o", ms=4, lw=1.5,
                    label=f"local DMD ({cell})")
            g = [np.min(d["global_vs_jac"]) for d in files if len(d["global_vs_jac"])]
            ax.axhline(np.mean(g), color=color, ls=":", lw=1, label=f"global DMD ({cell})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("perturbation size σ (× state std)")
        ax.set_ylabel("distance to Jacobian spectrum")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Top: k-step prediction error (mean over seeds; sine: over 3 frequencies). "
                 "Bottom: DMD vs Jacobian spectrum (10 slowest eigenvalues)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(FIG / "fig3_prediction_local.png")
    plt.close(fig)


def fig_flipflop_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for cell, ls in [("gru", "-"), ("tanh", "--")]:
        sig = _mean_over("flipflop", cell, "sweep_sigma")[0]
        for key, label, color in [("sweep_hidden_dmd", "affine DMD", BLUE),
                                  ("sweep_hidden_persist", "persistence", MUTED)]:
            axes[0].plot(sig, _mean_over("flipflop", cell, key).mean(0), color=color, ls=ls,
                         marker="o", ms=4, label=f"{label} ({cell})")
        axes[1].plot(sig, _mean_over("flipflop", cell, "sweep_basin_change").mean(0), color=ORANGE,
                     ls=ls, marker="o", ms=4, label=f"trials with ≥1 bit flipped ({cell})")
        axes[1].plot(sig, 1 - _mean_over("flipflop", cell, "sweep_final_sign_dmd").mean(0),
                     color=BLUE, ls=ls, marker="o", ms=4, label=f"DMD final-bit error, per bit ({cell})")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("perturbation size σ")
    axes[0].set_ylabel("hidden NMSE at 60 steps")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("perturbation size σ")
    axes[1].set_ylabel("fraction")
    axes[0].set_title("hidden-state error after 60 steps")
    axes[1].set_title("bit flips (true) vs DMD errors on final bits")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Flip-flop: prediction error vs perturbation size, and how often perturbations flip bits",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_flipflop_sweep.png")
    plt.close(fig)


def _kae_files(task, cell, sigma):
    cond = "omega0.30" if task == "sine" else "auto"
    return [np.load(p) for run in runs_for(task, cell)
            for p in [KAE / run / f"{cond}_sigma{sigma:g}_L16.npz"] if p.exists()]


def fig4_kae():
    """KAE vs DMD: prediction error for two perturbation sizes, and spectra (seed 0)."""
    sigmas = [0.5, 2.0]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.6))
    methods = [("kae", "Koopman AE (L = 16)", AQUA), ("dmd", "affine DMD", BLUE),
               ("persistence", "persistence", MUTED)]
    for j, (task, _, title) in enumerate(TASKS):
        for i, sigma in enumerate(sigmas):
            ax = axes[i, j]
            for key, label, color in methods:
                for cell, ls in [("gru", "-"), ("tanh", "--")]:
                    files = _kae_files(task, cell, sigma)
                    if not files:
                        continue
                    c = np.mean([d[f"curve_{key}"] for d in files], 0)
                    ax.plot(np.arange(1, len(c)), c[1:], color=color, ls=ls, lw=1.5,
                            label=f"{label} ({cell})")
            ax.set_yscale("log")
            ax.set_title(f"{title} · σ = {sigma:g}")
            ax.set_xlabel("prediction horizon (steps)")
            ax.set_ylabel("hidden-state NMSE")
        ax = axes[2, j]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.8)
        p = KAE / f"{task}_gru_h64_s0" / f"{'omega0.30' if task == 'sine' else 'auto'}_sigma0.5_L16.npz"
        if p.exists():
            d = np.load(p)
            lam = d["dmd_eigvals"]
            ax.scatter(lam.real, lam.imag, s=14, facecolor="none", edgecolor=BLUE, lw=1,
                       label="affine DMD (64 eigenvalues)")
            lam = d["kae_eigvals"]
            ax.scatter(lam.real, lam.imag, s=30, color=AQUA, edgecolor="white", lw=1, zorder=3,
                       label="Koopman AE (16 eigenvalues)")
            n_slow = int((np.abs(lam) > 0.97).sum())
            ax.text(0.57, 0.38, f"KAE: {n_slow} eigenvalues with |λ| > 0.97", fontsize=7)
        ax.plot(0.9 * np.cos(th), 0.9 * np.sin(th), color=MUTED, lw=0.6, ls=":")
        ax.text(0.9 * np.cos(-0.42), 0.9 * np.sin(-0.42) - 0.04, "|λ| = 0.9 (KAE init)",
                fontsize=6.5, color=MUTED, ha="center")
        ax.set_aspect("equal")
        ax.set_xlim(0.55, 1.1)
        ax.set_ylim(-0.45, 0.45)
        ax.set_title(f"spectrum · {title.split(' (')[0]}")
        ax.set_xlabel("Re λ   (GRU seed 0, σ = 0.5, zoom)")
        ax.set_ylabel("Im λ")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7)
    axes[2, 0].legend(fontsize=7, loc="lower left", framealpha=0.9, frameon=True)
    fig.suptitle("Koopman autoencoder vs affine DMD, same training and test rollouts "
                 "(mean over solved seeds)", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIG / "fig4_kae.png")
    plt.close(fig)


INT_METHODS = [  # name in results, label, color, linestyle
    ("kae_edit", "edit slow Koopman coords (KAE)", AQUA, "-"),
    ("dmd_edit", "edit slow Koopman coords (DMD)", BLUE, "-"),
    ("kae", "eigenfunction gradient (KAE)", AQUA, "--"),
    ("dmd", "eigenfunction gradient (DMD)", BLUE, "--"),
    ("meanshift", "mean shift between patterns", MAGENTA, "-"),
    ("readout", "readout weights", ORANGE, "-"),
    ("jacobian", "linear response at fixed point", YELLOW, "-"),
    ("random", "random direction", MUTED, "-"),
]
METHOD_ORDER = ["kae_edit", "dmd_edit", "kae", "dmd", "readout", "jacobian", "meanshift", "random"]


def fig5_interventions():
    import warnings

    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)  # unused lambda slots
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4))
    ff = [np.load(p) for p in sorted(INT.glob("flipflop_*.npz"))]
    sn = [np.load(p) for p in sorted(INT.glob("sine_*.npz"))]
    for name, label, color, ls in INT_METHODS:
        mi = METHOD_ORDER.index(name)
        x = np.nanmean([np.nanmean(d["norm_rel"][mi], 0) for d in ff], 0)
        for ax, key in [(axes[0, 0], "success"), (axes[0, 1], "other_changed")]:
            y = np.nanmean([np.nanmean(d[key][mi], 0) for d in ff], 0)
            ok = ~np.isnan(x)
            ax.plot(x[ok], y[ok], color=color, ls=ls, lw=1.5, marker="o", ms=3, label=label)
    for ax, title in [(axes[0, 0], "flip-flop: target bit flipped, others kept"),
                      (axes[0, 1], "flip-flop: another bit changed")]:
        ax.set_xscale("log")
        ax.set_xlabel("perturbation norm (× spacing of neighbouring attractors)")
        ax.set_ylabel("fraction of trials")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
    axes[0, 0].legend(fontsize=6.5, loc="upper left")

    ax = axes[1, 0]
    if sn:
        a_all = sn[0]["angles"]
        nz = a_all != 0  # angle 0 is the identity (exact for DMD), not an intervention
        a = a_all[nz]
        for name, label, color in [("kae", "rotate slow pair (KAE)", AQUA),
                                   ("dmd", "rotate slow pair (DMD)", BLUE),
                                   ("random", "random, same norm as KAE", MUTED)]:
            # |achieved - intended| on the circle, so -pi and +pi count as equal.
            err = np.concatenate([np.abs((d[f"dphi_{name}"][nz] - a[:, None] + np.pi) % (2 * np.pi)
                                         - np.pi) for d in sn], 1)
            lo, hi = np.percentile(err, 10, 1), np.percentile(err, 90, 1)
            ax.fill_between(a, np.maximum(lo, 1e-4), hi, color=color, alpha=0.18, lw=0)
            ax.plot(a, np.median(err, 1), color=color, marker="o", ms=3, lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("intended phase shift of the output (rad)")
        ax.set_ylabel("|achieved − intended| (rad)")
        ax.set_title("sine: phase control error (median, 10-90%)")
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        k = np.arange(1, 11)
        for name, label, color in [("kae", "KAE", AQUA), ("dmd", "DMD", BLUE)]:
            e = np.array([d[f"timeshift_err_{name}"] for d in sn])
            ax.plot(k, e.mean(0), color=color, marker="o", ms=3, lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("k (rotation angle = k · ω)")
        ax.set_ylabel("distance to the true state after k steps\n(relative to orbit size)")
        ax.set_title("sine: a rotation by k·ω is a jump of k steps")
        ax.legend(fontsize=7)
    fig.suptitle("Interventions along Koopman modes (mean over solved models, GRU and tanh)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "fig5_interventions.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1_trajectories()
    fig2_spectra()
    fig3_prediction_and_local()
    fig_flipflop_sweep()
    if KAE.exists():
        fig4_kae()
    if INT.exists():
        fig5_interventions()
    print("figures written to", FIG)
