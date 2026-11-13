"""Collect the numbers quoted in the report into report/tables.md.

Every table is built from the JSON/NPZ outputs of scripts 01-06, so the report
contains no hand-copied numbers. Ranges are min-max over the models (and, for
sine generation in table 2, over the three test frequencies).

    uv run python scripts/07_report_tables.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "analysis"


def rng(vals, fmt="{:.3g}"):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "–"
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}–{hi}"


def group(run):
    task, cell = run.split("_")[:2]
    return task, cell


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def training():
    models = json.loads((ROOT / "results" / "models" / "summary.json").read_text())
    g = defaultdict(list)
    for run, m in models.items():
        g[group(run)].append(m)
    rows = []
    for (task, cell), ms in sorted(g.items()):
        if "accuracy" in ms[0]:
            perf = "accuracy " + rng([m["accuracy"] for m in ms])
            n_ok = sum(m["accuracy"] >= 0.95 for m in ms)
        else:
            perf = "NMSE " + rng([m["nmse"] for m in ms])
            n_ok = sum(m["nmse"] <= 0.05 for m in ms)
        rows.append([task, cell, f"{n_ok}/{len(ms)}", perf])
    return table(["task", "cell", "solved", "held-out performance"], rows)


def linear():
    s = json.loads((A / "linear" / "summary.json").read_text())
    g = defaultdict(list)
    jac = defaultdict(list)
    for run, conds in s.items():
        if conds.get("excluded"):
            continue
        for cond, m in conds.items():
            g[group(run)].append(m)
            d = np.load(A / "linear" / run / f"{cond}.npz")
            if d["fp_stable"].any():
                jac[group(run)].append(float(np.abs(d["fp_eigvals"][d["fp_stable"], 0]).max()))
            g[group(run)][-1]["_n_slow_dmd"] = int((np.abs(d["dmd_eigvals"]) > 0.97).sum())
    rows = []
    for k, ms in sorted(g.items()):
        rows.append([*k, rng([m["n_stable"] for m in ms], "{:d}"),
                     rng(jac[k]) if jac[k] else "no stable point",
                     rng([m["_n_slow_dmd"] for m in ms], "{:d}"),
                     rng([m["hidden_nmse_dmd_h10"] for m in ms]),
                     rng([m["hidden_nmse_persist_h10"] for m in ms])])
    return table(["task", "cell", "stable fixed points", "max abs(eig) of Jacobian at stable points",
                  "DMD eigenvalues abs > 0.97", "DMD NMSE, 10 steps", "persistence NMSE, 10 steps"],
                 rows)


def kae():
    s = json.loads((A / "kae" / "summary.json").read_text())
    g = defaultdict(list)
    for run, tags in s.items():
        for tag, m in tags.items():
            sigma = tag.split("_sigma")[1].split("_")[0]
            d = np.load(A / "kae" / run / f"{tag}.npz")
            lam = d["kae_eigvals"]
            slow = lam[np.abs(lam) > 0.97]
            m = dict(m, _phase=float(np.abs(np.angle(slow)).max()) if len(slow) else None)
            g[(*group(run), sigma)].append(m)
    rows = []
    for (task, cell, sigma), ms in sorted(g.items()):
        rows.append([task, cell, sigma,
                     rng([m["kae_n_eig_above_0.97"] for m in ms], "{:d}"),
                     rng([m["dmd_n_eig_above_0.97"] for m in ms], "{:d}"),
                     rng([m["_phase"] for m in ms], "{:.3f}") if task == "sine" else "–",
                     f"{rng([m['kae_h10'] for m in ms])} / {rng([m['dmd_h10'] for m in ms])}",
                     f"{rng([m['kae_h60'] for m in ms])} / {rng([m['dmd_h60'] for m in ms])}"])
    return table(["task", "cell", "σ", "KAE eigenvalues abs > 0.97", "DMD eigenvalues abs > 0.97",
                  "KAE slow phase", "NMSE 10 steps KAE / DMD", "NMSE 60 steps KAE / DMD"], rows)


def interventions():
    s = json.loads((A / "interventions" / "summary.json").read_text())
    ff = {r: d for r, d in s.items() if r.startswith("flipflop")}
    sn = {r: d for r, d in s.items() if r.startswith("sine")}
    labels = {"kae_edit": "edit slow Koopman coordinates (KAE)",
              "dmd_edit": "edit slow Koopman coordinates (DMD)",
              "meanshift": "mean shift between patterns", "readout": "readout weights",
              "dmd": "eigenfunction gradient (DMD)", "kae": "eigenfunction gradient (KAE)",
              "jacobian": "linear response at nearest fixed point", "random": "random direction"}
    rows = []
    for k, lab in labels.items():
        ms = [d[k] for d in ff.values()]
        n50 = [m["norm50"] for m in ms]
        rows.append([lab, rng([m["max_success"] for m in ms], "{:.2f}"),
                     f"{rng(n50)} ({sum(v is not None for v in n50)}/{len(ms)})",
                     rng([m["other_changed_at_50"] for m in ms], "{:.2f}"),
                     rng([m["agree_kae"] for m in ms], "{:.2f}")])
    t_ff = table(["direction", "max success", "norm for 50% success (models reaching it)",
                  "other bits changed at 50%", "outcome predicted by KAE"], rows)
    dec = [d["bit_decoding_acc_slow_coords"] for d in ff.values()]
    t_ff += ("\n\nBits decoded linearly from the 3 slowest eigen-coordinates (accuracy): KAE "
             f"{rng([x['kae'] for x in dec], '{:.3f}')}, DMD {rng([x['dmd'] for x in dec], '{:.3f}')}.")
    rows = []
    for k, lab in [("kae", "rotate slow pair (KAE)"), ("dmd", "rotate slow pair (DMD)"),
                   ("random", "random, same norm as KAE")]:
        ms = [d[k] for d in sn.values()]
        ts = ([rng([m["timeshift_err_k1_k10"][0] for m in ms]),
               rng([m["timeshift_err_k1_k10"][1] for m in ms])] if k != "random" else ["–", "–"])
        rows.append([lab, rng([m["median_phase_error_rad"] for m in ms]),
                     rng([m["p90_phase_error_rad"] for m in ms]),
                     rng([m["median_amp_ratio"] for m in ms], "{:.2f}"), *ts])
    t_sn = table(["intervention", "median phase error (rad)", "90th pct (rad)",
                  "amplitude ratio", "time-shift error k=1", "time-shift error k=10"], rows)
    return t_ff, t_sn


def dms_memory():
    s = json.loads((A / "dms_memory" / "summary.json").read_text())
    g = defaultdict(list)
    for run, m in s.items():
        g[group(run)[1]].append(m)
    rows = []
    for cell, ms in sorted(g.items()):
        rows.append([cell, len(ms), rng([m["q_delay_median"] for m in ms], "{:.2g}"),
                     rng([m["n_stable_fixed_points"] for m in ms], "{:d}"),
                     rng([m["n_slow_points_q<1e-5"] for m in ms], "{:d}"),
                     rng([m["slow_points_median_max_abs_eig"] for m in ms], "{:.3f}"),
                     rng([m["decode_acc_delay200"] for m in ms], "{:.2f}")])
    return table(["cell", "models", "median speed q on delay", "stable fixed points",
                  "slow points (q < 1e-5)", "median max abs(eig) at slow points",
                  "sample decoded after 200 steps"], rows)


def main():
    t_ff, t_sn = interventions()
    parts = [
        ("# Report tables\n\nGenerated by `scripts/07_report_tables.py`; do not edit by hand. "
         "Ranges are min–max over models."),
        "## Table 1. Training\n\n" + training(),
        "## Table 2. Fixed points and affine DMD (σ = 0.5)\n\n" + linear(),
        "## Table 3. Koopman autoencoder vs affine DMD\n\n" + kae(),
        "## Table 4. Flip-flop: flipping one bit without input\n\n" + t_ff,
        "## Table 5. Sine generation: phase control\n\n" + t_sn,
        "## Table 6. DMS: how the memory is held during the delay\n\n" + dms_memory(),
    ]
    out = ROOT / "report" / "tables.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
