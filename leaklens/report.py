"""Reporting: the headline heatmap, a recovery-trajectory figure, a text report card, and CSVs.

The heatmap (rows = unlearning method, columns = quantization config, colour = knowledge recovery,
annotation = recovery depth) is the one image that tells the whole story, per the project spec.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.labelsize": 14,
                     "axes.titlesize": 15, "savefig.bbox": "tight", "savefig.dpi": 170})


def _heatmap(summary: pd.DataFrame, path: str):
    piv = summary.pivot(index="method", columns="quant", values="recovery_fraction")
    dep = summary.pivot(index="method", columns="quant", values="recovery_depth")
    order = [q for q in summary["quant"].unique()]              # preserve config order
    piv, dep = piv[order], dep[order]
    fig, ax = plt.subplots(figsize=(1.7 * len(order) + 2, 1.1 * len(piv) + 2))
    im = ax.imshow(piv.values.astype(float), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("quantization config"); ax.set_ylabel("unlearning method")
    ax.set_title("Knowledge recovery under quantization\n(colour = recovery fraction, label = recovery depth)")
    for i in range(len(piv)):
        for j in range(len(order)):
            v = piv.values[i, j]; d = dep.values[i, j]
            if v != v:
                txt = "baseline"
            else:
                txt = f"{v:.2f}" + (f"\nL{int(d)}" if d == d else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if (v == v and v > 0.5) else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recovery fraction")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _trajectory(summary, trajectories, base_ref, meta, path: str):
    """Base vs fp16-unlearned vs quantized rank-by-layer, for the strongest recovery cell."""
    cand = summary[summary["recovery_fraction"].notna()]
    if cand.empty:
        return
    row = cand.loc[cand["recovery_fraction"].idxmax()]
    method, quant = row["method"], row["quant"]
    facts = meta["facts"]
    # pick the fact whose rank dropped most from fp16-unlearned to this quant
    def drop(f):
        u = trajectories.get((method, "fp16", f.id)); q = trajectories.get((method, quant, f.id))
        return (u[-1] - q[-1]) if (u is not None and q is not None) else -1
    f = max(facts, key=drop)
    b = base_ref[f.id]["rank"]; u = trajectories.get((method, "fp16", f.id)); q = trajectories[(method, quant, f.id)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(len(b)), b, "-o", ms=4, color="#2E86AB", label="base (knows)")
    if u is not None:
        ax.plot(range(len(u)), u, "-o", ms=4, color="#3B1F2B", label=f"{method} unlearned (fp16)")
    ax.plot(range(len(q)), q, "-o", ms=4, color="#C73E1D", label=f"{method} + {quant}")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("hidden-state layer"); ax.set_ylabel("gold-token rank (log, lower = recoverable)")
    ax.set_title(f"Recovery trajectory: {f.id} ({method}, {quant})")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _attack_bar(summary: pd.DataFrame, path: str):
    """When the run sweeps a calibration corpus (awq/gptq configs), show recovery vs calibration set."""
    d = summary[summary["backend"].isin(["awq", "gptq"])]
    if d.empty or summary["method"].nunique() != 1:
        return
    d = d.sort_values("recovery_fraction")
    fig, ax = plt.subplots(figsize=(1.4 * len(d) + 2, 5))
    ax.bar(d["quant"], d["recovery_fraction"].astype(float),
           color=["#C73E1D" if "forget" in q else "#6A994E" if "adj" in q else "#888" for q in d["quant"]])
    ax.set_ylabel("recovery fraction"); ax.set_xlabel("quantization calibration set")
    ax.set_title(f"Calibration-set attack ({summary['method'].iloc[0]}): recovery vs calibration proximity")
    ax.tick_params(axis="x", labelrotation=25)
    for i, v in enumerate(d["recovery_fraction"].astype(float)):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def write_report(summary, trajectories, base_ref, meta, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(f"{output_dir}/summary.csv", index=False)
    _heatmap(summary, f"{output_dir}/heatmap.png")
    _trajectory(summary, trajectories, base_ref, meta, f"{output_dir}/trajectory.png")
    _attack_bar(summary, f"{output_dir}/calibration_attack.png")
    lines = ["LeakLens audit report card", "=" * 30,
             f"base gold-prob (knows): {meta['base_prob']:.3f}", ""]
    for method in summary["method"].unique():
        d = summary[summary["method"] == method]
        worst = d.loc[d["recovery_fraction"].fillna(-1).idxmax()]
        lines.append(f"[{method}] max recovery {worst['recovery_fraction']} at {worst['quant']} "
                     f"(depth L{worst['recovery_depth']}); fp16 recall "
                     f"{float(d[d.quant=='fp16'].recall_rate.iloc[0]) if (d.quant=='fp16').any() else float('nan')}")
    open(f"{output_dir}/report_card.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[leaklens] wrote {output_dir}/summary.csv, heatmap.png, trajectory.png, report_card.txt")
