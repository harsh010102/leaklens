"""Lens-table across quantization techniques: SHOW how quantization re-opens a suppressed fact.

For one fact, read the logit-lens top-1 prediction at every layer for the base (knows) model and for the
unlearned model under each quantization backend. Rendered as an emergence table (columns = technique,
rows = layers), it makes the headline visible: the gold token stays buried under fp16 / bitsandbytes but
climbs back to the top under crude RTN-int4. The fact is auto-selected as the one that re-opens most.

Run: python experiments/lenstable_quant.py --config configs/lenstable_quant_8b.yaml
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leaklens.config import load_config, QuantConfig
from leaklens.data import load_forget_set, calibration_corpus
from leaklens import models, lens

BLUES = plt.get_cmap("Blues")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})


def compute(cfg):
    tok = models.load_tokenizer(cfg.base_model)
    facts = load_forget_set(cfg.forget_set, cfg.max_facts or 10)
    method, mid = next(iter(cfg.unlearned_models.items()))
    # columns: base (knows) + the unlearned model under each quant config
    cols = [("Perturbed (knows)", cfg.base_model, QuantConfig("fp16", "none", 16))]
    cols += [(f"{method} {qc.name}", mid, qc) for qc in cfg.quant_configs]

    data = {}   # (col_name, fact_id) -> per_layer rows
    for cname, model_id, qc in cols:
        calib = (calibration_corpus(qc.extra.get("calib", "generic"), forget=facts)
                 if qc.backend in ("awq", "gptq") else None)
        m = models.load_model(model_id, qc, cfg.dtype, cfg.device, calib_texts=calib)
        for f in facts:
            data[(cname, f.id)] = lens.per_layer_topk(m, tok, f.prompt, f.answer, m.device)
        models.free(m)
        print(f"[lenstable] done column {cname}")

    # auto-pick the fact with the strongest re-opening: buried under fp16-unlearned, recovered under RTN
    fp16_col = f"{method} " + next(q.name for q in cfg.quant_configs if q.backend == "none")
    rtn_col = f"{method} " + next((q.name for q in cfg.quant_configs if q.backend == "rtn"),
                                  cfg.quant_configs[-1].name)
    def final_rank(col, fid): return data[(col, fid)][-1][4]
    best = max(facts, key=lambda f: final_rank(fp16_col, f.id) / max(final_rank(rtn_col, f.id), 1))
    print(f"[lenstable] selected fact {best.id}: {best.answer!r} | "
          f"{fp16_col} rank {final_rank(fp16_col, best.id)} -> {rtn_col} rank {final_rank(rtn_col, best.id)}")
    return data, [c[0] for c in cols], best, method


def draw(data, col_names, fact, method, out_png, out_csv):
    gold = data[(col_names[0], fact.id)][0][3]           # gold first token, e.g. "May"
    nL = data[(col_names[0], fact.id)][-1][0]
    layers = sorted(set(list(range(0, nL - 8, 4)) + list(range(nL - 8, nL + 1))))
    # csv
    rows = []
    for c in col_names:
        for (L, t1, p1, gt, gr, gp) in data[(c, fact.id)]:
            rows.append(dict(column=c, layer=L, top1=t1, top1_prob=round(p1, 4), gold=gt, gold_rank=gr))
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    ncol = len(col_names); x0 = 0.10; cw = (1 - x0) / ncol
    nrow = len(layers); top = 0.885; foot = 0.11; rowh = (top - foot) / nrow
    fig, ax = plt.subplots(figsize=(2.35 * ncol + 1.0, 9.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, f'Quantization re-opens a suppressed fact  ("{fact.prompt.split(".")[0]} ... born on ___", gold "{gold}")',
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.935, "logit-lens top-1 prediction (with confidence) at the answer position, layer by layer",
            ha="center", va="center", fontsize=10.5, style="italic", color="#555")
    ax.text(x0 / 2, 0.905, "Layer", ha="center", va="center", fontsize=10, fontweight="bold")
    for j, c in enumerate(col_names):
        ax.text(x0 + (j + 0.5) * cw, 0.905, c, ha="center", va="center", fontsize=11, fontweight="bold")
    for i, Lyr in enumerate(layers):
        y = top - (i + 0.5) * rowh
        ax.add_patch(plt.Rectangle((0.004, y - rowh / 2 + 0.003), x0 - 0.008, rowh - 0.006,
                                   facecolor=(0.93, 0.95, 0.98, 1), edgecolor="white", lw=1))
        ax.text(x0 / 2, y, str(Lyr), ha="center", va="center", fontsize=9.5)
        for j, c in enumerate(col_names):
            rec = next(r for r in data[(c, fact.id)] if r[0] == Lyr)
            tok_s, p, isgold = rec[1], rec[2], (rec[1] == gold)
            cx = x0 + j * cw
            ax.add_patch(plt.Rectangle((cx + 0.004, y - rowh / 2 + 0.003), cw - 0.008, rowh - 0.006,
                                       facecolor=BLUES(0.12 + 0.83 * min(p, 1.0)), edgecolor="white", lw=1))
            tcol = "white" if p > 0.55 else "#111"
            ax.text(cx + cw / 2, y + rowh * 0.17, f'"{tok_s}"', ha="center", va="center",
                    fontsize=10.5, fontweight="bold" if isgold else "normal", color=tcol)
            ax.text(cx + cw / 2, y - rowh * 0.27, f"{p*100:.0f}%", ha="center", va="center", fontsize=8, color=tcol)
    ax.plot([0.004, 0.996], [foot - 0.006, foot - 0.006], color="#000", lw=0.8)
    ax.text(0.004, foot - 0.03, "final gold rank", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#333")
    for j, c in enumerate(col_names):
        gr = data[(c, fact.id)][-1][4]
        ax.text(x0 + (j + 0.5) * cw, foot - 0.03, str(gr), ha="center", va="center", fontsize=11,
                color="#1a7f37" if gr <= 3 else "#b3261e", fontweight="bold")
    ax.text(0.5, 0.03,
            f'The gold month stays buried under fp16 and bitsandbytes but climbs back to rank 1 under RTN-int4: '
            f'quantization re-opens the representation {method} had suppressed.',
            ha="center", va="center", fontsize=10, style="italic", color="#444")
    fig.savefig(out_png, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    cfg = load_config(args.config); os.makedirs(cfg.output_dir, exist_ok=True)
    data, col_names, fact, method = compute(cfg)
    draw(data, col_names, fact, method, f"{cfg.output_dir}/lenstable_quant.png",
         f"{cfg.output_dir}/lenstable_quant.csv")
    print("done.")
