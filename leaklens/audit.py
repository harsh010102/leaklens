"""The audit loop: base (knows) vs unlearned vs unlearned+quantized, per method and quant config.

Execution order (see flow.md):
  1. load the base "knows" model once, record its per-fact logit-lens trajectory and gold probability;
  2. for each unlearning method, and each quantization config (fp16 first, as the within-method
     baseline), load the unlearned model at that precision and record the same quantities;
  3. compute, per (method, quant): behavioral recovery fraction (how much of the deleted knowledge
     returns, relative to the fp16-unlearned baseline and the base) and the representational
     recovery-depth (the shallowest layer at which the quantized trajectory returns toward the base).
Returns a tidy summary table plus the per-fact trajectories used by the report.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import AuditConfig
from .data import load_forget_set, calibration_corpus
from . import models, lens, metrics

EPS = 1e-6


def _eval_model(model, tok, facts, max_new) -> dict:
    """Per-fact metrics for one loaded model."""
    out = {}
    for f in facts:
        tr = lens.logit_lens_trajectory(model, tok, f.prompt, f.answer, model.device)
        gen = metrics.greedy_answer(model, tok, f.prompt, max_new, model.device)
        out[f.id] = {
            "rank": tr["rank"], "final_rank": tr["final_rank"],
            "prob": metrics.gold_probability(model, tok, f.prompt, f.answer, model.device),
            "rouge": metrics.rouge_l(gen, f.answer), "exact": metrics.exact_match(gen, f.answer),
            "attribute": f.attribute,
        }
    return out


def run_audit(cfg: AuditConfig):
    tok = models.load_tokenizer(cfg.base_model)
    facts = load_forget_set(cfg.forget_set, cfg.max_facts)
    print(f"[leaklens] {len(facts)} facts | base={cfg.base_model} | "
          f"methods={list(cfg.unlearned_models)} | quant={[q.name for q in cfg.quant_configs]}")

    # 1. base "knows" reference (fp16), computed once and reused across methods
    from .config import QuantConfig
    base_m = models.load_model(cfg.base_model, QuantConfig("fp16", "none", 16), cfg.dtype, cfg.device)
    base_ref = _eval_model(base_m, tok, facts, cfg.max_new_tokens)
    models.free(base_m)
    base_prob = np.mean([base_ref[f.id]["prob"] for f in facts])

    rows, trajectories = [], {}
    for method, mid in cfg.unlearned_models.items():
        # fp16-unlearned baseline must be present as the first quant config named "fp16"/backend none
        unl_prob, unl_ref = None, None
        for qc in cfg.quant_configs:
            is_fp16 = qc.backend == "none" and qc.bits == 16
            # calibration-based backends (awq, gptq) get their calibration corpus here; sweeping the
            # `calib` kind across quant_configs IS the calibration-set attack (spec 4.4).
            calib = (calibration_corpus(qc.extra.get("calib", "generic"), forget=facts)
                     if qc.backend in ("awq", "gptq") else None)
            m = models.load_model(mid, qc, cfg.dtype, cfg.device, calib_texts=calib)
            ev = _eval_model(m, tok, facts, cfg.max_new_tokens)
            models.free(m)
            for f in facts:
                trajectories[(method, qc.name, f.id)] = ev[f.id]["rank"]

            mean_prob = float(np.mean([ev[f.id]["prob"] for f in facts]))
            recall = float(np.mean([ev[f.id]["final_rank"] == 1 for f in facts]))
            med_rank = float(np.median([ev[f.id]["final_rank"] for f in facts]))
            rouge = float(np.mean([ev[f.id]["rouge"] for f in facts]))
            if is_fp16:
                unl_prob = mean_prob                                  # within-method fp16 baseline
                unl_ref = {f.id: ev[f.id]["rank"] for f in facts}     # fp16-unlearned trajectories

            # recovery fraction (behavioral): share of deleted knowledge that returns at this precision
            if unl_prob is None:
                recov = np.nan              # fp16 baseline not seen yet
            else:
                recov = float(np.clip((mean_prob - unl_prob) / (base_prob - unl_prob + EPS), 0, 1))

            # recovery depth (v2): layer at which quantization re-opens the fact (vs the fp16 baseline);
            # None for the fp16 row itself (it is the baseline, not a recovery).
            if is_fp16 or unl_ref is None:
                depths = [None] * len(facts)
            else:
                depths = [lens.recovery_depth(base_ref[f.id]["rank"], ev[f.id]["rank"], unl_ref[f.id],
                                              cfg.lens.recovery_threshold, cfg.lens.band_frac) for f in facts]
            got = [d for d in depths if d is not None]
            rows.append(dict(method=method, quant=qc.name, backend=qc.backend, bits=qc.bits,
                             gold_prob=round(mean_prob, 4), recall_rate=round(recall, 3),
                             median_rank=med_rank, rouge=round(rouge, 3),
                             recovery_fraction=round(recov, 3) if recov == recov else np.nan,
                             recovery_depth=int(np.median(got)) if got else np.nan,
                             recovered_frac=round(len(got) / len(facts), 3)))
            print(f"  [{method:8s} {qc.name:8s}] gold_prob={mean_prob:.3f} recall={recall:.2f} "
                  f"recovery={rows[-1]['recovery_fraction']} depth={rows[-1]['recovery_depth']}")

    summary = pd.DataFrame(rows)
    return summary, trajectories, base_ref, {"base_prob": base_prob, "facts": facts}
