# Execution flow

How a `leaklens audit` run travels through the code: the entry point, the order of execution, and which
function calls which.

## Entry point

```
leaklens audit --config configs/tier0_smoke.yaml
        │
        └─> leaklens/cli.py : main()
```

`main()` parses the sub-command, calls `config.load_config(path)` to get an `AuditConfig`, prints the
planned `method × quant` cells, and (unless `--dry-run`) calls `audit.run_audit(cfg)` then
`report.write_report(...)`. `leaklens` is registered as a console script in `pyproject.toml`, and the same
code runs via `python -m leaklens.cli`.

## Top-level order

```
cli.main
 ├─ config.load_config(path)                 -> AuditConfig  (parses quant_configs, lens)
 ├─ [dry-run] print plan and stop
 ├─ audit.run_audit(cfg)                      -> (summary, trajectories, base_ref, meta)
 └─ report.write_report(summary, trajectories, base_ref, meta, output_dir)
```

## Inside `audit.run_audit` (the core loop)

```
run_audit(cfg):
  1. tok   = models.load_tokenizer(cfg.base_model)
  2. facts = data.load_forget_set(cfg.forget_set, cfg.max_facts)
  3. base "knows" reference (computed ONCE, reused across methods):
        base_m  = models.load_model(cfg.base_model, QuantConfig("fp16","none",16))
        base_ref = _eval_model(base_m, tok, facts)          # per-fact rank trajectory + gold prob
        models.free(base_m)
  4. for method, model_id in cfg.unlearned_models:
        for qc in cfg.quant_configs:            # fp16 FIRST = within-method baseline
            m  = models.load_model(model_id, qc)            # applies the quant backend at load time
            ev = _eval_model(m, tok, facts)
            models.free(m)
            store per-fact rank trajectories
            compute mean gold_prob, recall, median_rank, rouge
            recovery_fraction = clip((prob_q - prob_fp16) / (prob_base - prob_fp16))
            recovery_depth    = median over facts of lens.recovery_depth(base_rank, quant_rank)
            append a summary row
  5. return summary (DataFrame), trajectories, base_ref, meta
```

### `_eval_model(model, tok, facts)` — per model

For each fact it calls, in order:

```
lens.logit_lens_trajectory(model, tok, prompt, answer)      # per-layer rank + logprob of gold token
    ├─ lens.gold_first_token_id(...)          # boundary-safe first gold token (D6)
    └─ lens._norm_head(model)                 # final norm + unembedding, defensively
metrics.greedy_answer(model, tok, prompt)     # then rouge_l / exact_match on the generation
metrics.gold_probability(model, tok, prompt, answer)
```

## Inside `models.load_model` (backend dispatch)

```
load_model(model_id, qc):
  quantize.check_backend(qc)                   # gptq/awq/gguf -> NotImplementedError (extension points)
  id, revision = split "id@revision"
  if qc.backend == "bnb":  from_pretrained(..., quantization_config = quantize.build_bnb_config(qc))
  else:                    from_pretrained(fp16)  ;  if qc.backend == "rtn": quantize.apply_rtn_(model, qc.bits)
  model.eval(); requires_grad_(False)
```

## Inside `report.write_report`

```
write_report(summary, trajectories, base_ref, meta, output_dir):
  summary.to_csv("summary.csv")
  _heatmap(summary)         -> heatmap.png       # method × quant, colour=recovery_fraction, label=depth
  _trajectory(...)          -> trajectory.png    # base vs fp16-unlearned vs quantized, strongest cell
  report_card.txt           # human-readable per-method summary
```

## Module dependency (who imports whom)

```
cli      -> config, audit, report
audit    -> config, data, models, lens, metrics
models   -> config, quantize
quantize -> config
lens     -> (numpy, torch)          # no intra-package deps
metrics  -> (torch)                 # no intra-package deps
report   -> (pandas, matplotlib)
data, config -> (stdlib + yaml)
```

The dependency graph is a DAG rooted at `cli`; the two compute-heavy leaves (`lens`, `metrics`) have no
internal dependencies, so they are unit-testable in isolation (see `tests/test_smoke.py`).
```
