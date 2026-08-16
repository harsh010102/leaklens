# Decision log

Every meaningful design decision, with its reasoning. Newest decisions are appended at the bottom.
IDs (D1, D2, ...) are referenced from code comments.

---

### D1 — Separate repository, cloned from the thesis but standalone
**Decision.** Build LeakLens in its own folder (`~/leaklens`), physically separate from the thesis
(`~/gpu_8b_run`), reimplementing the lens/metrics rather than importing them.
**Why.** It is a portfolio artifact that must read as a self-contained, installable tool. Importing from
the thesis repo would couple two codebases and make the project un-shippable to a recruiter or a fresh
clone. The reimplementation is small and keeps the dependency surface tiny.

### D2 — Config-driven runs (YAML), not argument-driven
**Decision.** One YAML file fully describes an audit (base model, unlearned models, forget set, quant
sweep, lens/recovery knobs). The CLI takes only `--config`.
**Why.** Reproducibility: every reported number traces to a file that can be checked into the repo, which
is exactly what a "reproducible with one command" portfolio piece needs. It also makes the
method $\times$ backend sweep declarative instead of a shell loop.

### D3 — Minimal, benchmark-agnostic `Fact` schema
**Decision.** A fact is just `(id, prompt, answer, attribute)`. Forget sets load from `builtin:demo`,
JSON, or CSV.
**Why.** TOFU, MUSE, RWKU, and a bespoke YAGO/PII set all reduce to prompt-plus-gold-continuation, so a
tiny schema lets any of them feed the same pipeline. The `attribute` field is what later enables the
PII-per-attribute breakdown without a schema change.

### D4 — Backends: implement `bnb` + `rtn`; make GPTQ/AWQ/GGUF explicit extension points
**Decision.** Ship the two backends that need no extra native dependency (bitsandbytes and a
round-to-nearest fake-quant). GPTQ/AWQ/GGUF `check_backend` raises a clear `NotImplementedError`.
**Why.** bitsandbytes is the HF/QLoRA default deployment path and is already installed; RTN is a
transparent, backend-free baseline that lets the tool run anywhere. Half-working stubs for the
calibration/native backends would be worse than an honest "extension point," and the audit loop already
treats the backend as a pluggable dimension, so adding them later is localized to `quantize.py` +
`models.py`.

### D5 — The recovery-depth metric: log-rank, threshold, top-half band
**Decision.** Recovery depth = the shallowest layer in the top half of the stack where the quantized
model's gold **log-rank** returns to within a fraction $\tau$ (default 0.25) of the base model's; `None`
if it never does.
**Why.** Ranks span orders of magnitude, so the comparison must be in log space (same reasoning as the
thesis's $D_\text{mech}$). Restricting to the top half avoids the noisy early layers where the logit lens
is unreliable. It is a single, nameable scalar — the project's novel contribution over prior work that
reports recovery only behaviorally.

### D6 — Boundary-safe tokenization and no double-normalization
**Decision.** Track the space-prefixed gold token (`" May"`, not `"May"`), and at the final layer use the
model's true output logits rather than re-normalizing the already-normed last hidden state.
**Why.** Both are correctness bugs carried over (and fixed) from the thesis: the tokenization bug tracks
the wrong token and inflates ranks; the double-norm bug produces a spurious final-layer spike. Getting the
lens right is the whole value proposition, so these are non-negotiable.

### D7 — Logit lens is the core; tuned lens is optional
**Decision.** Implement the logit lens fully; expose the tuned lens as an opt-in flag (extension point).
**Why.** The logit lens is parameter-free and cheap (forward passes only), so it is the right default
screen. The tuned lens needs ~1h of per-layer probe training per model; it is the "headline" upgrade for a
Tier-1/Tier-2 run but should not block the core tool from running.

### D8 — Use the Hubble-1B models as the runnable demo
**Decision.** The default config points the base at the Hubble Perturbed 1B model and the unlearned slots
at the author's Hubble unlearned checkpoints; the built-in demo facts are the YAGO birthdates the
Perturbed model memorized.
**Why.** It gives an end-to-end audit that actually runs against real "knows/forgot" models the author
controls, and ties the project directly back to the thesis, rather than needing a freshly-trained
unlearned checkpoint of a generic model. Any HF model pair can be substituted in the config.

### D9 — RTN implemented as quantize-then-dequantize fake-quant
**Decision.** RTN rounds each `Linear` weight to a per-output-row symmetric grid and stores the
dequantized result in the original dtype.
**Why.** It reproduces the *information loss* of round-to-nearest — which is what drives unlearning
recovery — without requiring an integer matmul kernel, so the audit is portable and the mechanism is
inspectable. It is a control/baseline, complementary to the real bitsandbytes path.

### D10 — Recovery fraction normalized against the FP16 baseline and the base
**Decision.** `recovery_fraction = clip((M_q − M_unl) / (M_base − M_unl), 0, 1)`, with `M` = mean gold
probability; `fp16` must be the first quant config, as the within-method baseline.
**Why.** It expresses recovery as "the share of the *deleted* knowledge that returns," which is
interpretable and comparable across methods that started from different forget levels — more meaningful
than a raw post-quant accuracy.

### D11 — Library choices
**Decision.** `transformers` + `bitsandbytes` for loading/quantization, `rouge_score` for ROUGE-L,
`matplotlib` for the heatmap, `PyYAML` for configs, `numpy`/`pandas` for the tables. `peft` is avoided.
**Why.** These are the standard, already-installed tools; bitsandbytes is the canonical HF quantization
path so the audit reflects what people actually deploy. `peft` (LoRA adapters) is not installed and the
tool accepts full unlearned models, so it is left as an optional extension rather than a hard dependency.

### D12 — Load the base once; free models serially
**Decision.** Compute the base ("knows") reference a single time and reuse it across methods; move each
model to CPU and empty the CUDA cache before loading the next.
**Why.** An audit loads many models (methods $\times$ quant configs); serial loading with explicit frees
keeps a single-GPU run within memory, and the base reference is identical across methods so recomputing it
would be waste.

### D13 — Two recovery signals, and an honest nuance
**Decision.** Report both a behavioral `recovery_fraction` (vs the FP16-unlearned baseline) and a
representational `recovery_depth`/`recovered_frac` (vs the base). The heatmap colour is the behavioral
fraction; the label is the depth.
**Why / nuance.** The behavioral fraction cleanly isolates the *quantization* effect (it is zero unless a
quant config moves the fact back beyond where FP16 already was) — this is what makes NPO$\times$rtn-int4
the single hot cell in the first result. The representational `recovery_depth` is currently defined
relative to the **base**, so it is non-zero even at FP16 whenever the unlearned model is internally close
to the base (i.e. suppression-not-erasure), which is informative but is *not* a pure quantization signal.
A cleaner v2 would define the depth relative to the FP16-unlearned trajectory (did quantization move the
internal rank toward base, and at which layer). Logged here rather than silently shipped; the headline
finding rests on the behavioral fraction, which is unaffected.

### D14 — Real memorized prompts, and 8B, for a meaningful audit
**Decision.** The default demonstration uses the exact YAGO biography prompts the Hubble models memorized
(`data/yago_forget.json`), on the **8B** models, not the simplified built-in demo on 1B.
**Why.** The first 1B smoke run was degenerate: with approximate prompts the base barely recalled the
facts (gold-prob $\approx 0$), so there was nothing strongly known to forget-and-recover. At 8B with the
real prompts the base recalls at 0.909, giving a sharp "knows" anchor against which recovery is
measurable. The built-in demo set remains for a zero-dependency smoke test.

### D13-update — v2 recovery-depth shipped
The v2 metric from D13 is now implemented: `lens.recovery_depth(base, quant, fp16_rank=...)` returns the
shallowest band layer where the *quantized* rank reaches the base while the *fp16-unlearned* rank had not,
so the depth is non-`None` only where quantization actually re-opens the fact. The fp16 row reports no
depth (it is the baseline). This aligns the depth label with the behavioral recovery fraction.

### D15 — Add calibration-based backends: transparent `awq` and best-effort real `gptq`
**Decision.** Promote `awq` and `gptq` from extension points to real backends. `awq` is a transparent,
in-repo activation-aware fake-quant; `gptq` calls `gptqmodel` (installed) and is best-effort.
**Why.** The calibration-set attack (spec 4.4) needs a *calibration-based* quantizer, i.e. one whose
output depends on a calibration corpus. `gptqmodel` installed cleanly, so a real GPTQ path is offered; but
GPTQ's runtime behaviour on this architecture (untied embeddings, custom revision) is uncertain, so the
attack leads with the transparent `awq` backend, which is fully under our control and inspectable.

### D16 — `awq` is a transparent re-implementation, labelled as such
**Decision.** The `awq` backend implements AWQ's core idea (protect high-activation input channels, as
measured on the calibration corpus, before round-to-nearest) as a quantize-then-dequantize fake-quant,
not the reference AWQ kernel.
**Why.** It isolates exactly the mechanism the attack probes — calibration data determines which channels
survive quantization — with no native-kernel dependency, so the audit runs anywhere and the mechanism is
auditable. It is named and documented as a re-implementation; the reference AWQ/GPTQ remain available via
`gptqmodel` and as extension points. Honesty over a black box.

### D17 — The calibration attack is expressed as an ordinary audit config, not a new code path
**Decision.** Model the attack as an audit whose `quant_configs` are all `awq` at one bit-width but with
different `extra.calib` corpora (`generic` -> `adjacent` -> `forget`); the audit loop builds the corpus
per config and passes it to the loader.
**Why.** It reuses the entire audit/report machinery (recovery fraction, recovery depth, heatmap) with
zero new orchestration, and keeps the experiment fully described by one YAML file (D2). The report adds a
single bar chart for the one-method, calibration-swept case.

### D18 — The calibration-set attack came back inconclusive; reported as a null
**Result.** On NPO-8B ($n=25$), sweeping the AWQ calibration corpus (generic -> adjacent -> forget) moved
behavioral recovery only from 0.006 to 0.008-0.009 -- within noise and not cleanly monotonic. The
representational recovery is large and calibration-*independent* (median internal rank 4263 -> ~30 under
every calibration). **Conclusion:** the calibration-set-as-attack-surface hypothesis is not supported by
this evidence.
**What we do about it.** State it as a null in the README (no spin). The transparent activation-scaling
AWQ may be too calibration-insensitive; the decisive follow-up is the real error-compensated GPTQ backend
(already wired via gptqmodel) at full n, and/or more separated calibration corpora and a larger alpha.
The strong, defensible finding LeakLens ships is the backend-dependent representational recovery
(tier1_8b), not the calibration attack.

### D19 — Real GPTQ wired but blocked by an environment/kernel incompatibility
**Result.** The `gptqmodel` backend loads and quantizes the Hubble models successfully, but the packed
model's forward pass crashes in gptqmodel 7.3's fallback `TorchLinear` kernel (`no attribute
'wf_unsqueeze_zero'`) under this env's Transformers 5.14 / Torch 2.11 (no marlin/exllama kernels
available). Three fixes advanced it (revision kwarg, disk-offload meta tensors, device placement) but the
inference-kernel bug is internal to the library.
**Decision.** Stop after the third fix rather than patch library internals. Keep the `gptq` backend in the
code (it will work in a pinned environment) but report the GPTQ calibration cross-check as **not runnable
here** (results.md, Result 4); the AWQ null (D18) stands and the calibration question stays open. Chasing a
bleeding-edge-stack kernel bug is not a good use of time relative to the firm backend-dependent finding.
