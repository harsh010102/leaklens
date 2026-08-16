# LeakLens: Auditing Whether Unlearning Survives Deployment Compression

**A portfolio project spec derived from the TUM machine unlearning thesis, scoped to LLM quantization.**

---

## 1. The one-sentence pitch

> Unlearning is validated at FP16. Models are deployed at 4-bit. LeakLens measures the gap, and shows you which layer it opens at.

Longer version for a README:

> Conklin et al. (ICLR 2026) argue LLM training is best understood as lossy compression: models retain only what serves the objective. Machine unlearning claims to surgically delete a specific piece of that retained information. Quantization then applies a second, much cruder compression before deployment. LeakLens asks whether the second compression undoes the first deletion, and uses logit-lens and tuned-lens probes to localize exactly where in the network the supposedly-forgotten information comes back.

## 2. Why this is a good project for you specifically

| Asset you already have | How it plugs in directly |
|---|---|
| TOFU / MUSE / RWKU benchmark pipelines | Become the forget-set evaluation layer. No rebuild needed. |
| Logit lens + tuned lens implementation | Becomes the core differentiator vs. all prior work, which is behavioral or weight-space. |
| YAGO PII entity dataset | Becomes the GDPR / EU AI Act compliance story. Nobody else has a PII-specific quantization result. |
| Thesis framing ("models that appear to forget vs. genuinely no longer encode") | Is *literally* the thesis of this project. Quantization is the sharpest available test of that distinction. |
| NXP engineering habits (pipelines, automation) | Justifies building a tool rather than running a one-off experiment. |

Estimated marginal effort over your existing thesis code: **2 to 5 weeks**, not a new project from zero.

## 3. Prior work you must cite (and position against)

Read these before writing a line of code. Being fluent in them is half the interview value.

| Paper | What it established | What it left open |
|---|---|---|
| Zhang et al., *Catastrophic Failure of LLM Unlearning via Quantization*, ICLR 2025 (arXiv 2410.16454) | The core phenomenon: 21% retention at full precision, 83% at 4-bit. Weight-space theoretical explanation. Proposes a large-learning-rate mitigation. | Purely behavioral + weight-space. No representational analysis. Limited quantization backend coverage. |
| *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution* (MANSU), arXiv 2605.15138, May 2026 | Circuit attribution to isolate knowledge subgraphs; "magnitude floor" constraint on updates. Argues Zhang's LR fix doesn't address root cause. | It's a *method*, not a *measurement instrument*. Doesn't generalize across deployment backends. |
| *From Signal Degradation to Computation Collapse: Two Failure Modes of LLM Quantization*, arXiv 2604.19884, April 2026 | Logit-lens analysis of quantization failure modes generally. | Explicitly notes mechanistic analysis of quantized models is "preliminary" and "fragmented." Not unlearning-specific. |
| Guo et al., *Mechanistic Unlearning via Mechanistic Localization* (arXiv 2410.12949) | Localized edits disrupt latent knowledge more robustly than output-preserving localization. | Doesn't test quantization at all. |
| Hong et al., *Intrinsic Evaluation of Unlearning using Parametric Knowledge Traces*, EMNLP 2025 | Representation-level unlearning evaluation. | No compression axis. |

**Your honest positioning:** "The phenomenon is known. The mechanism is partially characterized. What doesn't exist is a reproducible, multi-backend audit tool that reports representational recovery, so I built one."

## 4. Technical specification

### 4.1 Models (pick 2, add a third only if compute allows)

| Model | Params | Why |
|---|---|---|
| Llama-3.2-1B-Instruct | 1B | Fast iteration, fits anywhere, good logit-lens behavior |
| Qwen2.5-7B-Instruct | 7B | Realistic deployment size, strong GGUF/AWQ ecosystem support |
| Phi-3-mini-4k | 3.8B | TOFU-adjacent lineage, different training recipe = tests generality |

Deliberately span **different training recipes**. Conklin et al.'s result that different families compress differently predicts they should also *decompress* differently under quantization. That's your one cheap nod to their paper, and it's a real hypothesis.

### 4.2 Unlearning methods (4 is enough)

- **GA** (Gradient Ascent) — the weak baseline that fails loudly
- **GradDiff** (GA + retain loss)
- **NPO** (Negative Preference Optimization) — current strong baseline
- **RMU** or **IdkDPO** — representation-level, should in principle be more robust

Include at least one representation-space method. Your hypothesis is that representation-level unlearning survives quantization better than gradient-space suppression. If true, that's a clean, quotable finding.

### 4.3 Quantization backends (this is where you beat prior work)

Prior papers test one or two. Test the ones people actually ship.

| Backend | Configs | Notes |
|---|---|---|
| RTN (round-to-nearest) | INT8, INT4 | Baseline, no calibration |
| bitsandbytes | LLM.int8(), NF4, FP4, +double-quant | The QLoRA default path |
| GPTQ (via GPTQModel) | 4-bit, 3-bit, group sizes 32/128 | **Calibration-based** |
| AWQ | 4-bit | **Calibration-based**, protects salient weights |
| llama.cpp GGUF | Q8_0, Q5_K_M, Q4_K_M, Q3_K_M | What local deployment actually uses |
| SmoothQuant / W8A8 | 8-bit weights + activations | Tests activation quantization separately |

### 4.4 The novel angle, if you want a paper out of this

**Calibration-set choice as an attack surface.**

GPTQ and AWQ both require a calibration corpus. Prior work uses generic C4/WikiText and treats this as an implementation detail. But if an adversary controls or influences the calibration set, can they *maximize* recovery of forgotten knowledge?

Experiment:
1. Calibrate on generic C4 → measure recovery (baseline)
2. Calibrate on the **retain set** → measure recovery
3. Calibrate on data **semantically adjacent to the forget set** → measure recovery
4. Calibrate on the **forget set itself** (worst case / upper bound)

If recovery scales with calibration-set proximity to the forget domain, you have shown that quantization calibration is an unlearning attack vector. That is a genuine, publishable contribution, it is cheap to run, and as far as I can tell nobody has done it. This is your ICLR/NeurIPS workshop submission if you want one.

### 4.5 Metrics

**Behavioral (reuse from thesis)**
- TOFU Forget Quality (KS-test against retain-model distribution)
- TOFU Model Utility
- ROUGE-L and exact-match on forget set
- Probability assigned to ground-truth answer
- Verbatim extraction rate

**PII-specific (your YAGO differentiator)**
- Attribute extraction success rate per entity
- Recovery rate broken down by attribute type (name, date, location, relation)
- This is the number that a compliance audience cares about

**Representational (your core contribution)**
- **Logit lens trajectory**: at every layer, unembed the hidden state and record rank + log-prob of the target token. Produce three curves per fact: original model, unlearned model, unlearned+quantized model.
- **Tuned lens trajectory**: same, with trained affine probes. More faithful; use as the headline result and logit lens as the cheap screen.
- **Recovery depth**: the shallowest layer at which the quantized model's target-token rank returns to within X% of the original model's. This single scalar is your key novel metric. Name it and define it precisely.
- **Linear probe accuracy per layer** on forget-set attributes, pre- and post-quantization.

**Weight-space (reproduce Zhang et al. as a control)**
- Fraction of unlearning-modified weights that round back to their original-model quantized bucket
- Correlation between that fraction and observed recovery

**Cheap Conklin proxy (optional, one afternoon)**
- Effective rank / entropy of hidden-state covariance per layer, before and after quantization
- Hypothesis: layers where representational entropy collapses most under quantization are the layers where recovery happens

### 4.6 The headline figure

A heatmap. Rows = unlearning method. Columns = quantization config. Cell colour = knowledge recovery rate. Cell annotation = recovery depth (layer index).

One image that tells the whole story. Put it at the top of the README.

## 5. Scoping tiers

Pick based on how much time you actually have before September.

### Tier 0 — the weekend version (2 days)
- One model (Llama-3.2-1B), one unlearning method (NPO), TOFU forget10
- Three quantization configs: FP16, bnb-NF4, GGUF Q4_K_M
- Behavioral recovery only, plus a single logit-lens figure
- **Output:** a notebook + a 3-paragraph blog post with one chart

Even this is a credible portfolio piece. Ship it before attempting more.

### Tier 1 — the real project (2 to 3 weeks)
- Two models, four unlearning methods, six quantization backends
- Full behavioral + logit lens + tuned lens + recovery-depth metric
- Packaged as an installable CLI: `leaklens audit --model X --forget-set Y`
- **Output:** GitHub repo with README, the heatmap, reproducible configs, a written report

### Tier 2 — the paper (5 to 6 weeks)
- Everything in Tier 1
- Plus the calibration-set attack experiments (§4.4)
- Plus YAGO PII results
- **Output:** workshop paper submission + the tool

**Recommendation:** commit to Tier 1. Do Tier 0 first as a smoke test. Only escalate to Tier 2 if a supervisor or a lab bites.

## 6. Compute reality check

This is deliberately cheap.

- Unlearning runs with LoRA on a 1B model: ~15 to 30 min each on a single A100. Twelve runs is one long day.
- Quantization itself: minutes. GPTQ calibration on 128 samples is the slowest and is still under 20 minutes for 7B.
- Logit lens: pure forward passes, negligible.
- Tuned lens: training affine probes per layer, ~1 hour per model.

Total: comfortably within a TUM chair's cluster allocation, or roughly 50 to 150 EUR on Lambda / RunPod if you self-fund. A 24GB consumer GPU handles the 1B and 3B tiers entirely.

## 7. Deliverables that actually convert to interviews

Build these four, in this order.

1. **GitHub repo** — clean README opening with the heatmap and the three-line lossy-compression framing. Installable. Config-driven. Reproducible with one command. This is the artifact people actually click.
2. **A written report or blog post** — 1500 words. Title it for the finding, not the method. Something like *"Your model forgot. Then you quantized it, and it remembered."* Post to your academicpages site and cross-post to LinkedIn.
3. **A one-page PDF summary** — for attaching to applications and cold emails. Problem, method, headline number, link.
4. **A 90-second demo** — screen recording of the CLI running an audit and printing a report card. Optional but disproportionately effective for solutions-engineering roles.

## 8. How to pitch it, by role type

**AI assurance / governance (TÜV AI.Lab, appliedAI Institute, Fraunhofer IAIS)**
> "GDPR Article 17 erasure and EU AI Act documentation are validated on the full-precision model. The model that ships is 4-bit. I built the tool that measures whether the erasure survives that gap, and it frequently doesn't."

Lead with the PII/YAGO results. This is the strongest framing you have for this audience and it is genuinely underserved.

**Applied AI / forward-deployed engineer (Palantir, Celonis, NVIDIA, Aleph Alpha)**
> "I built a reproducible eval harness spanning six quantization backends, from bitsandbytes to GGUF, with a CI-style report card. It surfaces a class of silent regression that standard benchmarks miss."

Lead with the engineering: multi-backend abstraction, config-driven runs, reproducibility. De-emphasize the interpretability theory.

**Research engineer / applied scientist**
> "Prior work showed the phenomenon behaviorally and explained it in weight space. I localized it representationally and introduced a recovery-depth metric, then found that calibration-set choice modulates it."

Lead with §4.4 and the recovery-depth metric.

**Semiconductor / edge AI (Infineon, Qualcomm, Apple, NXP internally)**
> "Edge deployment means aggressive quantization. If your compliance story depends on unlearning, quantization can silently void it. Here's the measurement."

This one is worth raising inside NXP. Edge inference is their business, and a working-student who brings a novel safety-relevant finding about deployment compression is exactly the person who gets converted to full-time.

## 9. Immediate next steps

1. Read Zhang et al. 2410.16454 end to end. Clone their repo (github.com/zzwjames/FailureLLMUnlearning) and reproduce one number. This anchors everything.
2. Skim MANSU (2605.15138) and 2604.19884 for positioning, not method.
3. Do Tier 0 this weekend on Llama-3.2-1B.
4. If the Tier 0 chart looks interesting, show it to your thesis supervisor at the Chair of Responsible Data Science and ask whether it's worth a workshop submission. Supervisor buy-in converts this from a side project into a second publication.
5. Register the repo name and push the skeleton early, so the commit history shows sustained work rather than a weekend dump.
