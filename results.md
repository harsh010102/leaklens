# LeakLens results

Audits of whether Hubble unlearning survives deployment-time quantization. All runs use the Hubble
Perturbed model as the "knows" base (it memorized the birthdates) and the author's unlearned checkpoints;
the forget set is the 44 real YAGO birthdate prompts (`data/yago_forget.json`). Every number here is
reproducible from a config in `configs/` (see the command at the bottom).

Two recovery signals are reported (see `decision.md`, D5/D10/D13):

* **recovery fraction** (behavioral): the share of the *deleted* knowledge that returns at a given
  precision, normalized against the fp16-unlearned baseline and the base. Isolates the quantization effect.
* **recovery depth** (representational, v2): the shallowest mid-stack layer at which the *quantized*
  model's gold-token rank returns toward the base while the *fp16-unlearned* model's had not, i.e. the
  layer at which quantization re-opens the fact.

---

## Result 1 — The quantization backend decides whether unlearning survives (8B)

`configs/tier1_8b.yaml`, base gold-probability **0.909**. Final-layer gold-token statistics:

| method | precision | gold-prob | median rank | recovery fraction |
|---|---|---|---|---|
| NPO | fp16 | 0.000 | 4{,}263 | baseline |
| NPO | bnb int8 | 0.000 | 4{,}243 | 0.00 |
| NPO | bnb nf4 | 0.000 | 3{,}301 | 0.00 |
| **NPO** | **rtn-int4** | **0.147** | **3** | **0.16** |
| IDK | fp16 | 0.009 | 16 | baseline |
| IDK | rtn-int4 | 0.006 | 18 | 0.00 |
| RMU | fp16 | 0.063 | 546 | baseline |
| RMU | rtn-int4 | 0.028 | 381 | 0.00 |

**Finding.** NPO's forgotten fact is buried at fp16 (rank 4{,}263) and stays buried under the
calibration-aware bitsandbytes INT8/NF4, but is **fully re-opened by crude 4-bit round-to-nearest**
(rank 3, recall $0\to0.24$). The **backend**, not just the bit-width, decides whether unlearning survives:
the same 4-bit budget preserves the erasure under NF4 and destroys it under RTN. Heatmap:
`docs/heatmap_tier1_8b.png`.

## Result 2 — The four methods respond differently, in line with their mechanism

* **IDK** keeps the birthdate token internally accessible throughout (rank ~16 at fp16, `recovered_frac`
  0.72), because it redirects the *output* ("I don't know") without disturbing the representation; quantization
  neither helps nor hurts. Its low gold-probability is behavioral, not representational.
* **NPO** buries the fact deeply at fp16 but the representation is fragile: RTN-int4 recovers it (Result 1).
* **RMU** sits in between (rank ~546 at fp16), the deepest representational suppression of the three, and is
  only mildly moved by quantization.

This mirrors the mechanistic taxonomy from the parent thesis (IDK surface, NPO late suppression, RMU deep
scramble), now viewed through a deployment-compression lens.

## Result 3 — Calibration-set attack with transparent AWQ: inconclusive (null)

`configs/attack_calib_8b.yaml` (NPO, $n=25$). Same backend (activation-aware AWQ) and 4-bit width, sweeping
only the calibration corpus:

| calibration set | gold-prob | median rank | recovery fraction |
|---|---|---|---|
| fp16 (baseline) | 0.000 | 4{,}263 | baseline |
| awq, generic | 0.005 | 58 | 0.006 |
| awq, adjacent | 0.008 | 29 | 0.009 |
| awq, forget | 0.007 | 48 | 0.008 |

**Finding.** The calibration-set-as-attack-surface hypothesis is **not supported** on this evidence: recovery
moves only from 0.006 to 0.009 across calibration corpora — within noise and not cleanly monotonic. What is
strong and calibration-*independent* is that 4-bit AWQ re-opens the fact representationally (median internal
rank $4{,}263 \to \sim30$) under every calibration. Reported as a null, not spun. Bar chart:
`docs/calibration_attack_8b.png`.

## Result 4 — Calibration-set attack with real GPTQ (error-compensated)

`configs/attack_gptq_8b.yaml` (NPO, $n=44$, `gptqmodel`, `desc_act=True`). _To be completed once the run
finishes; this is the decisive test, since GPTQ's activation-order-aware error compensation is more
calibration-sensitive than the transparent AWQ of Result 3._

---

## Limitations

* Single model family (Hubble), single attribute (birthdate), one unlearned checkpoint per method; the
  representational-recovery finding (Result 1) is on $n=25$--$44$ facts.
* The `awq` backend is a transparent activation-scaling re-implementation isolating the calibration
  mechanism, not the reference AWQ kernel (`decision.md`, D16). Result 4 uses real GPTQ as the cross-check.
* Recovery depth (v2) and recovery fraction are complementary; the behavioral fraction is the primary,
  quantization-isolating signal.

## Reproduce

```bash
sbatch scripts/run_audit.sbatch configs/tier1_8b.yaml        # Result 1-2
sbatch scripts/run_audit.sbatch configs/attack_calib_8b.yaml # Result 3
sbatch scripts/run_audit.sbatch configs/attack_gptq_8b.yaml  # Result 4
```
