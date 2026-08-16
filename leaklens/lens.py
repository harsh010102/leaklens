"""Logit-lens trajectories and the recovery-depth metric (LeakLens's core contribution).

For a fact, we read at every layer the rank and log-probability the model assigns to the gold answer
token, giving a trajectory that shows *where* in the network the answer is (or is not) recoverable.
Running this on the original, the unlearned, and the unlearned+quantized model, and asking at which
layer the quantized trajectory returns toward the original, yields the recovery-depth metric.

Two correctness details are inherited from the thesis and matter (see decision.md, D6):
  * boundary-safe tokenization: track ' May' (space-prefixed), not 'May', or the rank is wrong;
  * no double-normalization at the final layer: HF's last hidden state is already normed, so use the
    model's true output logits there and head(norm(h)) only for intermediate layers.
"""
from __future__ import annotations
import numpy as np
import torch


def _norm_head(model):
    """Return (final_norm, unembedding_head) for a HF causal LM, defensively."""
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
    head = model.get_output_embeddings()
    if norm is None:                       # last resort: identity (rare architectures)
        norm = torch.nn.Identity()
    return norm, head


def gold_first_token_id(tok, prompt: str, answer: str, device) -> tuple[torch.Tensor, int, int]:
    """Boundary-safe: return (full_ids, answer_position, gold_first_token_id)."""
    prompt = prompt.rstrip()
    pre = tok(prompt, return_tensors="pt").input_ids.to(device)
    full = tok(prompt + " " + answer, return_tensors="pt").input_ids.to(device)
    pos = pre.shape[1] - 1                 # last prompt token predicts the answer
    gold_id = int(full[0, pre.shape[1]])
    return full, pos, gold_id


@torch.no_grad()
def logit_lens_trajectory(model, tok, prompt: str, answer: str, device) -> dict:
    """Per-layer rank and log-prob of the gold first token at the answer position."""
    full, pos, gold_id = gold_first_token_id(tok, prompt, answer, device)
    out = model(input_ids=full, output_hidden_states=True)
    hs = out.hidden_states
    true_logits = out.logits[0].float()
    norm, head = _norm_head(model)
    last = len(hs) - 1
    ranks, logps = [], []
    for L, h in enumerate(hs):
        logits = true_logits[pos] if L == last else head(norm(h[:, pos, :])).float()[0]
        ranks.append(int((logits > logits[gold_id]).sum()) + 1)
        logps.append(float(torch.log_softmax(logits, -1)[gold_id]))
    return {"rank": np.array(ranks), "logprob": np.array(logps),
            "n_layers": len(hs), "gold_id": gold_id, "final_rank": ranks[-1]}


def _within(a, b, threshold):
    return abs(a - b) <= threshold * max(b, np.log(2))


def recovery_depth(base_rank: np.ndarray, quant_rank: np.ndarray, fp16_rank: np.ndarray | None = None,
                   threshold: float = 0.25, band_frac: float = 0.5) -> int | None:
    """Shallowest layer (in the top 1-band_frac of the stack) at which quantization RECOVERS the fact.

    Defined on log-rank (rank spans orders of magnitude). If `fp16_rank` (the fp16-unlearned
    trajectory) is given, this is the v2 metric: the shallowest band layer at which the *quantized*
    model's gold log-rank returns to within `threshold` of the base while the *fp16-unlearned* model
    did NOT, i.e. the layer at which quantization specifically re-opens the fact (decision.md, D13).
    If `fp16_rank` is None it falls back to the v1 closeness-to-base definition. Returns the layer
    index, or None if there is no quantization-induced recovery in the band.
    """
    n = len(base_rank)
    lb = int(n * band_frac)
    lb_ = np.log(np.maximum(base_rank, 1))
    lq_ = np.log(np.maximum(quant_rank, 1))
    lu_ = np.log(np.maximum(fp16_rank, 1)) if fp16_rank is not None else None
    for L in range(lb, n):
        near_base = _within(lq_[L], lb_[L], threshold)
        if lu_ is None:
            if near_base:
                return L
        else:                                       # v2: quant near base AND fp16 was not (recovery)
            if near_base and not _within(lu_[L], lb_[L], threshold):
                return L
    return None
