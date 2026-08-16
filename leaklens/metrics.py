"""Behavioral (output-level) metrics, reused from the standard unlearning evaluation vocabulary.

These are the metrics prior quantization work relies on; LeakLens keeps them (so results are
comparable) but treats them as the *screen*, with the logit-lens recovery of lens.py as the headline.
Boundary-safe tokenization is used throughout for consistency with the lens (decision.md, D6).
"""
from __future__ import annotations
import math
import torch


def _boundary(tok, prompt, answer, device):
    prompt = prompt.rstrip()
    pre = tok(prompt, return_tensors="pt").input_ids.to(device)
    full = tok(prompt + " " + answer, return_tensors="pt").input_ids.to(device)
    return pre, full


@torch.no_grad()
def gold_probability(model, tok, prompt, answer, device) -> float:
    """Length-normalized teacher-forced probability of the gold answer."""
    pre, full = _boundary(tok, prompt, answer, device)
    if full.shape[1] <= pre.shape[1]:
        return 0.0
    logits = model(input_ids=full).logits[0]
    lps = [float(torch.log_softmax(logits[p].float(), -1)[full[0, p + 1]])
           for p in range(pre.shape[1] - 1, full.shape[1] - 1)]
    return math.exp(sum(lps) / max(len(lps), 1))


@torch.no_grad()
def greedy_answer(model, tok, prompt, max_new, device) -> str:
    ids = tok(prompt.rstrip(), return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def rouge_l(pred: str, gold: str) -> float:
    a, b = _norm(pred).split(), _norm(gold).split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> int:
    return int(_norm(gold) in _norm(pred))
