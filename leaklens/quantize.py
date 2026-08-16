"""Quantization backends.

The whole point of LeakLens is to sweep *deployment* quantization settings, so the backend is a
first-class, swappable dimension. We implement the two that need no extra native dependency and cover
the two important axes:

  * bitsandbytes ("bnb")  -> LLM.int8(), NF4, FP4 (+ double-quant): the QLoRA / HF default path.
  * round-to-nearest ("rtn") -> a transparent, backend-free fake-quant baseline (INT8/INT4), applied
    as quantize-then-dequantize on the linear weights, so the audit runs anywhere and the mechanism is
    inspectable.

GPTQ / AWQ / GGUF are calibration- or native-kernel-based and are left as explicit extension points
(they raise a clear NotImplementedError) rather than half-working stubs. See decision.md (D4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .config import QuantConfig

SUPPORTED = {"none", "bnb", "rtn"}
_EXTENSION = {"gptq", "awq", "gguf"}


def build_bnb_config(qc: QuantConfig):
    """Return a transformers BitsAndBytesConfig for backend == 'bnb'."""
    from transformers import BitsAndBytesConfig
    if qc.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if qc.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qc.extra.get("quant_type", "nf4"),   # "nf4" or "fp4"
            bnb_4bit_use_double_quant=qc.extra.get("double_quant", True),
            bnb_4bit_compute_dtype=torch.float16,
        )
    raise ValueError(f"bnb backend supports 4 or 8 bits, got {qc.bits}")


@torch.no_grad()
def apply_rtn_(model: nn.Module, bits: int) -> nn.Module:
    """In-place symmetric per-output-channel round-to-nearest fake-quant of all Linear weights.

    quantize-then-dequantize: w_q = clamp(round(w / s), -qmax, qmax) * s, with a per-row scale s.
    This is a transparent stand-in for a real INT kernel; it reproduces the *information loss* of RTN
    (which is what drives unlearning recovery) without needing an integer matmul backend.
    """
    qmax = 2 ** (bits - 1) - 1
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.float()
            s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax   # per-row scale
            w_q = torch.round(w / s).clamp_(-qmax, qmax) * s
            module.weight.data.copy_(w_q.to(module.weight.dtype))
    return model


def check_backend(qc: QuantConfig) -> None:
    if qc.backend in _EXTENSION:
        raise NotImplementedError(
            f"backend '{qc.backend}' is a documented extension point (needs a calibration/native "
            f"path). Implement it in quantize.py; the audit pipeline already treats backends as "
            f"pluggable.")
    if qc.backend not in SUPPORTED:
        raise ValueError(f"unknown backend {qc.backend!r}; supported: {sorted(SUPPORTED)}")
