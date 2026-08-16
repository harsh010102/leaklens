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

SUPPORTED = {"none", "bnb", "rtn", "awq", "gptq"}
_EXTENSION = {"gguf"}   # native-kernel backend, left as an extension point


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


@torch.no_grad()
def collect_input_scales(model, tok, calib_texts, device, max_len: int = 64) -> dict:
    """Run the calibration corpus and accumulate, per Linear, the mean absolute input activation per
    input channel. These calibration-derived statistics are what make the quantization data-dependent
    (and hence make the calibration set an attack surface, spec 4.4)."""
    scales: dict[int, torch.Tensor] = {}

    def hook(mod, inp, _out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        a = x.abs().mean(0)
        scales[id(mod)] = a if id(mod) not in scales else scales[id(mod)] + a

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
    for t in calib_texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] > 0:
            model(input_ids=ids)
    for h in handles:
        h.remove()
    return scales


@torch.no_grad()
def apply_awq_(model, tok, calib_texts, bits: int, alpha: float = 0.5, device="cuda") -> nn.Module:
    """Activation-aware round-to-nearest (an AWQ-style fake-quant). High-activation input channels,
    as measured on the CALIBRATION corpus, are protected by a per-channel scale before rounding and
    unscaled after, so the calibration data determines which computations survive quantization.
    This is a transparent re-implementation isolating the calibration mechanism (decision.md, D16)."""
    scales = collect_input_scales(model, tok, calib_texts, device)
    qmax = 2 ** (bits - 1) - 1
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and id(mod) in scales:
            a = scales[id(mod)].to(mod.weight.device)
            a = (a / a.mean().clamp_min(1e-8)).clamp_min(1e-4)
            s = a.pow(alpha).clamp(1e-2, 1e2)                    # per-input-channel protection factor
            w = mod.weight.data.float() * s.unsqueeze(0)         # scale columns up
            row_s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
            w_q = torch.round(w / row_s).clamp_(-qmax, qmax) * row_s
            mod.weight.data.copy_((w_q / s.unsqueeze(0)).to(mod.weight.dtype))   # unscale
    return model


def gptq_quantize(model_id: str, revision, tok, calib_texts, bits: int, group_size: int = 128, device="cuda"):
    """Real GPTQ via gptqmodel, calibrated on `calib_texts`. Best-effort: raises a clear error if the
    backend cannot run on this model so the caller can fall back to the transparent 'awq' backend."""
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except Exception as e:
        raise NotImplementedError(f"gptq backend needs gptqmodel: {e}")
    qcfg = QuantizeConfig(bits=bits, group_size=group_size)
    model = GPTQModel.load(model_id, qcfg, revision=revision, trust_remote_code=True)
    model.quantize([{"text": t} for t in calib_texts], batch_size=1)
    return model.model if hasattr(model, "model") else model


def check_backend(qc: QuantConfig) -> None:
    if qc.backend in _EXTENSION:
        raise NotImplementedError(
            f"backend '{qc.backend}' is a documented extension point (needs a calibration/native "
            f"path). Implement it in quantize.py; the audit pipeline already treats backends as "
            f"pluggable.")
    if qc.backend not in SUPPORTED:
        raise ValueError(f"unknown backend {qc.backend!r}; supported: {sorted(SUPPORTED)}")
