"""Model and tokenizer loading, with the quantization backend applied at load time.

One entry point, `load_model`, hides the backend differences from the rest of the pipeline: the audit
loop asks for a model "at quant config X" and gets back a ready-to-probe module, whether that means an
fp16 model, a bitsandbytes-quantized model, or an RTN fake-quantized one.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import QuantConfig
from . import quantize


def _split_rev(model_id: str):
    """Support an `id@revision` syntax (some checkpoints, e.g. Hubble base models, pin a revision)."""
    if "@" in model_id:
        rid, rev = model_id.split("@", 1)
        return rid, rev
    return model_id, None


def load_tokenizer(model_id: str):
    rid, rev = _split_rev(model_id)
    tok = AutoTokenizer.from_pretrained(rid, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str, qc: QuantConfig, dtype: str = "float16", device: str = "cuda",
               calib_texts=None):
    """Load `model_id` (optionally `id@revision`) under quantization config `qc`, in eval mode.

    `calib_texts` supplies the calibration corpus for the calibration-based backends (awq, gptq);
    if omitted they fall back to a generic corpus. Sweeping `calib_texts` is the calibration attack.
    """
    quantize.check_backend(qc)
    rid, rev = _split_rev(model_id)
    torch_dtype = getattr(torch, dtype)

    def _calib():
        from . import data
        return calib_texts if calib_texts is not None else data.calibration_corpus("generic")

    if qc.backend == "gptq":
        model = quantize.gptq_quantize(rid, rev, load_tokenizer(model_id), _calib(), qc.bits,
                                       qc.extra.get("group_size", 128), device)
    elif qc.backend == "bnb":
        model = AutoModelForCausalLM.from_pretrained(
            rid, revision=rev, quantization_config=quantize.build_bnb_config(qc),
            device_map={"": 0} if device.startswith("cuda") else "cpu",
            torch_dtype=torch.float16, attn_implementation="eager")
    else:
        # "none" (reference), "rtn" (fake-quant), or "awq" (calibration-aware fake-quant)
        model = AutoModelForCausalLM.from_pretrained(
            rid, revision=rev, torch_dtype=torch_dtype, attn_implementation="eager").to(device)
        if qc.backend == "rtn":
            quantize.apply_rtn_(model, qc.bits)
        elif qc.backend == "awq":
            quantize.apply_awq_(model, load_tokenizer(model_id), _calib(), qc.bits,
                                qc.extra.get("alpha", 0.5), device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def free(model) -> None:
    """Move a model off the GPU and release the cache (audits load many models serially)."""
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
