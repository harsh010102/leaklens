"""Configuration schema and YAML loader.

A LeakLens audit is fully described by a single YAML config: which base ("knows") model, which
unlearned models (one per unlearning method), which forget set, and which quantization configs to
sweep. Keeping the run config-driven (rather than argument-driven) is a deliberate choice so that
every reported number is reproducible from one file checked into the repo (see decision.md, D2).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class QuantConfig:
    """One deployment quantization setting to audit."""
    name: str                       # label shown in the report (e.g. "nf4", "int8", "fp16")
    backend: str                    # "none" | "bnb" | "rtn"  (gptq/awq/gguf are extension points)
    bits: int = 16                  # nominal bit-width
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific knobs (e.g. quant_type)


@dataclass
class LensConfig:
    enabled: bool = True
    use_tuned: bool = False          # tuned lens is optional; requires trained translators
    band_frac: float = 0.5           # recovery-depth searches the top (1-band_frac) of layers
    recovery_threshold: float = 0.25 # "within X% of base log-rank" defines recovery depth (D5)


@dataclass
class AuditConfig:
    base_model: str                              # original model that KNOWS the facts (reference)
    unlearned_models: dict[str, str]             # {method_name: hf_id_or_path} that FORGOT at fp16
    forget_set: str = "builtin:demo"             # path to json/csv, or "builtin:demo"
    quant_configs: list[QuantConfig] = field(default_factory=list)
    lens: LensConfig = field(default_factory=LensConfig)
    max_facts: int | None = None                 # cap facts for a fast smoke run
    max_new_tokens: int = 8                       # for greedy extraction metrics
    dtype: str = "float16"                        # fp16 reference dtype
    device: str = "cuda"
    seed: int = 0
    output_dir: str = "runs/latest"


def load_config(path: str) -> AuditConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = dict(raw or {})
    raw["quant_configs"] = [QuantConfig(**q) for q in raw.get("quant_configs", [])]
    if "lens" in raw and isinstance(raw["lens"], dict):
        raw["lens"] = LensConfig(**raw["lens"])
    return AuditConfig(**raw)
