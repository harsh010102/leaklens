"""CPU-only smoke tests: imports, config parsing, RTN fake-quant, recovery-depth logic, metrics.
No model download or GPU needed. Run: python -m pytest tests/  (or python tests/test_smoke.py)."""
import numpy as np
import torch
import torch.nn as nn

from leaklens.config import load_config, QuantConfig
from leaklens import quantize, lens, metrics


def test_config_roundtrip(tmp_path=None):
    import textwrap, tempfile, os
    y = textwrap.dedent("""
        base_model: dummy/base
        unlearned_models: {NPO: dummy/npo}
        quant_configs:
          - {name: fp16, backend: none, bits: 16}
          - {name: nf4, backend: bnb, bits: 4, extra: {quant_type: nf4}}
        lens: {band_frac: 0.5, recovery_threshold: 0.25}
    """)
    p = os.path.join(tempfile.mkdtemp(), "c.yaml"); open(p, "w").write(y)
    cfg = load_config(p)
    assert cfg.base_model == "dummy/base"
    assert cfg.quant_configs[1].backend == "bnb" and cfg.quant_configs[1].extra["quant_type"] == "nf4"


def test_rtn_changes_but_preserves_shape():
    lin = nn.Linear(64, 32)
    before = lin.weight.data.clone()
    m = nn.Sequential(lin)
    quantize.apply_rtn_(m, bits=4)
    assert lin.weight.shape == before.shape
    assert not torch.allclose(lin.weight.data, before)     # quantization changed the weights
    assert (lin.weight.data.abs() <= before.abs().amax() + 1e-4).all()


def test_recovery_depth():
    base = np.array([9000, 5000, 800, 40, 3, 1])           # base recovers late
    quant = np.array([9000, 5000, 900, 60, 3, 1])          # quant tracks base in the top half
    d = lens.recovery_depth(base, quant, threshold=0.25, band_frac=0.5)
    assert d is not None and d >= 3
    never = np.array([9000, 9000, 9000, 9000, 9000, 9000]) # quant stays buried -> no recovery
    assert lens.recovery_depth(base, never, threshold=0.05, band_frac=0.5) is None


def test_rouge_and_exact():
    assert metrics.rouge_l("May 16 1988", "May 16, 1988") > 0.5
    assert metrics.exact_match("born on May 16", "May 16") == 1
    assert metrics.rouge_l("", "x") == 0.0


def test_extension_backend_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        quantize.check_backend(QuantConfig("g", "gptq", 4))


if __name__ == "__main__":
    test_config_roundtrip(); test_rtn_changes_but_preserves_shape(); test_recovery_depth()
    test_rouge_and_exact()
    print("smoke tests passed")
