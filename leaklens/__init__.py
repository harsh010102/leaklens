"""LeakLens: auditing whether unlearning survives deployment-time quantization.

Unlearning is validated at FP16; models ship at 4-bit. LeakLens measures the gap and localizes,
with the logit lens, the layer at which supposedly-forgotten knowledge returns.
"""
__version__ = "0.1.0"
