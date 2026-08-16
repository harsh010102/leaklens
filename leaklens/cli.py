"""Command-line entry point:  leaklens audit --config configs/x.yaml

Also exposed as `python -m leaklens.cli`. A `--dry-run` validates the config and lists the planned
work without loading any model (useful on a login node without a GPU).
"""
from __future__ import annotations
import argparse
import sys

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="leaklens", description="Audit unlearning survival under quantization.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="run an audit from a config file")
    a.add_argument("--config", required=True)
    a.add_argument("--out", default=None, help="override output_dir")
    a.add_argument("--dry-run", action="store_true", help="validate config and print the plan; load nothing")
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        cfg = load_config(args.config)
        if args.out:
            cfg.output_dir = args.out
        plan = [f"{m} x {q.name}({q.backend}/{q.bits}b)" for m in cfg.unlearned_models for q in cfg.quant_configs]
        print(f"[leaklens] plan: {len(plan)} (method x quant) cells over base={cfg.base_model}")
        for p in plan:
            print("   -", p)
        if args.dry_run:
            print("[leaklens] dry-run: config valid; nothing loaded.")
            return 0
        from .audit import run_audit
        from .report import write_report
        summary, trajectories, base_ref, meta = run_audit(cfg)
        write_report(summary, trajectories, base_ref, meta, cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
