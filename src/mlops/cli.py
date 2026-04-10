"""
Unified MLOps entrypoint.

  uv run tracedata-mlops production   # 3-feature synthetic windows → MLflow
  uv run tracedata-mlops synthetic   # 18-feature research pipeline
  uv run tracedata-mlops sqlite      # Train from telemetry.db (3 features)
  uv run tracedata-mlops compliance  # AIVT2-aligned fairness/xai/robustness checks
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tracedata-mlops",
        description="TraceData ML training / MLflow workflows",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "production",
        help="Production 3-feature model (10-min windows) — see production_mlops.yaml",
    )
    sub.add_parser(
        "synthetic",
        help="18-feature synthetic trip pipeline — see mlops_config.yaml",
    )
    sub.add_parser(
        "sqlite",
        help="Train from SQLite trips (processor + trainer path)",
    )
    sub.add_parser(
        "compliance",
        help="Run AIVT2-aligned nightly compliance checks against latest MLflow run",
    )

    args = parser.parse_args(argv)

    if args.command == "production":
        from src.mlops.production_window_training import ProductionWindowTrainingPipeline

        ProductionWindowTrainingPipeline().run()
    elif args.command == "synthetic":
        from src.mlops.training_pipeline import MLOpsTrainingPipeline

        MLOpsTrainingPipeline("mlops_config.yaml").run_pipeline()
    elif args.command == "sqlite":
        from src.utils.trainer import train_model

        train_model()
    elif args.command == "compliance":
        from src.compliance.aivt2_runner import main as compliance_main

        compliance_main()
    else:
        parser.error("unknown command")
        return 2
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
