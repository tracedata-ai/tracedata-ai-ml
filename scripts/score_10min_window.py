#!/usr/bin/env python3
"""
One scoring path: **trip = series of 10-minute ping windows** → **one score + one explanation**.

  SmoothnessInference.from_local_paths(joblib, serving_dir)
      .score_trip_from_ping_windows(windows)

``windows`` is a list of windows; each window is a list of pings with ``acceleration_ms2`` and
``speed_kmh``. A single 10-minute bucket is ``[ pings ]``.

JSON file shape: ``{ "windows": [ [...], [...] ] }`` — or a bare array of pings for one window only.

Replace ``scripts/sample_serving`` with your MLflow ``serving/`` folder in production.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_paths() -> tuple[Path, Path]:
    model = _REPO_ROOT / "models" / "smoothness_model.joblib"
    serving = _REPO_ROOT / "scripts" / "sample_serving"
    return model, serving


def build_demo_window(
    *,
    seed: int,
    n_points: int = 20,
    interval_seconds: int = 30,
    t0: datetime | None = None,
) -> list[dict]:
    """One ~10 min window: pings every ``interval_seconds``."""
    rng = random.Random(seed)
    start = t0 or datetime(2026, 3, 7, 8, 0, 0)
    pings: list[dict] = []
    for i in range(n_points):
        pings.append(
            {
                "timestamp": (start + timedelta(seconds=i * interval_seconds)).isoformat(),
                "speed_kmh": 40.0 + rng.uniform(-5, 8),
                "acceleration_ms2": rng.uniform(-0.25, 0.25),
                "lat": 1.35,
                "lon": 103.84,
            }
        )
    return pings


def parse_trip_json(raw: object) -> list[list[dict]]:
    """Accept ``{windows: [...]}`` or a single flat list of pings (one window)."""
    if isinstance(raw, dict) and "windows" in raw:
        w = raw["windows"]
        if not isinstance(w, list) or not w:
            raise ValueError('"windows" must be a non-empty list')
        out: list[list[dict]] = []
        for i, win in enumerate(w):
            if not isinstance(win, list) or not win:
                raise ValueError(f"windows[{i}] must be a non-empty list of pings")
            out.append(win)
        return out
    if isinstance(raw, list):
        if not raw:
            raise ValueError("empty list")
        if isinstance(raw[0], dict):
            return [raw]
        if isinstance(raw[0], list):
            return raw
    raise ValueError('Expected {"windows": [[pings]...]} or [pings] or [[pings]...]')


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a full trip from one or more 10-minute ping windows.",
    )
    parser.add_argument("--model", type=Path, default=_default_paths()[0])
    parser.add_argument("--serving", type=Path, default=_default_paths()[1])
    parser.add_argument(
        "--trip-json",
        type=Path,
        default=None,
        help='JSON: {"windows": [[{ping}...], ...]} or one array of pings for a single window',
    )
    parser.add_argument(
        "--demo-windows",
        type=int,
        default=3,
        help="With no --trip-json, number of synthetic 10-min windows (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1
    if not (args.serving / "model_contract.json").is_file():
        print(f"Missing serving dir: {args.serving}", file=sys.stderr)
        return 1

    if args.trip_json:
        windows = parse_trip_json(json.loads(args.trip_json.read_text(encoding="utf-8")))
    else:
        base = datetime(2026, 3, 7, 8, 0, 0)
        windows = [
            build_demo_window(seed=args.seed + i, t0=base + timedelta(minutes=10 * i))
            for i in range(max(1, args.demo_windows))
        ]

    sys.path.insert(0, str(_REPO_ROOT))
    from src.inference.smoothness_inference import SmoothnessInference

    try:
        scorer = SmoothnessInference.from_local_paths(args.model, args.serving)
    except ValueError as e:
        print(f"Failed to load inference: {e}", file=sys.stderr)
        return 1

    result = scorer.score_trip_from_ping_windows(windows)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
