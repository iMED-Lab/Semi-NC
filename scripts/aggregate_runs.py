from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independently trained Semi-NC runs as paper mean±std"
    )
    parser.add_argument("runs", nargs="+", help="Evaluation JSON files (five for the paper protocol)")
    parser.add_argument("--output", default="results/five_run_summary.json")
    return parser.parse_args()


def aggregate(paths: list[str]) -> dict[str, dict[str, Union[float, int]]]:
    if len(paths) != 5:
        raise ValueError("The paper protocol requires exactly five independent runs")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    runs = [report.get("summary", report) for report in reports]
    metric_names = set(runs[0])
    if any(set(run) != metric_names for run in runs[1:]):
        raise ValueError("All run files must contain the same metric names")
    result = {}
    for name in sorted(metric_names):
        values = np.asarray(
            [run[name]["within_run_image_mean"] for run in runs], dtype=np.float64
        )
        result[name] = {
            "run_mean": float(values.mean()),
            "run_std": float(values.std(ddof=1)),
            "run_count": len(runs),
        }
    return result


def main() -> None:
    args = parse_args()
    summary = aggregate(args.runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
