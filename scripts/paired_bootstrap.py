from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RecordKey = Tuple[str, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired patient-cluster bootstrap significance testing"
    )
    parser.add_argument("--ours", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", default="results/paired_bootstrap.json")
    return parser.parse_args()


def _indexed_records(report: Dict[str, object]) -> Dict[RecordKey, Tuple[str, float]]:
    indexed = {}
    for record in report["records"]:  # type: ignore[index]
        key = (record["image"], int(record["class_index"]), record["metric"])
        patient_id = str(record.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(
                "Patient-cluster bootstrap requires patient_id for every manifest row"
            )
        indexed[key] = (patient_id, float(record["value"]))
    return indexed


def paired_patient_bootstrap(
    ours: Dict[str, object],
    baseline: Dict[str, object],
    samples: int = 10000,
    seed: int = 321,
) -> Dict[str, Dict[str, float]]:
    ours_records = _indexed_records(ours)
    baseline_records = _indexed_records(baseline)
    if set(ours_records) != set(baseline_records):
        raise ValueError("Paired reports must contain identical image/class/metric records")

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for key, (patient_id, ours_value) in ours_records.items():
        baseline_patient, baseline_value = baseline_records[key]
        if patient_id != baseline_patient:
            raise ValueError(f"Patient mismatch for paired record {key}")
        metric_name = f"class_{key[1]}/{key[2]}"
        improvement = ours_value - baseline_value
        if key[2] in {"fdr", "mhd"}:
            improvement = -improvement
        grouped[metric_name][patient_id].append(improvement)

    rng = np.random.default_rng(seed)
    result = {}
    for metric_name, patient_values in sorted(grouped.items()):
        patients = sorted(patient_values)
        draws = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            selected = rng.choice(patients, size=len(patients), replace=True)
            retained = [value for patient in selected for value in patient_values[patient]]
            draws[index] = np.mean(retained)
        observed = float(np.mean([value for values in patient_values.values() for value in values]))
        p_value = float(
            min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0)))
        )
        result[metric_name] = {
            "observed_improvement": observed,
            "p_value_two_sided": p_value,
            "bootstrap_samples": samples,
            "patient_count": len(patients),
        }
    return result


def main() -> None:
    args = parse_args()
    ours = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    result = paired_patient_bootstrap(ours, baseline, args.samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
