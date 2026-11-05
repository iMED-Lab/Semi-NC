import json

import pytest

from scripts.aggregate_runs import aggregate


def test_aggregate_runs_uses_run_means_not_within_run_std(tmp_path):
    paths = []
    values = (0.7, 0.75, 0.8, 0.85, 0.9)
    for index, value in enumerate(values):
        path = tmp_path / f"run_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "class_0/dice": {
                        "within_run_image_mean": value,
                        "within_run_image_std": 99.0,
                        "image_count": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
        paths.append(str(path))
    summary = aggregate(paths)
    assert summary["class_0/dice"]["run_mean"] == pytest.approx(0.8)
    assert summary["class_0/dice"]["run_std"] == pytest.approx(0.0790569415)
