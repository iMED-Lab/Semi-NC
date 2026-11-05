from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the complete YAML experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"num_classes", "model", "fia", "prototype", "training"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    training = config["training"]
    stream_total = int(training.get("labeled_batch_size", 0)) + int(
        training.get("unlabeled_batch_size", 0)
    )
    if stream_total and stream_total != int(training["batch_size"]):
        raise ValueError("labeled_batch_size + unlabeled_batch_size must equal batch_size")
    return config


def dynamic_gamma(epoch: int, floor: float = 0.7) -> float:
    """Paper Sec. 2.2.2: gamma=max(0.999-0.001*t, 0.7)."""
    return max(0.999 - 0.001 * int(epoch), float(floor))
