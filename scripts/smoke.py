from __future__ import annotations

import json
from pathlib import Path

import torch

from semi_nc import SemiNCNet, load_config


def main() -> None:
    config = load_config("configs/semi_nc.yaml")
    # Explicit execution-only synthetic input; this is not a scientific result.
    config["model"]["widths"] = [8, 16, 32, 64, 128]
    config["model"]["attention_dim"] = 16
    model = SemiNCNet(config).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 1, 64, 64))
    artifact = {
        "kind": "PLACEHOLDER_SMOKE_ONLY",
        "logits_shape": list(output["logits"].shape),
        "embedding_shape": list(output["embedding"].shape),
        "regularized_layers": len(output["regularization_features"]),
    }
    destination = Path("out/smoke.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
