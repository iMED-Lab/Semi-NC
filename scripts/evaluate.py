from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from semi_nc import SemiNCNet, load_config
from semi_nc.data import SegmentationManifestDataset
from semi_nc.metrics import segmentation_metrics
from semi_nc.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-NC Dice/FDR/MHD evaluation")
    parser.add_argument("--config", default="configs/semi_nc.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    dataset = SegmentationManifestDataset(
        args.manifest, config["num_classes"], config["image_size"]
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = SemiNCNet(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    values = defaultdict(list)
    records = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["image"].to(device))["probabilities"].cpu().numpy() >= args.threshold
            target = batch["mask"].numpy() >= 0.5
            for class_index in range(config["num_classes"]):
                metrics = segmentation_metrics(prediction[0, class_index], target[0, class_index])
                for name, value in metrics.items():
                    values[f"class_{class_index}/{name}"].append(value)
                    records.append(
                        {
                            "image": batch["image_path"][0],
                            "patient_id": batch["patient_id"][0],
                            "class_index": class_index,
                            "metric": name,
                            "value": value,
                        }
                    )
    summary = {
        name: {
            "within_run_image_mean": float(np.mean(v)),
            "within_run_image_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "image_count": len(v),
        }
        for name, v in values.items()
    }
    report = {"summary": summary, "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
