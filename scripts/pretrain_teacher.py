from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from semi_nc import SemiNCNet, load_config
from semi_nc.data import SegmentationManifestDataset, build_augmentation
from semi_nc.training import teacher_bootcamp_step
from semi_nc.utils import save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-NC teacher self-correcting bootcamp")
    parser.add_argument("--config", default="configs/semi_nc.yaml")
    parser.add_argument("--labeled-manifest", required=True)
    parser.add_argument("--output", default="checkpoints/teacher.pt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"] if args.seed is None else args.seed)
    seed_everything(seed)
    device = torch.device(args.device)
    dataset = SegmentationManifestDataset(
        args.labeled_manifest,
        int(config["num_classes"]),
        image_size=config["image_size"],
        augmentation=build_augmentation(config["augmentation"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )
    teacher = SemiNCNet(config).to(device)
    optimizer = torch.optim.Adam(
        teacher.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=(float(config["training"]["adam_beta1"]), float(config["training"]["adam_beta2"])),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["teacher_epochs"])
    for epoch in range(1, epochs + 1):
        teacher.train()
        running = 0.0
        for batch in loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = teacher_bootcamp_step(teacher, images, masks, config)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        print(f"teacher epoch={epoch}/{epochs} loss={running / len(loader):.6f}")
        save_checkpoint(args.output, teacher, optimizer, epoch, extra={"seed": seed})


if __name__ == "__main__":
    main()
