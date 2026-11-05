from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from semi_nc import SemiNCNet, load_config
from semi_nc.data import SegmentationManifestDataset, build_augmentation
from semi_nc.training import SemiNCTrainer
from semi_nc.utils import load_checkpoint, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Semi-NC Algorithm 2 training")
    parser.add_argument("--config", default="configs/semi_nc.yaml")
    parser.add_argument("--labeled-manifest", required=True)
    parser.add_argument("--unlabeled-manifest", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", default="checkpoints/student.pt")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _loader(dataset, config, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        drop_last=shuffle,
    )


def _repeat_loader(loader):
    """Repeat a loader without itertools.cycle retaining every batch in memory."""
    while True:
        yield from loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"] if args.seed is None else args.seed)
    seed_everything(seed)
    device = torch.device(args.device)
    augmentation = build_augmentation(config["augmentation"])
    labeled = SegmentationManifestDataset(
        args.labeled_manifest, config["num_classes"], config["image_size"], augmentation
    )
    unlabeled = SegmentationManifestDataset(
        args.unlabeled_manifest,
        config["num_classes"],
        config["image_size"],
        augmentation,
        unlabeled=True,
    )
    labeled_loader = _loader(
        labeled, config, config["training"]["labeled_batch_size"]
    )
    unlabeled_loader = _loader(
        unlabeled, config, config["training"]["unlabeled_batch_size"]
    )
    student, teacher = SemiNCNet(config).to(device), SemiNCNet(config).to(device)
    load_checkpoint(args.teacher, teacher, map_location=device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = SemiNCTrainer(student, teacher, config)
    start_epoch = 1
    if args.resume:
        payload = load_checkpoint(args.resume, student, optimizer, map_location=device)
        start_epoch = int(payload["epoch"]) + 1
        if payload.get("prototype_pools"):
            trainer.pool.load_state_dict(payload["prototype_pools"], device=device)

    total_epochs = int(config["training"]["total_epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        student.train()
        running = 0.0
        unlabeled_cycle = _repeat_loader(unlabeled_loader)
        for labeled_batch in labeled_loader:
            unlabeled_batch = next(unlabeled_cycle)
            optimizer.zero_grad(set_to_none=True)
            result = trainer.step(
                labeled_batch["image"].to(device),
                labeled_batch["mask"].to(device),
                unlabeled_batch["image"].to(device),
                epoch,
            )
            result.loss.backward()
            optimizer.step()
            running += float(result.loss.detach())
        print(
            f"student epoch={epoch}/{total_epochs} loss={running / len(labeled_loader):.6f} "
            f"gamma={result.gamma:.3f} pools={len(trainer.pool)}"
        )
        save_checkpoint(
            args.output,
            student,
            optimizer,
            epoch,
            prototype_state=trainer.pool.state_dict(),
            extra={"seed": seed},
        )


if __name__ == "__main__":
    main()
