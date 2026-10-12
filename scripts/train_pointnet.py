"""PointNet++ baseline on spatial split B; train/val only, epoch checkpoints."""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "PointnetPP"))
from lidar_dataloader import (LiDARTileDataset, Compose, RandomRotation,
                              RandomJitter, RandomScale)
from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss

CLASS_NAMES = ["water", "tree_canopy", "low_vegetation", "impervious", "buildings"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def metrics(cm):
    cm = cm.double()
    tp = cm.diag()
    actual, predicted = cm.sum(1), cm.sum(0)
    union = actual + predicted - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.zeros_like(tp))
    f1 = 2 * tp / (actual + predicted).clamp_min(1)
    return dict(accuracy=float(tp.sum() / cm.sum().clamp_min(1)),
                mean_iou=float(iou.mean()), macro_f1=float(f1.mean()),
                iou=iou.tolist(), f1=f1.tolist(), confusion_matrix=cm.long().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="Stop after this many additional complete epochs; 0 means no limit.")
    parser.add_argument("--resume", help="Trusted checkpoint created by this script.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or min(args.num_workers, args.stop_after) < 0:
        raise ValueError("Invalid positive/nonnegative arguments.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: submit the SLURM job.")
    torch.set_num_threads(4)
    data, output = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    metadata = json.loads((data / "metadata.json").read_text())
    if metadata.get("spatial_split", {}).get("variant") != "B":
        raise ValueError("Expected spatial split B.")
    if tuple(metadata[k] for k in ("points_per_tile", "num_features", "num_classes")) != (16384, 7, 5):
        raise ValueError("Expected 16384 points, seven features, five classes.")
    for split in ("train", "val"):
        for suffix in ("features", "labels"):
            if not (data / f"{split}_{suffix}.npy").is_file():
                raise FileNotFoundError(data / f"{split}_{suffix}.npy")
    if args.resume:
        if Path(args.resume).resolve().parent != output:
            raise ValueError("Resume into the original output directory.")
        if not (output / "history.csv").is_file():
            raise FileNotFoundError(output / "history.csv")
    else:
        output.mkdir(parents=True, exist_ok=False)

    config = dict(data_dir=str(data), epochs=args.epochs, batch_size=args.batch_size,
                  num_workers=args.num_workers, patience=args.patience, seed=args.seed,
                  lr=1e-3, weight_decay=5e-5, dropout=0.6,
                  class_weights=[2.0, 0.6, 0.2, 0.4, 0.45],
                  loss="focal(gamma=2) + 0.2 * dice", augmentation=dict(
                      rotation=180, jitter_sigma=0.02, jitter_clip=0.05, scale=[0.8, 1.2]))
    seed_all(args.seed)
    augmentation = Compose([RandomRotation(max_angle=180), RandomJitter(sigma=0.02, clip=0.05),
                            RandomScale(scale_low=0.8, scale_high=1.2)])
    datasets = {"train": LiDARTileDataset(str(data), "train", augmentation),
                "val": LiDARTileDataset(str(data), "val")}
    generators = {s: torch.Generator() for s in datasets}
    loaders = {}
    for split, dataset in datasets.items():
        n = metadata["split_sizes"][split]
        if dataset.features.shape != (n, 16384, 7) or dataset.labels.shape != (n, 16384):
            raise ValueError(f"Unexpected {split} shapes.")
        if dataset.features.dtype != np.float32 or dataset.labels.dtype != np.int64:
            raise ValueError(f"Unexpected {split} dtypes.")
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=split == "train", drop_last=False,
                                    num_workers=args.num_workers, pin_memory=True,
                                    worker_init_fn=seed_worker, generator=generators[split])
    model = PointNet2AerialSSG(num_classes=5, input_channels=4, use_xyz=True, dropout=0.6).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=args.epochs, steps_per_epoch=len(loaders["train"]),
        pct_start=0.1, div_factor=25.0, final_div_factor=1e4)
    alpha = torch.tensor(config["class_weights"], device="cuda")
    dice_fn = DiceLoss()
    start, best, bad_epochs = 0, float("inf"), 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        if saved["config"] != config or saved["metadata"] != metadata:
            raise ValueError("Resume configuration/data metadata differ from checkpoint.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start, best, bad_epochs = saved["epoch"], saved["best_val_loss"], saved["bad_epochs"]
        with (output / "history.csv").open(newline="") as f:
            history = list(csv.DictReader(f))
        if not history or int(history[-1]["epoch"]) != start:
            raise ValueError("History/checkpoint mismatch: inspect before resuming.")
    else:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
        except (OSError, subprocess.CalledProcessError):
            revision, status = "unknown", "unknown"
        (output / "config.json").write_text(json.dumps(dict(config=config, metadata=metadata,
             git_commit=revision, git_status=status, torch=torch.__version__, cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0), test_data_used=False,
             metric_note="Sampled entries include repeated points; macro metrics average five classes."), indent=2))
        (output / "train_pointnet_source.py").write_bytes(Path(__file__).read_bytes())
    print(f"GPU: {torch.cuda.get_device_name(0)}; train batches: {len(loaders['train'])}", flush=True)
    print("Full train/val epochs; test data are not loaded.", flush=True)
    if start >= args.epochs or bad_epochs >= args.patience:
        print("Run already finished or early stopping reached.", flush=True)
        return
    for epoch in range(start, args.epochs):
        tick = time.perf_counter()
        seed_all(args.seed + epoch)
        for i, generator in enumerate(generators.values()):
            generator.manual_seed(args.seed + epoch * 2 + i)
        torch.cuda.reset_peak_memory_stats()
        results = {}
        for split in ("train", "val"):
            training = split == "train"
            model.train(training)
            cm = torch.zeros(5, 5, dtype=torch.int64, device="cuda")
            total_loss, tiles = 0.0, 0
            for step, (features, labels) in enumerate(loaders[split], 1):
                features, labels = features.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                if not torch.isfinite(features).all() or not ((labels >= 0) & (labels < 5)).all():
                    raise ValueError(f"Invalid input in {split}, batch {step}.")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    logits = model(features)
                    ce = F.cross_entropy(logits, labels, reduction="none")
                    focal = (alpha[labels] * (1 - torch.exp(-ce)).pow(2) * ce).mean()
                    loss = focal + 0.2 * dice_fn(logits, labels)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss.")
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        scheduler.step()
                total_loss += loss.item() * features.shape[0]
                tiles += features.shape[0]
                predictions = logits.detach().argmax(1)
                cm += torch.bincount((labels * 5 + predictions).reshape(-1), minlength=25).reshape(5, 5)
                if step % 50 == 0 or step == len(loaders[split]):
                    print(f"epoch {epoch+1}/{args.epochs} {split} {step}/{len(loaders[split])} loss={total_loss/tiles:.6f}", flush=True)
            results[split] = dict(loss=total_loss / tiles, **metrics(cm.cpu()))
        val_loss = results["val"]["loss"]
        improved = val_loss < best
        best, bad_epochs = (val_loss, 0) if improved else (best, bad_epochs + 1)
        row = dict(epoch=epoch+1, train_loss=results["train"]["loss"], val_loss=val_loss,
                   train_accuracy=results["train"]["accuracy"], val_accuracy=results["val"]["accuracy"],
                   val_mean_iou=results["val"]["mean_iou"], val_macro_f1=results["val"]["macro_f1"],
                   lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter()-tick,
                   peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        state = dict(epoch=epoch+1, config=config, metadata=metadata,
                     model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(), best_val_loss=best, bad_epochs=bad_epochs)
        if improved:
            atomic_save(state, output / "best_checkpoint.pth")
        atomic_save(state, output / "last_checkpoint.pth")
        with (output / "history.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        (output / f"metrics_epoch_{epoch+1:03d}.json").write_text(json.dumps(
            dict(epoch=epoch+1, classes=CLASS_NAMES, results=results, **{"summary": row}), indent=2))
        print(f"EPOCH {epoch+1}: train={row['train_loss']:.6f} val={val_loss:.6f} "
              f"val mIoU={row['val_mean_iou']:.4f} time={row['seconds']:.1f}s "
              f"GPU={row['peak_gpu_gib']:.2f}GiB best={best:.6f}", flush=True)
        if bad_epochs >= args.patience:
            print("EARLY STOPPING", flush=True)
            break
        if args.stop_after and epoch+1-start >= args.stop_after:
            print("PLANNED PAUSE: resume from last_checkpoint.pth with the same configuration.", flush=True)
            break
    print("TRAINING/VALIDATION FINISHED; test evaluation remains separate.", flush=True)


if __name__ == "__main__":
    main()
