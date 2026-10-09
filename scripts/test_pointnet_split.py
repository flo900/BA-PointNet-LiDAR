"""Technischer GPU-Probelauf mit Training und Validierung; keine Testdaten."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "PointnetPP"))
from lidar_dataloader import LiDARTileDataset
from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.train_batches, args.val_batches) < 1:
        raise ValueError("Batch-Groessen und Anzahlen muessen positiv sein.")
    if not torch.cuda.is_available():
        raise RuntimeError("Keine CUDA-GPU vorhanden. Mit dem SLURM-Job starten.")
    source, output = Path(args.data_dir), Path(args.output_dir)
    with (source / "metadata.json").open() as f:
        metadata = json.load(f)
    if metadata.get("spatial_split", {}).get("variant") != "B":
        raise RuntimeError("Datensatz ist nicht als Variante B ausgewiesen.")
    if (metadata["points_per_tile"], metadata["num_features"], metadata["num_classes"]) != (16384, 7, 5):
        raise RuntimeError("Unerwartetes Eingabeformat.")
    # Verhindert, dass der Dataloader auf andere Datenordner ausweicht.
    for split in ("train", "val"):
        for suffix in ("features", "labels"):
            if not (source / f"{split}_{suffix}.npy").is_file():
                raise FileNotFoundError(source / f"{split}_{suffix}.npy")
    output.mkdir(parents=True, exist_ok=False)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    loaders, selected = {}, {}
    for split, batches in (("train", args.train_batches), ("val", args.val_batches)):
        dataset = LiDARTileDataset(str(source), split=split)
        if len(dataset) != metadata["split_sizes"][split]:
            raise RuntimeError(f"Falsche Tile-Anzahl in {split}.")
        n = batches * args.batch_size
        if n > len(dataset):
            raise ValueError(f"Zu wenige Tiles fuer {split}-Probelauf.")
        indices = rng.choice(len(dataset), size=n, replace=False).tolist()
        selected[split] = indices
        loaders[split] = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                                    shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model = PointNet2AerialSSG(num_classes=5, input_channels=4, dropout=0.6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    dice_fn = DiceLoss()
    # Fuer diesen technischen Test unveraendert aus test_pointnet_trainstep_realdata.py.
    # Diese Gewichte sind noch keine fuer das Volltraining getroffene Auswahl.
    alpha = torch.tensor([2.0, 0.6, 0.2, 0.4, 0.45], device=device)
    def loss_fn(logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        focal = (alpha[labels] * (1 - torch.exp(-ce)).pow(2) * ce).mean()
        return focal + 0.2 * dice_fn(logits, labels)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("Technischer Probelauf; Kennzahlen sind keine Modellbewertung.", flush=True)
    started = time.perf_counter()
    initial_weight = model.fc_layer[-1].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for split in ("train", "val"):
        training = split == "train"
        model.train(training)
        losses = []
        for step, (features, labels) in enumerate(loaders[split], 1):
            if not torch.isfinite(features).all() or not ((labels >= 0) & (labels < 5)).all():
                raise RuntimeError("Ungueltige Eingabewerte.")
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = model(features)
                if logits.shape != (features.shape[0], 5, 16384) or not torch.isfinite(logits).all():
                    raise RuntimeError("Ungueltige Modellausgabe.")
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss ist NaN oder unendlich.")
                if training:
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                            raise RuntimeError(f"Fehlender oder ungueltiger Gradient: {name}")
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError("Ungueltige Gewichte nach Optimizer-Schritt.")
            value = loss.item()
            losses.append(value)
            print(f"{split} {step}/{len(loaders[split])}: Loss={value:.6f}", flush=True)
        results[split] = dict(batches=len(losses), mean_loss=float(np.mean(losses)))
        if training and torch.equal(initial_weight, model.fc_layer[-1].weight.detach()):
            raise RuntimeError("Gewichte haben sich beim Training nicht veraendert.")

    checkpoint = output / "smoke_checkpoint.pth"
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    smoke_test_only=True, data_dir=str(source)), checkpoint)
    # Nur die gerade selbst erzeugte Datei laden; CPU schont GPU-Speicher.
    saved = torch.load(checkpoint, map_location="cpu")
    for name, value in model.state_dict().items():
        if not torch.equal(saved["model_state_dict"][name], value.detach().cpu()):
            raise RuntimeError("Gespeicherte Gewichte stimmen nicht ueberein.")
    report = dict(smoke_test_only=True, seed=seed, selected_indices=selected,
                  results=results, data_dir=str(source), gpu=torch.cuda.get_device_name(0),
                  elapsed_seconds=time.perf_counter()-started,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                  test_data_used=False, loss="focal + 0.2 * dice",
                  class_weights=alpha.tolist(), batch_size=args.batch_size)
    with (output / "smoke_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"GPU-Speicher (max. Tensorbelegung): {report['peak_gpu_gib']:.2f} GiB", flush=True)
    print("Checkpoint gespeichert und erfolgreich zurueckgelesen.", flush=True)
    print("TRAINING UND VALIDIERUNG: PROBELAUF ERFOLGREICH", flush=True)


if __name__ == "__main__":
    main()
