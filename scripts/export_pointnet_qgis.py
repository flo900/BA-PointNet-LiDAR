"""Export validation predictions to an EPSG:2056 point CSV for QGIS."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "PointnetPP"))
from lidar_dataloader import LiDARTileDataset
from pointnet2_aerial_optimized import PointNet2AerialSSG

NAMES = ["water", "tree_canopy", "low_vegetation", "impervious", "buildings"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise ValueError("Positive batch size and SLURM CUDA GPU required.")
    source, output = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    meta = json.loads((source / "metadata.json").read_text())
    if meta.get("spatial_split", {}).get("variant") != "B" or meta.get("crs") != "EPSG:2056":
        raise ValueError("Expected split B in EPSG:2056.")
    for name in ("val_features.npy", "val_labels.npy", "val_tiles.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    with (source / "val_tiles.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(r["array_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Tile CSV is not in validation array order.")
    # Only load a trusted checkpoint from our own training run.
    saved = torch.load(args.checkpoint, map_location="cpu")
    if saved["metadata"] != meta or Path(saved["config"]["data_dir"]).resolve() != source:
        raise ValueError("Checkpoint/data mismatch.")
    torch.set_num_threads(4)
    dataset = LiDARTileDataset(str(source), split="val")
    n, target = len(dataset), int(meta["points_per_tile"])
    if n != len(rows) or dataset.features.shape != (n, target, 7) or dataset.labels.shape != (n, target):
        raise ValueError("Unexpected validation shapes.")
    model = PointNet2AerialSSG(num_classes=5, input_channels=4, use_xyz=True,
                             dropout=saved["config"]["dropout"])
    model.load_state_dict(saved["model_state_dict"])
    model.cuda().eval()
    output.mkdir(parents=True, exist_ok=False)
    predictions = np.lib.format.open_memmap(output / "val_predictions.npy", mode="w+",
                                            dtype=np.uint8, shape=(n, target))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    offset = 0
    with torch.no_grad():
        for features, labels in loader:
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite validation features.")
            logits = model(features.cuda(non_blocking=True))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite predictions.")
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            predictions[offset:offset+len(pred)] = pred
            offset += len(pred)
            print(f"Validation predictions: {offset}/{n} tiles", flush=True)
    predictions.flush()
    if offset != n:
        raise ValueError("Incomplete prediction export.")
    cm = np.zeros((5, 5), dtype=np.int64)
    scores = []
    for i in range(n):
        truth, pred = dataset.labels[i], predictions[i]
        cm += np.bincount(truth * 5 + pred, minlength=25).reshape(5, 5)
        relevant = np.isin(truth, [1, 3]) | np.isin(pred, [1, 3])
        errors = (truth != pred) & relevant
        scores.append(dict(array_index=i, relevant_points=int(relevant.sum()),
                           relevant_errors=int(errors.sum()),
                           error_fraction=float(errors.sum() / max(1, relevant.sum()))))
    # Prefer substantial errors over a high percentage on just a few points.
    ranked = sorted((s for s in scores if s["relevant_errors"] > 0),
                    key=lambda s: (-s["relevant_errors"], s["array_index"]))
    selected = [(s["array_index"], "many_tree_or_impervious_errors") for s in ranked[:3]]
    remaining = sorted(set(range(n)) - {i for i, _ in selected})
    rng = np.random.default_rng(42)
    selected += [(int(i), "random_validation_tile") for i in
                 rng.choice(remaining, size=min(3, len(remaining)), replace=False)]
    fields = ["tile_index", "sample_point_index", "selection", "x_lv95", "y_lv95",
              "height_agl_m", "reference_id", "reference_class", "predicted_id",
              "predicted_class", "correct", "confusion"]
    with (output / "validation_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, selection in selected:
            features, truth, pred = dataset.features[i], dataset.labels[i], predictions[i]
            origin_x, origin_y = float(rows[i]["xmin"]), float(rows[i]["ymin"])
            size = float(meta["tile_size_m"])
            if not ((features[:, :2] >= 0).all() and (features[:, :2] < size).all()):
                raise ValueError("Expected local XY within tile; aborting coordinate reconstruction.")
            for j in range(target):
                ref, guess = int(truth[j]), int(pred[j])
                writer.writerow(dict(tile_index=i, sample_point_index=j, selection=selection,
                    x_lv95=origin_x+float(features[j, 0]), y_lv95=origin_y+float(features[j, 1]),
                    height_agl_m=float(features[j, 2]), reference_id=ref, reference_class=NAMES[ref],
                    predicted_id=guess, predicted_class=NAMES[guess], correct=int(ref == guess),
                    confusion="correct" if ref == guess else f"{NAMES[ref]} -> {NAMES[guess]}"))
    with (output / "selected_tiles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])+["selection"])
        writer.writeheader()
        writer.writerows(dict(rows[i], selection=selection) for i, selection in selected)
    (output / "export_metadata.json").write_text(json.dumps(dict(
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch=saved["epoch"],
        data_dir=str(source), split="val", crs="EPSG:2056", classes=NAMES,
        selected_tiles=selected, confusion_matrix=cm.tolist(),
        confusion_matrix_order="rows=reference, columns=prediction",
        sampled_entries_include_duplicates=True, test_data_used=False,
        selection_note="Three tiles with most errors involving tree/impervious plus three random tiles; not an unbiased accuracy sample.",
        complete=True), indent=2))
    print("QGIS EXPORT SUCCESSFUL", flush=True)
    print("Checkpoint epoch:", saved["epoch"], flush=True)
    print("Point CSV:", output / "validation_points.csv", flush=True)
    print("Load XY x_lv95/y_lv95 with EPSG:2056; height is above ground, not absolute Z.", flush=True)


if __name__ == "__main__":
    main()
