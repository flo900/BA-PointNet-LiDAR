"""Compare frozen baseline/RGB models on the SAME previously selected validation tiles."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "PointnetPP"))
from pointnet2_aerial_optimized import PointNet2AerialSSG

NAMES = ["water", "tree_canopy", "low_vegetation", "impervious", "buildings"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def predict(checkpoint, metadata, data_dir, features, channels, epoch):
    # Trusted local checkpoints only. No optimiser, backward pass or training.
    saved = torch.load(checkpoint, map_location="cpu")
    require(saved["epoch"] == epoch and saved["config"]["batch_size"] == 24,
            f"Unexpected checkpoint epoch/batch: {checkpoint}")
    require(saved["metadata"] == metadata and
            Path(saved["config"]["data_dir"]).resolve() == data_dir.resolve(),
            "Checkpoint/data mismatch")
    if channels == 7:
        require(saved["config"].get("experiment") == "rgb_v1", "Not RGB-v1 checkpoint")
        hashes = json.loads((checkpoint.parent / "source_hashes.json").read_text())
        require(digest(REPO / "src/PointnetPP/pointnet2_aerial_optimized.py") ==
                hashes["pointnet2_aerial_optimized_source.py"], "RGB model source changed")
    model = PointNet2AerialSSG(num_classes=5, input_channels=channels,
                             use_xyz=True, dropout=saved["config"]["dropout"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.cuda().eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), 2):
            points = torch.from_numpy(features[start:start+2].copy()).cuda()
            logits = model(points)
            require(tuple(logits.shape) == (len(points), 5, 16384) and
                    bool(torch.isfinite(logits).all()), "Invalid model output")
            chunks.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-data", required=True, type=Path)
    parser.add_argument("--rgb-data", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--rgb-checkpoint", required=True, type=Path)
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(torch.cuda.is_available(), "Submit a GPU SLURM job")
    torch.set_num_threads(4)
    require(not args.output_dir.exists(), "Output already exists")
    for protected in (args.baseline_data, args.rgb_data,
                      args.baseline_checkpoint.parent, args.rgb_checkpoint.parent):
        p, out = protected.resolve(), args.output_dir.resolve()
        require(p != out and p not in out.parents and out not in p.parents,
                "Output must be separate from data and training directories")
    selected = read_rows(args.selection_csv)
    indices = [int(row["array_index"]) for row in selected]
    require(len(indices) > 0 and len(set(indices)) == len(indices), "Invalid tile selection")
    rows = read_rows(args.baseline_data / "val_tiles.csv")
    rgb_rows = read_rows(args.rgb_data / "val_tiles.csv")
    require(rows == rgb_rows, "Baseline/RGB validation tile tables differ")
    require([int(r["array_index"]) for r in rows] == list(range(len(rows))),
            "Validation CSV not in array order")
    for row, i in zip(selected, indices):
        require(0 <= i < len(rows), "Selected index outside validation data")
        require(row["xmin"] == rows[i]["xmin"] and row["ymin"] == rows[i]["ymin"]
                and row.get("split", "val") == "val", "Old selection has different coordinates/split")
    metas = [json.loads((p / "metadata.json").read_text())
             for p in (args.baseline_data, args.rgb_data)]
    for meta, size in zip(metas, (7, 10)):
        require(meta.get("complete") is True and meta["num_features"] == size and
                meta["points_per_tile"] == 16384 and meta["crs"] == "EPSG:2056" and
                meta.get("spatial_split", {}).get("variant") == "B" and
                [meta["classes"][str(i)] for i in range(5)] == NAMES,
                "Unexpected metadata")
    arrays, truths = [], []
    for directory, size in zip((args.baseline_data, args.rgb_data), (7, 10)):
        features = np.load(directory / "val_features.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / "val_labels.npy", mmap_mode="r", allow_pickle=False)
        require(features.shape == (len(rows), 16384, size) and features.dtype == np.float32,
                "Unexpected feature shape/dtype")
        require(labels.shape == (len(rows), 16384) and labels.dtype == np.int64,
                "Unexpected label shape/dtype")
        arrays.append(features[indices])
        truths.append(labels[indices])
    require(np.array_equal(arrays[0], arrays[1][:, :, :7]) and
            np.array_equal(truths[0], truths[1]), "Selected points/reference labels differ")
    require(all(np.isfinite(a).all() for a in arrays) and
            np.isin(truths[0], range(5)).all(), "Invalid input")
    require(((arrays[0][:, :, :2] >= 0) & (arrays[0][:, :, :2] < 25)).all(),
            "Expected tile-local XY")
    require(((arrays[1][:, :, 7:] >= 0) & (arrays[1][:, :, 7:] <= 1)).all(),
            "Expected normalised RGB")
    predictions = [predict(cp, meta, directory, features, channels, epoch)
                   for cp, meta, directory, features, channels, epoch in zip(
                       (args.baseline_checkpoint, args.rgb_checkpoint), metas,
                       (args.baseline_data, args.rgb_data), arrays, (4, 7), (43, 48))]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    fields = ["tile_index", "sample_point_index", "selection", "x_lv95", "y_lv95",
              "height_agl_m", "reference_id", "reference_class", "baseline_predicted_id",
              "baseline_predicted_class", "rgb_predicted_id", "rgb_predicted_class",
              "baseline_correct", "rgb_correct", "prediction_changed", "comparison",
              "red", "green", "blue"]
    layer_fields = ["tile_index", "sample_point_index", "x_lv95", "y_lv95",
                    "reference_class", "predicted_class", "correct", "confusion"]
    counts = {name: 0 for name in ("both_match_reference", "rgb_only_matches_reference",
                                   "baseline_only_matches_reference", "both_differ_reference")}
    with (output / "comparison_points.csv").open("w", newline="") as stream, \
         (output / "baseline_points.csv").open("w", newline="") as base_stream, \
         (output / "rgb_points.csv").open("w", newline="") as rgb_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        layers = [csv.DictWriter(f, fieldnames=layer_fields) for f in (base_stream, rgb_stream)]
        for w in [writer] + layers:
            w.writeheader()
        for k, i in enumerate(indices):
            for j in range(16384):
                ref = int(truths[0][k, j])
                a, b = (int(p[k, j]) for p in predictions)
                key = ("both_match_reference" if a == ref and b == ref else
                       "rgb_only_matches_reference" if b == ref else
                       "baseline_only_matches_reference" if a == ref else "both_differ_reference")
                counts[key] += 1
                x = float(rows[i]["xmin"]) + float(arrays[0][k, j, 0])
                y = float(rows[i]["ymin"]) + float(arrays[0][k, j, 1])
                colour = np.rint(arrays[1][k, j, 7:] * 255).astype(int)
                writer.writerow(dict(tile_index=i, sample_point_index=j,
                    selection=selected[k].get("selection", "previous_validation_selection"),
                    x_lv95=x, y_lv95=y, height_agl_m=float(arrays[0][k, j, 2]),
                    reference_id=ref, reference_class=NAMES[ref], baseline_predicted_id=a,
                    baseline_predicted_class=NAMES[a], rgb_predicted_id=b, rgb_predicted_class=NAMES[b],
                    baseline_correct=int(a == ref), rgb_correct=int(b == ref),
                    prediction_changed=int(a != b), comparison=key,
                    red=int(colour[0]), green=int(colour[1]), blue=int(colour[2])))
                for w, pred in zip(layers, (a, b)):
                    w.writerow(dict(tile_index=i, sample_point_index=j, x_lv95=x, y_lv95=y,
                        reference_class=NAMES[ref], predicted_class=NAMES[pred], correct=int(pred == ref),
                        confusion="correct" if pred == ref else f"{NAMES[ref]} -> {NAMES[pred]}"))
    with (output / "selected_tiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (output / "export_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "export_metadata.json").write_text(json.dumps(dict(complete=True,
        split="val", crs="EPSG:2056", selected_indices=indices,
        selection_csv=str(args.selection_csv.resolve()), selection_sha256=digest(args.selection_csv),
        baseline_checkpoint=str(args.baseline_checkpoint.resolve()),
        baseline_checkpoint_sha256=digest(args.baseline_checkpoint), baseline_epoch=43,
        rgb_checkpoint=str(args.rgb_checkpoint.resolve()), rgb_checkpoint_sha256=digest(args.rgb_checkpoint),
        rgb_epoch=48, points=len(indices)*16384, comparison_counts=counts,
        test_data_used=False, repeated_entries_included=True,
        selection_note="Same tiles as prior QGIS export; selected using earlier batch-8 baseline errors and random tiles. Qualitative inspection, not an unbiased accuracy estimate.",
        reference_note="Correct means agreement with generated reference labels, not independently established ground truth."), indent=2))
    print("QGIS COMPARISON EXPORT SUCCESSFUL", flush=True)
    print(f"Tiles: {len(indices)}; sampled point entries: {len(indices)*16384}")
    print("Output:", output)
    print("Load point CSVs as XY x_lv95/y_lv95, EPSG:2056. Height is AGL, not absolute Z.")


if __name__ == "__main__":
    main()
