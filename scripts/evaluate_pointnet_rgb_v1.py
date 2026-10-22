"""RGB-v1 test evaluation: frozen epoch-48 checkpoint on the existing test split B."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "PointnetPP"))
from lidar_dataloader import LiDARTileDataset
from pointnet2_aerial_optimized import PointNet2AerialSSG

NAMES = ["water", "tree_canopy", "low_vegetation", "impervious", "buildings"]


def calculate_metrics(matrix):
    """Rows are reference labels, columns predictions; zero denominators -> zero."""
    cm = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    precision = divide(tp, predicted)
    recall = divide(tp, support)
    f1 = divide(2 * tp, support + predicted)
    iou = divide(tp, support + predicted - tp)
    total = float(cm.sum())
    if total == 0:
        raise ValueError("Empty confusion matrix.")
    return dict(accuracy=float(tp.sum() / total), macro_f1=float(f1.mean()),
                weighted_f1=float(np.sum(f1 * support) / total), mean_iou=float(iou.mean()),
                per_class=[dict(class_id=i, class_name=name, support=int(support[i]),
                                predicted_points=int(predicted[i]), precision=float(precision[i]),
                                recall=float(recall[i]), f1=float(f1[i]), iou=float(iou[i]))
                           for i, name in enumerate(NAMES)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-epoch", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError("A positive batch size and a SLURM CUDA GPU are required.")
    torch.set_num_threads(4)
    source, checkpoint, output = (Path(p).resolve() for p in
                                  (args.data_dir, args.checkpoint, args.output_dir))
    meta = json.loads((source / "metadata.json").read_text())
    if meta.get("spatial_split", {}).get("variant") != "B":
        raise ValueError("Expected spatial split B.")
    if [meta["classes"][str(i)] for i in range(5)] != NAMES:
        raise ValueError("Unexpected class order.")
    if tuple(meta[k] for k in ("points_per_tile", "num_features", "num_classes")) != (16384, 10, 5):
        raise ValueError("Unexpected data format.")
    if meta.get("complete") is not True:
        raise ValueError("Incomplete RGB dataset.")
    expected_features = ["x_local_m", "y_local_m", "height_agl_m", "intensity",
                         "return_num", "num_returns", "scan_angle", "red", "green", "blue"]
    if meta.get("features") != expected_features or meta.get("rgb", {}).get("visibility_mask") is not False:
        raise ValueError("Expected RGB-v1 feature order without a visibility mask.")
    if (output == source or source in output.parents or output in source.parents or
            output == checkpoint.parent or checkpoint.parent in output.parents or
            output in checkpoint.parent.parents):
        raise ValueError("Evaluation output must be separate from data and training directories.")
    if output.exists():
        raise FileExistsError(output)
    if meta["split_sizes"]["test"] != 896:
        raise ValueError("Expected 896 held-out test tiles.")
    for name in ("test_features.npy", "test_labels.npy"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    # Only load our own trusted train_pointnet_rgb_v1.py checkpoint.
    saved = torch.load(checkpoint, map_location="cpu")
    if saved["epoch"] != args.expected_epoch:
        raise ValueError(f"Expected epoch {args.expected_epoch}, found {saved['epoch']}.")
    if saved["config"]["batch_size"] != 24:
        raise ValueError("This evaluation fixes the selected batch-24 RGB model.")
    if saved["config"].get("experiment") != "rgb_v1" or saved["config"].get("model_input_channels") != 7:
        raise ValueError("Checkpoint is not the RGB-v1 model.")
    source_hashes = json.loads((checkpoint.parent / "source_hashes.json").read_text())
    for name in ("lidar_dataloader", "pointnet2_aerial_optimized"):
        content = (REPO / "src" / "PointnetPP" / f"{name}.py").read_bytes()
        if hashlib.sha256(content).hexdigest() != source_hashes[f"{name}_source.py"]:
            raise ValueError(f"Source changed since RGB training: {name}")
    label_digest = hashlib.sha256()
    with (source / "test_labels.npy").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            label_digest.update(block)
    if label_digest.hexdigest() != meta["verification"]["test"]["test_labels.npy_sha256"]:
        raise ValueError("Test labels changed since RGB export.")
    if saved["metadata"] != meta or Path(saved["config"]["data_dir"]).resolve() != source:
        raise ValueError("Checkpoint and dataset metadata differ.")
    dataset = LiDARTileDataset(str(source), split="test", transform=None)
    if dataset.features.shape != (896, 16384, 10) or dataset.labels.shape != (896, 16384):
        raise ValueError("Unexpected test shapes.")
    if dataset.features.dtype != np.float32 or dataset.labels.dtype != np.int64:
        raise ValueError("Unexpected test dtypes.")
    model = PointNet2AerialSSG(num_classes=5, input_channels=7, use_xyz=True,
                             dropout=saved["config"]["dropout"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.cuda().eval()
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation_source.py").write_bytes(Path(__file__).read_bytes())
    digest = hashlib.sha256()
    with checkpoint.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=2, pin_memory=True)
    cm = torch.zeros((5, 5), dtype=torch.int64, device="cuda")
    tile_results, offset = [], 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(f"Selected RGB v1 model: epoch {saved['epoch']}, training batch size 24", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            if not torch.isfinite(features).all() or not ((labels >= 0) & (labels < 5)).all():
                raise ValueError("Invalid test features/labels.")
            logits = model(features)
            if logits.shape != (len(features), 5, 16384) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model output.")
            pred = logits.argmax(dim=1)
            cm += torch.bincount((labels * 5 + pred).reshape(-1), minlength=25).reshape(5, 5)
            correct = (pred == labels).sum(dim=1).cpu().tolist()
            tile_results.extend(dict(array_index=offset+i, points=16384,
                                     correct_points=int(value), accuracy=value/16384)
                                for i, value in enumerate(correct))
            offset += len(features)
            if offset % 80 == 0 or offset == len(dataset):
                print(f"Test: {offset}/{len(dataset)} tiles", flush=True)
    matrix = cm.cpu().numpy()
    if offset != 896 or int(matrix.sum()) != 896 * 16384:
        raise ValueError("Incomplete evaluation.")
    results = calculate_metrics(matrix)
    report = dict(complete=True, split="test", tiles=896, sampled_point_entries=int(matrix.sum()),
                  checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
                  checkpoint_epoch=saved["epoch"], training_config=saved["config"],
                  data_dir=str(source), data_metadata=meta, evaluation_batch_size=args.batch_size,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  elapsed_seconds=time.perf_counter()-started,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                  metric_basis="All sampled entries, including repetitions, against existing generated reference labels.",
                  experiment="rgb_v1",
                  baseline_test_run="/cfs/earth/scratch/troxlflo/BA/runs/test_split_B_1065611",
                  test_labels_sha256=label_digest.hexdigest(),
                  model_selection="RGB epoch 48 selected by lowest validation loss; batch size 24 fixed for baseline comparison.",
                  test_reuse_note="Same spatial test split as the previously evaluated seven-feature baseline; not a new independent test region.",
                  confusion_matrix_order="rows=reference, columns=prediction",
                  confusion_matrix=matrix.tolist(), **results)
    with (output / "per_class.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results["per_class"][0]))
        writer.writeheader()
        writer.writerows(results["per_class"])
    with (output / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reference / prediction"] + NAMES)
        writer.writerows([name] + row.tolist() for name, row in zip(NAMES, matrix))
    with (output / "test_tiles_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tile_results[0]))
        writer.writeheader()
        writer.writerows(tile_results)
    summary = ["TEST EVALUATION: FIVE-CLASS RGB V1",
               f"Checkpoint epoch: {saved['epoch']}", "Test tiles: 896",
               f"Overall Accuracy: {results['accuracy']:.4%}",
               f"Macro F1: {results['macro_f1']:.6f}",
               f"Weighted F1: {results['weighted_f1']:.6f}",
               f"Mean IoU: {results['mean_iou']:.4%}"]
    for r in results["per_class"]:
        summary.append(f"{r['class_name']}: Precision={r['precision']:.4f} Recall={r['recall']:.4f} "
                       f"F1={r['f1']:.4f} IoU={r['iou']:.2%}")
    summary.append(report["metric_basis"])
    summary.append(report["test_reuse_note"])
    (output / "test_summary.txt").write_text("\n".join(summary)+"\n")
    (output / "test_results.json").write_text(json.dumps(report, indent=2))
    print("\n".join(summary), flush=True)
    print("TEST EVALUATION SUCCESSFUL\nOutput:", output, flush=True)


if __name__ == "__main__":
    main()

