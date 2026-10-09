"""Sampling-Zwischenexport als train_*.npy speichern; Tiles mit NaN-Hoehen ausschliessen."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-tiles", type=int)
    args = parser.parse_args()
    source, output = Path(args.sample_dir), Path(args.output_dir)
    with (source / "metadata.json").open() as f:
        sampling = json.load(f)
    if sampling.get("stage") != "sampled_intermediate" or not sampling.get("complete"):
        raise RuntimeError("Sampling-Zwischenexport nicht abgeschlossen.")
    with open(args.inventory, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "sampled"]
    filenames = [r["sample_file"] for r in rows]
    if len(rows) != sampling["tiles"] or len(set(filenames)) != len(rows):
        raise RuntimeError("Inventar unvollstaendig oder doppelte Dateinamen.")
    if any(Path(name).name != name or not name.endswith(".npz") for name in filenames):
        raise RuntimeError("Ungueltiger Dateiname im Inventar.")
    if set(filenames) != {p.name for p in source.glob("tile_*.npz")}:
        raise RuntimeError("Dateien stimmen nicht mit dem Inventar ueberein.")
    kept = [r for r in rows if int(r["sampled_missing_height"]) == 0]
    count = len(kept)
    if not count or (args.expected_tiles is not None and count != args.expected_tiles):
        raise RuntimeError(f"Unerwartete Anzahl verwendbarer Tiles: {count}")
    target = int(sampling["points_per_tile"])
    feature_names = sampling["features"]
    class_names = sampling["class_names"]
    if len(feature_names) != 7 or feature_names[2] != "height_agl_m":
        raise RuntimeError("Unerwartete Feature-Reihenfolge.")
    staging = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"Ausgabe existiert bereits: {output}")
    staging.mkdir(parents=True, exist_ok=False)
    features = np.lib.format.open_memmap(staging / "train_features.npy", mode="w+",
                                        dtype=np.float32, shape=(count, target, 7))
    labels = np.lib.format.open_memmap(staging / "train_labels.npy", mode="w+",
                                      dtype=np.int64, shape=(count, target))
    class_counts = np.zeros(len(class_names), dtype=np.int64)
    selected, excluded = [], []
    position = 0
    for number, row in enumerate(rows, 1):
        with np.load(source / row["sample_file"], allow_pickle=False) as data:
            tile_features, tile_labels = data["features"], data["labels"]
            if tile_features.shape != (target, 7) or tile_labels.shape != (target,):
                raise RuntimeError(f"Falsche Array-Form: {row['sample_file']}")
            if tile_features.dtype != np.float32 or tile_labels.dtype != np.int64:
                raise RuntimeError(f"Unerwarteter Datentyp: {row['sample_file']}")
            if not np.isin(tile_labels, range(len(class_names))).all():
                raise RuntimeError(f"Ungueltige Labels: {row['sample_file']}")
            if not np.array_equal(data["origin_xy"], [int(row["xmin"]), int(row["ymin"])]):
                raise RuntimeError("Tile-Ursprung stimmt nicht mit Inventar ueberein.")
            missing = int(np.isnan(tile_features[:, 2]).sum())
            if missing != int(row["sampled_missing_height"]):
                raise RuntimeError("NaN-Anzahl stimmt nicht mit Inventar ueberein.")
            if np.isinf(tile_features).any() or not np.isfinite(tile_features[:, [0, 1, 3, 4, 5, 6]]).all():
                raise RuntimeError("Ungueltige Werte ausserhalb der erlaubten NaN-Hoehen.")
            histogram = np.bincount(tile_labels, minlength=len(class_names))
            if any(int(histogram[j]) != int(row[f"sampled_{name}"]) for j, name in enumerate(class_names)):
                raise RuntimeError("Klassenhaeufigkeiten stimmen nicht mit Inventar ueberein.")
            if missing:
                excluded.append(dict(row, exclusion_reason="missing_height"))
            else:
                features[position] = tile_features
                labels[position] = tile_labels
                class_counts += histogram
                selected.append(dict(array_index=position, **row))
                position += 1
        if number % 500 == 0 or number == len(rows):
            print(f"{number}/{len(rows)} Dateien geprueft; {position} Training-Tiles geschrieben", flush=True)
    if position != count:
        raise RuntimeError("Exportierte Tile-Anzahl stimmt nicht.")
    features.flush()
    labels.flush()
    del features, labels
    for filename, entries in [("train_tiles.csv", selected), ("excluded_tiles.csv", excluded)]:
        fields = list(entries[0]) if entries else list(rows[0]) + ["exclusion_reason"]
        with (staging / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
    metadata = dict(points_per_tile=target, num_features=7, num_classes=len(class_names),
                    features=feature_names, classes={str(i): name for i, name in enumerate(class_names)},
                    tile_size_m=sampling["tile_size_m"], split_sizes={"train": count},
                    available_splits=["train"], pending_splits=["val", "test"],
                    complete=True, height_policy="exclude_tiles_with_missing_height",
                    excluded_tiles=len(excluded), train_class_counts=class_counts.tolist(),
                    source_sample_dir=str(source.resolve()),
                    source_inventory=str(Path(args.inventory).resolve()), sampling_metadata=sampling)
    with (staging / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    # Erst nach erfolgreichem Export unter dem endgueltigen Namen bereitstellen.
    if output.exists():
        raise FileExistsError(f"Ausgabe inzwischen vorhanden: {output}")
    staging.rename(output)
    print("\nTRAININGSEXPORT ERFOLGREICH")
    print(f"Tiles: {count}; wegen fehlender Hoehen ausgeschlossen: {len(excluded)}")
    print(f"Features: ({count}, {target}, 7), float32; Labels: ({count}, {target}), int64")
    for name, value in zip(class_names, class_counts):
        print(f"  {name}: {value:,}")
    print("Ausgabe:", output)
    print("Validierungs- und Testdaten werden separat vorbereitet.")


if __name__ == "__main__":
    main()
