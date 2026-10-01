#!/bin/bash
#SBATCH --job-name=check_waedi
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/check_waedi_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/check_waedi_%j.err

set -e

source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python -u - <<'PY'
import ast
import csv
from pathlib import Path
import numpy as np

root = Path("/cfs/earth/scratch/troxlflo/BA/data/intermediate/waedenswil_full")
script = Path("/net/home/troxlflo/BA/Projektarbeit-2/scripts/preprocess_lidar_water_nan.py")

# Klassen-IDs direkt aus dem Preprocessing-Skript lesen.
label_map = None
for node in ast.parse(script.read_text()).body:
    if isinstance(node, ast.Assign):
        if any(isinstance(t, ast.Name) and t.id == "LABEL_MAP" for t in node.targets):
            label_map = ast.literal_eval(node.value)
if label_map is None:
    raise RuntimeError("LABEL_MAP im Skript nicht gefunden.")

classes = {
    "Water": "water",
    "Tree canopy": "tree_canopy",
    "Low vegetation": "low_vegetation",
    "Impervious": "impervious",
    "Buildings": "buildings",
}
fields = [
    "x", "y", "height_agl_m", "intensity",
    "return_num", "num_returns", "scan_angle", "labels",
]

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

summaries = sorted(root.glob("tile_*/chunk_summary.csv"))
check(len(summaries) == 4, "Es werden vier Zusammenfassungen erwartet.")
files_checked = points_checked = missing_total = 0

for summary in summaries:
    with summary.open(newline="") as f:
        rows = list(csv.DictReader(f))

    tile_files = tile_points = tile_missing = 0
    for row in rows:
        path = summary.parent / f"chunk_{int(row['chunk_id']):03d}.npz"
        if int(row["total_points"]) == 0:
            check(not path.exists(), f"{path}: Datei für leeren Chunk vorhanden")
            continue

        with np.load(path, allow_pickle=False) as data:
            check(set(fields) <= set(data.files), f"{path}: Felder fehlen")
            n = int(row["labeled_points"])
            arrays = {field: data[field] for field in fields}

            for field, values in arrays.items():
                check(values.shape == (n,), f"{path}: falsche Form bei {field}")
                if field != "height_agl_m":
                    check(np.isfinite(values).all(), f"{path}: ungültige Werte bei {field}")

            labels = arrays["labels"]
            check(np.issubdtype(labels.dtype, np.integer), f"{path}: Labels nicht ganzzahlig")
            check(np.isin(labels, list(label_map.values())).all(), f"{path}: unbekannte Labels")

            for name, column in classes.items():
                count = int(np.count_nonzero(labels == label_map[name]))
                check(count == int(row[column]), f"{path}: Klassenanzahl für {name} stimmt nicht")

            height = arrays["height_agl_m"]
            check(not np.isinf(height).any(), f"{path}: unendliche Höhen")
            missing = np.isnan(height)
            missing_count = int(missing.sum())
            expected_missing = int(float(row.get("height_agl_missing_labeled_points") or 0))
            check(missing_count == expected_missing, f"{path}: NaN-Anzahl stimmt nicht")
            if missing_count:
                check(
                    missing.all() and (labels == label_map["Water"]).all(),
                    f"{path}: fehlende Höhen ausserhalb der Water-Sonderregel",
                )

        tile_files += 1
        tile_points += n
        tile_missing += missing_count

    files_checked += tile_files
    points_checked += tile_points
    missing_total += tile_missing
    print(
        f"{summary.parent.name}: {tile_files} Dateien OK, "
        f"{tile_points:,} Punkte, {tile_missing:,} fehlende Höhen"
    )

print(f"\nDATENPRÜFUNG ERFOLGREICH")
print(f"Dateien: {files_checked}")
print(f"Gelabelte Punkte: {points_checked:,}")
print(f"Fehlende Höhen: {missing_total:,}")
PY
