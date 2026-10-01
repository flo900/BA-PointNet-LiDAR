#!/bin/bash
#SBATCH --job-name=export_qgis
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/export_qgis_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/export_qgis_%j.err

set -e

source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python -u - <<'PY'
import ast
import csv
from pathlib import Path
import numpy as np
import geopandas as gpd

root = Path("/cfs/earth/scratch/troxlflo/BA/data/intermediate/waedenswil_full")
output = Path("/cfs/earth/scratch/troxlflo/BA/data/qa")
output.mkdir(parents=True, exist_ok=True)

script = Path("/net/home/troxlflo/BA/Projektarbeit-2/scripts/preprocess_lidar_water_nan.py")
label_map = None
for node in ast.parse(script.read_text()).body:
    if isinstance(node, ast.Assign):
        if any(isinstance(t, ast.Name) and t.id == "LABEL_MAP" for t in node.targets):
            label_map = ast.literal_eval(node.value)
if label_map is None:
    raise RuntimeError("LABEL_MAP nicht gefunden.")
names = {value: name for name, value in label_map.items()}

# Chunk auswählen, in dem alle vier Landklassen gut vertreten sind.
candidates = []
for summary in sorted(root.glob("tile_*/chunk_summary.csv")):
    with summary.open(newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["labeled_points"])
            if n == 0:
                continue
            counts = [
                int(row[field]) for field in
                ["tree_canopy", "low_vegetation", "impervious", "buildings"]
            ]
            if min(counts) > 0:
                candidates.append((min(counts) / n, summary.parent, row))

if not candidates:
    raise RuntimeError("Kein Chunk mit allen vier Landklassen gefunden.")

_, folder, row = max(candidates, key=lambda item: item[0])
chunk_id = int(row["chunk_id"])
source = folder / f"chunk_{chunk_id:03d}.npz"
target = output / f"qgis_check_{folder.name}_chunk_{chunk_id:03d}.gpkg"

if target.exists():
    raise RuntimeError(f"Export existiert bereits: {target}")

with np.load(source, allow_pickle=False) as data:
    labels = data["labels"]
    gdf = gpd.GeoDataFrame(
        {
            "label_id": labels,
            "class_name": [names[int(value)] for value in labels],
            "height_agl_m": data["height_agl_m"],
        },
        geometry=gpd.points_from_xy(data["x"], data["y"]),
        crs="EPSG:2056",
    )

gdf.to_file(target, layer="labeled_points", driver="GPKG", index=False)
print("Quelle:", source)
print("Punkte:", len(gdf))
print(gdf["class_name"].value_counts().to_string())
print("\nEXPORT ERFOLGREICH")
print("Datei:", target)
PY
