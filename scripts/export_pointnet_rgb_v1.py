"""Append orthophoto RGB to an unchanged, existing PointNet split.

RGB is sampled from the containing pixel (no interpolation), divided by 255.
No visibility mask is applied. Existing output directories are never reused.
"""
import argparse
import copy
import csv
import hashlib
import json
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window


FEATURES = ["x_local_m", "y_local_m", "height_agl_m", "intensity",
            "return_num", "num_returns", "scan_angle"]
SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)



def sample_rgb(features, row, get_raster):
    xy = features[:, :2].astype(np.float64)
    require(np.isfinite(features).all(), "Nicht-endliche Eingabefeatures")
    require(((xy >= 0) & (xy < 25)).all(), "Lokale XY ausserhalb [0, 25)")
    x = xy[:, 0] + float(row["xmin"])
    y = xy[:, 1] + float(row["ymin"])

    # Raster: linker Rand inklusive, unterer Rand exklusive.
    tile_x = np.floor(x / 1000).astype(np.int64)
    tile_y = (np.ceil(y / 1000) - 1).astype(np.int64)
    keys = np.column_stack((tile_x, tile_y))
    rgb = np.empty((len(x), 3), dtype=np.float32)

    for tx, ty in np.unique(keys, axis=0):
        take = (tile_x == tx) & (tile_y == ty)
        raster = get_raster(f"{tx}-{ty}")
        rgb[take] = read_pixels(x[take], y[take], raster)
    return rgb


def read_pixels(x, y, raster):
    rr, cc = rasterio.transform.rowcol(raster.transform, x, y)
    rr, cc = np.asarray(rr), np.asarray(cc)
    require(((rr >= 0) & (rr < raster.height) &
             (cc >= 0) & (cc < raster.width)).all(),
            "Punkte ausserhalb Orthofoto")
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    image = raster.read(
        (1, 2, 3),
        window=Window(c0, r0, c1-c0, r1-r0),
        masked=True,
    )
    selected = image[:, rr-r0, cc-c0].T
    require(not np.ma.getmaskarray(selected).any(), "Ungueltige Bildpixel")
    return np.asarray(selected, dtype=np.float32) / np.float32(255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="Headers aller Splits und erste Kachel jedes Splits pruefen; nichts schreiben")
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            "Quelle und Ausgabe muessen getrennte Verzeichnisse sein")
    require(not output.exists(), f"Ausgabe existiert bereits: {output}")
    meta_path = source / "metadata.json"
    meta = json.loads(meta_path.read_text())
    require(meta.get("complete") is True, "Quelldatensatz nicht vollstaendig")
    require(meta["features"] == FEATURES and meta["num_features"] == 7,
            "Unerwartete Featuredefinition")
    require(meta["tile_size_m"] == 25 and meta["points_per_tile"] == 16384,
            "Unerwartete Tile- oder Punktgroesse")
    require(meta["crs"] == "EPSG:2056", "Unerwartetes CRS")
    for split in SPLITS:
        for name in (f"{split}_features.npy", f"{split}_labels.npy", f"{split}_tiles.csv"):
            require((source / name).is_file(), f"Fehlende Quelldatei: {name}")

    result = copy.deepcopy(meta)
    result.update(num_features=10, features=FEATURES + ["red", "green", "blue"],
                  complete=False, source_dir=str(source), source_metadata=meta,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  rgb={"version": "rgb_v1", "image_dir": str(args.image_dir.resolve()),
                       "filename_year": 2025, "lidar_year": 2024,
                       "sampling": "containing_pixel_no_interpolation",
                       "boundary_policy": "left <= x < right; bottom < y <= top; use adjacent image at seams",
                       "normalization": "uint8 / 255 -> float32 [0,1]",
                       "coordinates": "float64(x_local_m) + xmin; float64(y_local_m) + ymin",
                       "visibility_mask": False,
                       "limitations": ["Occluded points may receive canopy/roof colours",
                                       "LiDAR and imagery have different acquisition years"],
                       "images": {}, "source_metadata_sha256": sha256(meta_path)},
                  verification={})
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=False)
        (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
        shutil.copyfile(Path(__file__), output / "export_source.py")

    with ExitStack() as stack:
        rasters = {}

        def get_raster(tile):
            if tile not in rasters:
                path = args.image_dir / f"swissimage-dop10_2025_{tile}_0.1_2056.tif"
                require(path.is_file(), f"Benoetigtes Orthofoto fehlt: {path}")
                raster = stack.enter_context(rasterio.open(path))
                require(raster.crs == rasterio.crs.CRS.from_epsg(2056),
                        "Bild-CRS falsch")
                require(raster.count == 3 and raster.dtypes == ("uint8",)*3,
                        "Bild muss drei uint8-Kanaele enthalten")
                require(raster.colorinterp == (
                    ColorInterp.red, ColorInterp.green, ColorInterp.blue),
                    "RGB-Kanalreihenfolge falsch")
                tx, ty = map(int, tile.split("-"))
                require(
                    raster.width == 10000 and raster.height == 10000 and
                    tuple(raster.transform)[:6] ==
                    (0.1, 0.0, tx*1000.0, 0.0, -0.1, (ty+1)*1000.0),
                    "Unerwartetes 1-km-Rastergitter",
                )
                rasters[tile] = raster
                print(f"Orthofoto geoeffnet: {path.name}", flush=True)
                if not args.check_only:
                    result["rgb"]["images"][tile] = {
                        "path": str(path.resolve()),
                        "sha256": sha256(path),
                    }
            return rasters[tile]

        for split in SPLITS:
            with (source / f"{split}_tiles.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            n = meta["split_sizes"][split]
            require(len(rows) == n, f"{split}: CSV-Laenge stimmt nicht")
            require([int(r["array_index"]) for r in rows] == list(range(n)),
                    f"{split}: array_index nicht in Array-Reihenfolge")
            require(all(r["split"] == split for r in rows), "Falsche Split-Zuordnung")
            features = np.load(source / f"{split}_features.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(source / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
            require(features.shape == (n, 16384, 7) and features.dtype == np.float32,
                    f"{split}: Unerwartetes Featurearray")
            require(labels.shape == (n, 16384) and labels.dtype == np.int64,
                    f"{split}: Unerwartetes Labelarray")
            if not args.check_only:
                target = np.lib.format.open_memmap(output / f"{split}_features.npy",
                                                  mode="w+", dtype=np.float32,
                                                  shape=(n, 16384, 10))
            counts = np.zeros(5, dtype=np.int64)
            for i, row in enumerate(rows[:1] if args.check_only else rows):
                rgb = sample_rgb(features[i], row, get_raster)
                require(np.isin(labels[i], np.arange(5)).all(), "Ungueltige Labels")
                counts += np.bincount(labels[i], minlength=5)
                if args.check_only:
                    print(f"{split}: erste Kachel OK; RGB-Bereich {rgb.min():.4f}–{rgb.max():.4f}", flush=True)
                else:
                    target[i, :, :7] = features[i]
                    target[i, :, 7:] = rgb
                    if (i+1) % 250 == 0 or i+1 == n:
                        print(f"{split}: {i+1}/{n} Tiles ergaenzt", flush=True)
            if args.check_only:
                continue
            target.flush()
            # Read saved output, not just the in-memory RGB buffer.
            saved = np.load(output / f"{split}_features.npy", mmap_mode="r", allow_pickle=False)
            for i in range(n):
                require(saved[i, :, :7].tobytes() == features[i].tobytes(),
                        f"{split}/{i}: Originalfeatures veraendert")
                require(np.isfinite(saved[i, :, 7:]).all() and
                        ((saved[i, :, 7:] >= 0) & (saved[i, :, 7:] <= 1)).all(),
                        "Ungueltige gespeicherte RGB-Werte")
            require(counts.tolist() == meta["class_counts"][split], "Klassenhaeufigkeiten veraendert")
            checks = {"original_features_bit_identical": True, "class_counts_identical": True}
            for name in (f"{split}_labels.npy", f"{split}_tiles.csv"):
                shutil.copyfile(source / name, output / name)
                digest = sha256(source / name)
                require(sha256(output / name) == digest, f"Kopie weicht ab: {name}")
                checks[name + "_sha256"] = digest
            result["verification"][split] = checks
            del target, saved
            print(f"{split}: Originalfeatures, Labels und Kacheltabelle unveraendert", flush=True)

    if args.check_only:
        print("PROBELAUF ERFOLGREICH: erste Kachel je Split; keine Dateien geschrieben.")
        return
    if (source / "split_tiles.csv").is_file():
        shutil.copyfile(source / "split_tiles.csv", output / "split_tiles.csv")
        require(sha256(source / "split_tiles.csv") == sha256(output / "split_tiles.csv"),
                "split_tiles.csv weicht ab")
    result["complete"] = True
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"RGB-EXPORT ERFOLGREICH: {output}")
    print("10 Features; keine Sichtbarkeitsmaske. Noch kein RGB-Training ausgefuehrt.")


if __name__ == "__main__":
    main()
