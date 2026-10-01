from pathlib import Path
import yaml


CONFIG_PATH = "configs/waedenswil.yaml"


with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


# --------------------------------------------------
# AOI
# --------------------------------------------------

xmin = cfg["aoi"]["xmin"]
xmax = cfg["aoi"]["xmax"]
ymin = cfg["aoi"]["ymin"]
ymax = cfg["aoi"]["ymax"]

tile_size = cfg["tiling"]["tile_size_m"]

width = xmax - xmin
height = ymax - ymin

tiles_x = width // tile_size
tiles_y = height // tile_size

print("Projekt:", cfg["project"]["name"])
print()

print("AOI:")
print(f"  Breite: {width} m")
print(f"  Höhe:   {height} m")
print(f"  Fläche: {width * height / 1_000_000:.2f} km²")
print()

print("25-m-Tiles:")
print(f"  X: {tiles_x}")
print(f"  Y: {tiles_y}")
print(f"  Total: {tiles_x * tiles_y}")
print()


# --------------------------------------------------
# LiDAR-Dateien
# --------------------------------------------------

lidar_dir = Path(cfg["lidar"]["directory"])

print("LiDAR:")

all_lidar_ok = True

for filename in cfg["lidar"]["files"]:
    path = lidar_dir / filename

    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"  OK  {filename}  ({size_mb:.1f} MB)")
    else:
        print(f"  FEHLT  {filename}")
        all_lidar_ok = False

print()


# --------------------------------------------------
# AV
# --------------------------------------------------

av_path = Path(cfg["av"]["path"])

print("AV:")
print(" ", "OK" if av_path.exists() else "FEHLT", av_path)
print()


# --------------------------------------------------
# Baumkronen
# --------------------------------------------------

tree_path = Path(cfg["tree_canopy"]["path"])

print("Baumkronen:")
print(" ", "OK" if tree_path.exists() else "FEHLT", tree_path)
print()


# --------------------------------------------------
# Plausibilitätschecks
# --------------------------------------------------

assert width > 0
assert height > 0

assert width % tile_size == 0, \
    "AOI-Breite ist nicht durch Tile-Grösse teilbar."

assert height % tile_size == 0, \
    "AOI-Höhe ist nicht durch Tile-Grösse teilbar."

assert all_lidar_ok, \
    "Mindestens eine LiDAR-Datei fehlt."

assert av_path.exists(), \
    "AV-Datei fehlt."

assert tree_path.exists(), \
    "Baumkronen-Datensatz fehlt."


print("CONFIG-CHECK ERFOLGREICH")
