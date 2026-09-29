import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# --------------------------------------------------
# Eingabedaten
# --------------------------------------------------

TARGET_PATH = "/cfs/earth/scratch/troxlflo/BA/data/intermediate/target.gpkg"
TREE_PATH = "/cfs/earth/scratch/troxlflo/BA/data/raw/tree_canopy/Waedi_Veg.gdb"

# Ausdehnung der LiDAR-Kachel 2692-1232
XMIN = 2692000
XMAX = 2693000
YMIN = 1232000
YMAX = 1233000

GRID_SIZE = 100  # Meter


# --------------------------------------------------
# Daten laden
# --------------------------------------------------

target = gpd.read_file(TARGET_PATH, layer="target")
trees = gpd.read_file(TREE_PATH, layer="Baumkronen")

# Polygone ohne Zielklasse ignorieren
target = target[target["target_class"].notna()].copy()

# Nur Daten innerhalb unserer 1-km²-LiDAR-Kachel behalten
tile = box(XMIN, YMIN, XMAX, YMAX)

target = target[target.intersects(tile)].copy()
trees = trees[trees.intersects(tile)].copy()

print(f"AV-Polygone in Kachel: {len(target)}")
print(f"Baumkronen in Kachel: {len(trees)}")
print()


# --------------------------------------------------
# 100 x 100 m Felder untersuchen
# --------------------------------------------------

results = []

for x in range(XMIN, XMAX, GRID_SIZE):
    for y in range(YMIN, YMAX, GRID_SIZE):

        cell = box(x, y, x + GRID_SIZE, y + GRID_SIZE)

        av = target[target.intersects(cell)].copy()
        tree = trees[trees.intersects(cell)].copy()

        # Tatsächliche Schnittfläche mit dem 100x100-m-Feld
        class_areas = {}

        for class_name in [
            "Building",
            "Impervious",
            "Low vegetation",
            "Tree canopy",
            "Water",
        ]:
            polygons = av[av["target_class"] == class_name]

            if len(polygons) == 0:
                area = 0
            else:
                area = polygons.geometry.intersection(cell).area.sum()

            class_areas[class_name] = area

        # Klasse gilt als vorhanden, wenn mindestens 10 m² im Feld liegen
        useful_classes = [
            name
            for name, area in class_areas.items()
            if area >= 10
        ]

        results.append({
            "xmin": x,
            "ymin": y,
            "xmax": x + GRID_SIZE,
            "ymax": y + GRID_SIZE,
            "class_count": len(useful_classes),
            "classes": ", ".join(useful_classes),
            "tree_crowns": len(tree),
            "building_m2": round(class_areas["Building"], 1),
            "impervious_m2": round(class_areas["Impervious"], 1),
            "low_vegetation_m2": round(class_areas["Low vegetation"], 1),
            "tree_canopy_m2": round(class_areas["Tree canopy"], 1),
            "water_m2": round(class_areas["Water"], 1),
        })


# --------------------------------------------------
# Kandidaten sortieren
# --------------------------------------------------

df = pd.DataFrame(results)

df = df.sort_values(
    by=["class_count", "tree_crowns"],
    ascending=[False, False]
)

print("Beste 10 Testbereiche:")
print()

print(
    df.head(10).to_string(index=False)
)

output = "/cfs/earth/scratch/troxlflo/BA/data/intermediate/test_area_candidates.csv"
df.to_csv(output, index=False)

print()
print(f"Alle 100 Felder gespeichert unter:")
print(output)
