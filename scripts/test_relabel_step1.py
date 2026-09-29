import numpy as np
from laspy import CopcReader
from laspy.copc import Bounds

# --------------------------------------------------
# Testgebiet: 100 x 100 m
# --------------------------------------------------

LIDAR_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/raw/lidar/"
    "swisssurface3d_2024_2692-1232_2056_5728.copc.laz"
)

bounds = Bounds(
    mins=np.array([2692000, 1232700]),
    maxs=np.array([2692100, 1232800])
)

# --------------------------------------------------
# LiDAR-Punkte laden
# --------------------------------------------------

with CopcReader.open(LIDAR_PATH) as reader:
    points = reader.query(bounds=bounds)

original_class = np.asarray(points.classification)

print("Punkte ursprünglich:", len(original_class))


# --------------------------------------------------
# Preprocessing:
# Klassen 1, 14, 15 und 27 entfernen
# --------------------------------------------------

remove_classes = [1, 14, 15, 27]

keep = ~np.isin(original_class, remove_classes)

classes = original_class[keep]

print("Entfernte Punkte:", (~keep).sum())
print("Verbleibende Punkte:", len(classes))


# --------------------------------------------------
# Step 1: direkte Zuordnung
# --------------------------------------------------

building = np.isin(classes, [6, 26])
water = classes == 9
impervious = classes == 17

directly_labeled = building | water | impervious
unresolved = ~directly_labeled

print("\nDirekt zugewiesen:")
print("Building:", building.sum())
print("Water:", water.sum())
print("Impervious:", impervious.sum())

print("\nNoch nicht zugewiesen:", unresolved.sum())

remaining_classes, counts = np.unique(
    classes[unresolved],
    return_counts=True
)

print("\nUrsprüngliche Klassen der noch nicht zugewiesenen Punkte:")

for cls, count in zip(remaining_classes, counts):
    print(f"Klasse {cls}: {count:,}")
