# 01 – Nicola Noger: GitHub-Repository und PointNet++-Workflow

**Referenzrepository:** `NicolaNoger/Projektarbeit-2`  
**Stand der Dokumentation:** 18.09.2026  
**Zweck:** Orientierung für die Bachelorarbeit und klare Trennung zwischen vorhandenem Referenzcode und eigener Weiterentwicklung.

---

## 1. Was macht Nicolas Projekt?

Nicolas Projekt vergleicht zwei Deep-Learning-Ansätze für Landbedeckungsklassifikation:

- **PointNet++** für 3D-LiDAR-Punktwolken
- **U-Net** für multispektrale 2D-Orthofotos (RGB + NIR)

Beide Ansätze klassifizieren in Nicolas Arbeit fünf Klassen:

1. Water
2. Tree canopy
3. Low vegetation
4. Impervious
5. Buildings

Für meine Bachelorarbeit ist zunächst vor allem der **PointNet++-Teil** relevant. Ziel ist nicht, Nicolas Arbeit einfach zu kopieren, sondern seinen vorhandenen Workflow zuerst reproduzierbar zum Laufen zu bringen und anschließend kontrolliert auf die eigene Fragestellung anzupassen.

---

## 2. Wichtigste Repository-Struktur

```text
Projektarbeit-2/
│
├── README.md
├── .gitignore
├── data_exploration.ipynb
│
└── src/
    ├── pointnet_requirements.txt
    ├── unet_requirements.txt
    │
    ├── PointnetPP/
    │   ├── lidar_dataloader.py
    │   ├── pointnet2_aerial_optimized.py
    │   ├── pointnet_train_aerial_optimized.py
    │   ├── pointnet_train_aerial_optimized.sh
    │   ├── pointnet_visualize_aerial.py
    │   ├── pointnet_visualize_aerial.sh
    │   └── PointNet2_PyTorch/
    │
    └── U-net/
        ├── U_net.py
        ├── dataloader.py
        ├── image_snipper.py
        ├── train_unet.py
        ├── train_unet.sh
        └── visualize_results.py
```

Für die aktuelle BA-Phase ist der Ordner `src/PointnetPP/` der zentrale Teil.

---

## 3. Rolle der einzelnen PointNet++-Dateien

### `lidar_dataloader.py`

Diese Datei lädt **bereits vorbereitete** Punktwolken-Tiles.

Erwartete Struktur:

```text
data_dir/
├── train_features.npy
├── train_labels.npy
├── val_features.npy
├── val_labels.npy
├── test_features.npy
├── test_labels.npy
└── metadata.json
```

Erwartete Dimensionen:

```text
features: (Anzahl Tiles, 16384, 7)
labels:   (Anzahl Tiles, 16384)
```

Ein einzelnes Tile enthält damit:

```text
16'384 Punkte × 7 Werte
```

Die sieben Werte sind:

```text
X
Y
Z
Intensity
ReturnNumber
NumberOfReturns
ScanAngle
```

XYZ werden als geometrische Koordinaten verwendet. Die vier restlichen Werte sind zusätzliche Punktattribute.

Der Dataloader nutzt `numpy.load(..., mmap_mode='r')`. Dadurch muss nicht der gesamte Datensatz gleichzeitig in den Arbeitsspeicher geladen werden.

Für Training, Validation und Test werden getrennte PyTorch-DataLoader erzeugt. Nur der Trainingsdatensatz erhält Datenaugmentation.

Implementierte Augmentationen:

- zufällige Rotation um die Z-Achse
- Jitter auf XYZ
- Skalierung von XYZ

---

### `pointnet2_aerial_optimized.py`

Hier ist das eigentliche angepasste PointNet++-Modell definiert.

Klasse:

```python
PointNet2AerialSSG
```

Eingabe:

```text
(Batch, 16384, 7)
```

Ausgabe:

```text
(Batch, 5, 16384)
```

Die fünf Ausgabekanäle entsprechen den fünf Landbedeckungsklassen. Für jeden Punkt werden fünf Logits erzeugt. Mit `argmax(dim=1)` wird daraus die vorhergesagte Klasse pro Punkt.

### Encoder / Set Abstraction

Nicolas finales Modell verwendet vier Set-Abstraction-Stufen:

| Stufe | Punkte | Radius | Nachbarn | Bedeutung |
|---|---:|---:|---:|---|
| SA1 | 2048 | 1.0 m | 64 | feine Strukturen |
| SA2 | 512 | 2.5 m | 48 | lokale Muster |
| SA3 | 128 | 5.0 m | 32 | mittlerer Kontext |
| SA4 | 32 | 10.0 m | 24 | Tile-Kontext |

Die Punkte werden unter anderem mit **Farthest Point Sampling (FPS)** ausgewählt.

### Decoder / Feature Propagation

Nach dem schrittweisen Downsampling werden die Merkmale wieder auf die ursprünglichen 16'384 Punkte hochgerechnet. PointNet++ kombiniert dabei grobe semantische Information mit feineren Merkmalen aus früheren Ebenen.

Am Ende folgt eine 1D-Convolution-basierte Klassifikationsschicht mit Dropout `0.6`.

### Dice Loss

In derselben Datei ist `DiceLoss` implementiert. Diese Loss-Funktion soll insbesondere Segmentierungsqualität, Klassengrenzen und seltene Klassen unterstützen.

---

### `pointnet_train_aerial_optimized.py`

Dies ist Nicolas zentrales Trainingsskript für die finale PointNet++-Variante.

Wichtige Konfiguration im aktuellen GitHub-Code:

```text
NUM_CLASSES      = 5
NUM_FEATURES     = 7
INPUT_CHANNELS   = 4
BATCH_SIZE       = 24
MAX_EPOCHS       = 50
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 5e-5
DROPOUT          = 0.6
FOCAL_GAMMA      = 2.0
DICE_WEIGHT      = 0.2
CLASS_WEIGHTS    = [2.0, 0.6, 0.2, 0.4, 0.45]
```

Trainingsaugmentation:

```text
Rotation: ±180°
Jitter: sigma 0.02, clip 0.05
Scale: 0.8–1.2
```

Im Code wird ein `AdamW`-Optimizer verwendet und ein `OneCycleLR`-Scheduler konfiguriert.

Das Training läuft über ein PyTorch-Lightning-Modul:

```text
PointNet2AerialOptimized
```

Nach dem Training wird ein `.pth`-Checkpoint gespeichert und der Testdatensatz ausgewertet.

Ausgegeben werden unter anderem:

- Overall Accuracy
- F1 pro Klasse
- Macro F1
- Weighted F1
- Classification Report
- Confusion Matrix

---

### `pointnet_train_aerial_optimized.sh`

SLURM-Wrapper für das Training auf dem HPC.

Nicolas Datei fordert unter anderem an:

```text
Partition: earth-5
GPU:       1
RAM:       64 GB
CPUs:      8
Zeit:      3 h
```

Das Skript lädt die Cluster-Module, aktiviert Nicolas virtuelle Umgebung und startet anschließend:

```bash
python pointnet_train_aerial_optimized.py
```

Die absoluten Pfade in diesem Skript zeigen auf Nicolas eigenes Scratch-Verzeichnis und müssen für meine Umgebung angepasst werden.

---

### `pointnet_visualize_aerial.py`

Lädt ein gespeichertes Modell, verarbeitet ausgewählte Test-Tiles und erzeugt Visualisierungen der Ground-Truth- und Modellklassen.

Auch hier sind Nicolas absolute Pfade direkt im Skript eingetragen. Vor Verwendung müssen sie auf die eigenen Verzeichnisse umgestellt werden.

---

### `PointNet2_PyTorch/`

Dies ist nicht primär Nicolas eigener PointNet++-Code, sondern die zugrunde liegende PointNet++-Implementierung, auf der sein Modell aufbaut.

Besonders wichtig ist:

```text
PointNet2_PyTorch/
└── pointnet2_ops_lib/
```

Darin befinden sich C++-/CUDA-Erweiterungen für rechenintensive Punktwolkenoperationen, beispielsweise Farthest Point Sampling.

Diese Extensions müssen für die jeweilige GPU-Architektur kompiliert werden.

---

## 4. Datenfluss im vorhandenen Workflow

Der **veröffentlichte** Teil des Workflows beginnt im Wesentlichen hier:

```text
bereits vorbereitete .npy-Dateien
        ↓
lidar_dataloader.py
        ↓
PyTorch Batch
        ↓
PointNet2AerialSSG
        ↓
Logits pro Punkt
        ↓
Loss
        ↓
Backpropagation
        ↓
Optimizer
        ↓
trainiertes Modell (.pth)
        ↓
Evaluation / Visualisierung
```

Wichtig: Das GitHub-Repository enthält **nicht den vollständigen Rohdaten-zu-NPY-Workflow**.

Das ist für die Bachelorarbeit entscheidend.

---

## 5. Was Nicolas schriftliche Arbeit zum Preprocessing beschreibt

Die schriftliche PA2 beschreibt Schritte, die im öffentlichen GitHub nicht vollständig als reproduzierbarer Preprocessing-Code vorhanden sind.

### Ausgangsdaten

Für PointNet++ verwendete Nicola:

- SwissSURFACE3D LiDAR
- AV-Daten als Ground Truth
- zusätzlichen Baumkronen-Datensatz

Die LiDAR-Daten für Wädenswil stammen laut Arbeit aus 2024.

### Zielklassen

Die ursprünglichen Landbedeckungsklassen wurden zu fünf Zielklassen zusammengeführt:

```text
Building
Impervious
Tree canopy
Low vegetation
Water
```

### Hierarchisches Relabeling

Die Arbeit beschreibt einen dreistufigen Prozess.

**Vorbereitung:** sehr seltene bzw. für die Zielsetzung irrelevante LiDAR-Klassen wurden entfernt.

**Schritt 1 – Direct Remapping**

Eindeutige LiDAR-Klassen wurden direkt auf Zielklassen abgebildet, z. B.:

```text
Building / facade → Buildings
Water             → Water
Bridge deck       → Impervious
```

**Schritt 2 – Polygon-basiertes Relabeling**

Übrige Punkte wurden räumlich mit Polygonen aus `target.gpkg` verknüpft.

Der wichtige Punkt dabei ist: Es genügt nicht, jedem Punkt einfach die Polygonklasse zu geben.

Beispiel Wald:

```text
Punkt am Boden unter Baumkrone
≠ automatisch Tree canopy
```

Bodenpunkte unter Waldkronen sollen beispielsweise weiterhin die bodennahe Klasse erhalten.

**Schritt 3 – Urban Tree Refinement**

Ein zweiter Datensatz `target_with_trees.gpkg` wurde verwendet, um isolierte urbane Bäume zu ergänzen.

---

## 6. Tiles und Eingabeformat

Das finale Modell verwendet:

```text
Tile-Größe:          25 × 25 m
Punkte pro Tile:     16'384
Features pro Punkt:  7
```

Die XYZ-Koordinaten werden laut Arbeit auf Tile-lokale Koordinaten bezogen.

Der vollständige Weg von:

```text
SwissSURFACE3D / Polygone
        ↓
25 × 25 m Tiles
        ↓
exakt 16'384 Punkte
        ↓
7 Features
        ↓
Train / Val / Test
        ↓
.npy
```

ist jedoch nicht als vollständiges öffentliches Skript im Repository vorhanden.

**Das ist der nächste große Arbeitsschritt für meine BA.**

---

## 7. Train / Validation / Test

Laut schriftlicher Arbeit:

```text
70 % Training
15 % Validation
15 % Test
```

Der Testdatensatz bleibt bis zur abschließenden Bewertung unberührt.

---

## 8. Ergebnisse aus Nicolas Arbeit

Für die finale PointNet++-Variante werden berichtet:

```text
Overall Accuracy: 88.2 %
Macro F1:         0.814
```

Die Arbeit beschreibt insbesondere gute Ergebnisse für dreidimensional gut erkennbare Strukturen wie Bäume und Gebäude.

Diese Werte dienen als Referenz für Nicolas konkreten Datensatz und seine fünf Klassen. Sie sind **kein erwarteter Zielwert für meine eigene binäre Versiegelungsklassifikation**.

---

## 9. Wichtige Unterschiede zwischen Arbeit und aktuellem GitHub-Code

Diese Punkte sollten bei einer wissenschaftlichen Reproduktion dokumentiert und später überprüft werden.

### Optimizer

In der schriftlichen Tabelle wird für PointNet++ **Adam** genannt.

Im aktuellen GitHub-Trainingscode steht:

```python
torch.optim.AdamW(...)
```

### Gewichtung von Focal + Dice

Die Arbeit beschreibt sinngemäß:

```text
80 % Focal + 20 % Dice
```

Der aktuelle Code berechnet jedoch:

```python
loss = focal_loss + 0.2 * dice_loss
```

Das entspricht mathematisch nicht exakt `0.8 * Focal + 0.2 * Dice`.

### Early Stopping

Die Arbeit nennt:

```text
Patience = 10 Epochen
```

Im Code existiert:

```python
PATIENCE = 10
```

Im derzeit veröffentlichten finalen Trainingsskript ist jedoch zu prüfen, ob dieser Wert tatsächlich über einen Early-Stopping-Callback an den Lightning-Trainer übergeben wird.

**Konsequenz:** Für eine saubere Reproduktion immer zwischen „in der Arbeit beschrieben“ und „im veröffentlichten Code tatsächlich implementiert“ unterscheiden.

---

## 10. Bedeutung für die Bachelorarbeit

Der sinnvollste Ablauf ist:

```text
1. Nicolas technischen Workflow reproduzieren       ✓
2. Cluster / CUDA / Modell technisch prüfen          ✓
3. Dataloader mit erwarteter Datenstruktur prüfen    ✓
4. vollständige Dummy-Pipeline prüfen                ✓
5. echtes LiDAR-Preprocessing rekonstruieren         ← nächster Schritt
6. Workflow mit echten Daten reproduzieren
7. Zieldefinition Versiegelt / Unversiegelt umsetzen
8. kontrollierte Experimente mit/ohne LiDAR bzw. mit geeigneten Features
9. Ergebnisse evaluieren und dokumentieren
```

Für die BA sollen die finalen Zielklassen nicht einfach Nicolas fünf Klassen bleiben. Die Baselland-Dokumentation verwendet für die Versiegelungsklassifikation eine binäre Codierung:

```text
0 = versiegelt
1 = unversiegelt
```

Die fachliche Definition der Klassen und der genaue Umgang mit Grenzfällen müssen vor dem finalen Training sauber festgelegt werden.

---

## 11. Quellenbasis dieser Zusammenfassung

- GitHub: `https://github.com/NicolaNoger/Projektarbeit-2`
- `PA2_Nicola_Noger.pdf`
- `Dokumentation LBK Baselland_rev.pdf`
- eigene, auf dem ZHAW-HPC erfolgreich ausgeführte Tests vom 17./18.09.2026

---

## 12. Merksatz

**Nicolas GitHub liefert bereits Dataloader, PointNet++-Architektur, Training und Evaluation. Der wichtigste fehlende Teil für die eigene Arbeit ist die reproduzierbare Aufbereitung echter LiDAR- und Ground-Truth-Daten bis zu den `.npy`-Arrays.**
