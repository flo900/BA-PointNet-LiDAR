# 03 – Erklärung aller Testdateien

**Stand:** 18.09.2026  
**Zweck:** nachvollziehen, warum jede Testdatei erstellt wurde und was ihr erfolgreicher Lauf tatsächlich beweist.

Die Tests wurden bewusst schrittweise aufgebaut. Dadurch konnte bei Problemen klar getrennt werden zwischen:

```text
Cluster
CUDA
PointNet++ Extension
Modell
Training
Dataloader
gesamter Pipeline
```

---

## Übersicht

| Datei | Prüft | Status |
|---|---|---|
| `gpu_test.sh` | PyTorch sieht A100 | ✓ erfolgreich |
| `compile_pointnet.sh` | CUDA/C++ PointNet++-Extension kompiliert | ✓ erfolgreich |
| `pointnet_gpu_test.sh` | echte PointNet++ CUDA-Operation | ✓ erfolgreich |
| `model_forward_test.sh` | vollständiger PointNet++ Forward Pass | ✓ erfolgreich |
| `mini_training_test.sh` | Backward + Optimizer | ✓ erfolgreich |
| `create_dummy_lidar_data.py` | erwartetes `.npy`-Datenformat | ✓ erfolgreich |
| `dataloader_test.py` | Nicolas echter Dataloader | ✓ erfolgreich |
| `end_to_end_test.sh` | `.npy` bis Prediction | ✓ erfolgreich |

---

# 1. `gpu_test.sh`

## Zweck

Erste Frage:

> Kann ein SLURM-Job mit meiner Python-Umgebung tatsächlich eine GPU verwenden?

Das Skript prüft typischerweise:

```python
torch.cuda.is_available()
torch.cuda.get_device_name(0)
torch.__version__
torch.version.cuda
```

## Erfolgreiches Ergebnis

Getestete GPU:

```text
NVIDIA A100-PCIE-40GB
```

PyTorch:

```text
1.12.1+cu116
```

Damit war klar:

```text
SLURM → GPU → PyTorch
```

funktioniert.

## Was dieser Test noch nicht beweist

Er beweist **nicht**, dass PointNet++ funktioniert. Er testet nur die allgemeine GPU-/PyTorch-Verbindung.

---

# 2. `compile_pointnet.sh`

## Zweck

PointNet++ nutzt eigene CUDA-/C++-Operationen. Diese können nicht nur als normale Python-Dateien ausgeführt werden, sondern müssen kompiliert werden.

Das Skript:

1. fordert eine GPU/Compute-Umgebung über SLURM an
2. lädt Compiler, CUDA und Conda
3. aktiviert `ba_pointnet`
4. setzt `TORCH_CUDA_ARCH_LIST=8.0`
5. kopiert die Build-Quelle auf Scratch
6. führt `pip install` aus
7. testet den Import der Extension
8. führt einen ersten CUDA-Test aus

## Warum auf Scratch kopieren?

Build-Prozesse erzeugen viele temporäre Dateien. Scratch ist dafür geeigneter als das Home-Verzeichnis.

## Erfolgreiches Ergebnis

Die Extension wurde gebaut:

```text
pointnet2_ops/_ext.cpython-39-x86_64-linux-gnu.so
```

Im Build war zu sehen:

```text
compute_80 / sm_80
```

Damit war klar, dass der Build für die A100 kompiliert wurde.

## Wichtige Folge

Nach diesem erfolgreichen Build muss die Extension nicht vor jedem Test neu kompiliert werden.

Nur bei relevanten Änderungen an den CUDA-/C++-Quellen oder bei einer anderen inkompatiblen Umgebung wäre ein erneuter Build nötig.

---

# 3. `pointnet_gpu_test.sh`

## Zweck

Nach dem erfolgreichen Import sollte geprüft werden, ob eine **echte PointNet++-CUDA-Operation** auf der A100 funktioniert.

Getestet wurde:

```text
Farthest Point Sampling (FPS)
```

Beispiel:

```text
128 künstliche XYZ-Punkte
        ↓
FPS
        ↓
16 räumlich repräsentative Punkte
```

## Warum ist FPS wichtig?

PointNet++ reduziert die Punktzahl in den Set-Abstraction-Schichten. FPS ist eine zentrale Operation dafür.

Wenn nur `import pointnet2_ops` funktioniert, heißt das noch nicht zwingend, dass die CUDA-Kernel korrekt auf der GPU ausgeführt werden.

Dieser Test ging deshalb einen Schritt weiter.

## Erfolgreiches Ergebnis

FPS lief auf der A100.

Damit war bestätigt:

```text
Python
↓
PyTorch
↓
PointNet++ Extension
↓
CUDA Kernel
↓
A100
```

---

# 4. `model_forward_test.sh`

## Zweck

Jetzt wurde nicht mehr nur eine einzelne CUDA-Funktion getestet, sondern Nicolas **vollständiges PointNet++-Modell**.

Erzeugt wurde ein künstlicher Tensor:

```text
Batch = 1
Punkte = 16'384
Features = 7
```

also:

```text
(1, 16384, 7)
```

Das Modell wurde erstellt mit:

```python
PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    use_xyz=True,
    dropout=0.6
)
```

## Erwartete Ausgabe

```text
(1, 5, 16384)
```

Bedeutung:

```text
1 Tile
5 Klassenwerte
für jeden der 16'384 Punkte
```

## Was `model.eval()` bedeutet

Das Modell wird in Evaluationsmodus versetzt.

Dadurch verhalten sich unter anderem Dropout und BatchNorm passend für Evaluation.

## Was `torch.no_grad()` bedeutet

PyTorch speichert keine Informationen für Backpropagation.

Das spart Speicher und Rechenaufwand.

## Erfolgreiches Ergebnis

Das vollständige Modell führte einen Forward Pass aus.

Damit war klar:

```text
Modellarchitektur + PointNet++ CUDA-Operationen
```

funktionieren zusammen.

---

# 5. `mini_training_test.sh`

## Zweck

Ein Forward Pass beweist noch nicht, dass Training funktioniert.

Beim Training braucht man zusätzlich:

```text
Prediction
↓
Loss
↓
Gradienten
↓
Backward
↓
Optimizer
↓
Gewichte ändern sich
```

Dieser Test verwendete künstliche Daten und zufällige Labels.

## Geprüfte Elemente

- PointNet++ im `train()`-Modus
- Focal Loss
- Dice Loss
- kombinierter Loss
- `loss.backward()`
- AdamW
- `optimizer.step()`
- Prüfung, ob sich ein Modellparameter verändert hat

## Erfolgreiches Ergebnis

Ausgabe:

```text
Forward -> Loss -> Backward -> Optimizer funktioniert
```

Damit war die komplette technische Trainingsmechanik bestätigt.

## Wichtig

Das war **kein echtes Training**.

Die Daten und Klassen waren zufällig. Das entstandene Modell hatte keine fachliche Bedeutung und musste nicht gespeichert werden.

---

# 6. `create_dummy_lidar_data.py`

## Zweck

Bis hierhin wurden künstliche Tensoren direkt im Python-Code erzeugt.

Als Nächstes sollte das echte Dateiformat von Nicolas Workflow geprüft werden.

Das Skript erzeugte:

```text
/cfs/earth/scratch/troxlflo/BA/dummy_lidar/
│
├── train_features.npy
├── train_labels.npy
├── val_features.npy
├── val_labels.npy
├── test_features.npy
├── test_labels.npy
└── metadata.json
```

## Erzeugte Shapes

Train:

```text
features: (8, 16384, 7)
labels:   (8, 16384)
```

Validation:

```text
features: (2, 16384, 7)
labels:   (2, 16384)
```

Test:

```text
features: (2, 16384, 7)
labels:   (2, 16384)
```

## Wichtig

Die Werte sind Zufallszahlen.

Der Test simuliert nur die **Struktur** echter Daten.

---

# 7. `dataloader_test.py`

## Zweck

Diese Datei ist **kein zweiter Dataloader**.

Sie importiert Nicolas vorhandenen:

```python
from src.PointnetPP.lidar_dataloader import ...
```

und prüft, ob dieser die künstlichen `.npy`-Dateien korrekt versteht.

## Erfolgreiches Ergebnis

Tatsächliche Ausgabe:

```text
Features: torch.Size([4, 16384, 7])
Features dtype: torch.float32

Labels: torch.Size([4, 16384])
Labels dtype: torch.int64

Min Label: 0
Max Label: 4
```

Metadata:

```text
points_per_tile: 16384
num_features: 7
num_classes: 5
```

Damit war bestätigt:

```text
.npy-Dateien
↓
Nicolas lidar_dataloader.py
↓
PyTorch Dataset
↓
PyTorch DataLoader
↓
korrekter Batch
```

---

# 8. `end_to_end_test.sh`

## Zweck

Letzter Integrationstest.

Vorher wurden einzelne Komponenten getrennt getestet.

Jetzt wurde die gesamte technische Strecke verbunden:

```text
.npy
↓
Nicolas Dataloader
↓
PyTorch Batch
↓
GPU
↓
Nicolas PointNet++
↓
Prediction
```

## Ablauf

### Dataloader

Ein künstliches `.npy`-Tile wird geladen.

Shape:

```text
(1, 16384, 7)
```

Labels:

```text
(1, 16384)
```

### GPU

Features und Labels werden mit:

```python
.to(device)
```

auf die A100 verschoben.

### Modell

```python
PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    use_xyz=True,
    dropout=0.6
)
```

### Prediction

Modellausgabe:

```text
(1, 5, 16384)
```

Danach:

```python
torch.argmax(output, dim=1)
```

Ergebnis:

```text
(1, 16384)
```

Also genau eine vorhergesagte Klasse pro Punkt.

## Erfolgreiches Ergebnis

```text
=== END-TO-END TEST ERFOLGREICH ===
.npy -> Dataloader -> GPU -> PointNet++ -> Prediction funktioniert.
```

Das ist der wichtigste bisherige technische Meilenstein.

---

# 9. Was wurde damit insgesamt bewiesen?

```text
ZHAW HPC                     ✓
SLURM                        ✓
A100                         ✓
PyTorch/CUDA                 ✓
PointNet++ CUDA Extensions   ✓
Modell                       ✓
Forward Pass                 ✓
Loss                         ✓
Backward                     ✓
Optimizer                    ✓
.npy Datenformat             ✓
Nicolas Dataloader           ✓
komplette Pipeline           ✓
```

---

# 10. Was wurde ausdrücklich noch NICHT getestet?

Noch offen:

```text
echte SwissSURFACE3D-Daten
echtes Labeling
echtes Preprocessing
25 × 25 m Tiling aus Rohdaten
Sampling auf exakt 16'384 reale Punkte
Erzeugung der 7 realen Features
räumlich saubere Train/Val/Test-Aufteilung
vollständiges Training auf echten Daten
fachliche Modellgüte
binäre Klassen Versiegelt / Unversiegelt
```

Damit ist klar: Die nächsten Probleme sind **nicht mehr grundlegende Cluster-Probleme**, sondern betreffen primär die Datenaufbereitung und die fachliche Methodik.

---

# 11. Soll man die Testdateien behalten?

Ja, zumindest vorerst.

Sinnvoll ist später eine Struktur wie:

```text
scripts/
└── tests/
    ├── gpu_test.sh
    ├── compile_pointnet.sh
    ├── pointnet_gpu_test.sh
    ├── model_forward_test.sh
    ├── mini_training_test.sh
    ├── create_dummy_lidar_data.py
    ├── dataloader_test.py
    └── end_to_end_test.sh
```

Sie dokumentieren, dass die Umgebung schrittweise validiert wurde und helfen bei späteren Fehlern.

Beispiel:

> Nach einem Update funktioniert das Training plötzlich nicht mehr.

Dann kann man von unten nach oben erneut prüfen:

```text
GPU?
↓
CUDA Extension?
↓
Modell?
↓
Dataloader?
↓
End-to-End?
```

So lässt sich der Fehler deutlich schneller eingrenzen.
