# 02 – ZHAW-HPC: Einrichtung der PointNet++-Umgebung

**Stand:** 18.09.2026  
**Getestete Zielhardware:** NVIDIA A100-PCIE-40GB  
**Ziel:** reproduzierbare Umgebung für Nicolas PointNet++-Workflow.

---

## 1. Grundidee der Umgebung

Die Arbeit ist auf drei Ebenen aufgeteilt:

```text
VS Code auf eigenem Computer
        ↓ Remote-SSH
ZHAW Login Node
        ↓ SLURM
GPU Compute Node (A100)
```

### Login Node

Verwendet für:

- Dateien bearbeiten
- Git
- kleine Python-Tests
- Conda-Verwaltung
- Jobs mit `sbatch` einreichen
- Logs lesen

### Compute Node / GPU

Verwendet für:

- CUDA
- PointNet++-CUDA-Operationen
- Forward Pass
- Training
- größere Berechnungen

**GPU-Training nicht direkt auf `login02` starten.**

---

## 2. Verzeichnisse

### Code

Aktueller Code:

```text
/net/home/troxlflo/BA/Projektarbeit-2
```

Kurzform im eigenen Home:

```text
~/BA/Projektarbeit-2
```

### Scratch

Für große Dateien, Builds, Daten, Logs und Modelle:

```text
/cfs/earth/scratch/troxlflo/BA
```

Beispiel aus den Tests:

```text
/cfs/earth/scratch/troxlflo/BA/gpu_test
```

### Conda-Environment

Die Umgebung liegt auf Scratch:

```text
/cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet
```

Das ist sinnvoll, da ML-Umgebungen viele und teilweise große Dateien enthalten.

---

## 3. VS Code Remote-SSH

Verbindung wurde erfolgreich hergestellt zu:

```text
login-rhel8.hpc.zhaw.ch
```

Nach der Verbindung erschien als konkreter Login-Host:

```text
login02
```

Im VS-Code-Terminal sieht der Prompt beispielsweise so aus:

```text
(ba_pointnet) [15:35:14 troxlflo@login02: ~/BA/Projektarbeit-2]
```

Bedeutung:

```text
(ba_pointnet)  → Conda-Environment aktiv
troxlflo       → Benutzer
login02        → aktueller Cluster-Rechner
~/BA/...       → aktuelles Verzeichnis
```

Wenn VS Code oben beim Terminal `bash` anzeigt, bedeutet das nur, dass die Bash-Shell verwendet wird.

---

## 4. Cluster-Module

Auf dem ZHAW-HPC wurden folgende Module verwendet:

```bash
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0
```

Danach:

```bash
conda activate ba_pointnet
```

### Warum Module?

Ein HPC-System stellt nicht jede Software global in einer einzigen Version bereit.

`module load` schaltet eine bestimmte Softwareversion für die aktuelle Shell frei.

Beispiele:

```text
gcc       → C/C++ Compiler
cuda      → NVIDIA CUDA Toolkit
miniconda → Python-Umgebungen
```

Die geladenen Module gelten für die aktuelle Shell-Sitzung und müssen nach einer neuen Anmeldung erneut geladen werden.

---

## 5. Python / Conda

Auf dem Basissystem war zunächst ein altes Python vorhanden:

```text
Python 3.6.8
```

Für PointNet++ wurde deshalb eine eigene Umgebung erstellt:

```text
Environment: ba_pointnet
Python:      3.9.25
```

Aktivieren:

```bash
conda activate ba_pointnet
```

Kontrolle:

```bash
which python
python --version
```

Der Python-Pfad sollte auf die Conda-Umgebung zeigen und **nicht** auf `/usr/bin/python`.

---

## 6. Bestätigte Kernversionen

Erfolgreich getestet:

```text
Python:         3.9.x
PyTorch:        1.12.1+cu116
CUDA in Torch:  11.6
NumPy:          1.21.6
```

Passende PyTorch-Komponenten:

```text
torchvision 0.13.1+cu116
torchaudio  0.12.1
```

Nicolas `src/pointnet_requirements.txt` verwendet ebenfalls:

```text
torch==1.12.1+cu116
torchvision==0.13.1+cu116
torchaudio==0.12.1+cu116
numpy==1.21.6
```

**Wichtig:** Nicht automatisch davon ausgehen, dass bereits jedes Paket aus `pointnet_requirements.txt` in meiner Umgebung installiert und getestet ist. Sicher bestätigt wurden die Komponenten, die für die bisherigen Tests gebraucht wurden.

---

## 7. NumPy-Kompatibilität

Nach der PointNet++-Kompilierung trat eine NumPy-Warnung auf.

Problem:

```text
PyTorch / Extension erwartet NumPy 1.x
vorhanden war NumPy 2.x
```

Lösung:

```bash
python -m pip install "numpy==1.21.6"
```

Danach funktionierten Import und GPU-Tests.

**Für diese alte PyTorch-Version NumPy nicht unkontrolliert auf 2.x aktualisieren.**

---

## 8. CUDA und GPU

Geladenes CUDA-Toolkit:

```text
CUDA 11.6.2
nvcc V11.6.124
```

Der GPU-Test auf einem SLURM-Compute-Node ergab:

```text
NVIDIA A100-PCIE-40GB
torch.cuda.is_available() == True
```

`nvidia-smi` kann eine deutlich neuere „CUDA Version“ anzeigen. Das ist die vom NVIDIA-Treiber unterstützte maximale CUDA-Version und muss nicht mit der Toolkit-Version identisch sein, gegen die PyTorch gebaut wurde.

Für unsere Umgebung relevant:

```text
PyTorch → cu116
Toolkit → CUDA 11.6
```

---

## 9. SLURM

SLURM verteilt Jobs auf die Compute-Nodes.

Ein Job wird mit:

```bash
sbatch script.sh
```

eingereicht.

Status:

```bash
squeue -u troxlflo
```

Abgeschlossene Jobs:

```bash
sacct
```

Typische Header:

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
```

### Wichtige Erkenntnis

Auf diesem Cluster mussten die GPU-Testjobs mit einem Working Directory auf Scratch gestartet werden.

Verwendet:

```bash
sbatch --chdir=/cfs/earth/scratch/troxlflo/BA/gpu_test script.sh
```

Wenn ein Job vom Home-Verzeichnis aus Probleme macht, zuerst prüfen, ob `--chdir` auf ein Scratch-Verzeichnis gesetzt wurde.

---

## 10. PointNet++ CUDA-Extensions

Die PointNet++-Basisbibliothek enthält CUDA-/C++-Operationen, die lokal kompiliert werden müssen.

Quellordner:

```text
src/PointnetPP/PointNet2_PyTorch/pointnet2_ops_lib
```

### Problem mit A100

Nicolas / die verwendete Basisimplementierung hatte die unterstützten GPU-Architekturen nicht passend für eine A100 vorkonfiguriert.

A100:

```text
Compute Capability 8.0
```

Deshalb wurden zwei Stellen auf:

```python
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
```

angepasst.

Geänderte Dateien:

```text
src/PointnetPP/PointNet2_PyTorch/pointnet2_ops_lib/setup.py
```

und

```text
src/PointnetPP/PointNet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py
```

Zusätzlich wurde beim Build gesetzt:

```bash
export TORCH_CUDA_ARCH_LIST="8.0"
```

---

## 11. Kompilierung

Die CUDA-Extension wurde über einen SLURM-Job kompiliert.

Wichtiger Installationsbefehl:

```bash
python -m pip install -v --no-build-isolation .
```

Warum `python -m pip`?

Damit ist eindeutig, dass `pip` zur aktuell aktiven Python-/Conda-Umgebung gehört.

Warum `--no-build-isolation`?

Damit der Build die bereits installierte, passende PyTorch-/CUDA-Umgebung verwendet, anstatt eine isolierte Build-Umgebung mit möglicherweise unpassenden Versionen anzulegen.

Der Compiler erzeugte unter anderem eine `.so`-Bibliothek:

```text
pointnet2_ops/_ext.cpython-39-x86_64-linux-gnu.so
```

Im Compiler-Output war entscheidend:

```text
-gencode=arch=compute_80,code=sm_80
```

Das bestätigte, dass für die A100-Architektur kompiliert wurde.

---

## 12. Warnungen beim Build

### `ninja` nicht installiert

Der Build meldete, dass `ninja` fehlt und auf einen langsameren Build-Prozess zurückfällt.

Das war **keine kritische Fehlermeldung**.

### CUDA/G++ Versionshinweis

Es gab eine Warnung bezüglich fehlender definierter Compiler-Grenzen für CUDA 11.6.

Der Build lief trotzdem erfolgreich.

### NumPy `_ARRAY_API`

Das war die relevante Kompatibilitätswarnung und wurde durch:

```bash
python -m pip install "numpy==1.21.6"
```

behoben.

---

## 13. Umgebungs-Schnellstart nach neuer Anmeldung

Nach einem Logout oder einem neuen VS-Code-Fenster:

```bash
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0
conda activate ba_pointnet
cd ~/BA/Projektarbeit-2
```

Danach kontrollieren:

```bash
which python
python --version
git status
```

Für GPU-Arbeit zusätzlich Job über SLURM einreichen.

---

## 14. Was bleibt dauerhaft, was nicht?

### Dauerhaft

Bleibt nach Logout erhalten:

- Dateien
- Git-Commits
- Conda-Environment
- installierte Python-Pakete
- kompilierte PointNet++-Extension
- Daten auf Home/Scratch
- SLURM-Jobs laufen auch nach Trennung der SSH-Verbindung weiter

### Nicht dauerhaft pro Shell

Nach einer neuen Anmeldung erneut nötig:

- `module load ...`
- `conda activate ba_pointnet`
- ggf. `cd` ins Projekt

---

## 15. Aktueller technischer Status

Erfolgreich:

```text
VS Code Remote-SSH                  ✓
Conda / Python 3.9                  ✓
PyTorch + CUDA 11.6                 ✓
A100-Zugriff über SLURM             ✓
PointNet++ CUDA-Extensions          ✓
Farthest Point Sampling             ✓
komplettes PointNet++ Modell        ✓
Backward / Optimizer                ✓
Dataloader                          ✓
End-to-End Pipeline                 ✓
```

Der Cluster ist damit technisch bereit für den nächsten Schritt: **echte LiDAR-Daten vorbereiten und den Workflow mit realen Daten ausführen.**
