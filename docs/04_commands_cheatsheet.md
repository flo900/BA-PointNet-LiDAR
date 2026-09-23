# 04 – Commands, Git, SLURM und Schnellstart

**Stand:** 18.09.2026  
**Zweck:** Dieses Dokument beim nächsten Arbeitsbeginn öffnen und ohne langes Suchen weiterarbeiten.

---

# 1. Schnellstart – neue Sitzung

Nach Verbindung mit dem ZHAW-HPC:

```bash
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0
conda activate ba_pointnet
cd ~/BA/Projektarbeit-2
```

Dann:

```bash
which python
python --version
git status
```

Erwartung:

```text
Conda-Environment: ba_pointnet
Python: 3.9.x
Projekt: ~/BA/Projektarbeit-2
```

---

# 2. Wo bin ich?

Aktuelles Verzeichnis:

```bash
pwd
```

Dateien anzeigen:

```bash
ls
```

Mit Details:

```bash
ls -lh
```

Auch versteckte Dateien:

```bash
ls -la
```

In Verzeichnis wechseln:

```bash
cd ORDNER
```

Eine Ebene zurück:

```bash
cd ..
```

Ins Home:

```bash
cd ~
```

---

# 3. Wichtige Projektpfade

Code:

```text
/net/home/troxlflo/BA/Projektarbeit-2
```

bzw.:

```bash
cd ~/BA/Projektarbeit-2
```

Scratch:

```text
/cfs/earth/scratch/troxlflo/BA
```

Dummy-Daten:

```text
/cfs/earth/scratch/troxlflo/BA/dummy_lidar
```

GPU-Test-Arbeitsordner:

```text
/cfs/earth/scratch/troxlflo/BA/gpu_test
```

Conda-Environment:

```text
/cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet
```

---

# 4. Dateien prüfen

Datei anzeigen:

```bash
cat DATEI
```

Anfang einer Datei:

```bash
head DATEI
```

Ende einer Datei:

```bash
tail DATEI
```

Log live verfolgen:

```bash
tail -f DATEI
```

Datei suchen:

```bash
find . -name "DATEINAME"
```

Beispiel:

```bash
find . -name "lidar_dataloader.py"
```

Text in Dateien suchen:

```bash
grep -R "SUCHTEXT" .
```

Mit Zeilennummer:

```bash
grep -Rn "SUCHTEXT" .
```

Beispiel:

```bash
grep -Rn "TORCH_CUDA_ARCH_LIST" .
```

---

# 5. Ordner / Dateien erstellen und löschen

Ordner:

```bash
mkdir NAME
```

Auch verschachtelte Pfade:

```bash
mkdir -p pfad/zum/ordner
```

Datei löschen:

```bash
rm DATEI
```

Ordner rekursiv löschen:

```bash
rm -rf ORDNER
```

**Achtung bei `rm -rf`: keine Rückfrage und kein Papierkorb.**

Kopieren:

```bash
cp QUELLE ZIEL
```

Ordner komplett kopieren:

```bash
cp -a QUELLE ZIEL
```

---

# 6. Module

Geladene Module anzeigen:

```bash
module list
```

Modul laden:

```bash
module load MODUL
```

Alle entfernen:

```bash
module purge
```

Verfügbare Versionen suchen:

```bash
module spider Python
```

oder:

```bash
module spider cuda
```

---

# 7. Conda

Environment aktivieren:

```bash
conda activate ba_pointnet
```

Deaktivieren:

```bash
conda deactivate
```

Umgebungen anzeigen:

```bash
conda env list
```

Python kontrollieren:

```bash
which python
python --version
```

Installierte Pakete:

```bash
python -m pip list
```

Bestimmtes Paket:

```bash
python -m pip show torch
```

Warum möglichst:

```bash
python -m pip
```

statt nur:

```bash
pip
```

verwenden?

Weil damit `pip` eindeutig zum aktuell ausgewählten Python gehört.

---

# 8. PyTorch / CUDA prüfen

Kurztest:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Auf einem Login-Node kann `torch.cuda.is_available()` erwartungsgemäß `False` sein, weil dort keine GPU für die Arbeit reserviert wurde.

GPU-Informationen gehören deshalb in einen SLURM-GPU-Job.

CUDA-Compiler:

```bash
nvcc --version
```

---

# 9. SLURM – Job starten

Syntax eines Jobskripts:

```bash
#!/bin/bash
#SBATCH --job-name=name
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
```

Vor Ausführung Bash-Syntax prüfen:

```bash
bash -n SCRIPT.sh
```

Wenn keine Ausgabe kommt, wurde kein Syntaxfehler gefunden.

Job einreichen:

```bash
sbatch --chdir=/cfs/earth/scratch/troxlflo/BA/gpu_test SCRIPT.sh
```

---

# 10. SLURM – Job überwachen

Eigene laufende Jobs:

```bash
squeue -u troxlflo
```

Bestimmten Job:

```bash
squeue -j JOBID
```

Historie / Status:

```bash
sacct
```

Genauer:

```bash
sacct -j JOBID
```

Job abbrechen:

```bash
scancel JOBID
```

---

# 11. SLURM-Ausgabe lesen

Beispiel:

```bash
cat /cfs/earth/scratch/troxlflo/BA/gpu_test/end_to_end_test.out
```

Fehler:

```bash
cat /cfs/earth/scratch/troxlflo/BA/gpu_test/end_to_end_test.err
```

Live beobachten:

```bash
tail -f /PFAD/ZUR/DATEI.out
```

---

# 12. Git – das Grundprinzip

```text
Arbeitsdateien
↓
git add
↓
Staging Area
↓
git commit
↓
gesicherter Git-Zwischenstand
↓
git push
↓
externe Sicherung auf GitHub
```

---

# 13. Git – aktuellen Zustand prüfen

```bash
git status
```

Das sollte einer der meistbenutzten Git-Befehle werden.

Er zeigt:

- geänderte Dateien
- neue Dateien
- für Commit vorgemerkte Dateien

---

# 14. Git – Änderungen ansehen

Noch nicht gestagte Änderungen:

```bash
git diff
```

Bereits mit `git add` vorgemerkte Änderungen:

```bash
git diff --staged
```

---

# 15. Git – Zwischenstand speichern

Alle Änderungen vormerken:

```bash
git add .
```

Besser bei wichtigen Änderungen auch gezielt:

```bash
git add DATEI
```

Commit:

```bash
git commit -m "Kurze sinnvolle Beschreibung"
```

Beispiel:

```bash
git commit -m "Set up and validate PointNet++ workflow on ZHAW HPC"
```

---

# 16. Git – Commit-Historie

Kompakt:

```bash
git log --oneline
```

Mit Verzweigungen:

```bash
git log --oneline --graph --decorate --all
```

---

# 17. Git – was ist seit letztem Commit passiert?

```bash
git status
git diff
```

Wenn beide sauber sind:

```text
working tree clean
```

Dann entspricht der aktuelle Arbeitsstand dem letzten Commit.

---

# 18. Git – zu älterem Stand schauen

**Nicht sofort mit harten Reset-Befehlen experimentieren.**

Zuerst Historie:

```bash
git log --oneline
```

Eine alte Datei ansehen:

```bash
git show COMMIT_ID:PFAD/ZUR/DATEI
```

Bevor ein kompletter Stand zurückgesetzt wird, zuerst einen neuen Commit oder Branch machen.

---

# 19. Eigenes GitHub-Repository – empfohlene Struktur

Bestehender lokaler Clone kann Nicolas Historie behalten.

Empfehlung:

```text
upstream → NicolaNoger/Projektarbeit-2
origin   → flo900/BA-PointNet-LiDAR
```

Aktuelle Remotes prüfen:

```bash
git remote -v
```

Wenn das eigene **leere private** GitHub-Repository erstellt wurde:

```bash
git remote rename origin upstream
```

Dann eigenes Repo hinzufügen:

```bash
git remote add origin git@github.com:flo900/BA-PointNet-LiDAR.git
```

Prüfen:

```bash
git remote -v
```

Erster Push:

```bash
git push -u origin main
```

Danach normalerweise:

```bash
git push
```

### Wichtig

Das eigene GitHub-Repository beim Erstellen **leer lassen**:

- kein README automatisch erzeugen
- keine `.gitignore` automatisch erzeugen
- keine License automatisch erzeugen

So gibt es beim ersten Push keine unnötigen Konflikte.

---

# 20. `origin` und `upstream`

Danach bedeutet:

```text
origin   = mein BA-Repository
upstream = Nicolas Original
```

Neue Änderungen von Nicola könnte man später gezielt holen:

```bash
git fetch upstream
```

Nicht automatisch mergen, ohne vorher zu prüfen.

---

# 21. GitHub-Sicherung

Nach sinnvoller Arbeitseinheit:

```bash
git status
git add .
git commit -m "Beschreibung"
git push
```

Ein sinnvoller Rhythmus ist nicht „jede Tastatureingabe committen“, sondern funktionierende, nachvollziehbare Arbeitsschritte.

Beispiele:

```text
Validate PointNet++ workflow on HPC
Add real LiDAR reader
Implement 25m tiling
Generate PointNet training arrays
Add binary sealing labels
Train first real-data model
```

---

# 22. Was gehört NICHT ins GitHub?

Nicht committen:

```text
große .npy-Dateien
LAS/LAZ/COPC
Modelle / Checkpoints
SLURM Logs
temporäre Outputs
vertrauliche Berichte
große Rohdaten
```

Deshalb `.gitignore` verwenden.

Vor einem Commit immer:

```bash
git status
```

ansehen.

---

# 23. Testdateien erneut ausführen

Dataloader lokal auf Login-Node:

```bash
python dataloader_test.py
```

Dummy-Daten neu erzeugen:

```bash
python create_dummy_lidar_data.py
```

GPU-/Modelltests über SLURM:

```bash
sbatch --chdir=/cfs/earth/scratch/troxlflo/BA/gpu_test end_to_end_test.sh
```

Dann:

```bash
squeue -u troxlflo
```

und:

```bash
cat /cfs/earth/scratch/troxlflo/BA/gpu_test/end_to_end_test.out
```

---

# 24. Python, Bash und SLURM unterscheiden

Python-Skript starten:

```bash
python SCRIPT.py
```

Shell-Skript direkt mit Bash:

```bash
bash SCRIPT.sh
```

SLURM-Skript über Scheduler:

```bash
sbatch SCRIPT.sh
```

Das sind drei unterschiedliche Dinge.

---

# 25. Bash – wichtige Zeichen

```bash
$VARIABLE
```

liest eine Variable.

```bash
export NAME=WERT
```

setzt eine Umgebungsvariable für nachfolgende Prozesse.

```bash
&&
```

zweiter Befehl läuft nur, wenn erster erfolgreich war.

```bash
;
```

trennt Befehle, unabhängig vom Erfolg.

```bash
\
```

kann lange Befehle über mehrere Zeilen fortsetzen.

```bash
#
```

Kommentar.

Aber:

```bash
#SBATCH
```

wird von SLURM als Direktive gelesen, wenn das Skript über `sbatch` gestartet wird.

---

# 26. `set -euo pipefail`

In unseren Bash-Testskripten verwendet:

```bash
set -euo pipefail
```

Bedeutung grob:

```text
-e  → bei Fehler abbrechen
-u  → undefinierte Variablen als Fehler behandeln
-o pipefail → Fehler innerhalb von Pipelines nicht verstecken
```

Für reproduzierbare Batch-Skripte sehr nützlich.

---

# 27. Was tun, wenn etwas plötzlich nicht mehr funktioniert?

Nicht sofort alles neu installieren.

In dieser Reihenfolge prüfen:

```text
1. Bin ich auf dem Cluster?
2. Bin ich im richtigen Ordner?
3. Sind die Module geladen?
4. Ist ba_pointnet aktiv?
5. Zeigt which python auf die Conda-Umgebung?
6. Was sagt git status?
7. Ist es ein Login-Node- oder GPU-Problem?
8. Was steht in .out?
9. Was steht in .err?
```

Commands:

```bash
hostname
pwd
module list
conda env list
which python
python --version
git status
```

---

# 28. Technischer Stand, an dem weitergearbeitet wird

Bis hierher erfolgreich:

```text
PointNet++ Umgebung eingerichtet
↓
CUDA Extensions für A100 kompiliert
↓
FPS auf GPU
↓
vollständiger Forward Pass
↓
Trainingsschritt
↓
Nicolas Dataloader
↓
End-to-End Test
```

**Nächster Schritt: echtes LiDAR-Preprocessing.**

Nicht noch mehr künstliche Infrastruktur bauen, solange kein konkretes technisches Problem auftritt.

---

# 29. Startblock zum Kopieren

Wenn beim nächsten Mal einfach weitergearbeitet werden soll:

```bash
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0
conda activate ba_pointnet
cd ~/BA/Projektarbeit-2
git status
```

Danach diese Dokumentation öffnen und beim Abschnitt „Nächster Schritt“ weitermachen.
