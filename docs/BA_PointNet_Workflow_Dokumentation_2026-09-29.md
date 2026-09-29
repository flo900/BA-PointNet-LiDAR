# BA – LiDAR/PointNet++ Preprocessing und technischer Prototyp

**Stand:** 29.09.2026  
**Projekt:** Bachelorarbeit – LiDAR-basierte Landbedeckungsklassifikation  
**Referenz:** Nicola Noger, *Projektarbeit 2* und öffentliches GitHub-Repository `NicolaNoger/Projektarbeit-2`

---

## 1. Ziel dieses Arbeitsschritts

Ziel war nicht, bereits ein fachlich aussagekräftiges Modell zu trainieren, sondern zuerst einen **vollständig reproduzierbaren technischen Workflow** aufzubauen:

```text
SwissSURFACE3D
→ Zielklassen/Relabeling
→ Qualitätskontrolle der Labels
→ lokale Höhe über Boden
→ 25×25-m-Tiles
→ 16'384 Punkte pro Tile
→ 7 PointNet++-Features
→ NumPy-Datensatz
→ Nicola-Dataloader
→ PointNet++ Forward Pass auf GPU
→ Loss + Backward + Optimizer-Step
```

Der Workflow ist am 29.09.2026 bis zum **erfolgreichen Trainingsschritt auf echten LiDAR-Daten** durchgelaufen.

Wichtig: Dieser Prototyp verwendet weiterhin die fünf Klassen aus Nicolas Arbeit:

| ID | Klasse |
|---:|---|
| 0 | Water |
| 1 | Tree canopy |
| 2 | Low vegetation |
| 3 | Impervious |
| 4 | Buildings |

Eine spätere Reduktion auf **Versiegelt / Unversiegelt** ist für die Bachelorarbeit weiterhin möglich, wurde hier aber bewusst noch nicht vorgenommen.

---

# 2. Was Nicola dokumentiert

Aus Nicolas Arbeit und seinem öffentlichen Repository lassen sich folgende Punkte sicher ableiten:

### Zielklassen und Relabeling

Nicola definiert fünf Zielklassen:

- Building
- Impervious
- Tree canopy
- Low vegetation
- Water

Für die LiDAR-Daten beschreibt er eine hierarchische Relabeling-Logik:

1. Entfernen der Klassen `1`, `14`, `15`, `27`
2. Direkte Zuordnung:
   - `6`, `26` → Buildings
   - `9` → Water
   - `17` → Impervious
3. Räumlicher Join mit `target.gpkg`
4. Noch ungelabelte Punkte werden mit einem Datensatz einzelner Baumkronen (`target_with_trees.gpkg`) verfeinert.

Explizit dokumentierte Beispiele:

- Ground innerhalb `Tree canopy` → Low vegetation
- Vegetation innerhalb `Tree canopy` → Tree canopy
- Ground innerhalb `Impervious` → Impervious
- Vegetation innerhalb `Impervious` bleibt zunächst ungelabelt, um überhängende Äste nicht als versiegelt zu klassifizieren.

### PointNet++-Eingabe

Nicola dokumentiert:

- Tilegröße: **25 × 25 m**
- Punkte pro Tile: **16'384**
- Features pro Punkt:
  1. X
  2. Y
  3. Z
  4. Intensity
  5. ReturnNumber
  6. NumberOfReturns
  7. ScanAngle
- XYZ als **tile-lokale Koordinaten**
- PointNet++ mit 5 Klassen
- Set-Abstraction-Radien: 1.0 m, 2.5 m, 5.0 m, 10.0 m

Sein Dataloader erwartet bereits vollständig vorbereitete Arrays:

```text
train_features.npy   (N_train, 16384, 7)
train_labels.npy     (N_train, 16384)

val_features.npy     (N_val, 16384, 7)
val_labels.npy       (N_val, 16384)

test_features.npy    (N_test, 16384, 7)
test_labels.npy      (N_test, 16384)

metadata.json
```

### Training

Der öffentliche optimierte Trainingscode verwendet unter anderem:

- Focal Loss
- Dice Loss mit Gewicht `0.2`
- AdamW
- Learning Rate `1e-3`
- Weight Decay `5e-5`
- Dropout `0.6`
- Klassenreihenfolge:
  `Water, Tree canopy, Low vegetation, Impervious, Buildings`

---

# 3. Wichtige Einschränkung der Reproduktion

Der **öffentliche GitHub-Stand enthält nicht den vollständigen Rohdaten-Preprocessing-Code**, mit dem Nicola aus SwissSURFACE3D, AV und Baumkronen die fertigen `.npy`-Arrays erzeugt hat.

Dadurch sind einige Details nicht exakt rekonstruierbar, insbesondere:

- vollständige Relabeling-Matrix für alle Kombinationen aus LiDAR-Klasse und AV-Klasse
- genaue Konstruktion von `target_with_trees.gpkg`
- genaue Formel der XYZ-Normalisierung
- exakte Behandlung von Tiles mit weniger als 16'384 brauchbaren Punkten
- exakte Sampling-Methode auf 16'384 Punkte

Diese Punkte wurden in unserem Workflow **nicht stillschweigend erfunden**, sondern entweder:
1. als eigene methodische Entscheidung dokumentiert oder
2. bewusst noch offengelassen.

---

# 4. Datengrundlagen des Prototyps

## 4.1 SwissSURFACE3D

COPC-Datei des Testgebiets:

```text
swisssurface3d_2024_2692-1232_2056_5728.copc.laz
```

Relevante ursprüngliche Klassen:

| SwissSURFACE3D-Klasse | Bedeutung |
|---:|---|
| 1 | Unclassified |
| 2 | Ground |
| 3 | Vegetation |
| 6 | Building |
| 9 | Water |
| 17 | Bridge Deck |
| 26 | Building facade |
| 27 | Bridge supports |

**Wichtig:** SwissSURFACE3D-Klasse `3` wird in dieser Dokumentation als **Vegetation** bezeichnet. Sie darf nicht mit der Zielklasse `Low vegetation` verwechselt werden.

## 4.2 Amtliche Vermessung

Verwendeter Layer:

```text
Bodenbedeckung_BoFlaeche_Area
```

Daraus wurde `target.gpkg` mit dem Feld `target_class` erzeugt.

Resultat:

```text
25'287 Polygone
Building        9'439
Impervious      7'900
Low vegetation  6'346
Water             914
Tree canopy       686
NULL                2
```

Die zwei NULL-Polygone entsprechen:

```text
vegetationslos.Geroell_Sand
```

Diese Klasse wurde bewusst nicht zugeordnet, da Nicolas Arbeit keine eindeutige Zuordnung dafür dokumentiert.

## 4.3 Baumkronen

Quelle:

```text
Waedi_Veg.gdb
```

Verwendeter Layer:

```text
Baumkronen
```

Eigenschaften:

```text
7'588 MultiPolygone
CRS EPSG:2056
```

Der Datensatz wird ausschließlich als **Hilfsdatensatz für die Label-Verfeinerung** verwendet. Er ist **kein PointNet++-Inputfeature**.

---

# 5. Auswahl des 100×100-m-Testgebiets

Skript:

```text
scripts/find_test_area.py
```

Testgebiet:

```text
X: 2692000–2692100
Y: 1232700–1232800
```

Auswahlkriterium:

- heterogene Fläche
- alle fünf AV-Zielklassen vorhanden
- Baumkronen vorhanden
- geeignet, mehrere Konfliktfälle zu testen

AV-Flächen innerhalb des Testgebiets:

| Zielklasse | Fläche |
|---|---:|
| Building | 1'130.5 m² |
| Impervious | 2'381.3 m² |
| Low vegetation | 5'716.0 m² |
| Tree canopy | 639.7 m² |
| Water | 132.5 m² |

Zusätzlich:

```text
33 Baumkronen
```

---

# 6. LiDAR-Auszug und Relabeling

Im 100×100-m-Gebiet wurden zunächst:

```text
373'317 LiDAR-Punkte
```

gelesen.

Ursprüngliche Klassen:

```text
1       1'805
2     166'513
3     169'068
6      33'377
9         122
17        111
26      2'315
27          6
```

## 6.1 Step 1 – direktes Relabeling

Skript:

```text
scripts/test_relabel_step1.py
```

Entsprechend Nicolas Dokumentation:

```text
1 / 14 / 15 / 27 → entfernt
6 / 26           → Buildings
9                → Water
17               → Impervious
```

Ergebnis:

```text
Entfernt:      1'811

Buildings:    35'692
Water:           122
Impervious:      111

Noch offen:  335'581
```

## 6.2 Step 2 – Kombination LiDAR + AV untersuchen

Skript:

```text
scripts/check_relabel_step2.py
```

Für die ursprünglichen Klassen `2 = Ground` und `3 = Vegetation` wurde geprüft, in welcher AV-Zielklasse die Punkte liegen.

Ergebnis:

```text
Ground + Low vegetation       109'632
Ground + Impervious            46'337
Ground + Tree canopy            8'756
Ground + Water                  1'536
Ground + Building                 252

Vegetation + Low vegetation   105'254
Vegetation + Tree canopy       43'182
Vegetation + Impervious        13'262
Vegetation + Water              6'412
Vegetation + Building             958
```

Diese Tabelle war zentral, weil Nicolas Arbeit **nicht für jede dieser Kombinationen eine explizite Regel angibt**.

---

# 7. Konfliktanalyse

Die Konfliktanalyse wurde nicht durchgeführt, um ein neues Forschungsthema zu eröffnen, sondern um zu vermeiden, dass problematische Überlagerungen blind als Ground Truth verwendet werden.

## 7.1 Konfliktpunkte exportieren

Skript:

```text
scripts/export_conflicts.py
```

Untersuchte Kombinationen:

```text
Vegetation + Impervious
Vegetation + Water
Vegetation + Building
Ground + Water
Ground + Building
```

Ergebnis:

```text
Vegetation + Impervious    13'262
Vegetation + Water          6'412
Ground + Water              1'536
Vegetation + Building         958
Ground + Building             252
--------------------------------
Total                      22'420
```

Davon lagen:

```text
16'210 / 22'420
```

innerhalb einer bekannten Baumkrone.

Das entspricht rund 72 %.

Besonders auffällig:

```text
Vegetation + Water:
5'871 / 6'412 innerhalb Baumkrone

Vegetation + Impervious:
9'116 / 13'262 innerhalb Baumkrone
```

Interpretation:

Viele scheinbare Konflikte sind **vertikale Überlagerungen**, beispielsweise Baumkronen über Straßen oder Gewässern, und nicht zwingend Fehler der AV.

## 7.2 Distanz zur AV-Grenze

Skript:

```text
scripts/analyze_conflict_distance.py
```

Analysiert wurden nur Konflikte **außerhalb** bekannter Baumkronen:

```text
6'210 Punkte
```

Distanz zur Grenze des zugehörigen AV-Polygons:

```text
0–0.5 m      2'636
0.5–1 m      1'322
1–2 m        1'593
>2 m           659
```

Daraus:

```text
42.4 % innerhalb 0.5 m
63.7 % innerhalb 1 m
89.4 % innerhalb 2 m
10.6 % weiter als 2 m
```

Die verbleibenden >2-m-Punkte bestanden nur aus:

```text
Vegetation + Impervious    548
Vegetation + Building      111
```

Die visuelle Kontrolle in QGIS zeigte auch dort überwiegend Baumkronen bzw. überhängende Vegetation.

**Schlussfolgerung:**  
`Konflikt` bedeutet nicht automatisch `AV falsch`.

---

# 8. Baumkronen-Refinement

Skript:

```text
scripts/check_tree_refinement.py
```

Ziel:

Prüfen, welche LiDAR/AV-Kombinationen innerhalb des separaten Baumkronendatensatzes liegen.

Besonders wichtig:

```text
Vegetation + AV Low vegetation
gesamt:              105'254
innerhalb Baumkrone:  68'837
```

Damit war klar:

`Vegetation + Low vegetation` darf nicht sofort vollständig zu `Low vegetation` gemacht werden, da sonst viele Einzelbäume in Wiesen/Gärten verloren gehen.

Weitere Beispiele:

```text
Vegetation + Impervious:
9'116 / 13'262 innerhalb Baumkrone

Vegetation + Water:
5'871 / 6'412 innerhalb Baumkrone
```

---

# 9. Vollständiges Relabeling des Prototyps

Skript:

```text
scripts/relabel_test_area.py
```

Das Skript speichert bewusst nicht nur das Endlabel, sondern zusätzlich:

```text
original_class
av_class
inside_tree_crown
final_label
status
rule_basis
```

Damit bleibt nachvollziehbar, **warum** ein Punkt sein Label erhalten hat.

## 9.1 Dokumentierte Regeln

Als `rule_basis = documented` gelten Regeln, die durch Nicolas Arbeit explizit unterstützt werden:

```text
1 / 14 / 15 / 27 → removed

6 / 26 → Buildings
9      → Water
17     → Impervious

Ground + Tree canopy      → Low vegetation
Vegetation + Tree canopy  → Tree canopy
Ground + Impervious       → Impervious

noch ungelabelte Vegetation innerhalb Baumkrone
→ Tree canopy

noch ungelabelter Ground innerhalb Baumkrone
→ Low vegetation
```

## 9.2 Eigene Ergänzung

Als:

```text
rule_basis = inferred
```

wurde markiert:

```text
Ground + AV Low vegetation
→ Low vegetation
```

sowie:

```text
Vegetation + AV Low vegetation
außerhalb Baumkrone
→ Low vegetation
```

Diese Zuordnung ist fachlich konsistent mit Nicolas Definition der Zielklasse, wird in seiner Arbeit aber **nicht als vollständige Relabeling-Regel explizit ausgeschrieben**.

## 9.3 Unresolved

Nicht eindeutig belegte Konflikte wurden bewusst nicht erzwungen:

```text
rule_basis = not_assigned
status = unresolved_conflict
```

Ergebnis:

```text
Low vegetation    155'821
Tree canopy       127'213
Impervious         46'448
Buildings          35'692
Water                 122
```

Ohne finales Label:

```text
removed             1'811
unresolved           6'210
--------------------------
NULL final_label     8'021
```

Regelbasis:

```text
documented      221'058
inferred        146'049
not_assigned      6'210
```

---

# 10. 25×25-m-Tiling

Skript:

```text
scripts/check_25m_tiles.py
```

Das 100×100-m-Testgebiet enthält theoretisch:

```text
4 × 4 = 16 Tiles
```

## 10.1 Randproblem

Zunächst entstanden zusätzliche Tiles mit Index `4`, beispielsweise:

```text
4_0
0_4
```

Ursache:

Punkte exakt auf:

```text
x = xmax
oder
y = ymax
```

wurden rechnerisch bereits dem nächsten Tile zugeordnet.

Korrektur:

Verwendung **halboffener Intervalle**:

```text
xmin <= x < xmax
ymin <= y < ymax
```

Damit werden Grenzpunkte eindeutig nur einer Seite zugeordnet und spätere Doppelzählungen verhindert.

Nach Korrektur:

```text
16 echte Tiles
```

## 10.2 Punktzahl pro Tile

Von den 16 Tiles besaßen:

```text
12 Tiles ≥ 16'384 verwendbare Punkte
4 Tiles < 16'384 verwendbare Punkte
```

Unterbesetzte Tiles:

```text
0_3   16'091
1_3   14'605
3_0   15'823
3_3   15'385
```

Für den technischen Prototyp wurden **nur die 12 vollständigen Tiles verwendet**.

Die endgültige Regel für unterbesetzte Tiles im gesamten Untersuchungsgebiet ist noch offen.

---

# 11. Höhe über lokalem Boden

## 11.1 Motivation

Nicola schreibt, dass XYZ in tile-lokalen Koordinaten vorliegen. Die exakte Z-Normalisierung ist jedoch nicht dokumentiert.

Für unsere Bachelorarbeit wurde entschieden:

```text
X_local = X - tile_xmin
Y_local = Y - tile_ymin
Z_local = Höhe über lokal geschätztem Boden
```

Damit bleiben alle räumlichen Dimensionen in **Metern**.

## 11.2 Methode

Skript:

```text
scripts/test_ground_height.py
```

Vorgehen:

1. SwissSURFACE3D-Punkte mit `original_class = 2` als Ground verwenden
2. Um das 100×100-m-Gebiet einen **25-m-Puffer** laden
3. Für jeden LiDAR-Punkt die **8 nächsten Ground-Punkte** mittels `scipy.spatial.cKDTree` suchen
4. Bodenhöhe invers-distanzgewichtet schätzen
5. berechnen:

```text
height_agl_m = Z_point - estimated_ground_z
```

Zusätzliche Qualitätsvariable:

```text
ground_distance_m
```

= Distanz zum nächstgelegenen Ground-Punkt.

## 11.3 Ergebnis

Ground-Punkte im gepufferten Gebiet:

```text
300'549
```

Distanz zum nächsten Ground-Punkt:

```text
50 %     0.05 m
90 %     0.38 m
95 %     1.53 m
99 %     4.74 m
Maximum  8.85 m
```

Höhe über Boden:

```text
Mittelwert    4.67 m
Median        1.63 m
75 %          7.14 m
Minimum      -0.32 m
Maximum      35.16 m
```

Die resultierenden Höhen wurden zusätzlich stichprobenartig in QGIS geprüft und als räumlich plausibel beurteilt.

Kleine negative Werte wurden nicht künstlich auf 0 gesetzt.

---

# 12. Erstellung des PointNet++-Prototyps

Skript:

```text
scripts/create_pointnet_prototype.py
```

Verwendet wurden nur die 12 Tiles mit mindestens 16'384 verwendbaren Punkten.

Pro Tile:

1. zufällige Auswahl von exakt `16'384` Punkten
2. Sampling **ohne Wiederholung**
3. fester Seed:

```text
RANDOM_SEED = 42
```

Dadurch ist das Sampling reproduzierbar.

## 12.1 Features

```text
1 X_local_m
2 Y_local_m
3 Height_AGL_m
4 Intensity
5 ReturnNumber
6 NumberOfReturns
7 ScanAngle
```

Datentyp:

```text
float32
```

## 12.2 Labels

```text
0 Water
1 Tree canopy
2 Low vegetation
3 Impervious
4 Buildings
```

Datentyp:

```text
int64
```

## 12.3 Ergebnis

```text
Features:
(12, 16384, 7)

Labels:
(12, 16384)
```

Ausgabeordner:

```text
/cfs/earth/scratch/troxlflo/BA/data/processed/pointnet_prototype
```

Dateien:

```text
prototype_features.npy
prototype_labels.npy
prototype_tiles.csv
metadata.json
```

---

# 13. Klassenverteilung im gesampelten Prototyp

Über alle 12 Tiles:

```text
Water              48
Tree canopy    65'971
Low vegetation 83'404
Impervious     26'116
Buildings      21'069
```

Summe:

```text
196'608 Punkte
```

Kontrolle:

```text
12 × 16'384 = 196'608
```

Die extrem geringe Wasserzahl ist für diesen kleinen Testausschnitt erwartbar und wurde nicht als Modellbewertung interpretiert.

---

# 14. Test mit Nicolas Dataloader

Da Nicolas Dataloader Dateinamen nach dem Schema:

```text
train_features.npy
train_labels.npy
```

erwartet, wurden für den Prototyp nur symbolische Links angelegt:

```text
train_features.npy -> prototype_features.npy
train_labels.npy   -> prototype_labels.npy
```

Damit wurde kein Datensatz dupliziert.

Resultat:

```text
Features: (12, 16384, 7) float32
Labels:   (12, 16384) int64
Classes:  5
Tiles:    12
```

Erstes Tile:

```text
X:       0.01 bis 24.99 m
Y:       0.00 bis 24.99 m
Z/AGL:  -0.04 bis 29.30 m
```

Der Dataloader konnte den selbst erzeugten Datensatz unverändert laden.

---

# 15. GPU Forward Pass mit echten Daten

Python-Skript:

```text
scripts/test_pointnet_forward_realdata.py
```

SLURM-Skript:

```text
scripts/forward_realdata_job.sh
```

SLURM-Job:

```text
1064175
```

Ergebnis:

```text
Device: cuda

Input:
torch.Size([1, 16384, 7])

Output:
torch.Size([1, 5, 16384])

Predictions:
torch.Size([1, 16384])

Forward Pass erfolgreich.
```

Die Ausgabe `[1]` als einzige vorhergesagte Klasse wurde **nicht fachlich interpretiert**, weil das Modell zu diesem Zeitpunkt zufällig initialisiert und untrainiert war.

---

# 16. Echter Trainingsschritt

Python-Skript:

```text
scripts/test_pointnet_trainstep_realdata.py
```

SLURM-Skript:

```text
scripts/trainstep_realdata_job.sh
```

SLURM-Job:

```text
1064176
```

Für den technischen Test wurden Nicolas öffentliche Trainingseinstellungen verwendet:

```text
Focal Loss gamma = 2
Dice Weight = 0.2

Class weights:
[2.0, 0.6, 0.2, 0.4, 0.45]

Optimizer:
AdamW

Learning rate:
1e-3

Weight decay:
5e-5
```

Ergebnis:

```text
Focal Loss: 0.525341
Dice Loss:  0.837128
Total Loss: 0.692767

Input:
torch.Size([1, 16384, 7])

Output:
torch.Size([1, 5, 16384])

Backward erfolgreich.
Optimizer-Step erfolgreich.
TRAININGSSCHRITT ERFOLGREICH
```

Die Loss-Werte wurden nicht als Modellleistung interpretiert. Relevant war ausschließlich der erfolgreiche technische Durchlauf von:

```text
Forward
→ Loss
→ Backward
→ Optimizer-Step
```

---

# 17. Übersicht der erstellten Skripte

| Skript | Zweck | Status |
|---|---|---|
| `find_test_area.py` | heterogenes 100×100-m-Testgebiet auswählen | Diagnose / Auswahl |
| `test_relabel_step1.py` | Nicolas direkten Relabeling-Schritt prüfen | Test |
| `check_relabel_step2.py` | LiDAR/AV-Kombinationen zählen | Diagnose |
| `export_conflicts.py` | problematische Kombinationen als GPKG exportieren | Diagnose |
| `analyze_conflict_distance.py` | Konflikte nach Distanz zur AV-Grenze untersuchen | Diagnose |
| `check_tree_refinement.py` | Einfluss des Baumkronendatensatzes prüfen | Diagnose |
| `relabel_test_area.py` | vollständige Prototyp-Relabelinglogik | **Pipeline** |
| `check_25m_tiles.py` | Punktzahlen und Klassen pro 25×25-m-Tile prüfen | Diagnose |
| `test_ground_height.py` | lokale Bodenhöhe / Height AGL testen | Prototyp / später Pipeline |
| `create_pointnet_prototype.py` | PointNet-kompatible `.npy`-Arrays erzeugen | **Pipeline** |
| `test_pointnet_forward_realdata.py` | Forward-Pass mit echtem Tile | Smoke-Test |
| `forward_realdata_job.sh` | GPU-SLURM-Job für Forward-Pass | Smoke-Test |
| `test_pointnet_trainstep_realdata.py` | Forward + Loss + Backward + Optimizer | Smoke-Test |
| `trainstep_realdata_job.sh` | GPU-SLURM-Job für Trainingsschritt | Smoke-Test |

Später sollte die Repo-Struktur bereinigt werden, zum Beispiel:

```text
scripts/
├── preprocessing/
├── diagnostics/
└── tests/
```

Die Diagnose- und Smoke-Test-Skripte sollten bis zur Stabilisierung des Workflows erhalten bleiben.

---

# 18. Wichtigste Abweichungen zu Nicolas Arbeit

Dieser Abschnitt ist für die Bachelorarbeit besonders wichtig.

| Thema | Nicola | Unser aktueller Workflow | Bewertung |
|---|---|---|---|
| Rohdaten-Preprocessing-Code | in der Arbeit beschrieben, aber öffentlich nicht vollständig vorhanden | selbst rekonstruiert | notwendige Reimplementierung |
| AV-Zielklassen | 5 Klassen | gleiche 5 Klassen | weitgehend reproduziert |
| `Geroell_Sand` | keine eindeutige Zuordnung dokumentiert | bleibt NULL | bewusst nicht erfunden |
| Tree Refinement | `target_with_trees.gpkg` | separater `Baumkronen`-Layer wird sequenziell angewendet | Implementierungsabweichung |
| vollständige Relabeling-Matrix | nur teilweise dokumentiert | dokumentierte Regeln + klar markierte `inferred`-Regeln | transparent ergänzt |
| problematische AV/LiDAR-Kombinationen | hierarchische Logik beschrieben | unresolved Konflikte bewusst erhalten | konservativer als blindes Overlay |
| Konfliktanalyse | keine quantitative Grenzdistanzanalyse dokumentiert | Baumkronen- und Grenzdistanzanalyse durchgeführt | eigene QA-Erweiterung |
| X/Y-Normalisierung | „tile-local coordinates“ | Tile-Ursprung wird abgezogen, Einheit bleibt Meter | plausible Konkretisierung |
| Z-Normalisierung | genaue Formel nicht dokumentiert | Height AGL aus 8 lokalen Ground-Nachbarn | **eigene methodische Abweichung** |
| Ground-Schätzung | nicht dokumentiert | 8 nächste Ground-Punkte, invers-distanzgewichtet, 25-m-Puffer | eigene Methode |
| Tile-Randbehandlung | nicht dokumentiert | halboffene Intervalle | Implementierungsentscheidung |
| Punkte pro Tile | 16'384 | 16'384 | reproduziert |
| Sampling auf 16'384 | genaue Methode nicht öffentlich dokumentiert | random without replacement, Seed 42 | eigene reproduzierbare Methode |
| Tiles <16'384 | Behandlung nicht dokumentiert | im Prototyp ausgeschlossen | vorläufige Entscheidung |
| Train/Val/Test | getrennte Splits vorgesehen | Prototyp noch ohne echten Split | technischer Test, noch keine finale Datenaufteilung |
| Dataloader | Nicolas Code | unverändert genutzt | reproduziert |
| PointNet++ Architektur | Nicolas optimierte Architektur | unverändert für Smoke-Test genutzt | reproduziert |
| Loss/Optimizer | Focal + 0.2 Dice, AdamW | im Trainings-Smoke-Test gleich verwendet | reproduziert |
| Augmentation | Rotation, Jitter, Scale im Training | im einzelnen Trainingsschritt nicht verwendet | bewusst noch nicht Teil des Smoke-Tests |
| finale BA-Klassen | Nicola: 5 Klassen | aktuell ebenfalls 5; Versiegelt/Unversiegelt noch offen | spätere BA-Anpassung möglich |

---

# 19. Methodisch wichtigste eigene Entscheidungen

Die folgenden Punkte müssen später im Methodikteil der BA explizit als **eigene Entscheidungen** beschrieben werden:

### A. Unvollständig dokumentierte Labels nicht blind ergänzen

Nicht jede Kombination von LiDAR- und AV-Klasse wird automatisch zu einem Zielwert gezwungen.

Stattdessen:

```text
documented
inferred
not_assigned
```

Damit bleibt die Herkunft jedes Labels nachvollziehbar.

### B. Baumkronen vor pauschaler Low-Vegetation-Zuordnung berücksichtigen

Ein Punkt mit:

```text
LiDAR Vegetation
+
AV Low vegetation
```

kann trotzdem zu einer Baumkrone gehören.

Die Baumkroneninformation verhindert hier eine systematische Fehlzuordnung.

### C. Z als Höhe über lokalem Boden

Statt absolute LV95-Höhe oder einfache `Z - Zmin`-Normalisierung:

```text
Height AGL
```

mit lokaler Ground-Schätzung.

Ziel:

- topographischen Einfluss reduzieren
- Gebäuden und Bäumen eine geometrisch besser interpretierbare Höhe geben
- Einheit Meter beibehalten

### D. Unterbesetzte Tiles vorläufig nicht künstlich auffüllen

Im Prototyp werden Tiles mit weniger als 16'384 verwendbaren Punkten verworfen.

Noch offen für den Gesamtdatensatz:

- Tile verwerfen
- Sampling mit Replacement
- andere Mindestschwelle

Diese Entscheidung soll erst anhand des gesamten Wädenswil-Gebiets getroffen werden.

---

# 20. Aktueller Meilenstein

Am Ende dieses Arbeitstages ist technisch nachgewiesen:

> Aus SwissSURFACE3D, AV und Baumkronendaten kann ein selbst erzeugter LiDAR-Datensatz aufgebaut werden, der die von Nicolas PointNet++-Implementierung erwartete Datenstruktur besitzt und auf dem ZHAW-HPC erfolgreich durch Forward Pass, Loss-Berechnung, Backpropagation und Optimizer-Step läuft.

Damit ist die **technische Machbarkeit der eigenen Preprocessing-Pipeline bestätigt**.

---

# 21. Nächste Schritte

Noch nicht erledigt:

1. Prototyp-Logik auf das tatsächliche Untersuchungsgebiet skalieren.
2. Anteil unterbesetzter 25×25-m-Tiles im Gesamtdatensatz bestimmen.
3. endgültige Regel für Tiles `<16'384` festlegen.
4. robuste Train-/Validation-/Test-Aufteilung festlegen.
5. Preprocessing-Skripte konsolidieren und Repo-Struktur bereinigen.
6. erst danach echtes Training durchführen.
7. später entscheiden, ob die BA endgültig mit fünf Klassen oder mit einer abgeleiteten binären Versiegelt/Unversiegelt-Klassifikation arbeitet.

---

# 22. Quellenbasis

## Nicola Noger

**Projektarbeit 2**

Relevant insbesondere:

- Abschnitt Daten-Preprocessing / Zielklassen / hierarchisches LiDAR-Relabeling
- Beschreibung der PointNet++-Architektur
- 25×25-m-Tiles
- 16'384 Punkte
- sieben Features
- tile-lokale XYZ-Koordinaten

## Öffentliches GitHub-Repository

```text
NicolaNoger/Projektarbeit-2
```

Besonders relevante Dateien:

```text
src/PointnetPP/lidar_dataloader.py
src/PointnetPP/pointnet2_aerial_optimized.py
src/PointnetPP/pointnet_train_aerial_optimized.py
```

Wichtig:

Der öffentliche Repository-Stand enthält nicht den vollständigen Rohdaten-Preprocessing-Code. Alle in dieser Dokumentation als eigene Ergänzung oder Abweichung gekennzeichneten Schritte dürfen daher **nicht Nicola zugeschrieben werden**.

---

## Kurzfassung der Abgrenzung

**Von Nicola übernommen bzw. direkt gestützt:**

```text
5 Zielklassen
hierarchische Label-Idee
direktes Remapping eindeutiger LiDAR-Klassen
AV-Kontext
Baumkronen-Refinement
25×25 m
16'384 Punkte
7 Features
PointNet++ Architektur
Dataloader-Struktur
Focal + Dice
```

**Von uns ergänzt / konkretisiert:**

```text
vollständige operative Relabelinglogik
rule_basis documented/inferred/not_assigned
Konflikt- und Grenzdistanzanalyse
separater Baumkronen-Layer statt target_with_trees.gpkg
halboffene Tile-Grenzen
Sampling ohne Replacement mit Seed 42
Height AGL über 8 Ground-Nachbarn
25-m-Ground-Puffer
vorläufiges Verwerfen unterbesetzter Tiles
technischer Prototyp ohne echten Train/Val/Test-Split
```
