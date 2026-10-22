# PointNet++: Baseline und einfache RGB-Erweiterung

Stand: 22.10.2026. Projekt: BA Florian Troxler, Wädenswil.

## Fragestellung und Versionen

Verbessert die Ergänzung von RGB aus SWISSIMAGE die bestehende fünfklassige
PointNet++-Klassifikation gegenüber denselben LiDAR-Punkten ohne RGB?

Die bestehende Baseline bleibt unverändert. RGB v1 verwendet eigene Skripte,
Datenverzeichnisse und Trainings-/Auswertungsordner. Es wurden keine Gewichte
der Baseline weitertrainiert. RGB v1 wurde frisch initialisiert. Eine spätere
Version mit RGB-Gültigkeitskennzeichnung ist geplant, noch nicht umgesetzt.

| Bestandteil | Baseline | RGB v1 |
|---|---|---|
| Eingabefeatures | 7 | 10: bisherige 7 plus RGB |
| Modell-Attributkanäle neben XYZ | 4 | 7 |
| Training | 1065610 | 1073682 |
| Ausgewählte Epoche | 43 | 48 |
| Auswahlkriterium | niedrigster Validierungs-Loss | niedrigster Validierungs-Loss |
| Testauswertung | 1065611 | 1073698 |
| Sichtbarkeits-/Gültigkeitsmaske | keine | keine |

## Daten und Referenz

Modellgebiet: 4 km², X 2691000–2693000, Y 1231000–1233000, EPSG:2056.
Dieses Modellgebiet ist ein Ausschnitt des Untersuchungsgebiets Wädenswil.
25 × 25 m Tiles, je 16’384 gesampelte Einträge. Dünn besetzte Tiles enthalten
wiederholte Punkte; 128 reine Wasser-Tiles mit fehlender AGL-Höhe wurden bereits
im ursprünglichen Trainingsdatenexport ausgeschlossen.

Train 4302, Validierung 912, Test 896 Tiles. Räumlicher Split B wurde exakt
beibehalten; kein Abstandspuffer. Die Teilgebiete können sich an Grenzen berühren.

Referenzlabels: regelbasiert aus AV, ursprünglichen LiDAR-Klassen und
Baumkronendaten erzeugt. Fünf Klassen: water, tree_canopy, low_vegetation,
impervious, buildings. Diese Referenz ist keine unabhängig fehlerfrei erhobene
Ground Truth. Die ursprünglichen LiDAR-Klassen sind keine Modellvorhersagen.

Der RGB-Export prüfte die bitidentische Übernahme der sieben Originalfeatures,
identische Klassenhäufigkeiten und per SHA-256 identische Kopien der Labelarrays
und Kacheltabellen für alle drei Splits. Punktreihenfolge, Sampling inklusive
Wiederholungen und räumliche Zuordnung bleiben erhalten.

## RGB-Aufbereitung

- Download: 58 GeoTIFF-Dateien aus der bereitgestellten Linkliste.
- Produktdateien SWISSIMAGE-DOP10, Jahrkennung 2025; LiDAR 2024. Das genaue
  Flugdatum wurde in diesem Arbeitsschritt nicht geprüft.
- Originalbilder: EPSG:2056, 0.1 m Pixel, drei uint8-Kanäle Rot/Grün/Blau.
- Vier zentrale Bilder decken das Modellgebiet ab; südliche Nachbarbilder werden
  bei Punkten exakt auf dem südlichen Bildrand benötigt und ebenfalls verwendet.
- LV95-Position: lokale XY zuerst in float64 umwandeln, dann xmin/ymin addieren.
- Farbe des enthaltenden Pixels, keine Interpolation; Normalisierung RGB / 255
  als float32. Alte Features werden nicht neu skaliert.
- Rasterrandregel: links inklusive, rechts exklusive; unten exklusive, oben
  inklusive. Passendes Nachbarbild verwenden, keine Punktverschiebung und kein
  pauschales Abschneiden von Pixelindizes.
- Keine Sichtbarkeitsmaske: verdeckte Bodenpunkte können Baumkronen-/Dachfarben
  erhalten. Unterschiedliche Aufnahmejahre und lagebedingte Bildabweichungen sind
  weitere mögliche Einschränkungen; ihr Effekt wurde nicht isoliert gemessen.
- RGB-Export Job 1073681: COMPLETED, ExitCode 0:0, 1 min 48 s.
- Separate Umgebung ba_rgb: Rasterio 1.4.4, NumPy 2.2.6. Training und Inferenz
  weiterhin in bestehendem ba_pointnet, Python 3.9.25, PyTorch 1.12.1+cu116.

## Training

Batchgrösse 24, maximal 50 Epochen, Early-Stopping-Patience 10, Seed 42,
AdamW LR 0.001, Weight Decay 0.00005, Dropout 0.6. Focal Loss gamma 2 mit
Klassenfaktoren [2.0, 0.6, 0.2, 0.4, 0.45] plus 0.2 × Dice Loss.
OneCycleLR und geometrische Augmentation wurden beibehalten. Keine Farbaugmentation.

RGB-Trainingsskript prüft Trainingseinstellungen gegen config.json des
ausgewählten Baseline-Laufs. Beide Läufe durchliefen 50 Epochen. Kein Testsplit
wurde im Training geladen. Gleicher Seed bedeutet bei unterschiedlichen
Eingabedimensionen nicht identische Initialgewichte; es liegt ein Lauf je Variante
vor. Modell und Dataloader wurden als gemeinsame Quelldateien nicht bearbeitet;
die vorhandene Parametrisierung unterstützt die zusätzlichen Kanäle.

## Validierung: jeweils Epoche mit niedrigstem Loss

| Kennzahl | Baseline, Epoche 43 | RGB v1, Epoche 48 |
|---|---:|---:|
| Loss | 0.078453 | 0.077955 |
| Accuracy | 92.27 % | 92.07 % |
| Macro-F1 | 0.9194 | 0.9183 |
| mIoU | 85.76 % | 85.55 % |
| Wasser IoU | 99.22 % | 99.05 % |
| Baumkronen IoU | 72.73 % | 72.69 % |
| Niedrige Vegetation IoU | 88.62 % | 88.28 % |
| Befestigte Flächen IoU | 72.94 % | 72.70 % |
| Gebäude IoU | 95.26 % | 95.05 % |

## Test: dieselben 896 Tiles, 14’680’064 gesampelte Einträge

| Kennzahl | Baseline | RGB v1 | Differenz RGB minus Baseline |
|---|---:|---:|---:|
| Accuracy | 92.2860 % | 92.2977 % | +0.0117 Prozentpunkte |
| Macro-F1 | 0.935180 | 0.935584 | +0.000404 |
| Weighted-F1 | 0.923398 | 0.923600 | +0.000202 |
| mIoU | 88.1385 % | 88.2033 % | +0.0648 Prozentpunkte |

| Klasse | Baseline P/R/F1/IoU | RGB P/R/F1/IoU |
|---|---|---|
| water | 0.9965 / 0.9847 / 0.9906 / 98.13 % | 0.9988 / 0.9835 / 0.9911 / 98.23 % |
| tree_canopy | 0.8419 / 0.9606 / 0.8973 / 81.38 % | 0.8437 / 0.9596 / 0.8980 / 81.48 % |
| low_vegetation | 0.9560 / 0.9092 / 0.9320 / 87.27 % | 0.9595 / 0.9058 / 0.9319 / 87.24 % |
| impervious | 0.8819 / 0.8760 / 0.8790 / 78.41 % | 0.8662 / 0.8954 / 0.8805 / 78.66 % |
| buildings | 0.9841 / 0.9701 / 0.9770 / 95.51 % | 0.9813 / 0.9717 / 0.9765 / 95.40 % |

P = Precision, R = Recall. Metriken pro gesampeltem Eintrag, Wiederholungen
mitgezählt. Identische Metrikberechnung; Confusion Matrix: Zeilen Referenz,
Spalten Vorhersage. Gegenstand ist Übereinstimmung mit erzeugten Referenzlabels.

Der Testsplit wurde bereits für die Baseline ausgewertet. Der RGB-Checkpoint
wurde anhand der Validierung festgelegt und danach auf denselben Testdaten
ausgewertet. Dies ist ein kontrollierter Vergleich auf einem bestehenden Testsplit,
kein neues unabhängiges Testgebiet. Künftige Varianten über die Validierung
entwickeln; keine Auswahl von Regeln oder Parametern anhand dieser Testresultate.

## Interpretation und nächste Schritte

In diesem Einzelvergleich kein klarer Genauigkeitsgewinn durch einfaches RGB.
Validierung geringfügig schlechter bei IoU/Accuracy/F1, Test geringfügig besser
insgesamt. Unterschiede sind klein; keine statistische Absicherung oder Aussage
zur allgemeinen Überlegenheit. Mehrere Seeds wurden noch nicht untersucht.
Verdeckung und Zeitunterschiede sind mögliche Einflussfaktoren, keine belegten
Ursachen des Ergebnisses. Baseline bleibt Referenz und unverändert verfügbar.

Bei impervious höherer Recall, niedrigere Precision: mehr Referenzpunkte dieser
Klasse erkannt, zugleich mehr andere Referenzpunkte als impervious vorhergesagt.

Geplant: QGIS-Vergleich beider Batch-24-Modelle auf denselben sechs bereits früher
ausgewählten Validierungstiles. Die frühere QGIS-Auswahl stammt vom Batch-8-Modell
(Epoche 38, Exportjob 1065048); dieses Mal werden beide Batch-24-Modelle neu auf
diesen Tiles inferiert. Keine direkte Gegenüberstellung alter Batch-8-Vorhersagen
mit neuen RGB-Vorhersagen. Auswahl war drei Tiles mit vielen Baumkronen-/impervious-
Fehlern und drei zufällige Tiles; qualitative Sichtung, keine repräsentative
Genauigkeitsstichprobe. Exportskripte vorbereitet; tatsächlicher Export noch offen.

Danach separate Version mit RGB_valid und klar definierter, zu prüfender
Verdeckungsregel erwägen. Keine nachträgliche Änderung der Baseline oder RGB v1.

## Ablage auf dem Cluster

Gemeinsamer Präfix: `/cfs/earth/scratch/troxlflo/BA/`

| Artefakt | Pfad relativ zum Präfix |
|---|---|
| Baseline-Daten | data/processed/waedenswil_pointnet_split_B_v1 |
| RGB-Daten | data/processed/waedenswil_pointnet_split_B_rgb_v1 |
| Orthofotos | data/raw/swissimage_2025_rgb |
| Baseline-Training | runs/train_split_B_b24_1065610 |
| RGB-Training | runs/train_split_B_rgb_v1_1073682 |
| Baseline-Test | runs/test_split_B_1065611 |
| RGB-Test | runs/test_split_B_rgb_v1_1073698 |
| Frühere QGIS-Auswahl | data/qa/pointnet_qgis_1065048/selected_tiles.csv |

Jeweils best_checkpoint.pth und last_checkpoint.pth im Trainingsordner erhalten.
Quellcode, Dokumentation und kleine Ergebnisdateien werden versioniert; grosse
Punktarrays, Orthofotos und Checkpoints bleiben ausserhalb von Git. Git ersetzt
keine Sicherung dieser grossen Dateien. RGB-Export protokolliert Bild-/Labelhashes;
Trainingslauf speichert Konfiguration und Quellcodeschnappschüsse;
Testauswertung protokolliert den Checkpoint-SHA-256.

Diese Dokumentation wurde aus den im Chat bereitgestellten Jobausgaben erstellt.
Vorbereitete lokale Dateien sind erst nach Übertragung und Commit auf dem Cluster
im Benutzer-Repository versioniert; kein Commit oder Push wurde durch den
Assistenten auf dem Cluster ausgeführt.
