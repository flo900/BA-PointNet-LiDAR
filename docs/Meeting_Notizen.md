# Bachelorarbeit – Meeting-Notizen und offene Fragen

**Projekt:** Landbedeckungsklassifikation Wädenswil mit LiDAR / PointNet++  
**Betreuung:** Dr. Johann Junghardt  
**Angelegt:** 13.10.2026  
**Zweck:** Offene Fragen, Besprechungsergebnisse und daraus folgende Entscheidungen nachvollziehbar festhalten.

## Offene Fragen für das nächste Meeting

### 1. Gründacherkennung: widersprüchliche Auswahlkriterien
- **Quelle:** *Abschlussbericht Landbedeckungsklassifikation Basel-Landschaft*, Kapitel 4, S. 15 (Workflow-Abbildung) und S. 18 (Methodentext).
- **Beobachtung:** Im Workflow-Schema steht für potenzielle Flachdächer eine **Dachneigung < 3 %**. Im Methodentext wird stattdessen ein **Höhenunterschied < 1 m** innerhalb der Dachfläche genannt.
- **Frage an Johann:** Welches Kriterium wurde tatsächlich angewendet? Warum wurde es gewählt? Beziehen sich beide Angaben auf unterschiedliche Verarbeitungsschritte?
- **Antwort:** _Noch offen_
- **Konsequenz / Entscheidung:** _Noch offen_

## Methodische Festlegungen (vorläufig)

- **Untersuchungsgebiet:** Wädenswil (ZH).
- **Ziel:** Bestehende GIS-Klassifikation versiegelt/unversiegelt gezielt innerhalb unsicherer AV-Bodenbedeckungskategorien durch LiDAR ergänzen; AV nicht vollständig ersetzen.
- **AV-Einteilung:** Die Experteneinteilung aus Tabelle 1 des Basel-Landschaft-Berichts vorläufig auf Wädenswil übertragen. Regionale Übertragbarkeit in der Bachelorarbeit ausdrücklich als Annahme diskutieren.
- **Inferenz:** PointNet++ zunächst auf die gesamte verfügbare LiDAR-Punktwolke anwenden; bei der Erstellung der finalen Karte nur Ergebnisse innerhalb ausgewählter unsicherer AV-Kategorien übernehmen.
- **Trainingsmodell:** Bestehendes Fünfklassenmodell vorerst als Ausgangspunkt behalten. Binäre Labels und mögliche Nachtrainingsstrategie erst nach Prüfung problematischer AV-Kategorien festlegen.
- **Evaluationshinweis:** Unabhängige manuelle Referenzflächen anhand vorhandener Daten vorsehen; keine eigene Befliegung geplant.
- **Methodischer Prüfpunkt:** Zweimaliger Majority Filter des NDVI-Workflows könnte kleine reale Versiegelungsflächen entfernen; bei der Evaluation gezielt prüfen.

## Besprechungsprotokolle

### Nächstes Meeting – Datum noch offen
- **Teilnehmende:** _Eintragen_
- **Antworten und Entscheidungen:** _Nach dem Gespräch ergänzen_
- **Aufgaben / Verantwortliche / Fristen:** _Nach dem Gespräch ergänzen_

## Neue Frage hinzufügen – Vorlage

### N. Kurztitel
- **Quelle / Kontext:**
- **Beobachtung:**
- **Frage:**
- **Antwort:** _Noch offen_
- **Konsequenz / Entscheidung:** _Noch offen_
