# Consumer Complaints NLP Analysis

Eine NLP-basierte Analyse von Verbraucherbeschwerden mit Python. Ziel des Projekts ist es, Beschwerdetexte automatisch zu laden, zu bereinigen, zu vektorisieren und mit Topic-Modeling-Verfahren häufige Themen zu extrahieren.

Der Datensatz wird automatisch über Kaggle geladen:

```python
import kagglehub

path = kagglehub.dataset_download("shashwatwork/consume-complaints-dataset-fo-nlp")
```

Alternativ kann auch eine lokale CSV-Datei verwendet werden.

---

## Projektziel

Dieses Projekt erfüllt folgende Anforderungen:

- Einrichtung eines offenen GitHub-Repositories zur Versionskontrolle
- Dokumentation der benötigten Python-Abhängigkeiten in `requirements.txt`
- Laden eines Consumer-Complaints-Datensatzes in eine Python-Umgebung
- Bereinigung der Beschwerdetexte zu analysierbaren Textdaten
- Anwendung von mindestens zwei Vektorisierungsmethoden:
  - CountVectorizer
  - TF-IDF
- Anwendung von mindestens zwei semantischen Analysetechniken:
  - LDA Topic Modeling
  - NMF Topic Modeling
- Export der Analyseergebnisse in den Ordner `output/`

---

## Projektbeschreibung

In diesem Projekt werden unstrukturierte Verbraucherbeschwerden aus einem Kaggle-Datensatz analysiert. Die Analyse umfasst folgende Arbeitsschritte:

1. Laden der Daten über Kaggle oder aus einer lokalen CSV-Datei
2. Erkennen der relevanten Textspalte
3. Bereinigung der Beschwerdetexte
4. Umwandlung der Texte in numerische Merkmale
5. Vergleich von CountVectorizer und TF-IDF
6. Extraktion häufiger Themen mit LDA und NMF
7. Speicherung der Ergebnisse als CSV-Dateien im Ordner `output/`

Die Analyse dient als reproduzierbare NLP-Pipeline, mit der aus unstrukturierten Beschwerdetexten strukturierte Ergebnisse gewonnen werden können.

---

## Verwendeter Datensatz

Der verwendete Datensatz ist der Kaggle-Datensatz:

`shashwatwork/consume-complaints-dataset-fo-nlp`

Das Skript `analysis.py` lädt diesen Datensatz automatisch über `kagglehub`, wenn beim Start keine lokale CSV-Datei angegeben wird.

Beispiel:

```bash
python analysis.py --topics 5
```

Wird eine eigene CSV-Datei verwendet, kann diese über den Parameter `--input` angegeben werden.

Beispiel:

```bash
python analysis.py --input data/consumer_complaints.csv --topics 5
```

---

## Technologie und Tools

Die Analyse verwendet folgende Python-Bibliotheken:

- `pandas` für Datenverarbeitung
- `numpy` für numerische Berechnungen
- `scikit-learn` für Vektorisierung und Topic Modeling
- `matplotlib` für optionale Diagramme
- `kagglehub` zum automatischen Laden des Kaggle-Datensatzes

---

## Projektstruktur

```text
consumer-complaints-nlp-analysis/
├── README.md
├── requirements.txt
├── analysis.py
└── output/
    ├── cleaned_complaints.csv
    ├── top_terms_count.csv
    ├── top_terms_tfidf.csv
    ├── topics_lda.csv
    ├── topics_nmf.csv
    ├── method_comparison.csv
    └── class_distribution.png
```

Der Ordner `output/` wird automatisch erzeugt, sobald das Skript ausgeführt wird.

---

## Installation

Zuerst müssen die benötigten Python-Bibliotheken installiert werden:

```bash
python -m pip install -r requirements.txt
```

Falls `python` nicht funktioniert, kann alternativ verwendet werden:

```bash
python3 -m pip install -r requirements.txt
```

Die Datei `requirements.txt` enthält die benötigten Abhängigkeiten:

```text
pandas
numpy
scikit-learn
matplotlib
kagglehub
```

---

## Hinweis zur Ausführung in GitHub Codespaces

Das Skript sollte in GitHub Codespaces über das Terminal ausgeführt werden, nicht über den kleinen Play-Button von VS Code/Codespaces.

Der Play-Button kann eine andere Python-Umgebung verwenden. Dadurch können Fehlermeldungen wie diese entstehen:

```text
ModuleNotFoundError: No module named 'numpy'
```

Das bedeutet nicht zwingend, dass der Code falsch ist. Es bedeutet meistens nur, dass der Play-Button einen anderen Python-Interpreter nutzt als das Terminal.

Deshalb sollte die Analyse über das Terminal gestartet werden.

---

## Ausführung mit Kaggle-Datensatz

Wenn keine lokale CSV-Datei angegeben wird, lädt das Skript automatisch den Kaggle-Datensatz.

```bash
python analysis.py --topics 5
```

Optional mit Diagramm:

```bash
python analysis.py --topics 5 --plot
```

Falls `python` nicht funktioniert:

```bash
python3 analysis.py --topics 5
```

oder mit Diagramm:

```bash
python3 analysis.py --topics 5 --plot
```

---

## Ausführung mit lokaler CSV-Datei

Alternativ kann eine eigene CSV-Datei verwendet werden:

```bash
python analysis.py --input data/consumer_complaints.csv --topics 5
```

Falls die Textspalte einen bestimmten Namen hat, kann sie manuell angegeben werden:

```bash
python analysis.py --input data/consumer_complaints.csv --text-column "Consumer complaint narrative" --topics 5
```

Falls zusätzlich eine Kategorie- oder Produktspalte für ein Diagramm verwendet werden soll:

```bash
python analysis.py --input data/consumer_complaints.csv --text-column "Consumer complaint narrative" --label-column "Product" --topics 5 --plot
```

---

## Ablauf der Analyse

Das Skript `analysis.py` führt folgende Schritte durch:

### 1. Laden der Daten

Die Daten werden entweder automatisch über Kaggle oder aus einer lokalen CSV-Datei geladen.

### 2. Erkennen der Textspalte

Das Skript versucht automatisch, eine passende Textspalte zu erkennen. Unterstützt werden typische Spaltennamen wie:

- `consumer_complaint_narrative`
- `Consumer complaint narrative`
- `complaint_text`
- `text`
- `narrative`
- `description`

Falls die automatische Erkennung nicht funktioniert, kann die Textspalte manuell über `--text-column` angegeben werden.

### 3. Textbereinigung

Die Beschwerdetexte werden bereinigt. Dabei werden unter anderem:

- Großbuchstaben in Kleinbuchstaben umgewandelt
- URLs entfernt
- Sonderzeichen entfernt
- überflüssige Leerzeichen entfernt
- sehr kurze oder leere Texte entfernt

Die bereinigten Texte werden in der Spalte `clean_text` gespeichert.

### 4. Vektorisierung

Zur Umwandlung der Texte in numerische Formate werden zwei Verfahren verwendet.

#### CountVectorizer

Der CountVectorizer zählt, wie häufig Wörter in den Texten vorkommen. Dadurch lassen sich häufige Begriffe im gesamten Textkorpus erkennen.

#### TF-IDF

TF-IDF gewichtet Wörter danach, wie charakteristisch sie für einzelne Texte sind. Begriffe, die in vielen Dokumenten häufig vorkommen, werden geringer gewichtet. Begriffe, die für einzelne Texte besonders typisch sind, erhalten ein höheres Gewicht.

### 5. Semantische Analyse / Topic Modeling

Zur Themenextraktion werden zwei Verfahren verwendet.

#### LDA

LDA steht für Latent Dirichlet Allocation. Das Verfahren erkennt Themen auf Basis gemeinsamer Wortverteilungen. Es arbeitet auf der CountVectorizer-Matrix.

#### NMF

NMF steht für Non-negative Matrix Factorization. Das Verfahren extrahiert Themen auf Basis der TF-IDF-Matrix. Die Ergebnisse sind häufig gut interpretierbar, weil die Themen direkt über gewichtete Begriffe beschrieben werden.

---

## Ergebnisdateien

Nach erfolgreicher Ausführung werden im Ordner `output/` folgende Dateien erzeugt:

- `cleaned_complaints.csv`
- `top_terms_count.csv`
- `top_terms_tfidf.csv`
- `topics_lda.csv`
- `topics_nmf.csv`
- `method_comparison.csv`
- `class_distribution.png`

### Bedeutung der Dateien

| Datei | Bedeutung |
|---|---|
| `cleaned_complaints.csv` | Datensatz mit bereinigter Textspalte `clean_text` |
| `top_terms_count.csv` | Häufigste Begriffe nach CountVectorizer |
| `top_terms_tfidf.csv` | Wichtigste Begriffe nach TF-IDF |
| `topics_lda.csv` | Extrahierte Themen mit LDA |
| `topics_nmf.csv` | Extrahierte Themen mit NMF |
| `method_comparison.csv` | Kurzer Vergleich der verwendeten Methoden |
| `class_distribution.png` | Optionales Diagramm zur Verteilung der Kategorien |

---

## Methodenvergleich

| Methode | Art | Grundlage | Ergebnis | Interpretation |
|---|---|---|---|---|
| CountVectorizer | Vektorisierung | Bereinigte Texte | Worthäufigkeiten | Zeigt häufig vorkommende Begriffe im gesamten Korpus. |
| TF-IDF | Vektorisierung | Bereinigte Texte | Gewichtete Begriffe | Hebt Begriffe hervor, die für einzelne Texte besonders charakteristisch sind. |
| LDA | Semantische Analyse / Topic Modeling | CountVectorizer-Matrix | Probabilistische Themen | Erkennt Themen auf Basis gemeinsamer Wortverteilungen. |
| NMF | Semantische Analyse / Topic Modeling | TF-IDF-Matrix | Faktorbasierte Themen | Liefert häufig gut interpretierbare Themen auf Basis gewichteter Begriffe. |

---

## Beispielhafte Interpretation

Die Analyse kann typische Themenbereiche in Verbraucherbeschwerden sichtbar machen, zum Beispiel:

- Bankkonten und Karten
- Kreditberichte und Bonität
- Darlehen und Hypotheken
- Zahlungsprobleme und Gebühren
- Beschwerden zu Abrechnungen oder Transaktionen

Die tatsächlichen Themen hängen vom verwendeten Datensatz, der Stichprobengröße und der gewählten Anzahl an Topics ab.

---

## Einschränkungen der Analyse

Die Analyse dient als technische Demonstration einer vollständigen NLP-Pipeline. Die Ergebnisse sind abhängig von der Größe und Qualität des verwendeten Datensatzes.

Bei kleineren Stichproben sind die extrahierten Themen nicht als endgültige fachliche Aussagen zu verstehen. Sie zeigen vor allem, dass die Daten erfolgreich geladen, bereinigt, vektorisiert und semantisch analysiert wurden.

Für belastbare fachliche Aussagen wäre eine größere und repräsentative Datenbasis erforderlich.

---

## Reproduzierbarkeit

Die Analyse ist reproduzierbar, weil:

- der Code in `analysis.py` versioniert ist
- die Abhängigkeiten in `requirements.txt` dokumentiert sind
- der Datensatz automatisch über `kagglehub` geladen werden kann
- die erzeugten Ergebnisdateien im Ordner `output/` gespeichert werden

---

## Beispielhafter Start in GitHub Codespaces

```bash
python -m pip install -r requirements.txt
python analysis.py --topics 5 --plot
```

Falls es mit `python` nicht funktioniert:

```bash
python3 -m pip install -r requirements.txt
python3 analysis.py --topics 5 --plot
```

---

## Autorin

Anna Katharina Schedler 