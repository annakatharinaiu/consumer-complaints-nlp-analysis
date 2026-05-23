# consumer-complaints-nlp-analysis

Eine umfassende NLP-basierte Analyse von Verbraucherbeschwerden mit fortgeschrittenen Text-Preprocessing-Techniken, Vektorisierungsmethoden und Topic-Modellierung.

## Projektbeschreibung

Dieses Projekt analysiert unstrukturierte Verbraucherbeschwerden aus dem Kaggle-Datensatz "Consumer Complaints Dataset for NLP". Das Ziel besteht darin:

- **Datenqualität prüfen**: Erkennung und Handling von fehlenden Werten, Duplikaten und Anomalien
- **Beschwerden vorzuverarbeiten**: Bereinigung von Texten, Normalisierung und standardisierte Tokenisierung
- **Verborgene Muster erkennen**: Automatische Identifikation von Beschwerdethemen und -kategorien mittels Machine Learning
- **Erkenntnisse extrahieren**: Statistische Analysen und Visualisierungen der Beschwerdepflichtenthemen

Das Projekt nutzt eine Kombination aus klassischen NLP-Techniken und Topic-Modeling-Algorithmen (LDA und NMF), um strukturierte Erkenntnisse aus unstrukturierten Texten zu gewinnen.

## Funktionsweise

### 1. **Dateneinlauf und Exploration**
- Laden des CSV-Datensatzes mit den Verbraucherbeschwerden
- Exploratorische Datenanalyse (EDA) zur Identifikation der Datenstruktur
- Überprüfung auf fehlende Werte, Duplikate und Datentypüberprüfung

### 2. **Textbereinigung und Normalisierung**
- Entfernung von Sonderzeichen, URLs und HTML-Tags
- Konvertierung zu Kleinbuchstaben
- Entfernung von Zahlen und unnötigen Leerzeichen
- Normalisierung von Umlauten und Sonderzeichen

### 3. **Tokenisierung und Lemmatisierung (spaCy)**
- Aufteilung der Texte in einzelne Tokens (Wörter/Phrasen)
- Entfernung von Stoppwörtern (z.B. "der", "die", "und")
- **Lemmatisierung**: Reduktion von Wörtern auf ihre Grundform (z.B. "Beschwerden" → "Beschwerde")
- Extraktion relevanter Wortarten (Nomen, Verben, Adjektive)

### 4. **Feature-Extraktion (Vektorisierung)**

#### CountVectorizer
- Konvertierung von Text zu einer Häufigkeitsmatrix
- Zählt, wie oft jedes Wort in jedem Dokument vorkommt
- Basis für LDA-Modellierung

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Gewichtet Wörter nach ihrer Relevanz im gesamten Datensatz
- Häufige Wörter über viele Dokumente erhalten niedrigere Gewichte
- Besonders aussagekräftige Wörter erhalten höhere Gewichte

### 5. **Topic-Modellierung**

#### LDA (Latent Dirichlet Allocation)
- Probabilistisches Modell zur Entdeckung verborgener Themen
- Identifiziert, welche Wörter zusammen erscheinen
- Ord Beschwerde lässt sich als Mischung verschiedener Themen darstellen
- Ergebnis: K latente Themen mit ihre charakteristischen Wörtern

#### NMF (Non-negative Matrix Factorization)
- Faktorisiert die Dokument-Wort-Matrix in zwei kleinere Matrizen
- Interpretierbar: Jedes Thema wird durch seine aussagekräftigsten Wörter definiert
- Oft einfacher zu interpretieren als LDA

### 6. **Visualisierung und Auswertung**
- Darstellung der Top-Wörter pro Thema
- Häufigkeitsverteilungen der Beschwerdekategorien
- Wordclouds für visuelle Patterns
- Clustering-Ergebnisse und thematische Zusammenhänge

## Setup

### Voraussetzungen
- Python 3.8 oder höher
- pip (Python Package Manager)

### Schritt-für-Schritt Anleitung

1. **Python-Umgebung erstellen und aktivieren:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# oder für Windows:
# .venv\Scripts\activate
```

2. **Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Die `requirements.txt` sollte folgende Pakete enthalten:
- `pandas` - Datenmanipulation und -analyse
- `numpy` - Numerische Berechnungen
- `scikit-learn` - Machine Learning (CountVectorizer, TfidfVectorizer, LDA, NMF)
- `spacy` - NLP-Bibliothek für Tokenisierung und Lemmatisierung
- `matplotlib` - Visualisierung
- `seaborn` - Statistische Visualisierungen
- `wordcloud` - Wort-Cloud-Generierung

## Verwendung

### Basis-Aufruf:
```bash
python analysis.py --input data/consumer_complaints.csv --text-column consumer_complaint_narrative --label-column category --topics 10 --plot
```

### Parameter-Erklärung:
- `--input`: Pfad zur CSV-Datei mit Beschwerdedaten
- `--text-column`: Name der Spalte, die die Beschwerde-Texte enthält
- `--label-column`: Name der Spalte mit Kategorien/Labels (optional)
- `--topics`: Anzahl der zu extrahierenden Themen (z.B. 10)
- `--plot`: Flag zum Aktivieren von Visualisierungen

### Die wichtigsten Verarbeitungsschritte im Skript:

1. **Daten einlesen und validieren**
   - CSV-Datei laden
   - Fehlende Einträge prüfen und handhaben
   - Datentypen überprüfen

2. **Textbereinigung und Normalisierung**
   - Spezialzeichen und Markup entfernen
   - Whitespace normalisieren
   - Text in Kleinbuchstaben konvertieren

3. **Tokenisierung, Stoppwort-Entfernung und Lemmatisierung**
   - spaCy-Modell (`en_core_web_sm`) laden
   - Texte in Tokens zerlegen
   - Lemmas extrahieren und Stoppwörter filtern

4. **Feature-Extraktion**
   - CountVectorizer: Häufigkeitsmatrix erstellen
   - TfidfVectorizer: TF-IDF-Gewichte berechnen

5. **Topic-Modellierung**
   - LDA-Modell trainieren: Identifiziert probabilistische Themenverteilungen
   - NMF-Modell trainieren: Faktorisiert Dokument-Wort-Matrix
   - Top-Wörter pro Thema extrahieren

6. **Ergebnisse visualisieren und exportieren**
   - Thema-Wort-Beziehungen visualisieren
   - Themenzuweisungen pro Dokument speichern
   - Statistische Zusammenfassungen generieren

## Projektstruktur

```
consumer-complaints-nlp-analysis/
├── README.md                      # Diese Datei - Dokumentation
├── requirements.txt               # Python-Abhängigkeiten
├── analysis.py                    # Hauptskript zur NLP-Analyse
├── data/
│   └── consumer_complaints.csv    # Input-Datensatz (nicht versioniert)
├── output/
│   ├── topics_lda.csv             # LDA-Ergebnisse
│   ├── topics_nmf.csv             # NMF-Ergebnisse
│   └── visualizations/            # Generierte Plots und Grafiken
└── notebooks/                     # Jupyter Notebooks für Exploration (optional)
```

## Dateien im Detail

### `analysis.py`
Das Hauptskript zur Analyse der Beschwerdedaten. Enthält:
- Funktionen für Dateneinlauf und Validierung
- Text-Preprocessing-Pipeline
- Modelltraining (LDA und NMF)
- Evaluierung und Visualisierung
- Command-Line-Interface für flexible Parameter

### `requirements.txt`
Liste aller benötigten Python-Pakete mit Versionsinformationen. Ermöglicht reproduzierbare Umgebung:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
spacy>=3.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
```

## Erwartete Ausgaben

Nach erfolgreichem Durchlauf erzeugt das Skript:
- **CSV-Dateien** mit Themenzuweisungen und -verteilungen
- **Visualisierungen** (Plots, Wordclouds, Heatmaps)
- **Konsolen-Output** mit statistischen Zusammenfassungen
- **Modell-Metriken** (z.B. Coherence-Score, Perplexity)

## Beispielergebnisse

### LDA Top-Wörter pro Thema:
- **Thema 1**: ["bank", "account", "card", "payment", ...]
- **Thema 2**: ["credit", "report", "score", "dispute", ...]
- **Thema 3**: ["loan", "mortgage", "interest", "debt", ...]

### Metriken:
- **Coherence Score**: Misst die semantische Zusammenhängigkeit der Themen
- **Perplexity**: Evaluiert die Modellgüte auf Test-Daten

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.

## Autor

Anna Katharina

## Hinweise

- Der Datensatz sollte mindestens 1.000 Einträge für aussagekräftige Ergebnisse enthalten
- Die Anzahl der Themen (--topics) sollte experimentell bestimmt werden
- Für große Datensätze (>100.000 Zeilen) können die Verarbeitungszeiten erheblich sein
- Die NLP-Pipeline ist derzeit auf englische Texte optimiert

