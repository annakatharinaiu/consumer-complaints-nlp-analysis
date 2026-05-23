# consumer-complaints-nlp-analysis

NLP-based analysis of consumer complaint narratives using text preprocessing, CountVectorizer, TF-IDF, LDA, and NMF.

## Projektbeschreibung

Dieses Projekt analysiert unstrukturierte Verbraucherbeschwerden aus dem Kaggle-Datensatz "Consumer Complaints Dataset for NLP". Ziel ist es, die Datenqualität zu prüfen, Beschwerden vorzuverarbeiten und Textmodelle zur Themenextraktion zu erstellen.

## Setup

1. Python-Umgebung erstellen:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Verwendung

```bash
python analysis.py --input data/consumer_complaints.csv --text-column consumer_complaint_narrative --label-column category --topics 10 --plot
```

Die wichtigsten Schritte im Skript sind:

- Daten einlesen und fehlende Einträge prüfen
- Textbereinigung und Normalisierung
- Tokenisierung, Entfernen von Stoppwörtern und Lemmatisierung mit spaCy
- Vektorisierung mittels CountVectorizer und TfidfVectorizer
- Themenmodellierung mit LDA und NMF

## Dateien

- `analysis.py`: Hauptskript zur Analyse der Beschwerdetexte
- `requirements.txt`: benötigte Python-Pakete
