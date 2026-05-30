# Ergebnisse der Beispielanalyse

## Setup

- Python-Umgebung wurde mit `python3 -m venv .venv` erstellt.
- Abhängigkeiten installiert mit `pip install -r requirements.txt`.
- Eine `environment.yml`-Datei wurde für Conda-kompatible Umgebungen erstellt.

## Daten

- Beispielhafte Textdaten werden in `data/sample_complaints.csv` bereitgestellt oder über Kaggle geladen.
- Die Datei enthält narrative Beschwerdetexte und eine optionale Kategorialspalte.
- Nach der Datenreinigung bleiben nur Texte mit mindestens 10 Zeichen erhalten.
- Die Klassenverteilung ist je nach Datensatz unterschiedlich.

## Vorverarbeitung

Die Textbereinigung umfasst folgende Schritte:

- **Kleinschreibung**: Alle Großbuchstaben werden in Kleinbuchstaben umgewandelt.
- **Entfernung von URLs**: Hyperlinks werden entfernt.
- **Entfernung von Sonderzeichen und Zahlen**: Nur alphabetische Zeichen und Leerzeichen bleiben erhalten.
- **Bereinigung mehrfacher Leerzeichen**: Mehrere aufeinanderfolgende Leerzeichen werden zu einem Leerzeichen reduziert.

Die bereinigten Texte werden in der Spalte `clean_text` gespeichert.

## Vektorisierung

### CountVectorizer

Der `CountVectorizer` zählt, wie häufig Wörter in jedem Dokument vorkommen. 
- **Eingabe**: Bereinigte Texte
- **Ausgabe**: Häufigkeiten von Wortvorkommnissen
- **Stopwords**: Englische Stopwords werden über scikit-learn entfernt.
- **Parameter**: `max_df=0.95`, `min_df=2`, `max_features=5000`

### TF-IDF

TF-IDF (Term Frequency – Inverse Document Frequency) gewichtet Wörter danach, wie charakteristisch sie für einzelne Texte sind.
- **Eingabe**: Bereinigte Texte
- **Ausgabe**: Gewichtete Begriffe pro Dokument
- **Stopwords**: Englische Stopwords werden über scikit-learn entfernt.
- **Parameter**: `max_df=0.95`, `min_df=2`, `max_features=5000`

**Unterschied zu CountVectorizer**: TF-IDF hebt Begriffe hervor, die in einzelnen Dokumenten häufig, aber insgesamt selten sind. Dies ermöglicht die Identifikation spezifischer Themenbereiche.

## Topic Modeling

### LDA (Latent Dirichlet Allocation)

LDA identifiziert probabilistische Themendistributionen basierend auf gemeinsamen Wortmustern.
- **Eingabe**: CountVectorizer-Matrix
- **Ausgabe**: Topics mit den wahrscheinlichsten Wörtern pro Topic
- **Charakteristik**: Erzeugt eher **breite, probabilistische Themenbereiche**. Ein Wort kann mit verschiedenen Stärken in mehreren Topics vorkommen.

### NMF (Non-negative Matrix Factorization)

NMF faktorisiert die TF-IDF-Matrix in eine Dokument-Topic- und eine Topic-Begriff-Matrix.
- **Eingabe**: TF-IDF-Matrix
- **Ausgabe**: Topics mit den gewichtigsten Wörtern pro Topic
- **Charakteristik**: Liefert häufig **klarer interpretierbare, fokussiertere Topics** mit deutlich unterschiedlichen Wortmengen pro Topic.

## Ergebnisdiskussion

Die Analyse erzeugt folgende Output-Dateien im Ordner `output/`:

- `cleaned_complaints.csv`: Datensatz mit bereinigten Texten
- `top_terms_count.csv`: Die 20 häufigsten Begriffe aus CountVectorizer
- `top_terms_tfidf.csv`: Die 20 wichtigsten Begriffe aus TF-IDF
- `topics_lda.csv`: Extrahierte Topics mit LDA
- `topics_nmf.csv`: Extrahierte Topics mit NMF
- `method_comparison.csv`: Tabellarischer Vergleich der verwendeten Methoden
- `class_distribution.png`: Optionales Balkendiagramm der Kategorienhäufigkeit

### Beobachtungen

1. **CountVectorizer** zeigt die Baseline der häufigsten Wörter im gesamten Korpus.
2. **TF-IDF** hebt speziellere und differenzierendere Begriffe hervor, die typisch für einzelne Beschwerden sind.
3. **LDA** identifiziert breitere, probabilistische Themenbereiche mit überlappenden Wortkonstellationen.
4. **NMF** liefert fokussiertere, oft distinctere Topics mit klareren Abgrenzungen zwischen den Themen.

Die Ergebnisse demonstrieren, dass bereits mit klassischen NLP-Methoden aus unstrukturierten Beschwerdetexten strukturierte Einblicke gewonnen werden können. Die Qualität und Aussagekraft der Topics hängen stark von der Größe, Qualität und Diversity des Datensatzes sowie von den gewählten Parametern (z. B. `min_df`, `max_df`, Anzahl Topics) ab.

Für vollständigere Analysen wird ein größerer und repräsentativer Datensatz empfohlen.
