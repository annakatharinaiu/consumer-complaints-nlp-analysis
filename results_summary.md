# Ergebnisse der Analyse

## Setup

- Python-Umgebung wurde mit `python3 -m venv .venv` erstellt.
- Abhängigkeiten installiert mit `pip install -r requirements.txt`.
- Eine `environment.yml`-Datei wurde für Conda-kompatible Umgebungen erstellt.

## Daten

- Textdaten werden über Kaggle geladen oder aus einer lokalen CSV-Datei bereitgestellt.
- Die Datei enthält narrative Beschwerdetexte und optionale Kategorialspalten.
- Nach der Datenreinigung bleiben nur Texte mit mindestens 10 Zeichen erhalten.
- Die Klassenverteilung ist je nach Datensatz unterschiedlich.

## Vorverarbeitung

Die Textbereinigung umfasst folgende Schritte:

- **Kleinschreibung**: Alle Großbuchstaben werden in Kleinbuchstaben umgewandelt.
- **Entfernung von URLs**: Hyperlinks werden entfernt.
- **Entfernung von Sonderzeichen und Zahlen**: Nur alphabetische Zeichen und Leerzeichen bleiben erhalten.
- **Bereinigung mehrfacher Leerzeichen**: Mehrere aufeinanderfolgende Leerzeichen werden zu einem Leerzeichen reduziert.
- **Entfernung von Stopwords**: Englische Stopwords werden über scikit-learn bei der Vektorisierung entfernt.

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

### Beobachtungen basierend auf den erzeugten Ergebnissen

**CountVectorizer** zeigt eine klare Baseline der häufigsten Wörter im gesamten Korpus. Die Top-Begriffe sind: `credit` (10.091 Vorkommen), `account` (9.800), `report` (5.693), `payment` (3.660), `debt` (2.265), `loan` (2.169), `card` (2.113) und `bank` (1.917). Diese Verteilung offenbart, dass Verbraucherbeschwerden hauptsächlich um Kreditberichte, Konten, Zahlungen und Schuldenfragen kreisen.

**TF-IDF** hebt speziellere und differenzierendere Begriffe hervor. Neben den bereits häufigen Begriffen erscheinen charakteristische Terme wie `identity` (108,6), `bureau` (105,2) und `item` (117,7), was darauf hindeutet, dass Identitätsdiebstahl und Fehler in Kreditberichten zentrale Beschwerdethemen sind. TF-IDF gewichtet diese selteneren Begriffe höher, obwohl sie insgesamt weniger häufig vorkommen.

**LDA** identifiziert breitere, probabilistische Themenbereiche. Die 10 Topics decken Themenkomplexe ab wie: (1) Kontoverwaltung und PayPal-Transaktionen, (2) Bank- und Kartenbetrug, (3) Kreditberichte und Bonitätsfragen, (4) Telefonische Kommunikation über Zahlungen, (5) Kreditschulden und Inkassoverfahren, (6) Kontodispute und Betrug, (7) Hypotheken und Darlehen, (8) Inkassopraxis und rechtliche Compliance, (9) Verbraucherschutzangelegenheiten, und (10) Identitätsdiebstahl und betrügerische Einträge. LDA erlaubt Überlappungen zwischen Topics.

**NMF** liefert fokussiertere, oft distinctere Topics mit klareren Abgrenzungen. Die Topics konzentrieren sich auf: Kreditbericht-Fehler, Bankgebühren und Betrug, betrügerische Konten, Inkasso, Streitbeilegung, Identitätsdiebstahl, Kreditzahlungen und Hypotheken, unbefugte Kreditanfragen, unbekannte Konten und Kontodispute. NMF erzeugt thematisch stärker segmentierte Ergebnisse.

Die Ergebnisse demonstrieren, dass bereits mit klassischen NLP-Methoden aus unstrukturierten Beschwerdetexten strukturierte Einblicke gewonnen werden können. Die Qualität und Aussagekraft der Topics hängen stark von der Größe und Repräsentativität des Datensatzes ab. Für vollständigere fachliche Analysen wird ein größerer Datensatz und eine Validierung durch Experten empfohlen.
