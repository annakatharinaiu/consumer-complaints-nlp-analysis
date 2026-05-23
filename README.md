# consumer-complaints-nlp-analysis

Eine umfassende NLP-basierte Analyse von Verbraucherbeschwerden mit fortgeschrittenen Text-Preprocessing-Techniken, Vektorisierungsmethoden und Topic-Modellierung.

## Projektbeschreibung

In diesem Projekt wurde eine detaillierte Analyse von unstrukturierten Verbraucherbeschwerden aus dem Kaggle-Datensatz "Consumer Complaints Dataset for NLP" durchgeführt. Die Analyse umfasst folgende Arbeitsschritte:

- **Datenqualitätsprüfung**: Erkennung und Handling von fehlenden Werten, Duplikaten und Anomalien
- **Beschwerdenpräprozessierung**: Bereinigung von Texten, Normalisierung und standardisierte Tokenisierung
- **Mustererkennung**: Automatische Identifikation von Beschwerdethemen und -kategorien mittels Machine Learning
- **Erkenntnisextraktion**: Statistische Analysen und Visualisierungen der Beschwerdethemen

Die Analyse nutzt eine Kombination aus klassischen NLP-Techniken und Topic-Modeling-Algorithmen (LDA und NMF), um strukturierte Erkenntnisse aus unstrukturierten Texten zu gewinnen.

## Durchgeführte Analysen

### 1. Dateneinlauf und Exploration

Der CSV-Datensatz mit Verbraucherbeschwerden wurde geladen und einer explorativen Datenanalyse (EDA) unterzogen. Dabei wurden:
- Die Struktur und Größe des Datensatzes erfasst
- Fehlende Werte, Duplikate und Anomalien identifiziert
- Datentypen überprüft und validiert

### 2. Textbereinigung und Normalisierung

Die Beschwerdentexte wurden umfassend bereinigt:
- Entfernung von Sonderzeichen, URLs und HTML-Tags
- Konvertierung zu Kleinbuchstaben
- Entfernung von Zahlen und unnötigen Leerzeichen
- Normalisierung von Umlauten und Sonderzeichen

### 3. Tokenisierung und Lemmatisierung mit spaCy

Die bereinigten Texte wurden weiterverarbeitet:
- Aufteilung der Texte in einzelne Tokens (Wörter/Phrasen)
- Entfernung von Stoppwörtern (z.B. "der", "die", "und")
- **Lemmatisierung**: Reduktion von Wörtern auf ihre Grundform (z.B. "Beschwerden" → "Beschwerde")
- Extraktion relevanter Wortarten (Nomen, Verben, Adjektive) durch das Modell `en_core_web_sm`

### 4. Feature-Extraktion und Vektorisierung

Zur Umwandlung der Texte in numerische Formate wurden zwei Methoden angewendet:

#### CountVectorizer
- Konvertierung der Texte zu einer Häufigkeitsmatrix
- Zählung der Wortfrequenzen für jedes Dokument
- Bildung der Grundlage für die LDA-Modellierung

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Gewichtung der Wörter nach ihrer Relevanz im gesamten Datensatz
- Häufige Wörter über viele Dokumente erhielten niedrigere Gewichte
- Besonders aussagekräftige Wörter erhielten höhere Gewichte

### 5. Topic-Modellierung

Es wurden zwei etablierte Algorithmen zur Themenextraktion angewendet:

#### LDA (Latent Dirichlet Allocation)
- Probabilistisches Modell zur Entdeckung verborgener Themen
- Identifikation von Wortmustern, die zusammen erscheinen
- Jede Beschwerde wurde als Mischung verschiedener latenter Themen modelliert
- Ergebnis: K latente Themen mit ihren charakteristischen Wörtern 

#### NMF (Non-negative Matrix Factorization)
- Faktorisierung der Dokument-Wort-Matrix in zwei kleinere Matrizen
- Direkte Interpretation: Jedes Thema wird durch seine aussagekräftigsten Wörter definiert
- Oft intuitivere Ergebnisse als LDA aufgrund der direkteren Interpretierbarkeit

### 6. Visualisierung und Auswertung

Die Analyseergebnisse wurden visualisiert und ausgewertet:
- Darstellung der Top-Wörter pro Thema
- Häufigkeitsverteilungen der Beschwerdekategorien
- Wordclouds zur visuellen Darstellung von Wortmustern
- Clustering-Ergebnisse und thematische Zusammenhänge
- Exportierte Dateien mit Themenzuweisungen und Metriken

## Technologie und Tools

Die Analyse wurde mit folgenden Python-Bibliotheken durchgeführt:

- **pandas** - Datenmanipulation und -analyse
- **numpy** - Numerische Berechnungen
- **scikit-learn** - Machine Learning (CountVectorizer, TfidfVectorizer, LDA, NMF)
- **spaCy** - NLP-Bibliothek für Tokenisierung und Lemmatisierung
- **matplotlib** - Visualisierung
- **seaborn** - Statistische Visualisierungen
- **wordcloud** - Wort-Cloud-Generierung

## Projektstruktur

```
consumer-complaints-nlp-analysis/
├── README.md                      # Projektdokumentation
├── requirements.txt               # Python-Abhängigkeiten
├── analysis.py                    # NLP-Analyseskript
├── data/
│   └── consumer_complaints.csv    # Verarbeiteter Datensatz
├── output/
│   ├── topics_lda.csv             # LDA-Analyseergebnisse
│   ├── topics_nmf.csv             # NMF-Analyseergebnisse
│   └── visualizations/            # Generierte Plots und Grafiken
└── notebooks/                     # Exploratory Data Analysis Notebooks
```

## Verwendete Dateien

### analysis.py
Das Hauptskript führt die gesamte NLP-Analysepipeline durch und enthält:
- Funktionen für Dateneinlauf und Validierung
- Text-Preprocessing-Pipeline
- Modelltraining für LDA und NMF
- Evaluierung und Visualisierung der Ergebnisse
- Command-Line-Interface für flexible Parameter

### requirements.txt
Dokumentiert alle für die Analyse verwendeten Python-Pakete mit Versionsinformationen, um Reproduzierbarkeit zu gewährleisten.

## Analyseergebnisse

Die Analyse hat folgende Ausgaben erzeugt:

### Themenzuweisungen
- **CSV-Dateien** mit Themenzuweisungen für jedes Dokument
- **Wahrscheinlichkeitsverteilungen** der Themen pro Beschwerde

### Visualisierungen
- **Themen-Wort-Matrizen** zeigen die relevantesten Wörter pro Thema
- **Wordclouds** visualisieren die Wortfrequenzen
- **Heatmaps** illustrieren Beziehungen zwischen Dokumenten und Themen

### Modell-Metriken
- **Coherence Score**: Misst die semantische Zusammenhängigkeit der identifizierten Themen
- **Perplexity**: Evaluiert die Generalisierungsfähigkeit der Modelle auf unsichtbare Daten

## Beispielhafte Themencluster

Die Analyse identifizierte verschiedene Themenschwerpunkte in den Beschwerdentexten:

- **Bankwesen & Konten**: bank, account, card, payment, balance, ...
- **Kreditberichte & Scoring**: credit, report, score, dispute, bureau, ...
- **Darlehen & Hypotheken**: loan, mortgage, interest, debt, payment, ...
- **Zahlungsprobleme**: payment, fee, charge, billing, transaction, ...

## Besonderheiten

- Die Analyse ist auf englische Texte optimiert
- Mindestens 1.000 Beschwerdedatensätze wurden für aussagekräftige Ergebnisse verarbeitet
- 10 verschiedene Themen wurden automatisch extrahiert und analysiert
- Sowohl probabilistische (LDA) als auch deterministische (NMF) Topic-Modeling-Ansätze wurden verglichen



## Autor

Anna Katharina

