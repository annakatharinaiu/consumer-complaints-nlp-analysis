# Ergebnisse der Beispielanalyse

## Setup

- Python-Umgebung wurde mit `python3 -m venv .venv` erstellt.
- Abhängigkeiten installiert mit `pip install -r requirements.txt`.
- Zusätzlich wurde das spaCy-Modell installiert: `python -m spacy download en_core_web_sm`.
- Eine `environment.yml`-Datei wurde für Conda-kompatible Umgebungen erstellt.

## Daten

- Beispielhafte Textdaten wurden in `data/sample_complaints.csv` angelegt.
- Die Datei enthält narrativ formulierte Beschwerdetexte und die Kategorie `category`.
- Nach der Datenreinigung blieben 8 Einträge erhalten.
- Die Klassenverteilung war leicht unausgewogen, aber repräsentativ für verschiedene Produktgruppen.

## Vorverarbeitung

- Textbereinigung: Kleinschreibung, Entfernen von URLs, Sonderzeichen und Ziffern.
- Tokenisierung und Lemmatisierung erfolgten mit `spaCy`.
- Stoppwörter wurden entfernt.

## Vektorisierung

- `CountVectorizer`: Zählte Wortvorkommen.
- `TfidfVectorizer`: Gewichtete Begriffe nach ihrer Trennschärfe.

### Beobachtungen

- Häufige Begriffe in beiden Repräsentationen waren `payment`, `account`, `bank`, `debt`, `call`.
- TF-IDF hob charakteristischere Begriffe wie `misapply`, `delinquent`, `unauthorized`, `investigate` hervor.

## Themenextraktion

- LDA und NMF wurden als semantische Techniken verwendet.
- LDA identifizierte Themen rund um:
  - Servicer- und Kontoprobleme
  - Inkasso-/Telefonbelästigung
  - Zins-/Kreditkartenprobleme
- NMF zeigte ähnliche Muster, betonte aber bei einzelnen Themen stärker Begriffe wie `unauthorized`, `misapply` und `investigate`.

## Ergebnisdiskussion

- Das Beispiel zeigt, dass einfache Vektorisierungsmethoden bereits den häufigsten Wortschatz sichtbar machen.
- TF-IDF eignet sich besser, um spezifische Probleme und typische Aussagen zu identifizieren.
- Beide Topic-Modelle liefern verwandte Themenstrukturen, wobei LDA tendenziell etwas breite, NMF etwas fokussiertere Themen anzeigt.
- Für den vollständigen Kaggle-Datensatz ist ein robusteres Tuning der Parameter (z. B. `min_df`, `max_df`, Anzahl Topics) empfehlenswert.
