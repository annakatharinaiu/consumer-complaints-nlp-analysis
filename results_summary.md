# Ergebnisse der Analyse

## Zusammenfassung

Das Projekt lädt Verbraucherbeschwerden automatisch über Kaggle oder aus einer lokalen CSV-Datei, bereinigt die Texte und führt klassische NLP-Analysen mit `scikit-learn` durch.

## Vorverarbeitung

Die folgenden Schritte wurden umgesetzt:

- Kleinschreibung aller Texte
- Entfernen von URLs
- Entfernen von Sonderzeichen
- Entfernen von Zahlen
- Bereinigung mehrfacher Leerzeichen
- Nutzung englischer Stopwords über `scikit-learn` bei der Vektorisierung

Bereinigte Texte werden in der Spalte `clean_text` gespeichert.

## Vektorisierung

- `CountVectorizer` zeigt die häufigsten Begriffe im gesamten Korpus.
- `TfidfVectorizer` hebt Begriffe hervor, die in einzelnen Dokumenten besonders charakteristisch sind.
- Die wichtigsten Begriffe werden in `output/top_terms_count.csv` und `output/top_terms_tfidf.csv` gespeichert.

Typische Begriffe in der Analyse sind z. B. `credit`, `account`, `report`, `payment`, `debt`, `loan`, `card` und `bank`.

## Topic Modeling

- `LDA` bildet eher breite, probabilistische Themencluster.
- `NMF` liefert oft klarer interpretierbare Themen anhand charakteristischer Begriffe.

Die Ergebnisse sind in `output/topics_lda.csv` und `output/topics_nmf.csv` dokumentiert.

## Ausgabe

Die Analyse erzeugt folgende Dateien im Ordner `output/`:

- `cleaned_complaints.csv`
- `top_terms_count.csv`
- `top_terms_tfidf.csv`
- `topics_lda.csv`
- `topics_nmf.csv`
- `method_comparison.csv`

## Fazit

Die Pipeline zeigt, dass mit klassischen NLP-Schritten und `scikit-learn` aus Verbraucherbeschwerden aussagekräftige Einblicke gewonnen werden können. CountVectorizer liefert häufige Begriffe, TF-IDF betont charakteristische Begriffe, LDA extrahiert breite Themen und NMF liefert oft klarer abgrenzbare Themen.

