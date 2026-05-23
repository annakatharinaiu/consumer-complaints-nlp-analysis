import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Bereinigt Rohtexte:
    - Kleinschreibung
    - Entfernen von URLs
    - Entfernen von Sonderzeichen und Zahlen
    - Entfernen mehrfacher Leerzeichen
    """
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Lädt das englische spaCy-Modell.
    Falls es fehlt, wird eine verständliche Fehlermeldung ausgegeben.
    """
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"Das spaCy-Modell '{model_name}' ist nicht installiert.\n"
            f"Installiere es mit folgendem Befehl:\n\n"
            f"python -m spacy download {model_name}"
        ) from exc


def preprocess_documents(df: pd.DataFrame, text_column: str, nlp) -> pd.DataFrame:
    """
    Erstellt eine bereinigte Textspalte 'clean_text':
    - Grundbereinigung mit Regex
    - Tokenisierung mit spaCy
    - Lemmatisierung
    - Entfernen von Stoppwörtern
    """
    df = df.copy()

    df[text_column] = df[text_column].astype(str).apply(clean_text)
    df = df[df[text_column].str.len() > 5].reset_index(drop=True)

    cleaned_texts = []

    for doc in nlp.pipe(df[text_column].tolist(), batch_size=64):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
        ]
        cleaned_texts.append(" ".join(tokens))

    df["clean_text"] = cleaned_texts
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)

    return df


def get_top_terms(vectorizer, matrix, top_n: int = 20) -> pd.DataFrame:
    """
    Gibt die wichtigsten Begriffe einer Vektorisierung zurück.
    Bei CountVectorizer sind das häufige Begriffe.
    Bei TF-IDF sind das gewichtete, charakteristische Begriffe.
    """
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.sum(axis=0)).ravel()

    result = pd.DataFrame({
        "term": terms,
        "score": scores
    })

    return result.sort_values(by="score", ascending=False).head(top_n)


def get_topics(model, vectorizer, n_top_words: int = 10) -> pd.DataFrame:
    """
    Extrahiert die wichtigsten Wörter je Topic aus einem Topic-Modell.
    """
    terms = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [terms[i] for i in top_indices]

        topics.append({
            "topic": f"Topic {topic_idx + 1}",
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topics)


def plot_class_distribution(df: pd.DataFrame, label_column: str, output_path: Path) -> None:
    """
    Erstellt ein Balkendiagramm zur Verteilung der Kategorien/Produkte.
    """
    counts = df[label_column].value_counts()

    plt.figure(figsize=(10, 6))
    counts.plot(kind="barh")
    plt.title("Class distribution")
    plt.xlabel("Number of complaints")
    plt.ylabel(label_column)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def detect_text_column(df: pd.DataFrame, requested_column: str) -> str:
    """
    Prüft, ob die angegebene Textspalte existiert.
    Falls nicht, werden typische Alternativen gesucht.
    """
    if requested_column in df.columns:
        return requested_column

    possible_columns = [
        "consumer_complaint_narrative",
        "complaint_text",
        "text",
        "complaint",
        "narrative"
    ]

    for column in possible_columns:
        if column in df.columns:
            print(f"Hinweis: Textspalte '{requested_column}' nicht gefunden.")
            print(f"Stattdessen wird '{column}' verwendet.")
            return column

    raise ValueError(
        f"Keine passende Textspalte gefunden.\n"
        f"Gesuchte Spalte: {requested_column}\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLP analysis pipeline for consumer complaints"
    )

    parser.add_argument(
        "--input",
        default="data/sample_complaints.csv",
        help="Pfad zur CSV-Datei mit den Beschwerdetexten"
    )

    parser.add_argument(
        "--text-column",
        default="consumer_complaint_narrative",
        help="Name der Textspalte in der CSV-Datei"
    )

    parser.add_argument(
        "--label-column",
        default="category",
        help="Name der Kategorie- oder Produktspalte"
    )

    parser.add_argument(
        "--topics",
        type=int,
        default=3,
        help="Anzahl der zu extrahierenden Topics"
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximale Anzahl der Merkmale für die Vektorisierung"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Speichert ein Diagramm zur Klassenverteilung"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Ordner für Ergebnisdateien"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV-Datei wurde nicht gefunden: {input_path}")

    print("Starte Analyse...")
    print(f"Eingabedatei: {input_path}")
    print(f"Ausgabeordner: {output_dir}")

    df = pd.read_csv(input_path)

    print(f"Geladene Zeilen: {len(df)}")
    print(f"Spalten in der Datei: {list(df.columns)}")

    text_column = detect_text_column(df, args.text_column)

    if len(df) < args.topics:
        print(
            f"Hinweis: Die Anzahl der Topics ({args.topics}) ist größer als die Anzahl der Texte ({len(df)}). "
            f"Die Topic-Anzahl wird auf {len(df)} reduziert."
        )
        args.topics = len(df)

    nlp = load_spacy_model()
    df_cleaned = preprocess_documents(df, text_column, nlp)

    if df_cleaned.empty:
        raise ValueError("Nach der Textbereinigung sind keine verwertbaren Texte übrig.")

    print(f"Zeilen nach Bereinigung: {len(df_cleaned)}")

    cleaned_output_path = output_dir / "cleaned_complaints.csv"
    df_cleaned.to_csv(cleaned_output_path, index=False)

    count_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=1,
        max_features=args.max_features
    )

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
        max_features=args.max_features
    )

    count_matrix = count_vectorizer.fit_transform(df_cleaned["clean_text"])
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned["clean_text"])

    top_count_terms = get_top_terms(count_vectorizer, count_matrix, top_n=20)
    top_tfidf_terms = get_top_terms(tfidf_vectorizer, tfidf_matrix, top_n=20)

    top_count_terms.to_csv(output_dir / "top_terms_count.csv", index=False)
    top_tfidf_terms.to_csv(output_dir / "top_terms_tfidf.csv", index=False)

    lda = LatentDirichletAllocation(
        n_components=args.topics,
        random_state=42,
        max_iter=20
    )

    nmf = NMF(
        n_components=args.topics,
        random_state=42,
        max_iter=500
    )

    lda.fit(count_matrix)
    nmf.fit(tfidf_matrix)

    lda_topics = get_topics(lda, count_vectorizer, n_top_words=10)
    nmf_topics = get_topics(nmf, tfidf_vectorizer, n_top_words=10)

    lda_topics.to_csv(output_dir / "topics_lda.csv", index=False)
    nmf_topics.to_csv(output_dir / "topics_nmf.csv", index=False)

    if args.plot and args.label_column in df_cleaned.columns:
        plot_class_distribution(
            df_cleaned,
            args.label_column,
            output_dir / "class_distribution.png"
        )

    print("\nAnalyse erfolgreich abgeschlossen.")
    print("\nErzeugte Dateien:")
    print(f"- {output_dir / 'cleaned_complaints.csv'}")
    print(f"- {output_dir / 'top_terms_count.csv'}")
    print(f"- {output_dir / 'top_terms_tfidf.csv'}")
    print(f"- {output_dir / 'topics_lda.csv'}")
    print(f"- {output_dir / 'topics_nmf.csv'}")

    if args.plot and args.label_column in df_cleaned.columns:
        print(f"- {output_dir / 'class_distribution.png'}")

    print("\nKurzinterpretation:")
    print("CountVectorizer zeigt häufig vorkommende Begriffe.")
    print("TF-IDF hebt charakteristische Begriffe einzelner Texte stärker hervor.")
    print("LDA extrahiert probabilistische Themen auf Basis von Worthäufigkeiten.")
    print("NMF extrahiert Themen auf Basis der TF-IDF-Gewichtung.")


if __name__ == "__main__":
    main()
