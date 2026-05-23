import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


KAGGLE_DATASET = "shashwatwork/consume-complaints-dataset-fo-nlp"


def clean_text(text: str) -> str:
    """
    Bereinigt Rohtexte für die NLP-Analyse:
    - Kleinschreibung
    - Entfernen von URLs
    - Entfernen von Sonderzeichen und Zahlen
    - Entfernen mehrfacher Leerzeichen
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def download_kaggle_dataset() -> Path:
    """
    Lädt den Consumer-Complaints-Datensatz von Kaggle mit kagglehub herunter
    und sucht automatisch nach einer CSV-Datei.
    """
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Das Paket 'kagglehub' ist nicht installiert.\n"
            "Installiere zuerst die Abhängigkeiten mit:\n\n"
            "python -m pip install -r requirements.txt"
        ) from exc

    print("Lade Kaggle-Datensatz herunter...")
    print(f"Datensatz: {KAGGLE_DATASET}")

    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    csv_files = list(dataset_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Im heruntergeladenen Kaggle-Datensatz wurde keine CSV-Datei gefunden: {dataset_path}"
        )

    csv_file = csv_files[0]

    print(f"Gefundene CSV-Datei: {csv_file}")

    return csv_file


def load_dataset(input_path: str | None) -> pd.DataFrame:
    """
    Lädt entweder eine lokale CSV-Datei oder automatisch den Kaggle-Datensatz.
    """
    if input_path:
        csv_path = Path(input_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Die angegebene Eingabedatei wurde nicht gefunden: {csv_path}")
    else:
        csv_path = download_kaggle_dataset()

    print(f"Lade Daten aus: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Geladene Zeilen: {len(df)}")
    print(f"Gefundene Spalten: {list(df.columns)}")

    return df


def detect_text_column(df: pd.DataFrame, requested_column: str | None) -> str:
    """
    Erkennt automatisch die Textspalte.
    Falls der Nutzer eine Spalte über --text-column angibt, wird diese bevorzugt.
    """
    if requested_column and requested_column in df.columns:
        return requested_column

    possible_columns = [
        "consumer_complaint_narrative",
        "Consumer complaint narrative",
        "complaint_text",
        "Complaint Text",
        "narrative",
        "Narrative",
        "complaint",
        "Complaint",
        "text",
        "Text",
        "description",
        "Description"
    ]

    for column in possible_columns:
        if column in df.columns:
            if requested_column:
                print(f"Hinweis: Die gewünschte Textspalte '{requested_column}' wurde nicht gefunden.")
            print(f"Verwendete Textspalte: '{column}'")
            return column

    raise ValueError(
        "Es wurde keine passende Textspalte gefunden.\n"
        f"Vorhandene Spalten: {list(df.columns)}\n\n"
        "Starte das Skript erneut und gib die Textspalte manuell an, z. B.:\n"
        'python analysis.py --text-column "Consumer complaint narrative"'
    )


def detect_label_column(df: pd.DataFrame, requested_column: str | None) -> str | None:
    """
    Erkennt optional eine Kategorie-/Produktspalte für eine Verteilungsgrafik.
    """
    if requested_column and requested_column in df.columns:
        return requested_column

    possible_columns = [
        "product",
        "Product",
        "category",
        "Category",
        "issue",
        "Issue",
        "sub_product",
        "Sub-product",
        "Sub Product"
    ]

    for column in possible_columns:
        if column in df.columns:
            print(f"Verwendete Kategorie-/Labelspalte: '{column}'")
            return column

    print("Keine Kategorie-/Labelspalte gefunden. Diagramm wird bei Bedarf übersprungen.")
    return None


def preprocess_data(df: pd.DataFrame, text_column: str, sample_size: int | None = None) -> pd.DataFrame:
    """
    Bereinigt die Texte und reduziert den Datensatz optional auf eine Stichprobe.
    Die Stichprobe ist sinnvoll, damit LDA und NMF im Codespace nicht zu langsam werden.
    """
    df = df.copy()

    df = df[df[text_column].notna()].reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        print(f"Datensatz wird für die Analyse auf {sample_size} Zeilen reduziert.")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df["clean_text"] = df[text_column].apply(clean_text)

    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    if df.empty:
        raise ValueError("Nach der Textbereinigung sind keine verwertbaren Texte übrig.")

    print(f"Zeilen nach Textbereinigung: {len(df)}")

    return df


def get_top_terms(vectorizer, matrix, top_n: int = 20) -> pd.DataFrame:
    """
    Ermittelt die wichtigsten Begriffe aus einer Vektorisierung.
    """
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.sum(axis=0)).ravel()

    result = pd.DataFrame({
        "term": terms,
        "score": scores
    })

    return result.sort_values(by="score", ascending=False).head(top_n)


def get_topics(model, vectorizer, top_n: int = 10) -> pd.DataFrame:
    """
    Gibt die wichtigsten Wörter je Topic zurück.
    """
    terms = vectorizer.get_feature_names_out()
    topics = []

    for topic_index, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [terms[i] for i in top_indices]

        topics.append({
            "topic": f"Topic {topic_index + 1}",
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topics)


def save_class_distribution_plot(df: pd.DataFrame, label_column: str, output_path: Path) -> None:
    """
    Erstellt optional ein Balkendiagramm zur Verteilung der Kategorien.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib ist nicht installiert. Diagramm wird übersprungen.")
        return

    counts = df[label_column].value_counts().head(20)

    plt.figure(figsize=(10, 6))
    counts.sort_values().plot(kind="barh")
    plt.title("Class distribution")
    plt.xlabel("Number of complaints")
    plt.ylabel(label_column)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Diagramm gespeichert: {output_path}")


def save_method_comparison(output_dir: Path) -> None:
    """
    Speichert eine kurze methodische Gegenüberstellung als CSV.
    """
    comparison = pd.DataFrame([
        {
            "method": "CountVectorizer",
            "type": "Vektorisierung",
            "input": "Bereinigte Texte",
            "output": "Worthäufigkeiten",
            "interpretation": "Zeigt häufig vorkommende Begriffe im Korpus."
        },
        {
            "method": "TF-IDF",
            "type": "Vektorisierung",
            "input": "Bereinigte Texte",
            "output": "Gewichtete Begriffe",
            "interpretation": "Hebt Begriffe hervor, die für einzelne Texte besonders charakteristisch sind."
        },
        {
            "method": "LDA",
            "type": "Semantische Analyse / Topic Modeling",
            "input": "CountVectorizer-Matrix",
            "output": "Probabilistische Themen",
            "interpretation": "Findet Themen auf Basis gemeinsamer Wortverteilungen."
        },
        {
            "method": "NMF",
            "type": "Semantische Analyse / Topic Modeling",
            "input": "TF-IDF-Matrix",
            "output": "Faktorbasierte Themen",
            "interpretation": "Findet häufig gut interpretierbare Themen auf Basis gewichteter Begriffe."
        }
    ])

    comparison.to_csv(output_dir / "method_comparison.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLP analysis for consumer complaints using Kaggle or local CSV data"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Optionaler Pfad zu einer lokalen CSV-Datei. Wenn nichts angegeben wird, wird der Kaggle-Datensatz verwendet."
    )

    parser.add_argument(
        "--text-column",
        default=None,
        help="Name der Textspalte. Wenn nichts angegeben wird, wird sie automatisch erkannt."
    )

    parser.add_argument(
        "--label-column",
        default=None,
        help="Name der Kategorie-/Produktspalte. Wenn nichts angegeben wird, wird sie automatisch erkannt."
    )

    parser.add_argument(
        "--topics",
        type=int,
        default=5,
        help="Anzahl der Topics für LDA und NMF"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Maximale Anzahl der zu analysierenden Zeilen. Verhindert zu lange Laufzeiten im Codespace."
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximale Anzahl der Wörter/Merkmale für die Vektorisierung"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Ordner für Ergebnisdateien"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Erstellt zusätzlich ein Diagramm zur Klassenverteilung"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starte NLP-Analyse der Consumer Complaints...")
    print(f"Ausgabeordner: {output_dir}")

    df = load_dataset(args.input)

    text_column = detect_text_column(df, args.text_column)
    label_column = detect_label_column(df, args.label_column)

    df_cleaned = preprocess_data(
        df=df,
        text_column=text_column,
        sample_size=args.sample_size
    )

    cleaned_file = output_dir / "cleaned_complaints.csv"
    df_cleaned.to_csv(cleaned_file, index=False)

    print("Erstelle CountVectorizer-Matrix...")
    count_vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=args.max_features
    )

    print("Erstelle TF-IDF-Matrix...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=args.max_features
    )

    count_matrix = count_vectorizer.fit_transform(df_cleaned["clean_text"])
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned["clean_text"])

    if count_matrix.shape[1] == 0 or tfidf_matrix.shape[1] == 0:
        raise ValueError(
            "Die Vektorisierung hat keine verwertbaren Begriffe ergeben. "
            "Reduziere ggf. min_df oder prüfe die Textspalte."
        )

    top_count_terms = get_top_terms(count_vectorizer, count_matrix, top_n=20)
    top_tfidf_terms = get_top_terms(tfidf_vectorizer, tfidf_matrix, top_n=20)

    top_count_file = output_dir / "top_terms_count.csv"
    top_tfidf_file = output_dir / "top_terms_tfidf.csv"

    top_count_terms.to_csv(top_count_file, index=False)
    top_tfidf_terms.to_csv(top_tfidf_file, index=False)

    n_documents = count_matrix.shape[0]
    n_features = count_matrix.shape[1]
    max_possible_topics = min(n_documents, n_features)

    if max_possible_topics < 2:
        raise ValueError("Es sind nicht genug Daten für Topic Modeling vorhanden.")

    n_topics = min(args.topics, max_possible_topics)

    if n_topics != args.topics:
        print(f"Topic-Anzahl wird von {args.topics} auf {n_topics} reduziert.")

    print("Trainiere LDA-Modell...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="batch"
    )

    lda.fit(count_matrix)

    print("Trainiere NMF-Modell...")
    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=500,
        init="nndsvda"
    )

    nmf.fit(tfidf_matrix)

    lda_topics = get_topics(lda, count_vectorizer, top_n=10)
    nmf_topics = get_topics(nmf, tfidf_vectorizer, top_n=10)

    lda_file = output_dir / "topics_lda.csv"
    nmf_file = output_dir / "topics_nmf.csv"

    lda_topics.to_csv(lda_file, index=False)
    nmf_topics.to_csv(nmf_file, index=False)

    save_method_comparison(output_dir)

    if args.plot:
        if label_column and label_column in df_cleaned.columns:
            plot_file = output_dir / "class_distribution.png"
            save_class_distribution_plot(df_cleaned, label_column, plot_file)
        else:
            print("Kein Diagramm erstellt, weil keine Kategorie-/Labelspalte erkannt wurde.")

    print("\nAnalyse erfolgreich abgeschlossen.")

    print("\nErzeugte Dateien:")
    print(f"- {cleaned_file}")
    print(f"- {top_count_file}")
    print(f"- {top_tfidf_file}")
    print(f"- {lda_file}")
    print(f"- {nmf_file}")
    print(f"- {output_dir / 'method_comparison.csv'}")

    if args.plot and label_column:
        print(f"- {output_dir / 'class_distribution.png'}")

    print("\nErfüllte Anforderungen:")
    print("- Daten wurden über Kaggle oder lokale CSV-Datei in Python geladen.")
    print("- Texte wurden bereinigt und als clean_text gespeichert.")
    print("- CountVectorizer wurde als erste Vektorisierung verwendet.")
    print("- TF-IDF wurde als zweite Vektorisierung verwendet.")
    print("- LDA wurde als erste semantische Analysetechnik verwendet.")
    print("- NMF wurde als zweite semantische Analysetechnik verwendet.")
    print("- Ergebnisse wurden im output-Ordner gespeichert.")


if __name__ == "__main__":
    main()