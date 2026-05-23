import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def preprocess_documents(df: pd.DataFrame, text_column: str, model: spacy.language.Language) -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].astype(str).map(clean_text)
    df = df[df[text_column].str.len() > 10].reset_index(drop=True)

    docs = list(model.pipe(df[text_column].tolist(), batch_size=64))
    lemmatized = []
    for doc in docs:
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        lemmatized.append(" ".join(tokens))

    df["clean_text"] = lemmatized
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    return df


def describe_dataset(df: pd.DataFrame, text_column: str, label_column: str | None = None) -> None:
    print("Dataset description")
    print("-------------------")
    print(f"Total rows: {len(df)}")
    print(f"Empty text rows: {df[text_column].isna().sum()}\n")

    if label_column and label_column in df.columns:
        print("Class distribution")
        print(df[label_column].value_counts(dropna=False).to_string())
        print("\nRelative class distribution")
        print(df[label_column].value_counts(normalize=True, dropna=False).to_string())
        print()

    missing = df.isna().sum()
    print("Missing values by column")
    print(missing[missing > 0].to_string() if not missing.empty else "None")
    print()


def plot_class_distribution(df: pd.DataFrame, label_column: str, output_path: Path | None = None) -> None:
    counts = df[label_column].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, color="#4c72b0", orient="h")
    plt.title("Class distribution")
    plt.xlabel("Number of complaints")
    plt.ylabel(label_column)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved class distribution to {output_path}")
    else:
        plt.show()
    plt.close()


def build_vectorizers(texts, max_df=0.95, min_df=1, max_features=20000):
    count_vectorizer = CountVectorizer(ngram_range=(1, 1), max_df=max_df, min_df=min_df, max_features=max_features)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=max_df, min_df=min_df, max_features=max_features)
    count_matrix = count_vectorizer.fit_transform(texts)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return count_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer


def top_terms(vectorizer, matrix, top_n=20):
    terms = vectorizer.get_feature_names_out()
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    top_indices = frequencies.argsort()[::-1][:top_n]
    return [(terms[i], float(frequencies[i])) for i in top_indices]


def display_top_terms(terms, title: str) -> None:
    print(title)
    print("-" * len(title))
    for term, freq in terms:
        print(f"{term}: {freq:.4f}")
    print()


def fit_topic_model(model_class, matrix, n_topics=10, random_state=42, max_iter=10):
    model = model_class(n_components=n_topics, random_state=random_state, max_iter=max_iter)
    model.fit(matrix)
    return model


def get_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]]
        topics.append((topic_idx + 1, top_features))
    return topics


def print_topics(topics, title: str) -> None:
    print(title)
    print("=" * len(title))
    for topic_idx, terms in topics:
        print(f"Topic {topic_idx}: {', '.join(terms)}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Consumer complaints NLP analysis pipeline")
    parser.add_argument("--input", required=True, help="Path to the complaint dataset CSV file")
    parser.add_argument("--text-column", default="consumer_complaint_narrative", help="Name of the text column")
    parser.add_argument("--label-column", default="category", help="Name of the label column")
    parser.add_argument("--topics", type=int, default=10, help="Number of latent topics")
    parser.add_argument("--max-features", type=int, default=15000, help="Maximum number of features for vectorizers")
    parser.add_argument("--plot", action="store_true", help="Save a class distribution plot")
    parser.add_argument("--output-dir", default="output", help="Directory to save figures and analysis output")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in dataset")

    print(f"Loading data from {input_path}")
    n_rows = len(df)
    print(f"Original rows: {n_rows}")

    n_missing = df[args.text_column].isna().sum()
    print(f"Missing text values: {n_missing}\n")

    n_labels = 0
    if args.label_column and args.label_column in df.columns:
        n_labels = df[args.label_column].isna().sum()
        print(f"Missing label values: {n_labels}\n")

    nlp = load_spacy_model()
    df = preprocess_documents(df, args.text_column, nlp)

    print(f"Rows after cleaning: {len(df)}\n")
    describe_dataset(df, args.text_column, args.label_column if args.label_column in df.columns else None)

    if args.label_column and args.label_column in df.columns and args.plot:
        plot_class_distribution(df, args.label_column, output_path=output_dir / "class_distribution.png")

    count_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer = build_vectorizers(
        df["clean_text"], max_features=args.max_features
    )

    display_top_terms(top_terms(count_vectorizer, count_matrix, top_n=20), "Top 20 terms from CountVectorizer")
    display_top_terms(top_terms(tfidf_vectorizer, tfidf_matrix, top_n=20), "Top 20 terms from TF-IDF")

    lda = fit_topic_model(LatentDirichletAllocation, count_matrix, n_topics=args.topics)
    nmf = fit_topic_model(NMF, tfidf_matrix, n_topics=args.topics)

    print_topics(get_topics(lda, count_vectorizer.get_feature_names_out(), n_top_words=10), "LDA Topics")
    print_topics(get_topics(nmf, tfidf_vectorizer.get_feature_names_out(), n_top_words=10), "NMF Topics")

    print(f"Analysis finished. Output files are available in: {output_dir}")


if __name__ == "__main__":
    main()
