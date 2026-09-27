"""
feature_extraction.py
----------------------
This is where text becomes numbers. We use TF-IDF (Term Frequency -
Inverse Document Frequency) on top of the lemmatized text produced by
preprocessing.py.

IMPORTANT (data leakage): the vectorizer must be `fit` only on the
training split. Validation and test data are only ever `transform`-ed with
the vectorizer that was already fit on train. Fitting on the full dataset
before splitting would leak information about the validation/test
vocabulary into training and make evaluation numbers overly optimistic --
train.py is careful to call fit_tfidf_vectorizer() on the train split only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_project.config import TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE
from nlp_project.preprocessing import preprocess_for_model


def build_vectorizer() -> TfidfVectorizer:
    """Create a fresh, unfit TF-IDF vectorizer with the project's settings.

    Text handed to the vectorizer is already fully preprocessed (cleaned,
    normalized, tokenized, stopword-filtered, lemmatized) by
    preprocess_for_model(), so we disable the vectorizer's own built-in
    lowercasing/tokenizing/stopword-filtering to avoid doing it twice.
    """
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_tfidf_vectorizer(train_texts: list[str]) -> TfidfVectorizer:
    """Fit a new TF-IDF vectorizer on TRAIN texts only. `train_texts`
    should already be raw texts -- this function runs the full
    preprocessing pipeline internally.
    """
    processed = [preprocess_for_model(t) for t in train_texts]
    vectorizer = build_vectorizer()
    vectorizer.fit(processed)
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: list[str]):
    """Transform (never fit) a list of raw texts using an already-fitted
    vectorizer. Used for validation, test, and live prediction text.
    """
    processed = [preprocess_for_model(t) for t in texts]
    return vectorizer.transform(processed)


def top_tfidf_features(vectorizer: TfidfVectorizer, tfidf_vector, top_n: int = 10) -> pd.DataFrame:
    """Given one row of a TF-IDF matrix (a single text's vector), return
    the top_n highest-scoring word/n-gram features as a small DataFrame
    with columns ["word", "tfidf_score"]. Used by the Streamlit app to
    show "which words mattered most" for a given piece of text.
    """
    feature_names = vectorizer.get_feature_names_out()
    row = tfidf_vector.toarray().ravel()
    top_indices = row.argsort()[::-1][:top_n]
    top_indices = [i for i in top_indices if row[i] > 0]
    return pd.DataFrame(
        {
            "word": [feature_names[i] for i in top_indices],
            "tfidf_score": [round(float(row[i]), 4) for i in top_indices],
        }
    )
