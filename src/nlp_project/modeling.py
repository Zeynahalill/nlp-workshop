"""
modeling.py
-----------
Classic NLP + ML models (no deep learning, no LLMs -- this workshop is
about the classic pipeline). Three models are trained and compared on the
VALIDATION set; the best one (by validation macro-F1) is selected, and
only then is the TEST set touched, in evaluation.py.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nlp_project.config import (
    LABEL_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    VALIDATION_SIZE,
)


def split_dataset(df: pd.DataFrame):
    """Stratified split into train / validation / test.

    Two-step split: first carve off TEST_SIZE as the test set, then split
    the remainder into train vs. validation so the final proportions match
    TRAIN_SIZE / VALIDATION_SIZE / TEST_SIZE from config.py (default
    60/20/20). Stratification keeps class proportions consistent across
    all three splits.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )
    relative_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_candidate_models() -> dict:
    """Return the three classic classifiers to compare, by name."""
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }


def train_model(model, X_train, y_train):
    """Fit a single model on the training features/labels."""
    model.fit(X_train, y_train)
    return model
