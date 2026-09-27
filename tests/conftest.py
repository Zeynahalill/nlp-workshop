import joblib
import pandas as pd
import pytest

from nlp_project.config import LABEL_COLUMN, TEXT_COLUMN
from nlp_project.feature_extraction import fit_tfidf_vectorizer, transform_texts
from nlp_project.modeling import train_model
from nlp_project.pipeline_signature import compute_pipeline_signature
from sklearn.naive_bayes import MultinomialNB


@pytest.fixture(scope="session")
def tiny_artifact_path(tmp_path_factory):
    """Build a small, fast model artifact purely for testing -- this does
    NOT touch the real project dataset/model in data/ or models/, so
    tests stay independent from `python scripts/train.py` having been run.
    """
    texts = (
        ["I was charged twice on my credit card"] * 15
        + ["My order never arrived at my address"] * 15
        + ["The app crashes every time I open it"] * 15
    )
    labels = (
        ["Billing"] * 15 + ["Delivery"] * 15 + ["Technical Support"] * 15
    )
    df = pd.DataFrame({TEXT_COLUMN: texts, LABEL_COLUMN: labels})

    vectorizer = fit_tfidf_vectorizer(df[TEXT_COLUMN].tolist())
    X = transform_texts(vectorizer, df[TEXT_COLUMN].tolist())
    model = train_model(MultinomialNB(), X, df[LABEL_COLUMN])

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "model_name": "Multinomial Naive Bayes",
        "classes": sorted(df[LABEL_COLUMN].unique()),
        "n_training_examples": len(df),
        "test_metrics": {"accuracy": 1.0},
        "pipeline_signature": compute_pipeline_signature(),
    }

    path = tmp_path_factory.mktemp("artifacts") / "model_artifacts.joblib"
    joblib.dump(artifact, path)
    return path