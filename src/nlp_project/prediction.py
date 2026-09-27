"""
prediction.py
-------------
Loads the saved model artifact and runs the LIVE PREDICTION flow for a
brand-new piece of text:

    NEW TEXT -> clean_text -> normalize_text -> tokenize_text
             -> remove_stopwords -> lemmatize_tokens
             -> saved TF-IDF vectorizer (transform only)
             -> saved model -> prediction

The new text is NEVER written to disk, appended to the dataset, or used
to refit the vectorizer or retrain the model. Only `transform` is called
on the vectorizer; the model is only ever asked to `predict`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from nlp_project.config import MODEL_ARTIFACT_PATH
from nlp_project.feature_extraction import top_tfidf_features
from nlp_project.preprocessing import run_pipeline


def load_artifact(path=MODEL_ARTIFACT_PATH) -> dict:
    """Load the joblib artifact saved by scripts/train.py. Contains the
    fitted vectorizer, the trained model, class names, and some metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact found at {path}. Run `python scripts/train.py` first."
        )
    return joblib.load(path)


def predict_text(text: str, artifact: dict, top_n_features: int = 10) -> dict:
    """Run the full live-prediction flow for one piece of text and return
    every intermediate stage plus the final prediction, so callers (the
    CLI script and the Streamlit app) can display the whole journey.
    """
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    classes = artifact["classes"]

    pipeline_result = run_pipeline(text)
    tfidf_vector = vectorizer.transform([pipeline_result.final_text])

    predicted_label = model.predict(tfidf_vector)[0]

    top3 = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(tfidf_vector)[0]
        ranking = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
        top3 = pd.DataFrame(ranking[:3], columns=["category", "probability"])
        top3["probability"] = top3["probability"].round(4)
    elif hasattr(model, "decision_function"):
        # LinearSVC has no predict_proba by default: decision_function
        # scores are NOT probabilities, so we label them explicitly as
        # "score" rather than implying a calibrated probability.
        scores = model.decision_function(tfidf_vector)[0]
        ranking = sorted(zip(classes, scores), key=lambda x: x[1], reverse=True)
        top3 = pd.DataFrame(ranking[:3], columns=["category", "score"])
        top3["score"] = top3["score"].round(4)

    top_features = top_tfidf_features(vectorizer, tfidf_vector, top_n=top_n_features)

    return {
        "pipeline": pipeline_result,
        "predicted_label": predicted_label,
        "top3": top3,
        "top_features": top_features,
    }
