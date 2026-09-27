"""
pipeline_signature.py
----------------------
Computes a short fingerprint of everything that determines how raw text
turns into a TF-IDF feature vector: the whole preprocessing.py module
(clean_text/normalize_text/tokenize_text/remove_stopwords/lemmatize_tokens,
including their regexes and the lemmatizer's suffix rules), the whole
feature_extraction.py module (build_vectorizer's settings), and the
STOPWORDS/CONTRACTIONS/TF-IDF constants in config.py.

WHY THIS EXISTS
----------------
scripts/train.py fits the TF-IDF vectorizer and the model *once* and saves
them to models/model_artifacts.joblib (gitignored -- it is never shipped,
only produced locally by running train.py). From then on, live prediction
(app.py, scripts/predict.py) only ever calls `vectorizer.transform(...)`;
it never re-fits anything, by design (see prediction.py's docstring) --
that part is correct and is NOT what this module changes.

The sharp edge this protects against: if preprocessing.py, config.py's
STOPWORDS/CONTRACTIONS, or the TF-IDF settings are edited *after* a model
was trained, the saved vectorizer's vocabulary was built from the OLD
pipeline's output. New text run through the NEW pipeline then usually
shares almost no tokens with that vocabulary, `transform()` returns a
near-all-zero row, and `predict_proba()` collapses toward the model's
learned class priors (the intercept term) -- probabilities barely move
between different inputs and hover close to 1/n_classes, with whichever
class has the strongest bias term winning almost every time. From the
outside this looks exactly like a broken live-prediction pipeline, even
though every individual piece of code is working correctly in isolation.

`compute_pipeline_signature()` is stored in the artifact at train time
(scripts/train.py) and re-checked every time the artifact is loaded
(prediction.load_artifact), so this situation is caught immediately with
a clear, actionable message instead of silently producing flat,
meaningless probabilities.
"""

from __future__ import annotations

import hashlib
import inspect

from nlp_project import feature_extraction, preprocessing
from nlp_project.config import (
    CONTRACTIONS,
    STOPWORDS,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)


class PipelineSignatureMismatch(RuntimeError):
    """Raised when a saved model artifact was trained with a different
    version of the preprocessing/TF-IDF pipeline than the one currently
    in the codebase. Retraining (python scripts/train.py) resolves it.
    """


def compute_pipeline_signature() -> str:
    """Hash everything that determines how raw text becomes TF-IDF
    features. Two runs produce the same signature if and only if none of
    preprocessing.py, feature_extraction.py, or the relevant config.py
    constants changed in between.
    """
    hasher = hashlib.sha256()
    hasher.update(inspect.getsource(preprocessing).encode("utf-8"))
    hasher.update(inspect.getsource(feature_extraction).encode("utf-8"))
    hasher.update(repr(sorted(STOPWORDS)).encode("utf-8"))
    hasher.update(repr(sorted(CONTRACTIONS.items())).encode("utf-8"))
    hasher.update(
        repr((TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF)).encode("utf-8")
    )
    return hasher.hexdigest()[:16]