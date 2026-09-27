"""
config.py
---------
Single place for paths and settings shared across the project. Keeping
these here (instead of scattered magic numbers) makes the pipeline easier
to follow during the workshop.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "model_artifacts.joblib"

OUTPUTS_DIR = ROOT_DIR / "outputs"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

TEXT_COLUMN = "text"
LABEL_COLUMN = "category"

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
# A small, self-contained English stopword list. We avoid depending on
# nltk's downloadable corpora so the pipeline works fully offline and the
# list itself stays easy to read and modify during the workshop.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "s", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves", "t", "ll",
    "re", "ve", "d", "m", "o",
}

# ---------------------------------------------------------------------------
# Common English contractions used during normalization
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "it's": "it is", "that's": "that is",
    "there's": "there is", "what's": "what is", "let's": "let us",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# ---------------------------------------------------------------------------
# Models to train and compare
# ---------------------------------------------------------------------------
MODEL_NAMES = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]
