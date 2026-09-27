"""
preprocessing.py
-----------------
Every NLP stage lives in its OWN function on purpose. Nothing is hidden
behind a single `preprocess(text)` call: during the workshop we want to be
able to run each function one at a time and print the text after every
step, so people can see exactly how it changes.

Stages in this file, in pipeline order:
    1. clean_text()        -> structural cleanup (HTML, URLs, punctuation, whitespace)
    2. normalize_text()     -> lowercase, contractions, whitespace normalization
    3. tokenize_text()      -> split into a list of word tokens
    4. remove_stopwords()   -> drop very common, low-information words
    5. lemmatize_tokens()   -> reduce words to their base/dictionary form

`run_pipeline()` at the bottom simply calls these in order and returns
every intermediate result, which is what app.py and scripts/train.py use
to show the step-by-step transformation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nlp_project.config import CONTRACTIONS, STOPWORDS

# ---------------------------------------------------------------------------
# 1. TEXT CLEANING
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_PUNCTUATION_RE = re.compile(r"[^\w\s']")  # keep apostrophes for contractions
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove structural noise: HTML tags, URLs, punctuation and extra
    whitespace. Deliberately does NOT lowercase yet -- that belongs to
    normalization, so the two concerns stay separate and inspectable.

    Apostrophes are deliberately kept here (not treated as "punctuation
    noise"): normalize_text() needs them intact to recognize contractions
    like "don't" / "can't". They get dropped naturally at the
    tokenization stage instead, since the token regex only matches
    letters.
    """
    if text is None:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. NORMALIZATION
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase the text, expand common English contractions, and
    collapse whitespace again (contraction expansion can introduce extra
    spaces). Cleaning removes *noise*; normalization makes different
    surface forms of the *same* meaning look identical (e.g. "Don't" and
    "do not" both become "do not").
    """
    if text is None:
        return ""
    text = text.lower()
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. TOKENIZATION
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def tokenize_text(text: str) -> list[str]:
    """Split normalized text into word tokens. A simple regex tokenizer is
    used on purpose (no external tokenizer library needed) -- it is easy
    to explain and easy to reason about during the workshop.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# 4. STOPWORD REMOVAL
# ---------------------------------------------------------------------------
def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out very common, low-information words (see STOPWORDS in
    config.py). These words carry little signal for classification but
    appear in almost every sentence.
    """
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# 5. LEMMATIZATION
# ---------------------------------------------------------------------------
# A small dictionary for irregular forms that suffix rules cannot handle.
_IRREGULAR_LEMMAS = {
    "was": "be", "were": "be", "is": "be", "are": "be", "been": "be",
    "am": "be", "has": "have", "had": "have", "did": "do", "does": "do",
    "went": "go", "gone": "go", "children": "child", "people": "person",
    "men": "man", "women": "woman", "feet": "foot", "mice": "mouse",
}

_VOWELS = set("aeiou")

# Words that happen to end in "-ing"/"-ed"/"-eed" but are NOT inflected
# verb forms, so the suffix rules below would otherwise mangle them.
_NON_INFLECTED = {
    "something", "nothing", "anything", "everything",
    "morning", "evening", "during", "king", "ring", "spring", "thing",
}


def _lemmatize_word(word: str) -> str:
    """Rule-based lemmatizer.

    We intentionally avoid nltk/spaCy here: their lemmatizers need
    downloadable corpora/models, which would make the workshop dependent
    on internet access during the session. This lightweight, transparent
    set of suffix rules covers the common cases well enough to teach the
    *concept* of lemmatization (see README for the difference vs.
    stemming). Swap in `nltk.WordNetLemmatizer` or spaCy any time you want
    higher linguistic accuracy.
    """
    if word in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[word]

    if word in _NON_INFLECTED:
        return word

    # plural "-ies" -> "y"          e.g. studies -> study
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # past tense "-ied" -> "y"      e.g. tried -> try
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"

    # words like "need"/"speed"/"indeed" are not simple past-tense verbs
    # even though they end in "-eed" -- leave them alone rather than
    # mangling them into a nonsense stem.
    if word.endswith("eed"):
        return word

    # progressive "-ing"           e.g. running -> run
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
            stem = stem[:-1]  # undo doubled consonant: runn -> run
        return stem

    # past tense "-ed"             e.g. charged -> charge, ordered -> order
    if word.endswith("ed") and len(word) > 3:
        stem = word[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
            stem = stem[:-1]  # doubled consonant: stopped -> stop
        elif stem and (stem[-1] in "cgvzsui" or stem.endswith("at")):
            # These endings are almost always a silent "e" that got
            # dropped before "-ed" was added (charge -> charged,
            # use -> used, notice -> noticed, create -> created).
            # Endings like "t" or "r" (expect -> expected,
            # order -> ordered) are left alone.
            stem = stem + "e"
        return stem

    # plural "-es": words ending in a sibilant sound take "-es" and lose
    # it entirely (box -> boxes, church -> churches); most other "-es"
    # words just add "s" to an already e-ending base (time -> times).
    if word.endswith("es") and len(word) > 3:
        if word.endswith(("ches", "shes", "sses", "xes", "zes")):
            return word[:-2]
        return word[:-1]

    # plural "-s"                  e.g. products -> product
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply the rule-based lemmatizer to every token in the list."""
    return [_lemmatize_word(t) for t in tokens]


# ---------------------------------------------------------------------------
# Pipeline result container + runner
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Every intermediate output of the preprocessing pipeline, so callers
    (the training script, the tests, and the Streamlit app) can display or
    inspect each stage individually.
    """

    raw_text: str
    cleaned_text: str = ""
    normalized_text: str = ""
    tokens: list[str] = field(default_factory=list)
    tokens_no_stopwords: list[str] = field(default_factory=list)
    lemmatized_tokens: list[str] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        """Space-joined lemmatized tokens -- this is what gets fed into
        the TF-IDF vectorizer."""
        return " ".join(self.lemmatized_tokens)


def run_pipeline(text: str) -> PipelineResult:
    """Run every preprocessing stage in order and keep every intermediate
    result. This is the function the workshop demo calls to show a text's
    full journey from raw input to model-ready features.
    """
    result = PipelineResult(raw_text=text)
    result.cleaned_text = clean_text(text)
    result.normalized_text = normalize_text(result.cleaned_text)
    result.tokens = tokenize_text(result.normalized_text)
    result.tokens_no_stopwords = remove_stopwords(result.tokens)
    result.lemmatized_tokens = lemmatize_tokens(result.tokens_no_stopwords)
    return result


def preprocess_for_model(text: str) -> str:
    """Convenience helper used by feature_extraction/prediction: runs the
    full pipeline and returns just the final space-joined string, ready
    for the TF-IDF vectorizer.
    """
    return run_pipeline(text).final_text
