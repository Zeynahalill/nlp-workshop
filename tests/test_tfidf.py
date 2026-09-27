from nlp_project.feature_extraction import fit_tfidf_vectorizer, transform_texts


TRAIN_TEXTS = [
    "I was charged an unexpected fee on my credit card",
    "I was charged twice for the same order this month",
    "My package has not arrived yet",
    "My package was never delivered to my address",
    "The app keeps crashing when I log in",
    "The app crashes every time I open it",
    "I cannot access my account",
    "I cannot log into my account anymore",
]


def test_vectorizer_fits_on_train_texts():
    vectorizer = fit_tfidf_vectorizer(TRAIN_TEXTS)
    assert len(vectorizer.vocabulary_) > 0


def test_transform_only_uses_existing_vocabulary():
    vectorizer = fit_tfidf_vectorizer(TRAIN_TEXTS)
    unseen_text = ["A completely different sentence about weather forecasts"]
    matrix = transform_texts(vectorizer, unseen_text)
    # Transform must never grow the vocabulary -- this is what prevents
    # validation/test data from leaking into the fitted vectorizer.
    assert matrix.shape[1] == len(vectorizer.vocabulary_)


def test_transform_output_shape_matches_input_count():
    vectorizer = fit_tfidf_vectorizer(TRAIN_TEXTS)
    matrix = transform_texts(vectorizer, TRAIN_TEXTS)
    assert matrix.shape[0] == len(TRAIN_TEXTS)
