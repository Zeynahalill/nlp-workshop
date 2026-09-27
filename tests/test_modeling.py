import pandas as pd

from nlp_project.config import LABEL_COLUMN, TEXT_COLUMN
from nlp_project.feature_extraction import fit_tfidf_vectorizer, transform_texts
from nlp_project.modeling import build_candidate_models, split_dataset, train_model


def _synthetic_df(n_per_class: int = 20) -> pd.DataFrame:
    billing = ["I was charged twice for my order"] * n_per_class
    delivery = ["My package never arrived at my house"] * n_per_class
    texts = billing + delivery
    labels = ["Billing"] * n_per_class + ["Delivery"] * n_per_class
    return pd.DataFrame({TEXT_COLUMN: texts, LABEL_COLUMN: labels})


def test_split_dataset_is_stratified_and_covers_all_rows():
    df = _synthetic_df()
    train_df, val_df, test_df = split_dataset(df)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df[LABEL_COLUMN].unique()) == {"Billing", "Delivery"}


def test_models_can_be_trained_on_small_data():
    df = _synthetic_df()
    train_df, _, _ = split_dataset(df)
    vectorizer = fit_tfidf_vectorizer(train_df[TEXT_COLUMN].tolist())
    X_train = transform_texts(vectorizer, train_df[TEXT_COLUMN].tolist())
    y_train = train_df[LABEL_COLUMN]

    models = build_candidate_models()
    assert len(models) == 3
    for name, model in models.items():
        trained = train_model(model, X_train, y_train)
        predictions = trained.predict(X_train)
        assert len(predictions) == len(y_train)
