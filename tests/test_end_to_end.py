import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from nlp_project.config import LABEL_COLUMN, TEXT_COLUMN
from nlp_project.evaluation import compute_metrics
from nlp_project.feature_extraction import fit_tfidf_vectorizer, transform_texts
from nlp_project.modeling import split_dataset, train_model
from nlp_project.preprocessing import run_pipeline


def test_full_pipeline_runs_without_error_on_small_dataset():
    texts = (
        ["I was charged twice on my credit card", "There is a duplicate charge on my invoice"] * 10
        + ["My package never arrived", "The tracking number does not work"] * 10
    )
    labels = ["Billing"] * 20 + ["Delivery"] * 20
    df = pd.DataFrame({TEXT_COLUMN: texts, LABEL_COLUMN: labels})

    # 1. preprocessing stage sanity check
    sample_result = run_pipeline(df[TEXT_COLUMN].iloc[0])
    assert sample_result.final_text != ""

    # 2. split
    train_df, val_df, test_df = split_dataset(df)
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

    # 3. feature extraction (fit on train only)
    vectorizer = fit_tfidf_vectorizer(train_df[TEXT_COLUMN].tolist())
    X_train = transform_texts(vectorizer, train_df[TEXT_COLUMN].tolist())
    X_test = transform_texts(vectorizer, test_df[TEXT_COLUMN].tolist())

    # 4. train
    model = train_model(MultinomialNB(), X_train, train_df[LABEL_COLUMN])

    # 5. evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(test_df[LABEL_COLUMN], y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
