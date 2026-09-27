"""
train.py
--------
The single entry point that runs the ENTIRE NLP pipeline end to end, and
prints what is happening at every stage so it can be followed live during
the workshop.

Run:
    python scripts/train.py
"""

import sys
from pathlib import Path

# Allow running as `python scripts/train.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from nlp_project import config
from nlp_project.evaluation import (
    build_comparison_table,
    compute_metrics,
    plot_confusion_matrix,
    select_best_model,
)
from nlp_project.feature_extraction import fit_tfidf_vectorizer, transform_texts
from nlp_project.modeling import build_candidate_models, split_dataset, train_model
from nlp_project.pipeline_signature import compute_pipeline_signature
from nlp_project.preprocessing import run_pipeline


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_pipeline_on_sample():
    """Show one example text moving through every preprocessing stage,
    exactly like the walkthrough in the workshop brief.
    """
    section("STEP-BY-STEP PIPELINE DEMO (one example)")
    sample = "I was charged an unexpected fee on my credit card!"
    result = run_pipeline(sample)
    print(f"RAW TEXT          : {result.raw_text!r}")
    print(f"CLEANED           : {result.cleaned_text!r}")
    print(f"NORMALIZED        : {result.normalized_text!r}")
    print(f"TOKENS            : {result.tokens}")
    print(f"STOPWORDS REMOVED : {result.tokens_no_stopwords}")
    print(f"LEMMATIZED        : {result.lemmatized_tokens}")
    print(f"FINAL (for TF-IDF): {result.final_text!r}")


def main():
    demo_pipeline_on_sample()

    # ------------------------------------------------------------------
    section("1. LOAD DATASET")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows from {config.RAW_DATA_PATH}")
    print(df[config.LABEL_COLUMN].value_counts())

    # ------------------------------------------------------------------
    section("2. TRAIN / VALIDATION / TEST SPLIT (stratified)")
    train_df, val_df, test_df = split_dataset(df)
    print(f"Train      : {len(train_df)} rows")
    print(f"Validation : {len(val_df)} rows")
    print(f"Test       : {len(test_df)} rows")

    # ------------------------------------------------------------------
    section("3. FEATURE EXTRACTION (TF-IDF, fit on TRAIN only)")
    vectorizer = fit_tfidf_vectorizer(train_df[config.TEXT_COLUMN].tolist())
    X_train = transform_texts(vectorizer, train_df[config.TEXT_COLUMN].tolist())
    X_val = transform_texts(vectorizer, val_df[config.TEXT_COLUMN].tolist())
    X_test = transform_texts(vectorizer, test_df[config.TEXT_COLUMN].tolist())
    y_train = train_df[config.LABEL_COLUMN]
    y_val = val_df[config.LABEL_COLUMN]
    y_test = test_df[config.LABEL_COLUMN]
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train feature matrix shape: {X_train.shape}")

    # ------------------------------------------------------------------
    section("4. TRAIN CANDIDATE MODELS")
    candidates = build_candidate_models()
    trained_models = {}
    validation_results = {}
    for name, model in candidates.items():
        print(f"Training: {name} ...")
        trained_models[name] = train_model(model, X_train, y_train)
        y_val_pred = trained_models[name].predict(X_val)
        validation_results[name] = compute_metrics(y_val, y_val_pred)

    # ------------------------------------------------------------------
    section("5. MODEL COMPARISON (on VALIDATION set)")
    comparison_df = build_comparison_table(validation_results)
    print(comparison_df.to_string(index=False))

    best_model_name = select_best_model(validation_results)
    print(f"\nSelected best model (highest validation macro F1): {best_model_name}")
    best_model = trained_models[best_model_name]

    # ------------------------------------------------------------------
    section("6. FINAL EVALUATION (on TEST set, touched only now)")
    y_test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name:>16}: {value:.4f}")

    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    class_labels = sorted(df[config.LABEL_COLUMN].unique())
    cm_path = config.EVALUATION_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_test_pred, class_labels, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    metrics_path = config.EVALUATION_DIR / "test_metrics.csv"
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    comparison_path = config.EVALUATION_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved metrics to {metrics_path} and {comparison_path}")

    # ------------------------------------------------------------------
    section("7. SAVE MODEL ARTIFACT")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_signature = compute_pipeline_signature()
    artifact = {
        "vectorizer": vectorizer,
        "model": best_model,
        "model_name": best_model_name,
        "classes": sorted(df[config.LABEL_COLUMN].unique()),
        "n_training_examples": len(train_df),
        "test_metrics": test_metrics,
        # Fingerprint of preprocessing.py / feature_extraction.py / the
        # TF-IDF & stopword/contraction settings at train time. Checked
        # again on every load_artifact() call (see prediction.py) so a
        # stale artifact is rejected loudly instead of silently producing
        # near-uniform, meaningless live predictions.
        "pipeline_signature": pipeline_signature,
    }
    joblib.dump(artifact, config.MODEL_ARTIFACT_PATH)
    print(f"Saved model artifact to {config.MODEL_ARTIFACT_PATH}")
    print(f"Pipeline signature: {pipeline_signature}")

    section("DONE")
    print("Run `streamlit run app.py` to try live predictions.")


if __name__ == "__main__":
    main()