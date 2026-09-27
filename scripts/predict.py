"""
predict.py
----------
Small CLI to try live prediction from the terminal without Streamlit.
Loads the saved artifact only; never retrains anything.

Run:
    python scripts/predict.py "I ordered something but it has not arrived yet."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp_project.prediction import load_artifact, predict_text


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/predict.py "your text here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    artifact = load_artifact()

    result = predict_text(text, artifact)
    pipeline = result["pipeline"]

    print(f"RAW TEXT          : {pipeline.raw_text!r}")
    print(f"CLEANED           : {pipeline.cleaned_text!r}")
    print(f"NORMALIZED        : {pipeline.normalized_text!r}")
    print(f"TOKENS            : {pipeline.tokens}")
    print(f"STOPWORDS REMOVED : {pipeline.tokens_no_stopwords}")
    print(f"LEMMATIZED        : {pipeline.lemmatized_tokens}")
    print()
    print(f"PREDICTED CATEGORY: {result['predicted_label']}")
    if result["top3"] is not None:
        print("\nTop 3:")
        print(result["top3"].to_string(index=False))
    print("\nTop TF-IDF features for this text:")
    print(result["top_features"].to_string(index=False))


if __name__ == "__main__":
    main()
