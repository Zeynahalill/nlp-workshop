"""
app.py
------
Streamlit demo for the workshop. Shows the full NLP pipeline running live
on whatever text the user types in.

Run:
    streamlit run app.py

Nothing the user types here is ever saved to disk, added to the dataset,
or used to retrain the model -- see prediction.py for the prediction-only
flow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from nlp_project.config import RAW_DATA_PATH
from nlp_project.prediction import load_artifact, predict_text

st.set_page_config(page_title="NLP Workshop: From Raw Text to Prediction", layout="wide")

st.title("NLP Workshop: From Raw Text to Prediction")
st.caption(
    "Type any customer-support-style message below and watch it move through "
    "every stage of the classic NLP pipeline, step by step."
)

# ---------------------------------------------------------------------------
# Load the trained artifact (never trains anything here)
# ---------------------------------------------------------------------------
try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(
        "No trained model found yet. Run `python scripts/train.py` first, "
        "then restart this app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Info")
    st.write(f"**Model used:** {artifact['model_name']}")
    st.write(f"**Number of classes:** {len(artifact['classes'])}")
    st.write(f"**Classes:** {', '.join(artifact['classes'])}")
    st.write(f"**Training examples:** {artifact['n_training_examples']}")
    if RAW_DATA_PATH.exists():
        n_total = sum(1 for _ in open(RAW_DATA_PATH, encoding="utf-8")) - 1
        st.write(f"**Total dataset size:** {n_total}")
    st.divider()
    st.subheader("Test-set metrics")
    for name, value in artifact.get("test_metrics", {}).items():
        st.write(f"{name}: {value:.4f}")

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
st.subheader("1. Enter a new text")
default_example = "I ordered something but it has not arrived yet."
user_text = st.text_area("Text to classify", value=default_example, height=100)
predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    result = predict_text(user_text, artifact, top_n_features=10)
    pipeline = result["pipeline"]

    st.subheader("2. NLP Pipeline: how your text was transformed")

    stage_cols = st.columns(3)
    with stage_cols[0]:
        st.markdown("**Raw text**")
        st.code(pipeline.raw_text, language=None)
        st.markdown("**Cleaned text**")
        st.caption("HTML tags, URLs, punctuation and extra whitespace removed.")
        st.code(pipeline.cleaned_text or "(empty)", language=None)
    with stage_cols[1]:
        st.markdown("**Normalized text**")
        st.caption("Lowercased and contractions expanded.")
        st.code(pipeline.normalized_text or "(empty)", language=None)
        st.markdown("**Tokens**")
        st.code(str(pipeline.tokens) if pipeline.tokens else "(none)", language=None)
    with stage_cols[2]:
        st.markdown("**Stopwords removed**")
        st.code(
            str(pipeline.tokens_no_stopwords) if pipeline.tokens_no_stopwords else "(none)",
            language=None,
        )
        st.markdown("**Lemmatized tokens**")
        st.caption("Final form fed into TF-IDF.")
        st.code(
            str(pipeline.lemmatized_tokens) if pipeline.lemmatized_tokens else "(none)",
            language=None,
        )

    st.subheader("3. TF-IDF: top features for this text")
    if result["top_features"].empty:
        st.info("No TF-IDF features matched the training vocabulary for this text.")
    else:
        st.dataframe(result["top_features"], use_container_width=True, hide_index=True)

    st.subheader("4. Prediction")
    pred_cols = st.columns([1, 2])
    with pred_cols[0]:
        st.metric("Predicted category", result["predicted_label"])
    with pred_cols[1]:
        if result["top3"] is not None:
            score_col = "probability" if "probability" in result["top3"].columns else "score"
            label = "Top 3 (probability)" if score_col == "probability" else (
                "Top 3 (decision score -- not a calibrated probability)"
            )
            st.markdown(f"**{label}**")
            st.dataframe(result["top3"], use_container_width=True, hide_index=True)

    st.caption(
        "This text was used only for this one prediction. It was not added "
        "to the dataset and the model was not retrained."
    )
else:
    st.info("Enter some text above and click **Predict** to see the pipeline in action.")
