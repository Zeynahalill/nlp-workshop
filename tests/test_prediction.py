import pytest

from nlp_project.prediction import load_artifact, predict_text


def test_artifact_loading(tiny_artifact_path):
    artifact = load_artifact(tiny_artifact_path)
    assert "vectorizer" in artifact
    assert "model" in artifact
    assert "classes" in artifact
    assert len(artifact["classes"]) == 3


def test_artifact_loading_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "does_not_exist.joblib")


def test_predict_returns_expected_keys(tiny_artifact_path):
    artifact = load_artifact(tiny_artifact_path)
    result = predict_text("I was charged for something I did not buy", artifact)
    assert "predicted_label" in result
    assert result["predicted_label"] in artifact["classes"]
    assert "pipeline" in result
    assert "top_features" in result


def test_predict_on_empty_input_does_not_crash(tiny_artifact_path):
    artifact = load_artifact(tiny_artifact_path)
    result = predict_text("", artifact)
    # An empty/near-empty input should still produce *some* prediction
    # rather than raising, since the vectorizer/model must handle
    # all-zero feature vectors gracefully.
    assert result["predicted_label"] in artifact["classes"]
    assert result["pipeline"].tokens == []


def test_predict_does_not_mutate_vectorizer_vocabulary(tiny_artifact_path):
    artifact = load_artifact(tiny_artifact_path)
    vocab_before = dict(artifact["vectorizer"].vocabulary_)
    predict_text("a completely new sentence never seen before", artifact)
    assert artifact["vectorizer"].vocabulary_ == vocab_before
