from nlp_project.preprocessing import remove_stopwords


def test_removes_common_stopwords():
    tokens = ["this", "is", "a", "good", "product"]
    assert remove_stopwords(tokens) == ["good", "product"]


def test_keeps_content_words():
    tokens = ["charge", "credit", "card"]
    assert remove_stopwords(tokens) == ["charge", "credit", "card"]


def test_empty_list():
    assert remove_stopwords([]) == []
