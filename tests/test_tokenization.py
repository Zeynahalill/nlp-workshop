from nlp_project.preprocessing import tokenize_text


def test_basic_tokenization():
    assert tokenize_text("this is a good product") == [
        "this", "is", "a", "good", "product",
    ]


def test_empty_string_returns_empty_list():
    assert tokenize_text("") == []


def test_ignores_numbers_and_symbols():
    tokens = tokenize_text("order 12345 was fine")
    assert "12345" not in tokens
    assert tokens == ["order", "was", "fine"]
