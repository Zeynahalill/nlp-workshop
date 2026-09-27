from nlp_project.preprocessing import normalize_text


def test_lowercases():
    assert normalize_text("HELLO World") == "hello world"


def test_expands_contractions():
    assert normalize_text("I don't like this") == "i do not like this"
    assert normalize_text("It's broken") == "it is broken"


def test_normalizes_whitespace_after_expansion():
    result = normalize_text("I  can't   do this")
    assert "  " not in result


def test_handles_empty_string():
    assert normalize_text("") == ""
