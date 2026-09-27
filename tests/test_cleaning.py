from nlp_project.preprocessing import clean_text


def test_removes_html_tags():
    assert clean_text("<p>Hello there</p>") == "Hello there"


def test_removes_urls():
    result = clean_text("Check https://example.com/support for help")
    assert "http" not in result
    assert "example.com" not in result


def test_removes_punctuation():
    assert clean_text("Wait, what?!") == "Wait what"


def test_collapses_whitespace():
    assert clean_text("too    many     spaces") == "too many spaces"


def test_handles_none_gracefully():
    assert clean_text(None) == ""
