from nlp_project.preprocessing import lemmatize_tokens


def test_studies_to_study():
    assert lemmatize_tokens(["studies"]) == ["study"]


def test_running_to_run():
    assert lemmatize_tokens(["running"]) == ["run"]


def test_charged_to_charge():
    assert lemmatize_tokens(["charged"]) == ["charge"]


def test_plural_products_to_product():
    assert lemmatize_tokens(["products"]) == ["product"]


def test_irregular_was_to_be():
    assert lemmatize_tokens(["was"]) == ["be"]


def test_empty_list():
    assert lemmatize_tokens([]) == []
