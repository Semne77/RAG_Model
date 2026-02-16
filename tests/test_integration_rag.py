from src.logic import ask_question


def test_serbia_question_returns_djokovic():
    answer = ask_question("Which player is from Serbia?")
    assert "Djokovic" in answer


def test_languages_question_returns_federer():
    answer = ask_question("Which player speaks German, Swiss German, and English?")
    assert "Federer" in answer
