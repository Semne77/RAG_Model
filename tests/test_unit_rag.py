import pytest
from src.logic import validate_question

def test_question_not_null_or_empty():
    ok, err = validate_question("")
    assert ok is False
    assert "required" in err.lower()

    ok, err = validate_question("   ")
    assert ok is False
    assert "required" in err.lower()

    ok, err = validate_question(None)
    assert ok is False
    assert "required" in err.lower()

    ok, err = validate_question("Hello")
    assert ok is True
    assert err == ""


def test_question_max_15_words():
    fifteen = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    ok, err = validate_question(fifteen)
    assert ok is True

    sixteen = fifteen + " sixteen"
    ok, err = validate_question(sixteen)
    assert ok is False
    assert "15" in err


def test_question_ascii_only():
    ok, err = validate_question("Federer 2008!!! #1?")
    assert ok is True

    ok, err = validate_question("Привет")  # Cyrillic
    assert ok is False
    assert "english" in err.lower() or "ascii" in err.lower()

    ok, err = validate_question("你好")  # Chinese
    assert ok is False
