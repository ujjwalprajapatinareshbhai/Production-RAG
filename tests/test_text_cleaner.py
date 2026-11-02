
from app.text_cleaner import clean_text


def test_removes_extra_spaces():
    result = clean_text("   Hello world   ")
    assert result == "Hello world"


def test_preserves_paragraphs():
    result = clean_text("First paragraph\n\n\n\nSecond paragraph")
    assert result == "First paragraph\n\nSecond paragraph"


def test_normalizes_line_endings():
    result = clean_text("First line\r\nSecond line\rThird line")
    assert result == "First line\nSecond line\nThird line"


def test_empty_text():
    assert clean_text("") == ""


if __name__ == "__main__":
    test_removes_extra_spaces()
    test_preserves_paragraphs()
    test_normalizes_line_endings()
    test_empty_text()

    print("All text cleaner tests passed!")