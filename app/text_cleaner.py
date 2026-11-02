
import re


def clean_text(text: str) -> str:
    """
    Normalize whitespace in extracted PDF text
    while preserving paragraph boundaries.
    """

    if not text:
        return ""

   
    text = text.replace("\r\n", "\n").replace("\r", "\n")

   
    lines = [line.strip() for line in text.split("\n")]

    
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    
    return cleaned_text.strip()