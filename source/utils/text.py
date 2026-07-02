# imports
import re
from typing import Any

# constants
TEXT_FIELDS = ("user_query", "chatgpt_response")
POSITIVE_LABEL = "yes"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_']+")

def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Lowercases the text, then extracts runs of letters, digits, underscores, and
    apostrophes (via ``TOKEN_PATTERN``), dropping punctuation and whitespace. For
    example, "Don't PANIC!" -> ["don't", "panic"]. This is the shared tokenizer
    the text-based models (Naive Bayes, TF-IDF) use so their vocabularies match.
    """
    return TOKEN_PATTERN.findall(text.lower())

def record_to_text(record: dict[str, Any]) -> str:
    """Flatten a record's text fields into a single string.

    Concatenates the ``TEXT_FIELDS`` ("user_query", "chatgpt_response") of a
    record into one newline-separated string, which is what the models actually
    tokenize and train on. Missing fields are treated as empty, and values are
    coerced to str so non-string fields don't break the join.
    """
    return "\n".join(str(record.get(field, "")) for field in TEXT_FIELDS)