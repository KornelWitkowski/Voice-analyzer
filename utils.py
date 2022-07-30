
from typing import Optional


def get_integer(text: Optional[str]) -> Optional[int]:
    if not text:
        return None

    string_with_digit = "".join([char for char in text if char.isdigit()])

    if not string_with_digit:
        return None

    return int(string_with_digit)
