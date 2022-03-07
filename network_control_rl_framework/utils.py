from typing import Any


def is_int(number: Any) -> bool:
    try:
        int(number)
        return True
    except (ValueError, TypeError):
        return False


def is_float(number: Any) -> bool:
    try:
        float(number)
        return True
    except (ValueError, TypeError):
        return False
