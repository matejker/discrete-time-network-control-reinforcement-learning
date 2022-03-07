from network_control_rl_framework.utils import is_int, is_float


def test_is_int():
    assert is_int("lorem") is False
    assert is_int(None) is False
    assert is_int(123) is True
    assert is_int(123.123) is True
    assert is_int(True) is True


def test_is_float():
    assert is_float("lorem") is False
    assert is_float(None) is False
    assert is_float(123) is True
    assert is_float(123.123) is True
    assert is_float(True) is True
