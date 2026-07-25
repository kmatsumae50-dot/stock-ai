import math

import numpy as np

from stock_ai.utils.safe_cast import safe_float, safe_int


def test_safe_int_handles_common_prediction_values():
    assert safe_int(1) == 1
    assert safe_int(np.int64(2)) == 2
    assert safe_int("3") == 3
    assert safe_int("4.0") == 4
    assert safe_int(None) is None
    assert safe_int(math.nan) is None


def test_safe_int_handles_sqlite_blob_integer_values():
    assert safe_int((1).to_bytes(8, byteorder="little", signed=True)) == 1
    assert safe_int((0).to_bytes(8, byteorder="little", signed=True)) == 0
    assert safe_int(b"2") == 2


def test_safe_int_returns_none_for_unconvertible_values():
    assert safe_int("not-a-class") is None
    assert safe_int(b"") is None


def test_safe_float_handles_common_prediction_values():
    assert safe_float(0.5) == 0.5
    assert safe_float(np.float64(0.25)) == 0.25
    assert safe_float("0.75") == 0.75
    assert safe_float(None) is None
    assert safe_float(math.nan) is None


def test_safe_float_handles_sqlite_blob_float_values():
    assert safe_float(np.float64(0.625).tobytes()) == 0.625
