from __future__ import annotations

import math
import struct


__all__ = ["safe_float", "safe_int"]


def safe_int(value) -> int | None:
    """Convert values from pandas/numpy/SQLite into a plain int when possible."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _safe_int_from_bytes(bytes(value))

    if hasattr(value, "item"):
        try:
            return safe_int(value.item())
        except (TypeError, ValueError):
            pass

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def safe_float(value) -> float | None:
    """Convert values from pandas/numpy/SQLite into a plain float when possible."""
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _safe_float_from_bytes(bytes(value))

    if hasattr(value, "item"):
        try:
            return safe_float(value.item())
        except (TypeError, ValueError):
            pass

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if math.isnan(result):
        return None
    return result


def _safe_int_from_bytes(raw: bytes) -> int | None:
    if not raw:
        return None

    try:
        text = raw.decode("utf-8").strip().strip("\x00")
    except UnicodeDecodeError:
        text = ""

    if text:
        try:
            return int(text)
        except ValueError:
            pass

    candidates: list[int] = []
    for byteorder in ("little", "big"):
        try:
            candidates.append(int.from_bytes(raw, byteorder=byteorder, signed=True))
        except (OverflowError, ValueError):
            continue

    for candidate in candidates:
        if -1 <= candidate <= 10:
            return candidate

    return candidates[0] if candidates else None


def _safe_float_from_bytes(raw: bytes) -> float | None:
    if not raw:
        return None

    try:
        text = raw.decode("utf-8").strip().strip("\x00")
    except UnicodeDecodeError:
        text = ""

    if text:
        try:
            result = float(text)
            return None if math.isnan(result) else result
        except ValueError:
            pass

    for format_code in ("<d", ">d", "<f", ">f"):
        try:
            result = struct.unpack(format_code, raw)[0]
        except struct.error:
            continue
        if not math.isnan(result):
            return float(result)

    return None
