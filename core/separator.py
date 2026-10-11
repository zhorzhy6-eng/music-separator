"""
Shim. Перенаправляет в core.engines.demucs_engine.

Логика Demucs переехала в пакет движков (этап 3 рефакторинга). Этот файл
оставлен, чтобы старые импорты продолжали работать:

    from core.separator import AudioSeparator      # работает
    from core.separator import SeparationError     # работает

Новый код должен импортировать напрямую:
    from core.engines import make_separator, get_engine
"""
from __future__ import annotations

from core.engines.demucs_engine import (  # noqa: F401
    INSTR_NOTCH_GAIN_DB_REAL,
    INSTR_NOTCH_HZ,
    INSTR_NOTCH_Q,
    MODEL_SR,
    VOCALS_HPF_HZ,
    VOCALS_HPF_ORDER,
    AudioSeparator,
    DemucsEngine,
    SeparationError,
)

__all__ = [
    "AudioSeparator", "DemucsEngine", "SeparationError",
    "MODEL_SR", "VOCALS_HPF_HZ", "VOCALS_HPF_ORDER",
    "INSTR_NOTCH_HZ", "INSTR_NOTCH_Q", "INSTR_NOTCH_GAIN_DB_REAL",
]
