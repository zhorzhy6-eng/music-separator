"""
Shim. Перенаправляет в core.engines.bs_engine.

Логика BS-Roformer переехала в пакет движков (этап 3 рефакторинга). Этот файл
оставлен, чтобы старые импорты продолжали работать:

    from core.separator_bs import BSAudioSeparator     # работает
    from core.separator_bs import find_dead_shebang    # работает

Новый код должен импортировать напрямую:
    from core.engines import make_separator, get_engine
"""
from __future__ import annotations

from core.engines.bs_engine import (  # noqa: F401
    CLIP_SAFETY_PEAK,
    CLI_NORMALIZATION,
    FIX_VENV_HINT,
    INSTR_HPF_HZ,
    INSTR_HPF_ORDER,
    INSTR_LPF_HZ,
    INSTR_LPF_ORDER,
    INSTR_NOTCH_GAIN_DB_REAL,
    INSTR_NOTCH_HZ,
    INSTR_NOTCH_Q,
    SUM_LIMITER_ATTACK_MS,
    SUM_LIMITER_LIMIT,
    SUM_LIMITER_RELEASE_MS,
    BSAudioSeparator,
    BSEngine,
    BSSeparationError,
    assert_launcher_alive,
    find_dead_shebang,
)

__all__ = [
    "BSAudioSeparator", "BSEngine", "BSSeparationError",
    "find_dead_shebang", "assert_launcher_alive", "FIX_VENV_HINT",
    "INSTR_HPF_HZ", "INSTR_HPF_ORDER", "INSTR_NOTCH_HZ", "INSTR_NOTCH_Q",
    "INSTR_NOTCH_GAIN_DB_REAL", "INSTR_LPF_HZ", "INSTR_LPF_ORDER",
    "CLIP_SAFETY_PEAK", "CLI_NORMALIZATION", "SUM_LIMITER_LIMIT",
    "SUM_LIMITER_ATTACK_MS", "SUM_LIMITER_RELEASE_MS",
]
