"""
Shim. Перенаправляет в core.engines.hybrid_engine.

Логика гибрида переехала в пакет движков (этап 3 рефакторинга). Этот файл
оставлен, чтобы старые импорты продолжали работать:

    from core.separator_hybrid import HybridSeparator   # работает

Новый код должен импортировать напрямую:
    from core.engines import make_separator, get_engine
"""
from __future__ import annotations

from core.engines.hybrid_engine import (  # noqa: F401
    HybridEngine,
    HybridSeparator,
    HybridSeparationError,
)

__all__ = ["HybridSeparator", "HybridEngine", "HybridSeparationError"]
