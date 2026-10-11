"""
Общие утилиты, которые раньше дублировались в analyzer.py, super_analyzer.py
и midi_analyzer.py.
"""
from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def to_db(x: float) -> float:
    """Амплитуда → дБ. Ноль и отрицательные дают -200 дБ вместо ошибки."""
    if x <= 1e-10:
        return -200.0
    return 20.0 * math.log10(x)


def dir_size_bytes(path: Path) -> int:
    """Суммарный размер всех файлов в каталоге."""
    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    except Exception:
        return 0


def cleanup_temp_dirs(dirs: Iterable[Path]) -> None:
    """
    Удаляет временные каталоги и пишет в лог, сколько места освободилось.

    Нужна, чтобы output/ не распухал: каждый прогон гибрида оставлял там
    ~163 МБ, а FLAC-и от BS лежали до следующего запуска.
    """
    removed = 0
    freed = 0
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        size = dir_size_bytes(p)
        try:
            shutil.rmtree(p)
            removed += 1
            freed += size
            logger.info("Временная папка удалена",
                        extra={"ctx_dir": str(p),
                               "ctx_freed_mb": round(size / 1024 / 1024, 1)})
        except Exception:
            logger.warning("Не удалось удалить временную папку %s", p, exc_info=True)
    if removed:
        logger.info("Очищено временных папок: %d, освобождено %.1f МБ",
                    removed, freed / 1024 / 1024)


def json_safe(obj):
    """
    Рекурсивно приводит numpy-типы к чистым Python-типам,
    чтобы json.dumps мог их сериализовать.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
