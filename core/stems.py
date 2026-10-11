"""
StemSet — типизированный результат разделения.

Раньше разделение возвращало «голый» словарь dict[str, Path], и каждый, кто его
получал, угадывал форму: есть ли ключ, как он называется, не опечатка ли в нём.
Теперь форма одна и проверяется сразу.

Совместимость: StemSet ведёт себя как прежний словарь — работает result["vocals"],
result.get("vocals"), "vocals" in result и result["vocals"] is not None.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class StemSet:
    """Пути к сохранённым стемам. None — стема нет."""

    vocals: Path | None = None
    instrumental: Path | None = None
    drums: Path | None = None
    bass: Path | None = None
    other: Path | None = None

    def as_dict(self) -> dict[str, Path]:
        """Только заполненные стемы: {имя: путь}."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __getitem__(self, key: str) -> Path:
        val = getattr(self, key, None)
        if val is None:
            raise KeyError(f"Стем не найден: {key}")
        return val

    def __contains__(self, key: str) -> bool:
        return getattr(self, key, None) is not None

    def __iter__(self):
        """Итерация по загруженным стемам — как у прежнего словаря."""
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def get(self, key: str, default=None):
        """
        Как у словаря: нет стема — вернём default.

        Незаполненный стем здесь равен отсутствующему: код вправе писать
        result.get("vocals", что-то), и это должно работать.
        """
        val = getattr(self, key, None)
        return default if val is None else val

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()


# Имена стемов, которые понимает StemSet (для проверок и тестов)
STEM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(StemSet))
