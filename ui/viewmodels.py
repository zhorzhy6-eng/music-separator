"""
UiState — единое состояние окна.

Зачем: раньше окно держало пути в четырёх отдельных полях и после каждой
операции вручную включало-выключало девять кнопок в шести местах. Ошибка в одном
месте — и кнопка «повисала» включённой, уводя работу на файлы прошлого трека
(баг №3 из QA-прогона). Теперь состояние одно, а кнопки рисуются из него
функцией _refresh_actions().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UiState:
    """Что сейчас загружено в окне и занято ли оно работой."""

    source: Path | None = None            # исходный файл
    vocals: Path | None = None            # вокал текущего трека
    instrumental: Path | None = None      # минус текущего трека
    midi: Path | None = None              # последний MIDI текущего трека
    busy: bool = False                    # идёт задача — кнопки не трогать

    # ---------- готовность шагов ----------
    # DECISION: окно спрашивает состояние («есть ли файл?»), а не проверяет
    # пути в каждом обработчике. Проверка идёт по факту: файл мог не сохраниться
    # или его удалили из output/ руками — тогда кнопка обязана быть выключена.

    @property
    def has_source(self) -> bool:
        """Исходный трек загружен и лежит на диске."""
        return self.source is not None and Path(self.source).exists()

    @property
    def has_stems(self) -> bool:
        """Есть и вокал, и минус — можно делать транскрипцию и супер-анализ."""
        return (self.vocals is not None and Path(self.vocals).exists()
                and self.instrumental is not None
                and Path(self.instrumental).exists())

    @property
    def has_midi(self) -> bool:
        """Есть готовый MIDI — можно делать MIDI-анализ."""
        return self.midi is not None and Path(self.midi).exists()

    def reset_for_new_file(self, path: Path) -> None:
        """
        Новый трек: старые результаты к нему не относятся.

        Без этого «Пианино» после загрузки другого трека брало минус прошлого,
        а сохраняло результат под именем нового.
        """
        self.source = Path(path)
        self.vocals = None
        self.instrumental = None
        self.midi = None
        self.busy = False

    def mark_busy(self, busy: bool = True) -> None:
        self.busy = busy
