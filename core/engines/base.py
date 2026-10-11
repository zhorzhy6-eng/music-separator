"""
Общий контракт сепараторов.

Любой движок (Demucs, BS-Roformer, гибрид) обязан уметь одно и то же:
принять файл, папку для результата и необязательный колбэк прогресса,
вернуть StemSet. Всё остальное — внутреннее дело движка.

Зачем: раньше у каждого сепаратора был свой набор методов и своя форма
результата, и вызывающий код (воркер, UI, CLI) должен был знать, с кем именно
он сейчас разговаривает. Теперь он знает только этот договор.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:  # только для аннотаций — на импорт не влияет
    from core.stems import StemSet

# progress_cb(доля 0..1, что делаем прямо сейчас)
ProgressCb = Callable[[float, str], None]


class EngineCancelled(Exception):
    """
    Работу прервал пользователь (кнопка «Отмена» или закрытие окна).

    Отдельный тип нужен, чтобы воркер и UI отличали отмену от настоящей ошибки:
    отменённая задача — это не сбой, пугать человека красным окном незачем.
    """


@runtime_checkable
class Separator(Protocol):
    """Договор: что должен уметь любой движок разделения."""

    name: str

    def separate(
        self,
        src: Path,
        out_dir: Path,
        progress_cb: ProgressCb | None = None,
    ) -> "StemSet": ...

    def cancel(self) -> bool:
        """Прервать работу, если её можно прервать. True — прервали."""
        ...
