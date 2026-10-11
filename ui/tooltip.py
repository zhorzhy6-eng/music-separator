"""
Всплывающая подсказка для виджетов CustomTkinter.

Зачем: раньше пояснения («torchcrepe — монофонический трекер, для полифонии
нужен другой путь») висели на экране всегда и занимали место независимо от того,
думает человек об этом или нет. Подсказка нужна ровно в тот момент, когда мышь
наведена на кнопку, — тогда она и появляется.

Устройство простое:
  * при наведении запускается таймер (по умолчанию 0.5 с) — случайное движение
    мыши через кнопку подсказок не вызывает;
  * подсказка — отдельное окно без рамки поверх всех окон; оно НЕ перехватывает
    мышь, поэтому навести курсор и уйти дальше можно спокойно;
  * при уходе курсора (или клике) подсказка гасится сразу.

Текст можно задать строкой или функцией без аргументов — функция вызывается
перед показом. Это нужно для подсказок, которые зависят от состояния
(например, «выключено, пока не загружен файл»).
"""
from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable

import customtkinter as ctk

logger = logging.getLogger(__name__)

# Насколько отступить от курсора/виджета, чтобы подсказка не липла к мыши
OFFSET_X = 14
OFFSET_Y = 22
SCREEN_MARGIN = 8


class ToolTip:
    """Подсказка, которая появляется при наведении мыши на виджет."""

    def __init__(self, widget, text: str | Callable[[], str], delay_ms: int = 500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tooltip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        # Клик — обычно открывает окно/меню поверх: подсказка там только мешает
        widget.bind("<Button-1>", self._on_leave, add="+")

    # ---------- события ----------
    def _on_enter(self, event=None) -> None:
        self._cancel_timer()
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except Exception:
            logger.debug("Не удалось запланировать подсказку", exc_info=True)

    def _on_leave(self, event=None) -> None:
        self._cancel_timer()
        self._hide()

    def _cancel_timer(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                logger.debug("Не удалось отменить таймер подсказки", exc_info=True)
            self._after_id = None

    # ---------- показ/скрытие ----------
    def _resolve_text(self) -> str:
        text = self.text
        if callable(text):
            try:
                text = text()
            except Exception:
                logger.exception("Текст подсказки не собрался")
                return ""
        return str(text or "").strip()

    def _show(self) -> None:
        self._after_id = None
        text = self._resolve_text()
        if not text or self._tooltip is not None:
            return
        try:
            self._tooltip = self._build(text)
            self._place(self._tooltip)
        except Exception:
            logger.exception("Не удалось показать подсказку")
            self._hide()

    def _build(self, text: str) -> tk.Toplevel:
        """
        Окно подсказки.

        DECISION: обычный tk.Toplevel с tk.Label, а не CTkFrame. Подсказка —
        служебное окошко без своей логики: ему нужны только цвет, шрифт и
        мгновенный отклик, а CustomTkinter рисует свои виджеты через Canvas и
        на таймингах наведения заметно «тормозит».
        """
        dark = ctk.get_appearance_mode() == "Dark"
        bg = "#2b2b2b" if dark else "#fffbe6"
        fg = "#f0f0f0" if dark else "#1a1a1a"
        border = "#555555" if dark else "#c9c28f"

        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)          # без заголовка и рамки
        tip.wm_attributes("-topmost", True)

        frame = tk.Frame(tip, background=border, borderwidth=1, relief="solid")
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame, text=text, justify="left", anchor="w",
            background=bg, foreground=fg,
            font=("Segoe UI", 9), wraplength=420,
            padx=10, pady=7,
        ).pack(fill="both", expand=True)
        return tip

    def _place(self, tip: tk.Toplevel) -> None:
        """Ставит подсказку рядом с виджетом, не давая уехать за край экрана."""
        tip.update_idletasks()
        w, h = tip.winfo_width(), tip.winfo_height()
        screen_w = tip.winfo_screenwidth()
        screen_h = tip.winfo_screenheight()

        try:
            x = self.widget.winfo_pointerx() + OFFSET_X
            y = self.widget.winfo_pointery() + OFFSET_Y
        except Exception:
            x = self.widget.winfo_rootx() + OFFSET_X
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        x = max(SCREEN_MARGIN, min(x, screen_w - w - SCREEN_MARGIN))
        y = max(SCREEN_MARGIN, min(y, screen_h - h - SCREEN_MARGIN))
        tip.wm_geometry(f"+{int(x)}+{int(y)}")

    def _hide(self) -> None:
        if self._tooltip is None:
            return
        try:
            self._tooltip.destroy()
        except Exception:
            logger.debug("Подсказка уже закрыта", exc_info=True)
        self._tooltip = None

    def destroy(self) -> None:
        """Убирает подсказку и её таймер (нужно, когда виджет удаляют)."""
        self._cancel_timer()
        self._hide()


__all__ = ["ToolTip"]
