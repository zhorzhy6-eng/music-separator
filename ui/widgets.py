"""
Мелкие виджеты с оформлением: заголовки, подписи, разделители.

Зачем: раньше в каждом файле интерфейса писался свой размер шрифта и свой цвет
(«13 bold», «11», кортеж `("#6a6a6a", "#9a9a9a")»). Теперь это одна функция на
каждую роль текста, и поменять их вид можно в одном месте.

Роли:
    title_label    — заголовок вкладки/блока: 15 px, жирный
    card_label     — заголовок карточки: 13 px
    body_label     — основной текст: 12 px
    caption_label  — подпись, пояснение: 11 px, приглушённый
"""
from __future__ import annotations

import logging

import customtkinter as ctk

from ui.button_styles import _font
from ui.theme import FONT_BODY, FONT_CAPTION, FONT_CARD, FONT_TAB

logger = logging.getLogger(__name__)


def title_label(master, text: str, theme: dict, **kw) -> ctk.CTkLabel:
    """Заголовок вкладки или крупного блока."""
    return ctk.CTkLabel(
        master, text=text, anchor="w",
        font=_font(FONT_TAB, bold=True), text_color=theme["text"],
        **kw,
    )


def card_label(master, text: str, theme: dict, **kw) -> ctk.CTkLabel:
    """Заголовок карточки."""
    return ctk.CTkLabel(
        master, text=text, anchor="w",
        font=_font(FONT_CARD, bold=True), text_color=theme["text_dim"],
        **kw,
    )


def body_label(master, text: str, theme: dict, **kw) -> ctk.CTkLabel:
    """Основной текст."""
    return ctk.CTkLabel(
        master, text=text, anchor="w", justify="left",
        font=_font(FONT_BODY), text_color=theme["text"],
        **kw,
    )


def caption_label(master, text: str, theme: dict, **kw) -> ctk.CTkLabel:
    """Подпись и пояснение: мельче и тише основного текста."""
    return ctk.CTkLabel(
        master, text=text, anchor="w", justify="left",
        font=_font(FONT_CAPTION), text_color=theme["text_dim"],
        **kw,
    )


def muted_label(master, text: str, theme: dict, **kw) -> ctk.CTkLabel:
    """Третичный текст: то, что читают один раз (подсказки в настройках)."""
    label = caption_label(master, text, theme, **kw)
    label.configure(text_color=theme["text_muted"])
    return label


def separator(master, theme: dict, **kw) -> ctk.CTkFrame:
    """Тонкая линия-разделитель."""
    return ctk.CTkFrame(master, height=1, fg_color=theme["border"],
                        corner_radius=0, **kw)


def restore_main_focus(widget) -> None:
    """
    Возвращает клавиатуру главному окну после закрытия всплывающего.

    Зачем: всплывающие окна (список моделей, меню «⚙») при открытии забирают
    фокус на себя. Когда их уничтожают, Tk фокус НЕ возвращает — он остаётся
    ни на чём (focus_get() → None), и горячие клавиши плеера (→ ← Shift+→
    Space Home End) перестают работать, пока человек не щёлкнет по окну.
    Проверено: после закрытия списка моделей стрелки перестали двигать плеер,
    а focus_get() возвращал None.

    Именно focus_force(), а не focus_set(): focus_set в этой ситуации молча
    ничего не делает (Tk считает, что окно «и так не в фокусе») — проверено
    на живом окне. Чужое окно при этом не выдёргивается: человек только что
    кликнул внутри нашего.
    """
    try:
        root = widget.winfo_toplevel()
        if root is not None and root.winfo_exists():
            root.focus_force()
    except Exception:
        logger.debug("Фокус не вернулся главному окну", exc_info=True)


__all__ = [
    "title_label", "card_label", "body_label", "caption_label", "muted_label",
    "separator", "restore_main_focus",
]
