"""
Карточка — панель со скруглением, тонкой рамкой и заголовком.

Зачем: кнопки, разбросанные по пустому фону, читаются как «панель инженера» —
непонятно, что к чему относится. Карточка собирает связанное вместе и даёт
экранной форме порядок: «Источник» → «Способ разделения» → «Готово».

Как пользоваться:

    card = Card(parent, "Источник", theme)
    card.pack(fill="x", pady=(0, SPACE_BETWEEN))
    ...кладите виджеты в card.body...

Заголовок и подпись необязательны. Внутри карточки отступ 16 (SPACE_CARD),
заголовок отделён от содержимого 8 (SPACE_BLOCK) — как в остальном интерфейсе.
"""
from __future__ import annotations

import logging

import customtkinter as ctk

from ui.button_styles import icon_image
from ui.theme import (
    FONT_CARD,
    RADIUS_CARD,
    SPACE_BLOCK,
    SPACE_CARD,
    font_spec,
)

logger = logging.getLogger(__name__)


class Card(ctk.CTkFrame):
    """Панель с заголовком. Содержимое складывать в `card.body`."""

    def __init__(self, master, title: str = "", theme: dict | None = None,
                 subtitle: str = "", icon: str | None = None, **kw):
        from ui.theme import get_theme

        theme = theme or get_theme("dark")
        self.theme = theme
        super().__init__(
            master,
            fg_color=theme["panel"],
            corner_radius=RADIUS_CARD,
            border_width=1,
            border_color=theme["border"],
            **kw,
        )

        self.header: ctk.CTkFrame | None = None
        self.icon_label: ctk.CTkLabel | None = None
        self.title_label: ctk.CTkLabel | None = None
        self.subtitle_label: ctk.CTkLabel | None = None

        if title or subtitle:
            self.header = ctk.CTkFrame(self, fg_color="transparent")
            self.header.pack(fill="x", padx=SPACE_CARD, pady=(SPACE_CARD, SPACE_BLOCK))

            if icon:
                img = icon_image(icon, 16, theme["text_dim"], master=self)
                if img is not None:
                    self.icon_label = ctk.CTkLabel(self.header, text="", image=img)
                    self.icon_label.pack(side="left", padx=(0, 6))

            if title:
                self.title_label = ctk.CTkLabel(
                    self.header, text=title, anchor="w",
                    font=font_spec(FONT_CARD, "bold"),
                    text_color=theme["text_dim"],
                )
                self.title_label.pack(side="left")

            if subtitle:
                self.subtitle_label = ctk.CTkLabel(
                    self.header, text=subtitle, anchor="e",
                    font=font_spec(11), text_color=theme["text_muted"],
                )
                self.subtitle_label.pack(side="right")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        top = 0 if self.header is not None else SPACE_CARD
        self.body.pack(fill="both", expand=True, padx=SPACE_CARD,
                       pady=(top, SPACE_CARD))

    # ---------- оформление ----------
    def apply_theme(self, theme: dict) -> None:
        """Перекрашивает карточку под новую тему."""
        self.theme = theme
        try:
            self.configure(fg_color=theme["panel"], border_color=theme["border"])
        except Exception:
            logger.debug("Карточку не удалось перекрасить", exc_info=True)
        if self.title_label is not None:
            self.title_label.configure(text_color=theme["text_dim"])
        if self.subtitle_label is not None:
            self.subtitle_label.configure(text_color=theme["text_muted"])

    def set_subtitle(self, text: str) -> None:
        """
        Подпись справа в заголовке (например, «готово»).

        Если подписи в карточке не было, она создаётся — вызывающему коду не
        надо помнить, заказывал он её при сборке или нет.
        """
        if self.subtitle_label is None:
            parent = self.header or self
            self.subtitle_label = ctk.CTkLabel(
                parent, text="", anchor="e",
                font=font_spec(11), text_color=self.theme["text_muted"])
            self.subtitle_label.pack(side="right")
        try:
            self.subtitle_label.configure(text=text)
        except Exception:
            logger.debug("Подпись карточки не обновилась", exc_info=True)

    # ---------- удобные сокращения ----------
    def add(self, widget):
        """Кладёт готовый виджет в тело карточки и возвращает его же."""
        widget.pack(fill="x")
        return widget


__all__ = ["Card"]
