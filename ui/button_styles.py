"""
Четыре вида кнопок: главная, второстепенная, тихая и опасная.

Зачем: раньше кнопки выглядели одинаково независимо от важности — «Разделить»
(ради чего всё и запускается) не отличалась от «Открыть папку». Теперь у каждой
кнопки есть роль, и по одному взгляду на экран понятно, что тут главное:

    primary_button   — заливка акцентом, белый текст. Одно главное действие:
                       «Разделить на вокал и минус», «Сохранить» в настройках.
    secondary_button — обводка акцентом, прозрачный фон. Действия второго плана:
                       получить ноты, проверить качество.
    ghost_button     — без рамки, серый текст. Служебное: открыть папку, меню.
    danger_button    — красная. Останавливает работу («Отмена») — чтобы не
                       нажать случайно, она выглядит иначе, чем все остальные.

Иконки подставляются автоматически: если PNG нет (не запускали
tools/make_icons.py), на кнопке останется запасной символ из icons_catalog.
"""
from __future__ import annotations

import logging

import customtkinter as ctk

from ui import icons as icon_lib
from ui.icons_catalog import EMOJI_FALLBACK, ICON_NAMES


def emoji_fallback(name: str) -> str:
    """Запасной символ вместо иконки. Реэкспорт: им пользуется окно и меню."""
    return EMOJI_FALLBACK.get(name, "")
from ui.theme import (
    FONT_BUTTON,
    HEIGHT_BUTTON,
    HEIGHT_BUTTON_SMALL,
    HEIGHT_INPUT,
    RADIUS_BUTTON,
    RADIUS_INPUT,
    font_spec,
)

logger = logging.getLogger(__name__)

# Запасное семейство на случай, если theme ещё не видела живого окна
_DEFAULT_FAMILY = "Segoe UI"


def _font(size: int = FONT_BUTTON, bold: bool = False, mono: bool = False):
    """Шрифт кнопки. Кортеж понимают и CustomTkinter, и обычные tk-виджеты."""
    try:
        return font_spec(size, "bold" if bold else "normal", mono=mono)
    except Exception:                                # noqa: BLE001
        logger.debug("Не удалось подобрать шрифт", exc_info=True)
        return (_DEFAULT_FAMILY, size, "bold") if bold else (_DEFAULT_FAMILY, size)


def icon_image(name: str, size: int, color: str, master=None):
    """Картинка иконки или None. Цвет — строка '#RRGGBB'."""
    try:
        return icon_lib.get_icon(name, size=size, color=color, master=master)
    except Exception:                                # noqa: BLE001
        logger.debug("Иконка %s не загрузилась", name, exc_info=True)
        return None


def button_text(text: str, icon: str | None) -> str:
    """
    Подпись кнопки с запасным символом.

    Если иконки готовы — текст остаётся чистым («Разделить на вокал и минус»),
    картинка встанет слева от него. Если иконок нет — подставляем ОДИН символ
    в начало, чтобы кнопку всё равно было видно (раньше их было по два-три).
    """
    if not icon:
        return text
    return text


def _label_with_fallback(text: str, icon: str | None, img) -> str:
    """Текст кнопки: с иконкой — как есть, без иконки — с запасным символом."""
    if not icon or img is not None:
        return text
    fallback = icon_lib.emoji_fallback(icon)
    return f"{fallback} {text}" if fallback else text


def _build(master, text: str, command, theme: dict, colors: dict,
           icon: str | None, icon_size: int, height: int, radius: int,
           font_size: int, bold: bool, **kw) -> ctk.CTkButton:
    """Общая сборка: различаются только цвета, высота и шрифт."""
    img = icon_image(icon, icon_size, colors["text_color"], master=master) if icon else None
    label = _label_with_fallback(text, icon, img)
    button = ctk.CTkButton(
        master,
        text=label,
        command=command,
        image=img,
        compound="left" if img is not None else "text",
        fg_color=colors["fg_color"],
        hover_color=colors["hover_color"],
        text_color=colors["text_color"],
        text_color_disabled=theme["text_muted"],
        border_color=colors.get("border_color", theme["border"]),
        border_width=colors.get("border_width", 0),
        corner_radius=radius,
        height=height,
        font=_font(font_size, bold),
        **kw,
    )
    # Ссылка на картинку: CustomTkinter держит её сам, но так надёжнее при смене темы
    button._ast_icon = img
    button._ast_icon_name = icon
    return button


def primary_button(master, text: str, command, theme: dict,
                   icon: str | None = "scissors", icon_size: int = 20,
                   height: int = HEIGHT_BUTTON, font_size: int = FONT_BUTTON,
                   **kw) -> ctk.CTkButton:
    """Главное действие: заливка акцентом, белый текст."""
    return _build(master, text, command, theme, {
        "fg_color": theme["accent"],
        "hover_color": theme["accent_hover"],
        "text_color": theme["on_accent"],
    }, icon, icon_size, height, RADIUS_BUTTON, font_size, True, **kw)


def secondary_button(master, text: str, command, theme: dict,
                     icon: str | None = None, icon_size: int = 20,
                     height: int = HEIGHT_BUTTON_SMALL,
                     font_size: int = FONT_BUTTON, **kw) -> ctk.CTkButton:
    """
    Действие второго плана: обводка акцентом, светлая подложка.

    DECISION: подложка — не цвет панели и не «transparent». «Прозрачная»
    заливка у CustomTkinter берётся из его темы (серый), а цвет панели на
    светлой теме давал белую кнопку на белой карточке — рамку почти не видно.
    Отдельный цвет accent_soft делает кнопку заметной в обеих темах.
    """
    return _build(master, text, command, theme, {
        "fg_color": theme["accent_soft"],
        "hover_color": theme["panel_hover"],
        "text_color": theme["accent"],
        "border_color": theme["accent"],
        "border_width": 1,
    }, icon, icon_size, height, RADIUS_BUTTON, font_size, False, **kw)


def ghost_button(master, text: str, command, theme: dict,
                 icon: str | None = None, icon_size: int = 16,
                 height: int = HEIGHT_INPUT, font_size: int = 12,
                 **kw) -> ctk.CTkButton:
    """
    Служебное действие: без рамки, приглушённый текст.

    Фон — цвет панели: «прозрачный» у CustomTkinter тоже серый, а не фон окна.
    """
    return _build(master, text, command, theme, {
        "fg_color": theme["panel"],
        "hover_color": theme["panel_hover"],
        "text_color": theme["text_dim"],
    }, icon, icon_size, height, RADIUS_INPUT, font_size, False, **kw)


def danger_button(master, text: str, command, theme: dict,
                  icon: str | None = "x", icon_size: int = 20,
                  height: int = HEIGHT_BUTTON, font_size: int = FONT_BUTTON,
                  **kw) -> ctk.CTkButton:
    """Опасное действие (остановить работу): красная заливка."""
    return _build(master, text, command, theme, {
        "fg_color": theme["danger"],
        "hover_color": theme["danger_hover"],
        "text_color": theme["on_accent"],
    }, icon, icon_size, height, RADIUS_BUTTON, font_size, True, **kw)


def icon_button(master, icon: str, command, theme: dict, tooltip: str = "",
                size: int = 40, icon_size: int = 20, round_: bool = False,
                **kw) -> ctk.CTkButton:
    """
    Кнопка-значок без текста: меню «⚙», круглые кнопки плеера.

    `round_=True` — полный круг (радиус = половина высоты), как у плееров.
    """
    img = icon_image(icon, icon_size, theme["text"], master=master)
    label = "" if img is not None else icon_lib.emoji_fallback(icon)
    button = ctk.CTkButton(
        master,
        text=label,
        command=command,
        image=img,
        fg_color="transparent",
        hover_color=theme["panel_hover"],
        text_color=theme["text"],
        corner_radius=(size // 2) if round_ else RADIUS_BUTTON,
        width=size,
        height=size,
        font=_font(15),
        **kw,
    )
    button._ast_icon = img
    button._ast_icon_name = icon
    return button


# =============================================================
#  Раскраска состояния (тема сменилась / кнопка выключилась)
# =============================================================
# Цвета каждой роли — в одном месте: и при сборке, и при переключении темы
ROLE_COLORS = {
    "primary": lambda t: (t["accent"], t["accent_hover"], t["on_accent"]),
    "secondary": lambda t: (t["accent_soft"], t["panel_hover"], t["accent"]),
    "ghost": lambda t: (t["panel"], t["panel_hover"], t["text_dim"]),
    "danger": lambda t: (t["danger"], t["danger_hover"], t["on_accent"]),
    "icon": lambda t: ("transparent", t["panel_hover"], t["text"]),
    "icon_accent": lambda t: (t["accent"], t["accent_hover"], t["on_accent"]),
}


def recolor(button: ctk.CTkButton, role: str, theme: dict) -> None:
    """
    Перекрашивает кнопку под новую тему.

    Выключенную кнопку не трогаем: у неё свой приглушённый вид, и «оживить» её
    цветом значило бы обещать действие, которого сейчас нет.
    """
    if role not in ROLE_COLORS:
        logger.warning("Неизвестная роль кнопки: %r", role)
        return
    fg, hover, text_color = ROLE_COLORS[role](theme)
    try:
        button.configure(fg_color=fg, hover_color=hover, text_color=text_color,
                         text_color_disabled=theme["text_muted"],
                         border_color=(theme["accent"] if role == "secondary"
                                       else theme["border"]))
    except Exception:
        logger.debug("Кнопку не удалось перекрасить", exc_info=True)

    # Иконка: у акцентных кнопок она белая, у остальных — цвета текста
    icon_name = getattr(button, "_ast_icon_name", None)
    if icon_name in ICON_NAMES and getattr(button, "_ast_icon", None) is not None:
        size = button._ast_icon.cget("size")[0]
        img = icon_image(icon_name, size, text_color, master=button)
        if img is not None:
            try:
                button.configure(image=img)
                button._ast_icon = img
            except Exception:
                logger.debug("Иконку кнопки не удалось заменить", exc_info=True)


__all__ = [
    "primary_button", "secondary_button", "ghost_button", "danger_button",
    "icon_button", "recolor", "button_text", "icon_image", "emoji_fallback",
    "ROLE_COLORS",
]
