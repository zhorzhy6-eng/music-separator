"""
Панель настроек — вкладка «⚙ Настройки».

Раньше это был отдельный модальный диалог: чтобы поменять устройство Demucs,
приходилось открывать окно поверх окна. Теперь настройки — обычная вкладка:
видно, что в них есть, и не мешает работать.

Что осталось прежним (и это важно):
  * поля описываются одним списком Field — по нему строится интерфейс, проверка
    значений и сброс. Добавить настройку = строка здесь и поле в core/config.py;
  * проверка и сборка значений — чистые функции (validate_values / apply_values),
    их проверяют тесты без окна;
  * панель НЕ пишет файл сама. Она отдаёт готовый AppConfig в on_save, а что с
    ним делать, решает окно (ui/app_window.py): сохранить, обновить кэш настроек,
    перечитать интерфейс.
"""
from __future__ import annotations

import logging
from tkinter import messagebox
from typing import Any

import customtkinter as ctk

from core.config import AppConfig
from core.engines import resolve_engine_id
from ui.button_styles import primary_button, recolor, secondary_button
from ui.card import Card
from ui.engine_selector import EngineSelector
from ui.settings_dialog import (
    FIELDS,
    Field,
    apply_values,
    config_to_values,
    validate_values,
)
from ui.theme import SPACE_BLOCK, SPACE_CARD, SPACE_TIGHT, font_spec, get_theme
from ui.tooltip import ToolTip
from ui.widgets import caption_label, muted_label

logger = logging.getLogger(__name__)

SECTION_ORDER = ("Разделение", "BS-Roformer (Python-API)", "Звук и файлы")

# Понятные названия разделов: «BS-Roformer (Python-API)» — это про внутренности
SECTION_TITLES = {
    "Разделение": "Как разделять",
    "BS-Roformer (Python-API)": "Тонкая настройка моделей Roformer",
    "Звук и файлы": "Звук и файлы",
}


class SettingsPanel(ctk.CTkFrame):
    """Панель настроек: все поля AppConfig, кнопки «Сохранить» и «Сбросить»."""

    def __init__(self, master, config: AppConfig, on_save,
                 theme: dict | None = None, **kw) -> None:
        super().__init__(master, fg_color="transparent", **kw)
        self.config_data = config
        self.on_save = on_save
        self.theme = theme or get_theme("dark")
        self.vars: dict[str, Any] = {}
        self._widgets: dict[str, Any] = {}
        self._value_labels: dict[str, ctk.CTkLabel] = {}
        self._hints: list[ctk.CTkLabel] = []
        self._field_labels: list[ctk.CTkLabel] = []
        self._cards: list[Card] = []
        self._build()
        self._load(config)

    # ---------- сборка ----------
    def _build(self) -> None:
        theme = self.theme
        body = ctk.CTkScrollableFrame(self, label_text="",
                                      fg_color=theme["bg"],
                                      scrollbar_button_color=theme["border"],
                                      scrollbar_button_hover_color=theme["text_muted"])
        body.pack(fill="both", expand=True, padx=SPACE_TIGHT, pady=SPACE_TIGHT)
        self._body = body

        caption_label(
            body,
            "Настройки нужны редко: по умолчанию приложение уже настроено. "
            "После изменений нажмите «Сохранить» — они начнут действовать сразу, "
            "без перезапуска.",
            theme, wraplength=900,
        ).pack(fill="x", padx=SPACE_BLOCK, pady=(SPACE_BLOCK, SPACE_BLOCK))

        for section in SECTION_ORDER:
            fields = [f for f in FIELDS if f.section == section]
            if not fields:
                continue
            card = Card(body, SECTION_TITLES.get(section, section), theme)
            card.pack(fill="x", padx=SPACE_TIGHT, pady=(0, SPACE_BLOCK))
            self._cards.append(card)
            for field in fields:
                self._build_field(card.body, field)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=SPACE_BLOCK, pady=(SPACE_TIGHT, SPACE_TIGHT))
        self.buttons_row = buttons

        self.btn_save = primary_button(buttons, "Сохранить", self._save, theme,
                                       icon="save", width=170)
        self.btn_save.pack(side="left", padx=(0, SPACE_BLOCK))
        ToolTip(self.btn_save, "Записывает настройки в config/settings.json. "
                               "Новые значения начинают действовать сразу.")

        self.btn_reset = secondary_button(buttons, "Вернуть как было", self._reset,
                                          theme, icon="rotate-ccw", width=200)
        self.btn_reset.pack(side="left")
        ToolTip(self.btn_reset, "Ставит значения по умолчанию в поля, "
                                "но файл не трогает — сначала «Сохранить».")

        self.status = ctk.CTkLabel(self, text="", anchor="w",
                                   font=font_spec(12), text_color=theme["success"])
        self.status.pack(fill="x", padx=SPACE_CARD, pady=(0, SPACE_TIGHT))

    def _build_field(self, parent, field: Field) -> None:
        theme = self.theme
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=SPACE_BLOCK)

        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        label = ctk.CTkLabel(top, text=field.label, width=250, anchor="w",
                             font=font_spec(12), text_color=theme["text"])
        label.pack(side="left")
        self._field_labels.append(label)

        if field.name == "default_engine":
            # DECISION: движок по умолчанию выбирается тем же группированным
            # списком, что и на вкладке «Разделение». Иначе здесь был бы снова
            # плоский список из 26 строк — ровно то, от чего уходили.
            selector = EngineSelector(top, width=300, theme=theme)
            selector.pack(side="left")
            self._widgets[field.name] = selector
            self.vars[field.name] = ctk.StringVar()
        elif field.kind == "choice":
            var = ctk.StringVar()
            widget = ctk.CTkOptionMenu(
                top, values=list(field.choices), variable=var, width=200,
                height=32, corner_radius=6, font=font_spec(12),
                fg_color=theme["panel_hover"], button_color=theme["panel_hover"],
                button_hover_color=theme["border"], text_color=theme["text"],
                dropdown_fg_color=theme["panel"],
                dropdown_hover_color=theme["panel_hover"],
                dropdown_text_color=theme["text"])
            widget.pack(side="left")
            self._widgets[field.name] = widget
            self.vars[field.name] = var
        elif field.kind == "slider":
            var = ctk.DoubleVar()
            value_label = ctk.CTkLabel(top, text="", width=60,
                                       font=font_spec(12), text_color=theme["text_dim"])
            value_label.pack(side="left", padx=(0, 6))
            self._value_labels[field.name] = value_label
            widget = ctk.CTkSlider(
                top, from_=field.minimum, to=field.maximum,
                number_of_steps=max(1, int(round((field.maximum - field.minimum) / field.step))),
                variable=var, width=190, height=16,
                progress_color=theme["accent"], button_color=theme["accent"],
                button_hover_color=theme["accent_hover"], fg_color=theme["border"],
                command=lambda value, f=field: self._on_slider(f, value),
            )
            widget.pack(side="left")
            self._widgets[field.name] = widget
            self.vars[field.name] = var
        else:
            var = ctk.BooleanVar()
            widget = ctk.CTkSwitch(
                top, text="", variable=var, width=48,
                progress_color=theme["accent"], button_color=theme["panel"],
                fg_color=theme["border"], font=font_spec(12))
            widget.pack(side="left")
            self._widgets[field.name] = widget
            self.vars[field.name] = var

        if field.hint:
            # DECISION: подсказка остаётся видимым текстом, а не tooltip — в
            # настройках человек читает и сравнивает, а не «наводит и угадывает».
            hint = muted_label(row, field.hint, theme, wraplength=560)
            hint.pack(fill="x", padx=(250, 4))
            self._hints.append(hint)
            ToolTip(label, field.hint)

    def apply_theme(self, theme: dict) -> None:
        """Перекрашивает панель настроек под новую тему."""
        self.theme = theme
        try:
            self._body.configure(fg_color=theme["bg"],
                                 scrollbar_button_color=theme["border"],
                                 scrollbar_button_hover_color=theme["text_muted"])
            self.status.configure(text_color=theme["success"])
        except Exception:
            logger.debug("Панель настроек не перекрасилась", exc_info=True)

        for hint in self._hints:
            hint.configure(text_color=theme["text_muted"])

        # Поля: список, слайдер, переключатель и подпись значения
        for name, widget in self._widgets.items():
            try:
                if name == "default_engine":
                    widget.apply_theme(theme)
                elif isinstance(widget, ctk.CTkOptionMenu):
                    widget.configure(fg_color=theme["panel_hover"],
                                     button_color=theme["panel_hover"],
                                     button_hover_color=theme["border"],
                                     text_color=theme["text"],
                                     dropdown_fg_color=theme["panel"],
                                     dropdown_hover_color=theme["panel_hover"],
                                     dropdown_text_color=theme["text"])
                elif isinstance(widget, ctk.CTkSlider):
                    widget.configure(progress_color=theme["accent"],
                                     button_color=theme["accent"],
                                     button_hover_color=theme["accent_hover"],
                                     fg_color=theme["border"])
                elif isinstance(widget, ctk.CTkSwitch):
                    widget.configure(progress_color=theme["accent"],
                                     button_color=theme["panel"],
                                     fg_color=theme["border"])
            except Exception:
                logger.debug("Поле %s не перекрасилось", name, exc_info=True)

        for label in self._value_labels.values():
            label.configure(text_color=theme["text_dim"])
        for label in self._field_labels:
            label.configure(text_color=theme["text"])

        # Кнопки панели — через их роли (см. ui/button_styles.py)
        recolor(self.btn_save, "primary", theme)
        recolor(self.btn_reset, "secondary", theme)

    def _on_slider(self, field: Field, value: float) -> None:
        """Показывает текущее значение слайдера рядом с ним."""
        label = self._value_labels.get(field.name)
        if label is not None:
            label.configure(text=f"{value:.{field.decimal}f}")

    # ---------- значения ----------
    def _load(self, config: AppConfig) -> None:
        """Раскладывает настройки по виджетам."""
        values = config_to_values(config)
        for field in FIELDS:
            value = values.get(field.name)
            var = self.vars[field.name]
            widget = self._widgets[field.name]

            if field.name == "default_engine":
                engine_id = self._engine_id_of(str(value))
                widget.set(engine_id)
                var.set(engine_id)
                continue
            if field.kind == "switch":
                var.set(bool(value))
            elif field.kind == "choice":
                text = str(value)
                if field.choices and text not in field.choices:
                    # Значение не из списка (файл настроек правили руками) —
                    # показываем как есть, чтобы пользователь видел правду.
                    logger.warning("Значение %s=%r вне списка допустимых", field.name, text)
                    try:
                        widget.configure(values=[*field.choices, text])
                    except Exception:
                        logger.debug("Не удалось добавить %r в список", text, exc_info=True)
                var.set(text)
            else:
                var.set(float(value))
                self._on_slider(field, float(value))

    @staticmethod
    def _engine_id_of(value: str) -> str:
        """Значение из настроек (id или старая подпись) → id движка."""
        try:
            return resolve_engine_id(value)
        except KeyError:
            logger.warning("Движок по умолчанию не найден в реестре: %r", value)
            return ""

    def collect(self) -> dict[str, Any]:
        """Текущие значения всех полей (то, что уйдёт в настройки)."""
        values: dict[str, Any] = {}
        for field in FIELDS:
            try:
                if field.name == "default_engine":
                    values[field.name] = self._widgets[field.name].get()
                else:
                    values[field.name] = self.vars[field.name].get()
            except Exception:
                logger.exception("Не удалось прочитать поле %s", field.name)
                values[field.name] = getattr(self.config_data, field.name)
        return values

    def _reset(self) -> None:
        """Ставит значения по умолчанию. Файл не трогает — сначала «Сохранить»."""
        self._load(AppConfig())
        self.status.configure(text="Поля сброшены к значениям по умолчанию — "
                                  "нажмите «Сохранить»")
        logger.info("Поля настроек сброшены к значениям по умолчанию")

    def reload(self, config: AppConfig, message: str = "") -> None:
        """
        Перечитывает настройки в поля панели.

        Нужно после сохранения: окно могло применить не то, что лежало в полях
        (например, значение пришло из другого источника). Показываем человеку
        то, что реально записано в файл, а не то, что он набрал.
        """
        self.config_data = config
        self._load(config)
        if message:
            self.status.configure(text=message)

    # ---------- сохранение ----------
    def _save(self) -> None:
        values = self.collect()
        problems = validate_values(values)
        if problems:
            messagebox.showerror(
                "Настройки не сохранены",
                "Проверьте значения:\n\n" + "\n".join(f"• {p}" for p in problems),
                parent=self.winfo_toplevel(),
            )
            return

        new_config = apply_values(self.config_data, values)
        logger.info(
            "Настройки изменены на вкладке «Настройки»",
            extra={"ctx_changed": _diff(self.config_data, new_config)},
        )
        try:
            self.on_save(new_config)
        except Exception as e:
            logger.exception("Не удалось применить настройки")
            messagebox.showerror("Настройки", f"Настройки не применены: {e}",
                                 parent=self.winfo_toplevel())
            return
        self.config_data = new_config
        self.status.configure(text="✅ Настройки сохранены")


def _diff(old: AppConfig, new: AppConfig) -> dict:
    """Что именно поменялось — для лога (и чтобы не писать в лог весь конфиг)."""
    changed = {}
    for field in FIELDS:
        before, after = getattr(old, field.name, None), getattr(new, field.name, None)
        if before != after:
            changed[field.name] = f"{before} → {after}"
    return changed


__all__ = ["SettingsPanel", "SECTION_ORDER"]
