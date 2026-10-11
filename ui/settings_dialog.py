"""
Диалог настроек: всё, что лежит в config/settings.json, правится мышкой.

Зачем: настройки влияли на результат (устройство Demucs, целевой пик, notch,
частота вывода), но менять их можно было только руками в JSON — с риском сломать
файл и без подсказок, что вообще допустимо.

Как устроено:
  * описание полей (Field) — один список, по нему строится и интерфейс, и
    проверка значений, и сброс к значениям по умолчанию. Добавить настройку =
    добавить одну строку сюда и поле в core/config.py;
  * типы значений берутся из AppConfig (dataclass), а не из интерфейса: если
    поле в конфиге целое, слайдер всё равно вернёт int;
  * проверка значений вынесена в чистые функции (validate_values/apply_values) —
    их можно тестировать без окна.

Применение: диалог НЕ пишет файл сам. Он отдаёт готовый AppConfig в on_save, а
что с ним делать, решает окно (ui/app_window.py): сохранить, обновить кэш
настроек и перечитать состояние интерфейса.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from tkinter import messagebox
from typing import Any, Literal

import customtkinter as ctk

from core.config import AppConfig
from core.engines import all_labels, resolve_engine_id

logger = logging.getLogger(__name__)

# Разделы диалога: заголовок и что в нём лежит
SECTION_ORDER = ("Разделение", "BS-Roformer (Python-API)", "Звук и файлы")


@dataclass(frozen=True)
class Field:
    """Описание одной настройки: как показать и как проверить."""

    name: str
    label: str
    kind: Literal["choice", "slider", "switch"]
    section: str
    hint: str = ""
    choices: tuple[str, ...] = ()
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 1.0
    decimal: int = 0                  # сколько знаков после запятой показывать


FIELDS: tuple[Field, ...] = (
    # ---------- Разделение ----------
    # DECISION: подписи переписаны простым языком («Считать на видеокарте или
    # процессоре» вместо «Устройство для Demucs»). Термины вроде «shifts» и
    # «target_peak» оставлены только в подсказке — человеку без музыкального
    # образования они ничего не объясняют, а настройку найти мешают.
    Field(
        name="default_engine", label="Способ разделения по умолчанию", kind="choice",
        section="Разделение", choices=tuple(all_labels()),
        hint="Какая модель подставляется в список при запуске приложения. "
             "Менять стоит, только если результат не устраивает.",
    ),
    Field(
        name="demucs_device", label="Считать на видеокарте или процессоре",
        kind="choice", section="Разделение", choices=("cuda", "cpu"),
        hint="Видеокарта (cuda) — быстро. Процессор (cpu) — медленно, но работает "
             "на любом компьютере.",
    ),
    Field(
        name="demucs_shifts", label="Точность разделения", kind="slider",
        section="Разделение", minimum=1, maximum=10, step=1,
        hint="Сколько раз модель прогоняет трек, чтобы уточнить результат. "
             "Больше — чище и медленнее (внутри это называется shifts).",
    ),
    Field(
        name="demucs_overlap", label="Сглаживание стыков", kind="slider",
        section="Разделение", minimum=0.0, maximum=0.9, step=0.05, decimal=2,
        hint="Насколько куски трека перекрывают друг друга при склейке. "
             "0.25 — обычное значение, трогать не обязательно.",
    ),

    # ---------- BS-Roformer через Python-API ----------
    Field(
        name="bs_use_fp16", label="Экономить видеопамять", kind="switch",
        section="BS-Roformer (Python-API)",
        hint="Считать в половинной точности. Помогает, если видеопамяти мало; "
             "быстрее от этого не станет.",
    ),
    Field(
        name="bs_batch_size", label="Сколько кусков считать за раз", kind="choice",
        section="BS-Roformer (Python-API)", choices=("0", "1", "2", "4", "8"),
        hint="0 — как задано в самой модели. Больше — быстрее, но требует больше "
             "видеопамяти.",
    ),
    Field(
        name="bs_keep_model_loaded", label="Держать модель в памяти", kind="switch",
        section="BS-Roformer (Python-API)",
        hint="Экономит 5–10 секунд на каждом следующем треке, но занимает "
             "видеопамять, пока приложение открыто.",
    ),

    # ---------- Звук и файлы ----------
    Field(
        name="notch_enabled", label="Убирать горб на 2.5 кГц в минусе",
        kind="switch", section="Звук и файлы",
        hint="По умолчанию выключено: этот вырез слышен в гитарах и в атаке "
             "барабанов. Включайте, только если минус звучит гулко.",
    ),
    Field(
        name="target_peak", label="Громкость результата", kind="slider",
        section="Звук и файлы", minimum=0.5, maximum=0.99, step=0.001, decimal=3,
        hint="Один уровень на пару «вокал + минус», чтобы они не разъезжались "
             "по громкости. 0.891 ≈ −1 дБ от предела — безопасно для mp3.",
    ),
    Field(
        name="output_sample_rate", label="Частота выходных файлов", kind="choice",
        section="Звук и файлы", choices=("44100", "48000"),
        hint="44100 — как на CD и как считают модели. 48000 — если так удобнее "
             "в вашем редакторе.",
    ),
    Field(
        name="keep_temp_dirs", label="Оставлять временные папки", kind="switch",
        section="Звук и файлы",
        hint="Для разбора проблем: рабочие папки движков останутся в output/ "
             "и не удалятся.",
    ),
)

FIELDS_BY_NAME = {f.name: f for f in FIELDS}


# ============================================================
#  Значения: чтение, проверка, применение (без интерфейса)
# ============================================================
def defaults_values() -> dict[str, Any]:
    """Значения по умолчанию — из AppConfig, а не из интерфейса."""
    cfg = AppConfig()
    return {f.name: getattr(cfg, f.name) for f in FIELDS}


def config_to_values(config: AppConfig) -> dict[str, Any]:
    """Текущие настройки → словарь значений для полей диалога."""
    values = defaults_values()
    for field in FIELDS:
        if hasattr(config, field.name):
            values[field.name] = getattr(config, field.name)
    return values


def _coerce(field: Field, value: Any) -> Any:
    """
    Приводит значение из виджета к типу поля в AppConfig.

    Тип берём из самого конфига: слайдер всегда возвращает float, а в настройке
    demucs_shifts лежит int — без приведения в settings.json уехало бы 3.0.
    """
    sample = getattr(AppConfig(), field.name)
    if isinstance(sample, bool):
        return bool(value)
    if isinstance(sample, int):
        return int(round(float(value)))
    if isinstance(sample, float):
        return float(value)
    return str(value)


def validate_values(values: dict[str, Any]) -> list[str]:
    """
    Проверяет значения ДО сохранения. Возвращает список понятных проблем.

    Пустой список — можно сохранять. Проверяем и то, что интерфейсом не задать
    (файл настроек можно править руками), поэтому диапазоны — как в коде.
    """
    problems: list[str] = []

    def number(name: str) -> float | None:
        try:
            return float(values.get(name))
        except (TypeError, ValueError):
            problems.append(f"{FIELDS_BY_NAME[name].label}: нужно число")
            return None

    shifts = number("demucs_shifts")
    if shifts is not None and not (1 <= shifts <= 20):
        problems.append("Точность разделения — допустимо от 1 до 20")

    overlap = number("demucs_overlap")
    if overlap is not None and not (0.0 <= overlap <= 0.9):
        problems.append("Сглаживание стыков — допустимо от 0.0 до 0.9")

    peak = number("target_peak")
    if peak is not None and not (0.1 <= peak <= 1.0):
        problems.append("Громкость результата — допустимо от 0.1 до 1.0")

    rate = number("output_sample_rate")
    if rate is not None and not (8000 <= rate <= 192000):
        problems.append("Частота выходных файлов — допустимо от 8000 до 192000 Гц")

    batch = number("bs_batch_size")
    if batch is not None and not (0 <= batch <= 32):
        problems.append("Сколько кусков считать за раз — допустимо от 0 до 32")

    engine = str(values.get("default_engine", ""))
    try:
        resolve_engine_id(engine)
    except KeyError:
        problems.append(f"Способ разделения не найден в списке моделей: {engine}")

    if str(values.get("demucs_device", "")).lower() not in ("cuda", "cpu"):
        problems.append("Считать на видеокарте или процессоре — только cuda или cpu")

    return problems


def apply_values(config: AppConfig, values: dict[str, Any]) -> AppConfig:
    """
    Собирает новый AppConfig из значений диалога.

    Поля, которых нет в диалоге, сохраняются как были — так ручные правки в
    settings.json не теряются при сохранении из интерфейса.
    """
    updates = {f.name: _coerce(f, values[f.name]) for f in FIELDS if f.name in values}
    return replace(config, **updates)


# ============================================================
#  Диалог
# ============================================================
class SettingsDialog(ctk.CTkToplevel):
    """Окно настроек. Все поля — из AppConfig (см. FIELDS выше)."""

    def __init__(self, master, config: AppConfig, on_save) -> None:
        super().__init__(master)
        self.title("⚙ Настройки")
        self.geometry("620x660")
        self.minsize(560, 520)
        self.config_data = config
        self.on_save = on_save

        # Значения живут в переменных Tk: их читает и сохранение, и тесты
        self.vars: dict[str, Any] = {}
        self._widgets: dict[str, Any] = {}
        self._value_labels: dict[str, ctk.CTkLabel] = {}

        self._build()
        self._load(config)

        # Модальное окно: пока настройки открыты, главное окно не трогаем
        self.transient(master)
        self.after(50, self._grab)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())

    def _grab(self) -> None:
        """grab_set после отрисовки: иначе на некоторых системах окно «залипает»."""
        try:
            self.grab_set()
            self.focus_force()
        except Exception:
            logger.debug("Не удалось сделать окно настроек модальным", exc_info=True)

    # ---------- сборка интерфейса ----------
    def _build(self) -> None:
        body = ctk.CTkScrollableFrame(self, label_text="")
        body.pack(fill="both", expand=True, padx=10, pady=(10, 4))

        for section in SECTION_ORDER:
            fields = [f for f in FIELDS if f.section == section]
            if not fields:
                continue
            ctk.CTkLabel(body, text=section,
                         font=ctk.CTkFont(size=15, weight="bold")).pack(
                anchor="w", padx=6, pady=(12, 4))
            for field in fields:
                self._build_field(body, field)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkButton(buttons, text="💾 Сохранить", width=140,
                      command=self._save).pack(side="left", padx=4)
        ctk.CTkButton(buttons, text="Отмена", width=110,
                      command=self.destroy,
                      fg_color="#555555", hover_color="#444444").pack(side="left", padx=4)
        ctk.CTkButton(buttons, text="↩ Сбросить поля", width=170,
                      command=self._reset,
                      fg_color="#7f5f3f", hover_color="#6a4f34").pack(side="right", padx=4)

    def _build_field(self, parent, field: Field) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=4)

        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=field.label, width=250, anchor="w").pack(side="left")

        if field.kind == "choice":
            var = ctk.StringVar()
            widget = ctk.CTkOptionMenu(top, values=list(field.choices), variable=var,
                                       width=200)
            widget.pack(side="left")
        elif field.kind == "slider":
            var = ctk.DoubleVar()
            label = ctk.CTkLabel(top, text="", width=60)
            label.pack(side="left", padx=(0, 6))
            self._value_labels[field.name] = label
            widget = ctk.CTkSlider(
                top, from_=field.minimum, to=field.maximum,
                number_of_steps=max(1, int(round((field.maximum - field.minimum) / field.step))),
                variable=var, width=190,
                command=lambda value, f=field: self._on_slider(f, value),
            )
            widget.pack(side="left")
        else:
            var = ctk.BooleanVar()
            widget = ctk.CTkSwitch(top, text="", variable=var, width=48)
            widget.pack(side="left")

        self._widgets[field.name] = widget

        if field.hint:
            ctk.CTkLabel(row, text=field.hint, justify="left", anchor="w",
                         font=ctk.CTkFont(size=11),
                         text_color=("#6a6a6a", "#9a9a9a"),
                         wraplength=560).pack(fill="x", padx=(250, 4))

        self.vars[field.name] = var

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
            if field.kind == "switch":
                var.set(bool(value))
            elif field.kind == "choice":
                text = str(value)
                if field.choices and text not in field.choices:
                    # Значение не из списка (файл настроек правили руками) —
                    # показываем как есть, чтобы пользователь видел правду.
                    logger.warning("Значение %s=%r вне списка допустимых", field.name, text)
                    try:
                        self._widgets[field.name].configure(
                            values=[*field.choices, text])
                    except Exception:
                        logger.debug("Не удалось добавить %r в список", text, exc_info=True)
                var.set(text)
            else:
                var.set(float(value))
                self._on_slider(field, float(value))

    def collect(self) -> dict[str, Any]:
        """Текущие значения всех полей (то, что уйдёт в настройки)."""
        values: dict[str, Any] = {}
        for field in FIELDS:
            var = self.vars[field.name]
            try:
                values[field.name] = var.get()
            except Exception:
                logger.exception("Не удалось прочитать поле %s", field.name)
                values[field.name] = getattr(self.config_data, field.name)
        return values

    def _reset(self) -> None:
        """Ставит значения по умолчанию. Файл не трогает — сначала «Сохранить»."""
        self._load(AppConfig())
        logger.info("Поля настроек сброшены к значениям по умолчанию")

    # ---------- сохранение ----------
    def _save(self) -> None:
        values = self.collect()
        problems = validate_values(values)
        if problems:
            messagebox.showerror(
                "Настройки не сохранены",
                "Проверьте значения:\n\n" + "\n".join(f"• {p}" for p in problems),
                parent=self,
            )
            return

        new_config = apply_values(self.config_data, values)
        logger.info(
            "Настройки изменены в диалоге",
            extra={"ctx_changed": _diff(self.config_data, new_config)},
        )
        try:
            self.on_save(new_config)
        except Exception as e:
            logger.exception("Не удалось применить настройки")
            messagebox.showerror("Настройки", f"Настройки не применены: {e}",
                                 parent=self)
            return
        self.destroy()


def _diff(old: AppConfig, new: AppConfig) -> dict:
    """Что именно поменялось — для лога (и чтобы не писать в лог весь конфиг)."""
    changed = {}
    for field in FIELDS:
        before, after = getattr(old, field.name, None), getattr(new, field.name, None)
        if before != after:
            changed[field.name] = f"{before} → {after}"
    return changed


__all__ = [
    "Field", "FIELDS", "FIELDS_BY_NAME", "SECTION_ORDER", "SettingsDialog",
    "apply_values", "config_to_values", "defaults_values", "validate_values",
]
