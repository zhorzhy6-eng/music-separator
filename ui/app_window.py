"""
Главное окно приложения: шапка, зона файла, волна, плеер и четыре вкладки.

Как устроено (и почему так):
  * общее для всех вкладок — сверху: файл, волна, плеер. Их не нужно искать по
    разделам, они всегда под рукой;
  * четыре вкладки — «Разделение», «Ноты», «Качество», «Настройки». На каждой
    свои кнопки: раньше девять кнопок висели на экране одновременно, хотя в
    работе обычно одна-две;
  * вкладки разбиты на КАРТОЧКИ (ui/card.py) — «Источник», «Способ разделения»,
    «Прослушать результат». По одному взгляду видно, что к чему относится;
  * кнопки различаются по важности (ui/button_styles.py): главное действие
    залито акцентом, второстепенные — с обводкой, служебные — без рамки;
  * видимость и доступность кнопок считается в одном месте (_refresh_actions)
    из состояния UiState. Кнопка включается не «слепо», а если для ТЕКУЩЕГО
    трека реально есть нужный файл на диске;
  * прогресс-бар появляется только на время задачи, пояснения живут в
    всплывающих подсказках (ui/tooltip.py).

Подсказки-шаги в строке состояния («Шаг 1: выберите файл») нужны человеку,
который открыл приложение впервые: они говорят, что делать дальше, а не что
произошло.

Горячие клавиши:
  Space — Play/Pause, → / ← — ±5 сек, Shift+→ / Shift+← — ±30 сек,
  Home — в начало, End — в конец.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from core.audio_utils import get_audio_info, AudioError
from core.config import AppConfig, SETTINGS_FILE, get_config
from core.engines import DEFAULT_ENGINE_ID, get_engine, resolve_engine_id
from core.paths import ANALYSIS_DIR, OUTPUT_DIR
from core.player import AudioPlayer
from core.reports import view_from_super_report
from core.worker import Worker, ProgressEvent, DoneEvent, ErrorEvent, TaskType
from ui import button_styles as bs
from ui.button_styles import danger_button, ghost_button, primary_button, secondary_button
from ui.card import Card
from ui.components import DropZone, WaveformView, PlayerBar, _fmt_time
from ui.engine_selector import EngineSelector
from ui.settings_panel import SettingsPanel
from ui.theme import (
    FONT_CAPTION,
    FONT_TITLE,
    SPACE_BETWEEN,
    SPACE_BLOCK,
    SPACE_CARD,
    SPACE_TIGHT,
    font_spec,
    get_theme,
)
from ui.tooltip import ToolTip
from ui.viewmodels import UiState
from ui.widgets import caption_label, restore_main_focus

logger = logging.getLogger(__name__)

SEEK_SMALL = 5
SEEK_LARGE = 30

# Названия вкладок. В одном месте — чтобы переключение по коду («Настройки»
# из меню) не разъезжалось с подписями на экране.
# DECISION: эмодзи из названий вкладок убраны. На разных системах они рисуются
# по-разному (а кое-где квадратиками), а смысла не добавляли: «🎼 Транскрипция»
# и «Ноты» — это одно и то же, только второе понятнее.
TAB_SEPARATE = "Разделение"
TAB_TRANSCRIBE = "Ноты"
TAB_ANALYZE = "Качество"
TAB_SETTINGS = "Настройки"

WAVE_PLACEHOLDER = "Загрузите файл, чтобы увидеть волну"

# Что делать дальше — простыми словами, в строке состояния
HINT_CHOOSE_FILE = "Шаг 1: выберите файл — нажмите на зону выше"
HINT_PRESS_SPLIT = "Шаг 2: нажмите синюю кнопку «Разделить на вокал и минус»"
HINT_LISTEN = "Готово: выберите «Вокал» или «Минус» и нажмите ▶"

# Список движков живёт в core/engines (реестр ENGINES). Здесь его нет:
# UI — только витрина реестра, чтобы списки моделей не разъезжались в трёх местах.


class AppWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # Настройки из config/settings.json (значения по умолчанию — в core/config.py)
        self.config_data = AppConfig.load()

        self.theme_name = "dark"
        self.theme = get_theme(self.theme_name)
        ctk.set_appearance_mode(self.theme_name)
        ctk.set_default_color_theme("blue")
        # DECISION: фон окна задаём НАШИМ цветом. По умолчанию CustomTkinter
        # красит окно в свой gray (gray14), и он не совпадал с панелями — окно
        # выглядело как набор серых прямоугольников на чужом фоне.
        self.configure(fg_color=self.theme["bg"])

        self.title("Audio Separator & Transcriber")
        self.geometry("1040x820")
        self.minsize(940, 700)

        self.current_file: str | None = None
        # Пути — из единой точки core.paths (корень проекта, а не рабочий каталог)
        self.output_dir: Path = OUTPUT_DIR
        self.analysis_dir: Path = ANALYSIS_DIR
        self._last_info: dict = {}

        # DECISION: атрибут называется ui_state, а не state. Имя state уже занято
        # собственным методом окна (self.state(...) — «свернуть/развернуть»), и
        # customtkinter вызывает его при показе окна: с полем state окно падало
        # с «'UiState' object is not callable» ещё до первой отрисовки.
        # ui_state — единственный источник правды о том, что загружено и занято ли
        # окно. Старые поля (vocals_path и др.) оставлены как зеркала состояния:
        # их читает плеер, и они нужны для совместимости.
        self.ui_state = UiState()
        self.vocals_path: Path | None = None
        self.instrumental_path: Path | None = None
        self.last_piano_path: Path | None = None

        # Номер текущей задачи: нужен, чтобы кнопка «Отмена» знала, что отменять
        self.current_task_id: int | None = None
        # Что за задача идёт: по нему строка состояния говорит простыми словами
        self._current_task_type: TaskType | None = None

        # Открытый диалог настроек (None — закрыт). Держим ссылку, чтобы второе
        # нажатие «Настройки» не открывало второе окно.
        self._settings_dialog = None
        # Выпадающее меню шапки и «плёнка», закрывающая его по клику мимо
        self._menu_popup: ctk.CTkToplevel | None = None
        self._menu_overlay = None
        # Кнопки с их ролью — для перекраски при смене темы
        self._styled: list[tuple] = []
        self._theme_icons: list[tuple] = []

        self.worker = Worker()
        self.player = AudioPlayer()

        self._build_ui()
        self._build_tabs()
        self._bind_hotkeys()
        self._apply_theme_labels()
        # Стартовое состояние: файла нет, волна — заглушка, кнопки выключены
        self.wave.show_placeholder(WAVE_PLACEHOLDER)
        self._refresh_actions()
        self._poll_events()
        self._tick_position()
        # Закрытие окна: гасим воркер и дочерние процессы (audio-separator.exe)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        """
        Верхняя часть окна — общая для всех вкладок.

        Порядок такой: шапка → зона файла → волна → плеер → вкладки → состояние.
        Файл, волна и плеер относятся к самому треку, а не к отдельной операции,
        поэтому переключение вкладок их не трогает.
        """
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=SPACE_CARD, pady=(SPACE_CARD, SPACE_BLOCK))
        self.header = top

        self.title_label = ctk.CTkLabel(
            top, text="Audio Separator & Transcriber",
            font=font_spec(FONT_TITLE, "bold"), text_color=self.theme["text"])
        self.title_label.pack(side="left", padx=(2, 0))
        self.subtitle_label = ctk.CTkLabel(
            top, text="Разделение музыки на вокал и минус",
            font=font_spec(FONT_CAPTION), text_color=self.theme["text_dim"])
        self.subtitle_label.pack(side="left", padx=(12, 0), pady=(6, 0))

        # DECISION: «Тема» и «Настройки» свёрнуты в одно меню «⚙». Две кнопки в
        # шапке дублировали то, что и так есть на вкладке настроек, и отнимали
        # место у главного — выбора файла и кнопки «Разделить».
        self.btn_menu = bs.icon_button(
            top, "settings", self._open_menu, self.theme, size=36, icon_size=20)
        # DECISION: запасной символ «⚙» здесь НЕ подставляем. icon_button уже
        # решает это сам: есть картинка — текста нет, нет картинки — покажет «⚙».
        # Строка `configure(text="⚙")` рисовала символ поверх загруженной иконки,
        # и в шапке было две шестерёнки одна на другой (видно на скриншотах
        # QA-прогона 11.10.2026, logs/qa_full_2026-10-11/screenshots/).
        self.btn_menu.pack(side="right")
        self._styled.append((self.btn_menu, "icon"))
        ToolTip(self.btn_menu, "Меню: тема, настройки, о программе, выход")

        # Одна кнопка «Открыть папку» на всё приложение (открывает output/).
        # Раньше их было две («Папка» и «Отчёты»), и обе открывали папки —
        # разница была видна только автору.
        self.btn_open_output = ghost_button(
            top, "Открыть папку", self._open_output, self.theme,
            icon="folder", width=150, height=36)
        self.btn_open_output.pack(side="right", padx=(0, SPACE_BLOCK))
        self._styled.append((self.btn_open_output, "ghost"))
        ToolTip(self.btn_open_output,
                "Открывает папку output/ — туда сохраняются вокал, минус и ноты.\n"
                "Отчёты о качестве лежат в logs/analysis/ "
                "(кнопка на вкладке «Качество»).")

        # --- Зона файла: до загрузки — «перетащите сюда», после — карточка файла
        self.drop = DropZone(self, on_file=self._load_file,
                             theme=self.theme, height=140)
        self.drop.pack(fill="x", padx=SPACE_CARD, pady=(0, SPACE_BLOCK))

        self.wave = WaveformView(self, theme=self.theme)
        self.wave.pack(fill="x", padx=SPACE_CARD, pady=(0, SPACE_BLOCK))

        # Плеер: карточка с круглыми кнопками, перемоткой и «Слушать:»
        self.player_bar = PlayerBar(self,
                                    on_play=self._player_play,
                                    on_pause=self._player_pause,
                                    on_stop=self._player_stop,
                                    on_volume=self._player_volume,
                                    on_seek=self._player_seek,
                                    on_listen=self._on_listen_change,
                                    theme=self.theme)
        self.player_bar.pack(fill="x", padx=SPACE_CARD, pady=(0, SPACE_BLOCK))
        # Короткое имя сохранено: на него смотрят тесты и код плеера
        self.combo_listen = self.player_bar.combo_listen

    def _build_tabs(self) -> None:
        """Четыре вкладки: у каждой свой набор кнопок."""
        self.tabs = ctk.CTkTabview(
            self, fg_color=self.theme["bg"],
            segmented_button_fg_color=self.theme["panel"],
            segmented_button_selected_color=self.theme["accent"],
            segmented_button_selected_hover_color=self.theme["accent_hover"],
            segmented_button_unselected_color=self.theme["panel"],
            segmented_button_unselected_hover_color=self.theme["panel_hover"],
            segmented_button_font=font_spec(14, "bold"),
            text_color=self.theme["text"],
            corner_radius=12,
        )
        self.tabs.pack(fill="both", expand=True, padx=SPACE_CARD, pady=(0, SPACE_TIGHT))

        self.tab_separate = self.tabs.add(TAB_SEPARATE)
        self.tab_transcribe = self.tabs.add(TAB_TRANSCRIBE)
        self.tab_analyze = self.tabs.add(TAB_ANALYZE)
        self.tab_settings = self.tabs.add(TAB_SETTINGS)

        self._build_tab_separate(self.tab_separate)
        self._build_tab_transcribe(self.tab_transcribe)
        self._build_tab_analyze(self.tab_analyze)
        self._build_tab_settings(self.tab_settings)

        # При смене вкладки пересчитываем кнопки: их доступность зависит от того,
        # что уже есть для текущего трека, а это меняется в том числе «само»
        # (файлы могли появиться на диске).
        self.tabs.configure(command=self._refresh_actions)

        # --- Строка состояния ---
        self.status_row = ctk.CTkFrame(self, fg_color="transparent")
        self.status_row.pack(fill="x", padx=SPACE_CARD, pady=(0, SPACE_BLOCK))
        self.status = ctk.CTkLabel(
            self.status_row, text="", anchor="w",
            font=font_spec(FONT_CAPTION), text_color=self.theme["text_dim"])
        self.status.pack(side="left")
        # Подсказка-шаг: что делать дальше. Живёт справа, чтобы не спорить
        # со строкой состояния за место.
        self.hint = ctk.CTkLabel(
            self.status_row, text=HINT_CHOOSE_FILE, anchor="e",
            font=font_spec(FONT_CAPTION), text_color=self.theme["accent"])
        self.hint.pack(side="right")

    def _build_tab_separate(self, parent) -> None:
        """Вкладка «Разделение»: источник, способ разделения, результат."""
        card = Card(parent, "Способ разделения", self.theme,
                    subtitle="можно не менять", icon="settings")
        card.pack(fill="x", pady=(SPACE_BLOCK, SPACE_BETWEEN))
        self.card_engine = card

        caption_label(
            card.body,
            "Готовый набор моделей: выбирать нужно, только если результат "
            "не устроит.",
            self.theme, wraplength=900,
        ).pack(fill="x", pady=(0, SPACE_BLOCK))

        self.engine_combo = EngineSelector(card.body,
                                           on_change=self._on_engine_change,
                                           theme=self.theme, width=460)
        self.engine_combo.pack(anchor="w")
        # Значение из настроек может устареть (модель убрали) — берём движок
        # по умолчанию из реестра, а не падаем.
        self.engine_combo.set(self._default_engine_id())
        card.set_subtitle(self.engine_combo.label() or "")

        action = ctk.CTkFrame(parent, fg_color="transparent")
        action.pack(fill="x", pady=(0, SPACE_BLOCK))
        self.action_row = action

        self.btn_separate = primary_button(
            action, "Разделить на вокал и минус", self._on_separate, self.theme,
            icon="scissors", icon_size=20)
        self.btn_separate.pack(side="left")
        self._styled.append((self.btn_separate, "primary"))
        ToolTip(self.btn_separate,
                "Разделяет текущий трек на две дорожки: голос и музыку без голоса.\n"
                "Результат: output/<имя трека>_vocals.wav и _instrumental.wav.\n"
                "Сначала выберите файл — зона вверху окна.")

        # Прогресс-бар появляется ВМЕСТЕ с задачей и исчезает вместе с ней.
        # Раньше пустая полоска висела на экране всегда и выглядела как дыра.
        self.progress_row = ctk.CTkFrame(action, fg_color="transparent")
        self.progress = ctk.CTkProgressBar(
            self.progress_row, height=14, corner_radius=7,
            progress_color=self.theme["accent"], fg_color=self.theme["border"])
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, SPACE_BLOCK))

        self.btn_cancel = danger_button(
            self.progress_row, "Отмена", self._on_cancel, self.theme,
            icon="x", icon_size=16, height=36, state="disabled")
        self.btn_cancel.pack(side="right")
        self._styled.append((self.btn_cancel, "danger"))
        ToolTip(self.btn_cancel, "Останавливает текущую работу. "
                                 "Уже готовые файлы останутся на месте.")

    def _build_tab_transcribe(self, parent) -> None:
        """Вкладка «Ноты»: превращение вокала и минуса в MIDI."""
        card = Card(parent, "Получить ноты", self.theme,
                    subtitle="нужны вокал и минус", icon="file-text")
        card.pack(fill="x", pady=(SPACE_BLOCK, SPACE_BETWEEN))
        self.card_transcribe = card

        caption_label(
            card.body,
            "Кнопки включатся, когда для текущего трека будет готово разделение "
            "(вкладка «Разделение»). Ноты открываются в любой программе-редакторе: "
            "MuseScore, FL Studio, Reaper.",
            self.theme, wraplength=900,
        ).pack(fill="x", pady=(0, SPACE_BLOCK))

        row = ctk.CTkFrame(card.body, fg_color="transparent")
        row.pack(fill="x")

        self.btn_voc_midi = secondary_button(
            row, "Вокал в ноты", self._on_transcribe_vocals, self.theme,
            icon="music", state="disabled", width=200, height=40)
        self.btn_voc_midi.pack(side="left", padx=(0, SPACE_BLOCK))
        self._styled.append((self.btn_voc_midi, "secondary"))
        ToolTip(self.btn_voc_midi,
                "Записывает мелодию голоса нотами (файл .mid).\n"
                "Подходит для соло и мелодий без аккордов — программа ведёт "
                "одну ноту за раз.")

        self.btn_inst_midi = secondary_button(
            row, "Минус в ноты", self._on_transcribe_instrumental, self.theme,
            icon="music", state="disabled", width=200, height=40)
        self.btn_inst_midi.pack(side="left", padx=(0, SPACE_BLOCK))
        self._styled.append((self.btn_inst_midi, "secondary"))
        ToolTip(self.btn_inst_midi,
                "Пытается записать нотами музыку без голоса.\n"
                "Если в треке аккорды или несколько инструментов сразу, нот "
                "получится мало — попробуйте «Минус как пианино» справа.")

        self.btn_piano = secondary_button(
            row, "Минус как пианино", self._on_piano, self.theme,
            icon="piano", state="disabled", width=200, height=40)
        self.btn_piano.pack(side="left")
        self._styled.append((self.btn_piano, "secondary"))
        ToolTip(self.btn_piano,
                "Разбирает музыку без голоса на ноты для фортепиано: аккорды и "
                "несколько инструментов сразу.\nСчитается дольше, зато нот больше.")

    def _build_tab_analyze(self, parent) -> None:
        """Вкладка «Качество»: отчёты о звуке, нотах и разделении."""
        card = Card(parent, "Проверить результат", self.theme,
                    subtitle="отчёты сохраняются в logs/analysis", icon="activity")
        card.pack(fill="x", pady=(SPACE_BLOCK, SPACE_BETWEEN))
        self.card_analyze = card

        caption_label(
            card.body,
            "Проверки ничего не меняют в файлах — они только считают и показывают, "
            "что получилось. Занимают меньше минуты.",
            self.theme, wraplength=900,
        ).pack(fill="x", pady=(0, SPACE_BLOCK))

        row = ctk.CTkFrame(card.body, fg_color="transparent")
        row.pack(fill="x")

        self.btn_analyze = secondary_button(
            row, "Проверить качество звука", self._on_analyze_current, self.theme,
            icon="activity", state="disabled", width=210, height=40)
        self.btn_analyze.pack(side="left", padx=(0, SPACE_BLOCK))
        self._styled.append((self.btn_analyze, "secondary"))
        ToolTip(self.btn_analyze,
                "Смотрит на громкость, перегрузку, тишину и частоты того файла, "
                "который сейчас в плеере.\nПолезно, чтобы понять, не «каша» ли "
                "получилась. Отчёт ложится в logs/analysis/.")

        self.btn_analyze_midi = secondary_button(
            row, "Проверить ноты", self._on_analyze_midi, self.theme,
            icon="file-text", state="disabled", width=180, height=40)
        self.btn_analyze_midi.pack(side="left", padx=(0, SPACE_BLOCK))
        self._styled.append((self.btn_analyze_midi, "secondary"))
        ToolTip(self.btn_analyze_midi,
                "Смотрит на готовые ноты: сколько их, в каком диапазоне, "
                "какой темп.\nВключается после того, как ноты сделаны.")

        self.btn_super_pair = secondary_button(
            row, "Проверить качество разделения", self._on_super_pair_analysis,
            self.theme, icon="sparkles", state="disabled",
            width=250, height=40)
        self.btn_super_pair.pack(side="left")
        self._styled.append((self.btn_super_pair, "secondary"))
        ToolTip(self.btn_super_pair,
                "Сравнивает вокал и минус и ищет огрехи разделения: остатки "
                "голоса в музыке, пропавшие удары, «размазанный» звук.\n"
                "В конце показывает простой вердикт: можно ли доверять результату.")

        bottom = ctk.CTkFrame(card.body, fg_color="transparent")
        bottom.pack(fill="x", pady=(SPACE_BLOCK, 0))
        self.btn_open_analysis = ghost_button(
            bottom, "Открыть отчёты", self._open_analysis, self.theme,
            icon="chart-column", width=170, height=32)
        self.btn_open_analysis.pack(side="left")
        self._styled.append((self.btn_open_analysis, "ghost"))
        ToolTip(self.btn_open_analysis,
                "Открывает папку logs/analysis/ со всеми отчётами: таблицы и "
                "картинки с графиками.")

    def _build_tab_settings(self, parent) -> None:
        """Вкладка «Настройки»: те же поля, что были в диалоге, но без окна поверх окна."""
        self.settings_panel = SettingsPanel(parent, self.config_data,
                                            on_save=self._apply_settings,
                                            theme=self.theme)
        self.settings_panel.pack(fill="both", expand=True)

    # ---------- оформление ----------
    def _apply_theme_labels(self) -> None:
        """Красит подписи, которые живут вне карточек и компонентов."""
        self.title_label.configure(text_color=self.theme["text"])
        self.subtitle_label.configure(text_color=self.theme["text_dim"])
        self.status.configure(text_color=self.theme["text_dim"])
        self.hint.configure(text_color=self.theme["accent"])

    def _refresh_theme(self) -> None:
        """
        Перекрашивает всё окно под новую тему.

        Идём по дереву виджетов и зовём `apply_theme` у тех, кто его умеет
        (карточки, волна, плеер, зона файла). Так новому компоненту достаточно
        завести этот метод, а не регистрироваться в списке где-то ещё.

        DECISION: отказались от второго набора «светлых» цветов через кортежи
        ("#fff", "#111"). Кортеж — это два фиксированных цвета, и в новой теме
        он не меняется: часть окна оставалась бы серой. Поэтому у нас один
        набор цветов на тему и явная перекраска.
        """
        theme = self.theme
        try:
            self.configure(fg_color=theme["bg"])
        except Exception:
            logger.debug("Фон окна не перекрасился", exc_info=True)

        for widget in self._walk(self):
            apply_theme = getattr(widget, "apply_theme", None)
            if callable(apply_theme):
                try:
                    apply_theme(theme)
                except Exception:
                    logger.debug("Виджет %r не перекрасился", widget, exc_info=True)

        for button, role in self._styled:
            bs.recolor(button, role, theme)

        self.tabs.configure(
            fg_color=theme["bg"],
            segmented_button_fg_color=theme["panel"],
            segmented_button_selected_color=theme["accent"],
            segmented_button_selected_hover_color=theme["accent_hover"],
            segmented_button_unselected_color=theme["panel"],
            segmented_button_unselected_hover_color=theme["panel_hover"],
            text_color=theme["text"])
        self.progress.configure(progress_color=theme["accent"],
                                fg_color=theme["border"])
        self.settings_panel.apply_theme(theme)
        self._apply_theme_labels()
        self.set_theme_icons()

    @staticmethod
    def _walk(widget):
        """Все виджеты окна, включая вложенные."""
        for child in widget.winfo_children():
            yield child
            yield from AppWindow._walk(child)

    # ---------- настройки движка ----------
    def _default_engine_id(self) -> str:
        """
        id движка по умолчанию из настроек.

        В настройке может лежать и подпись из UI, и id из реестра — понимаем оба
        варианта (resolve_engine_id). Значение может устареть (модель убрали) —
        тогда берём движок по умолчанию из реестра.
        """
        engine = getattr(self.config_data, "default_engine", "")
        try:
            return resolve_engine_id(engine)
        except KeyError:
            logger.warning("Движок из настроек не найден в реестре: %r", engine)
            return DEFAULT_ENGINE_ID

    def _on_engine_change(self, engine_id: str) -> None:
        """Человек выбрал другую модель — показываем это в подписи карточки."""
        try:
            spec = get_engine(engine_id)
        except KeyError:
            logger.warning("Выбран неизвестный движок: %r", engine_id)
            return
        # В подписи карточки — техническое имя: его видно и в настройках, и в
        # логе, поэтому по нему можно сверить, что реально запустится.
        try:
            self.card_engine.set_subtitle(spec.label)
        except Exception:
            logger.debug("Подпись движка не обновилась", exc_info=True)
        # А в подсказке-шаге — понятное название: человек должен видеть, что
        # выбор сработал, не сверяясь с техническим именем.
        try:
            if self.ui_state.busy:
                return
            self.hint.configure(text=f"Выбрано: {spec.display_name}. "
                                     f"Дальше — кнопка «Разделить».")
        except Exception:
            logger.debug("Подсказка о смене модели не обновилась", exc_info=True)

    # ---------- Файлы текущего трека ----------
    # Правило одно: любая кнопка работает ТОЛЬКО с файлами того трека, который
    # сейчас загружен. Иначе после неудачного разделения «Пианино» брало минус
    # прошлого трека, а результат сохраняло под именем нового.
    def _current_stem(self) -> str | None:
        return Path(self.current_file).stem if self.current_file else None

    def _path_belongs_to_current(self, path) -> bool:
        """Файл существует и назван по текущему треку (то есть он «наш»)?"""
        stem = self._current_stem()
        if not stem or not path:
            return False
        p = Path(path)
        return p.exists() and p.stem.startswith(stem + "_")

    def _current_stems_present(self) -> dict[str, Path]:
        """Какие стемы текущего трека реально лежат в output/."""
        stem = self._current_stem()
        found: dict[str, Path] = {}
        if not stem:
            return found
        for kind in ("vocals", "instrumental"):
            p = self.output_dir / f"{stem}_{kind}.wav"
            if p.exists():
                found[kind] = p
        return found

    def _effective_stems(self) -> tuple[Path | None, Path | None]:
        """
        Пути вокала и минуса текущего трека.

        Берём сохранённые после разделения, а если их нет (например, трек
        загрузили заново, а файлы уже есть на диске) — находим по именам.
        Чужие файлы не подставляются никогда.
        """
        found = self._current_stems_present()
        vocals = (self.vocals_path if self._path_belongs_to_current(self.vocals_path)
                  else found.get("vocals"))
        instr = (self.instrumental_path if self._path_belongs_to_current(self.instrumental_path)
                 else found.get("instrumental"))
        return vocals, instr

    def _midi_candidates(self) -> list[Path]:
        """MIDI текущего трека: пианино в приоритете, потом вокал, потом минус."""
        stem = self._current_stem()
        if not stem:
            return []
        out: list[Path] = []
        if self.last_piano_path and self._path_belongs_to_current(self.last_piano_path):
            out.append(Path(self.last_piano_path))
        for suffix in ("_piano.mid", "_vocals.mid", "_instrumental.mid"):
            p = self.output_dir / f"{stem}{suffix}"
            if p.exists() and p not in out:
                out.append(p)
        return out

    # ---------- состояние интерфейса ----------
    def _sync_state_from_disk(self) -> None:
        """
        Подтягивает состояние к тому, что реально лежит на диске.

        Так кнопки честно включаются и после перезапуска приложения, когда
        стемы для загруженного трека уже есть.
        """
        vocals, instr = self._effective_stems()
        self.ui_state.source = Path(self.current_file) if self.current_file else None
        self.ui_state.vocals = vocals
        self.ui_state.instrumental = instr
        midi = self._midi_candidates()
        self.ui_state.midi = midi[0] if midi else None

    def _refresh_actions(self) -> None:
        """
        Единая точка правды для кнопок и прогресс-бара — вызывается после КАЖДОГО
        изменения состояния (загрузка файла, старт задачи, успех, ошибка, отмена,
        смена вкладки).

        Три правила:
          1) пока идёт работа (state.busy) — кнопки запуска выключены, включена
             только «Отмена». Раньше каждая операция выключала кнопки вручную в
             своём месте, и после ошибки они включались безусловно (баг №3 из QA);
          2) когда работа кончилась — кнопка включается не «слепо», а если для
             ТЕКУЩЕГО трека реально есть нужный файл на диске;
          3) прогресс-бар и «Отмена» появляются только на время задачи.
        """
        self._sync_state_from_disk()
        s = self.ui_state
        busy = s.busy

        def run_state(available: bool) -> str:
            return "normal" if (available and not busy) else "disabled"

        # --- Вкладка «Разделение» ---
        self.btn_separate.configure(state=run_state(s.has_source))
        self.btn_cancel.configure(
            state="normal" if (busy and self.current_task_id is not None) else "disabled")

        # Прогресс-бар — только когда задача идёт
        if busy:
            self.progress_row.pack(side="left", fill="x", expand=True,
                                   padx=(SPACE_BLOCK, 0))
        else:
            self.progress_row.pack_forget()

        # --- Кнопка «Открыть папку» (одна на всё приложение) ---
        # Активна всегда: папка output/ существует с первого запуска, и заглянуть
        # в неё законно даже без разделения — там могут лежать прошлые результаты.
        self.btn_open_output.configure(state="normal")

        # --- Вкладка «Ноты» ---
        self.btn_voc_midi.configure(state=run_state(s.has_stems))
        self.btn_inst_midi.configure(state=run_state(s.has_stems))
        self.btn_piano.configure(state=run_state(s.instrumental is not None))

        # --- Вкладка «Качество» ---
        self.btn_analyze.configure(state=run_state(s.has_source))
        self.btn_analyze_midi.configure(state=run_state(s.has_midi))
        self.btn_super_pair.configure(state=run_state(s.has_stems))

        self._update_hint()

    def _update_hint(self) -> None:
        """Подсказка-шаг: что делать дальше. Простыми словами, без жаргона."""
        s = self.ui_state
        if s.busy:
            text = "Идёт работа — можно отменить кнопкой «Отмена»"
        elif not s.has_source:
            text = HINT_CHOOSE_FILE
        elif s.has_stems:
            text = HINT_LISTEN
        else:
            text = HINT_PRESS_SPLIT
        try:
            self.hint.configure(text=text)
        except Exception:
            logger.debug("Подсказка не обновилась", exc_info=True)

    def _set_busy(self, busy: bool) -> None:
        """Работа началась/закончилась: обновляем состояние и кнопки."""
        self.ui_state.mark_busy(busy)
        if not busy:
            self.current_task_id = None
        self._refresh_actions()

    # ---------- Меню «⚙» ----------
    def _open_menu(self) -> None:
        """
        Выпадающее меню шапки: тема, настройки, о программе, выход.

        DECISION: своё маленькое меню на Toplevel, а не tk.Menu. tk.Menu на
        тёмной теме Windows рисуется системными цветами и выбивается из окна;
        к тому же пункт «Тема» должен показывать текущее состояние.
        """
        if self._menu_popup is not None:
            self._close_menu()
            return

        theme = self.theme
        popup = ctk.CTkToplevel(self)
        popup.wm_overrideredirect(True)
        popup.wm_attributes("-topmost", True)
        # Рамка цветом границы: меню не должно сливаться с окном под ним
        popup.configure(fg_color=theme["border"])
        self._menu_popup = popup

        surface = ctk.CTkFrame(popup, fg_color=theme["panel"], corner_radius=10)
        surface.pack(fill="both", expand=True, padx=1, pady=1)

        self._menu_icon_widgets: list[tuple] = []
        theme_icon = "sun" if self.theme_name == "dark" else "moon"
        items = (
            (theme_icon, "Светлая тема" if self.theme_name == "dark" else "Тёмная тема",
             self._toggle_theme),
            ("settings", "Настройки…", self._go_to_settings),
            ("folder", "Открыть папку output", self._open_output),
            ("chart-column", "Открыть отчёты", self._open_analysis),
            ("info", "О программе", self._show_about),
            ("log-out", "Выход", self._on_close),
        )
        for icon, text, command in items:
            self._menu_item(surface, icon, text, command)

        popup.update_idletasks()
        w, h = popup.winfo_reqwidth(), popup.winfo_reqheight()
        x = self.btn_menu.winfo_rootx() + self.btn_menu.winfo_width() - w
        y = self.btn_menu.winfo_rooty() + self.btn_menu.winfo_height() + 6
        screen_w, screen_h = popup.winfo_screenwidth(), popup.winfo_screenheight()
        x = max(8, min(x, screen_w - w - 8))
        y = max(8, min(y, screen_h - h - 8))
        popup.wm_geometry(f"{w}x{h}+{int(x)}+{int(y)}")

        try:
            overlay = tk.Frame(self, background=theme["overlay"], bd=0,
                               highlightthickness=0)
            overlay.place(x=0, y=0, relwidth=1, relheight=1)
            overlay.bind("<Button-1>", lambda e: self._close_menu())
            self._menu_overlay = overlay
        except Exception:
            logger.debug("Плёнка под меню не создалась", exc_info=True)

        popup.bind("<Escape>", lambda e: self._close_menu())
        try:
            popup.focus_force()
        except Exception:
            logger.debug("Фокус на меню не перешёл", exc_info=True)

    def _menu_item(self, popup, icon: str, text: str, command) -> None:
        """Строка меню: иконка, подпись, действие. Иконка перекрашивается с темой."""
        theme = self.theme
        img = bs.icon_image(icon, 18, theme["text_dim"], master=popup)
        button = ctk.CTkButton(
            popup, text=text, anchor="w", width=260, height=34,
            image=img, compound="left" if img is not None else "text",
            fg_color="transparent", hover_color=theme["panel_hover"],
            text_color=theme["text"], font=font_spec(13),
            command=lambda c=command: self._run_menu_item(c))
        if img is None:
            button.configure(text=f"{bs.emoji_fallback(icon)} {text}")
        button.pack(fill="x", padx=6, pady=1)
        self._menu_icon_widgets.append((button, icon))

    def set_theme_icons(self) -> None:
        """Обновляет иконки, которые зависят от темы (переключатель в меню)."""
        for button, icon in getattr(self, "_menu_icon_widgets", []):
            try:
                button.configure(image=bs.icon_image(icon, 18, self.theme["text_dim"],
                                                     master=button))
            except Exception:
                logger.debug("Иконка меню не обновилась", exc_info=True)

    def _run_menu_item(self, command) -> None:
        """Пункт меню: сначала закрываем меню, потом делаем дело."""
        self._close_menu()
        try:
            command()
        except Exception:
            logger.exception("Пункт меню упал")

    def _close_menu(self) -> None:
        if self._menu_overlay is not None:
            try:
                self._menu_overlay.destroy()
            except Exception:
                logger.debug("Плёнка под меню уже удалена", exc_info=True)
            self._menu_overlay = None
        if self._menu_popup is not None:
            try:
                self._menu_popup.destroy()
            except Exception:
                logger.debug("Меню уже закрыто", exc_info=True)
            self._menu_popup = None
        # Меню забирало фокус на себя — возвращаем его окну, иначе перестают
        # работать горячие клавиши плеера (см. ui/widgets.restore_main_focus).
        restore_main_focus(self)

    def _go_to_settings(self) -> None:
        """«Настройки…» из меню — переключает на вкладку настроек."""
        try:
            self.tabs.set(TAB_SETTINGS)
        except Exception:
            logger.exception("Не удалось открыть вкладку настроек")
            return
        self._refresh_actions()

    def _show_about(self) -> None:
        messagebox.showinfo(
            "О программе",
            "Audio Separator & Transcriber\n\n"
            "Локальное приложение: разделяет музыку на вокал и минус,\n"
            "делает ноты (MIDI) и проверяет качество результата.\n\n"
            "Всё считается на этом компьютере — аудио никуда не отправляется.\n\n"
            f"Результаты: {self.output_dir}\n"
            f"Отчёты: {self.analysis_dir}",
            parent=self,
        )

    # ---------- Отмена и закрытие ----------
    def _on_cancel(self):
        """Кнопка «Отмена»: просим воркер остановить текущую задачу."""
        if self.current_task_id is None:
            return
        task_id = self.current_task_id
        # Движок гасит дочерний процесс не мгновенно, а окно своё действие уже
        # сделало — поэтому сразу выключаем кнопку, а не после ответа воркера.
        self.btn_cancel.configure(state="disabled")
        self.status.configure(text="Останавливаю…")
        if not self.worker.cancel(task_id):
            self.status.configure(text="Отменять нечего — работа уже закончилась")

    def _on_close(self):
        """
        Закрытие окна: гасим воркер вместе с дочерними процессами.

        Без этого audio-separator.exe продолжал работать в фоне и держал GPU.
        """
        try:
            self._close_menu()
        except Exception:
            logger.debug("Меню при закрытии окна", exc_info=True)
        try:
            self.worker.shutdown(wait=False)
        except Exception:
            logger.exception("Остановка воркера при закрытии окна")
        try:
            self.player.stop()
        except Exception:
            logger.exception("Остановка плеера при закрытии окна")
        self.destroy()

    # ---------- Горячие клавиши ----------
    def _bind_hotkeys(self):
        self.bind_all("<Right>", lambda e: self._seek_relative(+SEEK_SMALL))
        self.bind_all("<Left>",  lambda e: self._seek_relative(-SEEK_SMALL))
        self.bind_all("<Shift-Right>", lambda e: self._seek_relative(+SEEK_LARGE))
        self.bind_all("<Shift-Left>",  lambda e: self._seek_relative(-SEEK_LARGE))
        self.bind_all("<space>", lambda e: self._toggle_play_pause())
        self.bind_all("<Home>", lambda e: self._seek_home())
        self.bind_all("<End>",  lambda e: self._seek_end())

    def _seek_relative(self, delta):
        try:
            if self.player.duration <= 0:
                return "break"
            new_pos = max(0.0, min(self.player.position + delta, self.player.duration))
            self.player.seek(new_pos)
            self.player_bar.set_position(new_pos)
        except Exception:
            logger.exception("seek_relative")
        return "break"

    def _seek_home(self):
        try:
            self.player.seek(0.0)
            self.player_bar.set_position(0.0)
        except Exception:
            logger.exception("seek_home")
        return "break"

    def _seek_end(self):
        try:
            end = self.player.duration
            self.player.seek(end)
            self.player_bar.set_position(end)
        except Exception:
            logger.exception("seek_end")
        return "break"

    def _toggle_play_pause(self):
        try:
            if self.player.is_playing:
                self.player.pause()
                self.player_bar.set_playing(False)
            else:
                if self.combo_listen.get() == "—" and self.current_file:
                    self.combo_listen.set("Исходный трек")
                    self.player.load(self.current_file)
                    self.player_bar.set_duration(self.player.duration)
                    self.player_bar.set_position(0.0)
                self.player.play()
                self.player_bar.set_playing(True)
        except Exception:
            logger.exception("toggle_play_pause")
        return "break"

    # ---------- Обработчики ----------
    def _open_settings(self):
        """
        Открывает настройки. Теперь это вкладка, а не модальное окно.

        Метод оставлен ради обратной совместимости: на него смотрят тесты и
        старый код; из меню вызывается _go_to_settings().
        """
        self._go_to_settings()

    def _apply_settings(self, new_config: AppConfig):
        """
        Сохраняет настройки на диск и применяет их к работающему окну.

        Важно: кэш настроек (core.config.get_config) сбрасывается. Без этого
        движки работали бы на старых значениях до перезапуска приложения —
        а человек ждёт, что смена устройства Demucs подействует сразу.
        """
        try:
            new_config.save()
        except Exception as e:
            logger.exception("Не удалось сохранить настройки")
            # Понятный текст — на вкладку, подробности — в лог: файл может быть
            # занят другой программой или лежать в папке без прав на запись.
            raise RuntimeError(f"не удалось записать {SETTINGS_FILE.name}: {e}") from e
        self.config_data = get_config(reload=True)
        # Список движков строится из реестра, а вот какой выбран по умолчанию —
        # из настроек: после сохранения показываем актуальный.
        self.engine_combo.set(self._default_engine_id())
        # Панель настроек перечитывает то, что реально записалось: если файл
        # лежал только для чтения или значение подменилось, человек увидит правду.
        self.settings_panel.reload(self.config_data, message="Настройки сохранены")
        self.status.configure(text="Настройки сохранены")
        self._refresh_actions()
        logger.info("Настройки сохранены", extra={"ctx_file": str(SETTINGS_FILE)})

    def _toggle_theme(self):
        try:
            self.theme_name = "light" if self.theme_name == "dark" else "dark"
            ctk.set_appearance_mode(self.theme_name)
            self.theme = get_theme(self.theme_name)
            # Цвета Canvas, графика и кнопок задаются вручную, поэтому сообщаем
            # о смене темы всему окну, а не только «корню»
            self._refresh_theme()
            logger.info("Тема переключена", extra={"ctx_theme": self.theme_name})
            self.status.configure(
                text=f"Тема: {'тёмная' if self.theme_name == 'dark' else 'светлая'}")
        except Exception:
            logger.exception("toggle_theme")

    def _load_file(self, path):
        path = Path(path)
        try:
            info = get_audio_info(path)
        except AudioError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.current_file = path
        self._last_info = dict(info)
        # Новый трек — старые результаты к нему не относятся. Иначе кнопки увели бы
        # работу на файлы предыдущего трека (см. _path_belongs_to_current).
        self.ui_state.reset_for_new_file(Path(path))
        self.vocals_path = None
        self.instrumental_path = None
        self.last_piano_path = None
        # Зона файла превращается в карточку: имя, длительность, формат.
        # Высоту зона меняет сама — после загрузки она ужимается со 140 до 72 px.
        self.drop.show_file(path, info)
        try:
            self.drop.pack_configure(fill="x")
        except Exception:
            logger.debug("Высота зоны файла не пересчиталась", exc_info=True)
        self.wave.plot_file(path, caption=self._wave_caption("Исходный трек", path))
        self._refresh_actions()
        # Что за файл — одной строкой состояния, без «Гц» и «кан.»
        self.status.configure(text=self._file_summary(info))
        self.progress.set(0)
        try:
            self.player.load(path)
            self.combo_listen.set("Исходный трек")
            self.player_bar.set_duration(self.player.duration)
            self.player_bar.set_position(0.0)
            self.player_bar.set_listen_state()
        except Exception:
            logger.exception("загрузка в плеер")

    @staticmethod
    def _file_summary(info: dict) -> str:
        """«song.mp3 — 3:07, 44.1 кГц, стерео» — понятная сводка о файле."""
        parts = [str(info.get("name", "файл"))]
        duration = info.get("duration_sec")
        if duration:
            parts.append(_fmt_time(float(duration)))
        rate = info.get("sample_rate")
        if rate:
            parts.append(f"{int(rate) / 1000:.1f} кГц".replace(".0 ", " "))
        channels = info.get("channels")
        if channels:
            from ui.components import _channels_word
            parts.append(_channels_word(int(channels)))
        return " — " + ", ".join(parts[1:]) if len(parts) > 1 else parts[0]

    def _submit_task(self, submit, *args, kind: TaskType | None = None) -> int | None:
        """
        Общая обёртка постановки задачи: запомнить номер и перейти в «занято».

        Через неё проходят ВСЕ кнопки, поэтому «занято» и «Отмена» работают
        одинаково, чем бы задача ни была.

        `kind` — что именно запустили: по нему строка состояния называет работу
        простыми словами («Разделяем: обработка вокала… 75%»).
        """
        self._current_task_type = kind
        try:
            task_id = submit(*args)
        except Exception as e:
            logger.exception("Не удалось поставить задачу")
            messagebox.showerror("Ошибка", str(e))
            self._set_busy(False)
            return None
        self.current_task_id = task_id
        self._set_busy(True)
        return task_id

    def _on_separate(self):
        if not self.current_file:
            return
        engine_id = self.engine_combo.get()

        # id из селектора → описание движка из реестра. Воркер получает уже
        # готовое описание и не знает ни про подписи, ни про то, как они называются.
        try:
            spec = get_engine(engine_id)
        except KeyError:
            messagebox.showerror("Ошибка", f"Неизвестный движок: {engine_id}")
            self._refresh_actions()
            return

        self.status.configure(text=f"Разделяем… ({spec.label})")
        self._submit_task(self.worker.submit_separation, spec,
                          self.current_file, str(self.output_dir),
                          kind=TaskType.SEPARATE)

    def _on_transcribe_vocals(self):
        self._on_transcribe("vocals")

    def _on_transcribe_instrumental(self):
        self._on_transcribe("instrumental")

    def _on_transcribe(self, kind):
        vocals, instr = self._effective_stems()
        src = vocals if kind == "vocals" else instr
        if not src:
            messagebox.showwarning(
                "Пока нечего превращать в ноты",
                "Сначала разделите ЭТОТ трек — дорожек для него ещё нет.\n\n"
                f"Трек: {Path(self.current_file).name if self.current_file else '—'}")
            return
        out = self.output_dir / f"{Path(self.current_file).stem}_{kind}.mid"
        what = "голоса" if kind == "vocals" else "музыки без голоса"
        self.status.configure(text=f"Записываем ноты {what}…")
        self._submit_task(self.worker.submit_transcription, str(src), str(out),
                          kind=TaskType.TRANSCRIBE)

    def _on_piano(self):
        _, instr = self._effective_stems()
        if not instr:
            messagebox.showwarning(
                "Пока нечего превращать в ноты",
                "Сначала разделите ЭТОТ трек — музыки без голоса для него ещё нет.\n\n"
                f"Трек: {Path(self.current_file).name if self.current_file else '—'}")
            return
        out = self.output_dir / f"{Path(self.current_file).stem}_piano.mid"
        self.status.configure(text="Разбираем музыку на ноты для пианино…")
        self._submit_task(self.worker.submit_piano, str(instr), str(out),
                          kind=TaskType.PIANO)

    def _analysis_source(self) -> tuple[Path | None, str]:
        """
        Что анализировать и как это назвать.

        Берём файл из плеера, а если плеер пуст (трек только что загрузили, а
        стемы для него уже лежат в output/) — вокал, иначе минус, иначе исходник.
        Раньше кнопка была активна, но анализ падал с «Плеер пуст»: кнопка и
        действие смотрели на разные вещи.
        """
        src = self.player.current_file
        if src is not None:
            return Path(src), Path(src).name
        vocals, instr = self._effective_stems()
        if vocals is not None:
            return vocals, f"{vocals.name} (вокал)"
        if instr is not None:
            return instr, f"{instr.name} (минус)"
        if self.current_file:
            return Path(self.current_file), Path(self.current_file).name
        return None, ""

    def _on_analyze_current(self):
        src, label = self._analysis_source()
        if src is None:
            messagebox.showwarning("Нет файла", "Сначала выберите аудиофайл")
            return
        self.status.configure(text="Проверяем качество звука…")
        self._submit_task(self.worker.submit_analysis, str(src), str(self.analysis_dir),
                          kind=TaskType.ANALYZE)

    def _on_analyze_midi(self):
        candidates = self._midi_candidates()

        if not candidates:
            messagebox.showwarning(
                "Нот пока нет",
                "Сначала сделайте ноты для этого трека (вкладка «Ноты»).\n\n"
                f"Трек: {Path(self.current_file).name if self.current_file else '—'}")
            return

        midi_path = candidates[0]
        self.status.configure(text=f"Проверяем ноты: {midi_path.name}…")
        self._submit_task(self.worker.submit_midi_analysis, str(midi_path),
                          str(self.analysis_dir), kind=TaskType.MIDI_ANALYZE)

    def _on_super_pair_analysis(self):
        vocals, instr = self._effective_stems()
        if not vocals or not instr:
            messagebox.showwarning(
                "Нужны обе дорожки",
                "Сначала разделите ЭТОТ трек — нужны и голос, и музыка без голоса.\n\n"
                f"Трек: {Path(self.current_file).name if self.current_file else '—'}")
            return
        self.status.configure(text="Проверяем качество разделения…")
        self._submit_task(self.worker.submit_pair_analysis,
                          str(vocals), str(instr), str(self.analysis_dir),
                          kind=TaskType.ANALYZE_PAIR)

    def _open_output(self):
        self._open_folder(self.output_dir, "open_output")

    def _open_analysis(self):
        self._open_folder(self.analysis_dir, "open_analysis")

    @staticmethod
    def _open_folder(path: Path, context: str) -> None:
        """Открывает папку в системном проводнике (Windows / macOS / Linux)."""
        try:
            p = str(path)
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception:
            logger.exception("Не удалось открыть папку (%s)", context)

    # ---------- Плеер ----------
    def _listen_path(self, value: str) -> Path | None:
        """
        Файл, который соответствует выбору «Слушать».

        Пути берём из _effective_stems — как и все кнопки окна: если стемы
        текущего трека уже лежат в output/ (например, приложение перезапустили),
        они найдутся, а чужие файлы не подставятся.
        """
        if value == "Исходный трек":
            return Path(self.current_file) if self.current_file else None
        vocals, instr = self._effective_stems()
        if value == "Вокал":
            return Path(vocals) if vocals else None
        if value == "Минус":
            return Path(instr) if instr else None
        return None

    @staticmethod
    def _wave_caption(value: str, path) -> str:
        """Подпись под волной: «Вокал: track_vocals.wav — 3:07»."""
        name = Path(path).name
        try:
            seconds = get_audio_info(Path(path)).get("duration_sec")
        except Exception:
            logger.debug("Длительность для подписи волны не получена", exc_info=True)
            seconds = None
        return f"{value}: {name}" + (f" — {_fmt_time(float(seconds))}" if seconds else "")

    def _on_listen_change(self, value):
        """
        Переключение «Слушать»: плеер И волна показывают один и тот же файл.

        Раньше волна оставалась от исходного трека, даже когда играл вокал:
        глазами видно одно, ушами — другое.
        """
        try:
            if value == "—":
                # Ничего не слушаем — и показывать нечего
                self.player.stop()
                self.player_bar.set_position(0.0)
                self.player_bar.set_playing(False)
                self.player_bar.set_listen_state()
                self.wave.clear()
                return

            target = self._listen_path(value)
            if target is None:
                # Файла для этого выбора ещё нет (трек не разделён) — честно
                # говорим об этом и возвращаем выбор в исходное состояние.
                message = f"{value}: файла ещё нет — сначала разделите трек"
                self.status.configure(text=message)
                self.combo_listen.set("—")
                self.wave.clear(f"{message}")
                self.player.stop()
                self.player_bar.set_position(0.0)
                self.player_bar.set_playing(False)
                self.player_bar.set_listen_state()
                return

            was_playing = self.player.is_playing
            # Сначала волна, потом плеер: если со звуком что-то не так, картинка
            # всё равно покажет выбранный файл (и в логе будет видно причину).
            self.wave.plot_file(target, caption=self._wave_caption(value, target))
            try:
                self.player.load(target)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
                if was_playing:
                    self.player.play()
            except Exception:
                logger.exception("Не удалось открыть файл в плеере: %s", target)
                self.status.configure(text=f"{value}: файл не открылся в плеере")
            self.player_bar.set_listen_state()
        except Exception:
            logger.exception("listen_change")

    def _player_play(self):
        try:
            if self.combo_listen.get() == "—" and self.current_file:
                self.combo_listen.set("Исходный трек")
                self.player.load(self.current_file)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
                self.player_bar.set_listen_state()
            self.player.play()
            self.player_bar.set_playing(True)
        except Exception:
            logger.exception("player_play")

    def _player_pause(self):
        try:
            self.player.pause()
            self.player_bar.set_playing(False)
        except Exception:
            logger.exception("player_pause")

    def _player_stop(self):
        try:
            self.player.stop()
            self.player_bar.set_position(0.0)
            self.player_bar.set_playing(False)
        except Exception:
            logger.exception("player_stop")

    def _player_volume(self, v):
        self.player.set_volume(v)

    def _player_seek(self, seconds):
        try:
            self.player.seek(seconds)
        except Exception:
            logger.exception("seek")

    def _tick_position(self):
        try:
            if self.player.is_playing or self.player.position > 0:
                self.player_bar.set_position(self.player.position)
        except Exception:
            logger.exception("tick_position")
        self.after(500, self._tick_position)

    # ---------- Опрос событий воркера ----------
    def _poll_events(self):
        """
        Дренаж очереди событий. Ошибка при обработке ОДНОГО события не должна
        останавливать остальные: раньше весь цикл был обёрнут в общий try/except,
        из-за чего при сбое в _on_done прогресс замирал, а в логе было пусто.
        """
        while True:
            try:
                ev = self.worker.events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_event(ev)
            except Exception:
                logger.exception("Обработка события %r", ev)
        self.after(100, self._poll_events)

    def _handle_event(self, ev):
        """Обработка одного события воркера."""
        if isinstance(ev, ProgressEvent):
            self.progress.set(ev.progress)
            # Понятная запись: «Разделяем: обработка вокала… 75%».
            # Было «Demucs: разделение — 45%»: название движка и процент, из
            # которых непонятно ни что делается, ни зачем.
            self.status.configure(text=self._stage_text(ev.stage, ev.progress))
            return

        if isinstance(ev, DoneEvent):
            self._on_done(ev)
            return

        if isinstance(ev, ErrorEvent):
            self.progress.set(0)
            self._set_busy(False)
            if getattr(ev, "cancelled", False):
                # Отмена — не сбой: человека пугать красным окном незачем
                self.status.configure(text="Отменено")
                return
            self.status.configure(text=f"Не получилось: {ev.message}")
            messagebox.showerror("Не получилось", ev.message)
            # Кнопки включаем НЕ слепо, а по факту: что реально есть для текущего
            # трека. Иначе после падения разделения «Пианино» уводило работу
            # на файлы предыдущего трека.
            self._refresh_actions()
            return

        logger.warning("Неизвестное событие воркера: %r", ev)

    # Как называть работу простыми словами: тип задачи → начало фразы
    STAGE_VERBS = {
        TaskType.SEPARATE: "Разделяем",
        TaskType.SEPARATE_BS: "Разделяем",
        TaskType.SEPARATE_HYBRID: "Разделяем",
        TaskType.TRANSCRIBE: "Записываем ноты",
        TaskType.PIANO: "Разбираем музыку на ноты",
        TaskType.ANALYZE: "Проверяем качество звука",
        TaskType.ANALYZE_PAIR: "Проверяем качество разделения",
        TaskType.MIDI_ANALYZE: "Проверяем ноты",
    }

    # Технические названия стадий → простые слова. Ключи сверяем без регистра.
    STAGE_PLAIN = {
        "чтение аудио": "читаем файл",
        "подготовка аудио": "готовим файл",
        "разделение нейросетью": "разделяем нейросетью",
        "разделение нейросетью (bs-roformer)": "работает модель BS-Roformer",
        "разделение нейросетью (3-5 минут)": "разделяем нейросетью",
        "обработка вокала": "обрабатываем голос",
        "обработка минуса": "обрабатываем музыку",
        "постобработка минуса": "доводим музыку",
        "уровни и динамика": "смотрим громкость",
        "уровни и клиппинг": "смотрим громкость и перегрузку",
        "анализ тишины": "ищем паузы",
        "спектр": "разбираем частоты",
        "спектральный анализ": "разбираем частоты",
        "стерео": "смотрим левый и правый канал",
        "оценка f0": "определяем высоту голоса",
        "огибающие": "считаем громкость по секундам",
        "огибающие по сегментам": "считаем громкость по секундам",
        "bleed-анализ": "ищем остатки голоса в музыке",
        "vpi-анализ": "ищем отпечаток голоса",
        "phase smearing": "смотрим смазанность атаки",
        "spectral tilt": "смотрим затемнение верха",
        "transient loss": "смотрим чёткость ударов",
        "посекундный анализ": "сверяем по секундам",
        "определение высоты тона": "определяем ноты",
        "сборка midi": "собираем файл с нотами",
        "транскрипция нейросетью (muscriptor)": "разбираем музыку нейросетью",
        "сведение в фортепиано": "собираем партию пианино",
        "конвертация flac → wav": "переводим файл в другой формат",
        "сборка результата": "собираем результат",
        "нормализация пары": "выравниваем громкость",
        "графики": "рисуем картинки к отчёту",
        "вердикт": "готовим вывод",
        "разбор нот": "смотрим ноты",
    }

    def _stage_text(self, stage: str, progress: float) -> str:
        """
        «Разделяем: обрабатываем голос… 75%» — что делаем и сколько готово.

        Глагол берём по типу задачи (воркер знает его точно), подробность — из
        названия стадии движка (переведённого на простой язык), процент — из
        события. Так человек читает одну понятную строку вместо
        «Demucs: разделение — 45%».
        """
        verb = self.STAGE_VERBS.get(self._current_task_type, "Работаем")
        percent = int(max(0.0, min(1.0, float(progress))) * 100)
        detail = (stage or "").strip().rstrip(".")
        if not detail or detail.lower() in ("готово", "готов"):
            return f"{verb}… {percent}%"
        plain = self.STAGE_PLAIN.get(detail.lower())
        if plain is None:
            # DECISION: незнакомую стадию показываем КАК ЕСТЬ, не опуская первую
            # букву. Раньше «BS-Roformer (API): модель…» превращалось в
            # «bS-Roformer (API): модель…»: правило «без кричащих заглавных»
            # писалось для обычных слов, а на аббревиатуре ломало название модели.
            plain = detail
        # Процент уже может стоять в самой стадии (движок присылает «… — 38%»):
        # дописывать его второй раз нельзя, иначе в строке выходит «38%… 38%».
        if plain.rstrip().endswith("%"):
            return f"{verb}: {plain}"
        return f"{verb}: {plain}… {percent}%"

    def _on_done(self, ev):
        result = ev.result
        ttype = ev.task_type  # аудит P0.3: маршрутизация по типу задачи, а не по форме результата

        # Работа закончилась: снимаем «занято» и разрешаем кнопки (см. _refresh_actions)
        self._set_busy(False)

        # --- Разделение (Demucs / BS-Roformer / Hybrid) ---
        if ttype in (TaskType.SEPARATE, TaskType.SEPARATE_BS, TaskType.SEPARATE_HYBRID):
            # StemSet ведёт себя как прежний словарь, поэтому обращение по ключу
            # осталось прежним, а форма результата теперь одна для всех движков.
            self.ui_state.vocals = result["vocals"]
            self.ui_state.instrumental = result["instrumental"]
            self.vocals_path = self.ui_state.vocals
            self.instrumental_path = self.ui_state.instrumental
            self.status.configure(text="Готово: голос и музыка лежат в папке output")
            self._refresh_actions()
            # Сразу показываем и включаем вокал, а вместе с ним — его волну:
            # combo_listen.set() не вызывает команду, поэтому рисуем явно.
            self.player_bar.select_listen("Вокал")
            try:
                self.player.load(self.vocals_path)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
                self.wave.plot_file(self.vocals_path,
                                    caption=self._wave_caption("Вокал", self.vocals_path))
            except Exception:
                logger.exception("загрузка вокала")
            self.player_bar.set_listen_state()

        # --- Супер-анализ пары «вокал + минус» ---
        elif ttype is TaskType.ANALYZE_PAIR:
            # Схему отчёта разбирает core.reports, окно получает готовые числа
            view = view_from_super_report(result)
            self.status.configure(text="Проверка разделения готова")
            coverage_line = ""
            if not view.coverage_ok:
                coverage_line = (f"\n\nВнимание: голоса в треке мало "
                                 f"({view.coverage_pct:.1f}% времени), поэтому "
                                 f"вердикт может быть ненадёжным.")
            # DECISION: когда вокала в треке нет вовсе (на вход подали готовый
            # минус), список чисел не показываем. Раньше в этом случае человек
            # читал «Отпечаток голоса в минусе: 139.8 дБ» и «Подозрительных
            # секунд: 88» — числа, посчитанные от тишины, выглядели приговором.
            if not view.vocal_present:
                messagebox.showinfo(
                    "Качество разделения",
                    f"{view.kid}\n\n"
                    f"Что видно по уровням:\n"
                    f"• Вокал: {result.get('level_1_overall', {}).get('vocals_rms_db')} dBFS\n"
                    f"• Минус: {result.get('level_1_overall', {}).get('instrumental_rms_db')} dBFS\n\n"
                    f"Отчёты лежат здесь: {self.analysis_dir}"
                )
                self._refresh_actions()
                return
            vpi_line = (f"• Отпечаток голоса в минусе: {view.vpi_db} дБ"
                        if view.vpi_known
                        else "• Отпечаток голоса в минусе: посчитать не удалось "
                             "(мало тихих участков вокала)")
            messagebox.showinfo(
                "Качество разделения",
                f"Вердикт: {view.kid}\n\n"
                f"Что посчитали (везде меньше — лучше):\n"
                f"• Остатки голоса в музыке: {view.bleed_pct}%\n"
                f"{vpi_line}\n"
                f"• Размазанность звука: {view.phase_db} дБ\n"
                f"• Затемнение верха: {view.tilt_pct}%\n"
                f"• Потеря чёткости ударов: {view.transient_pct}%\n\n"
                f"Найдено огрехов: {view.real_artifacts} "
                f"(музыкальных событий: {view.musical_events})\n"
                f"Доля активного вокала в треке: {view.coverage_pct}%"
                f"{coverage_line}\n\n"
                f"Отчёты лежат здесь: {self.analysis_dir}"
            )
            self._refresh_actions()

        # --- Проверка нот ---
        elif ttype is TaskType.MIDI_ANALYZE:
            s = result["summary"]
            self.status.configure(
                text=f"Ноты: {s['total_notes']} шт., "
                     f"от {s['pitch_min_name']} до {s['pitch_max_name']}"
            )
            messagebox.showinfo(
                "Проверка нот готова",
                f"{result['name']}\n\n"
                f"Нот: {s['total_notes']}\n"
                f"Самая низкая: {s['pitch_min_name']}\n"
                f"Самая высокая: {s['pitch_max_name']}\n"
                f"Средняя: {s['pitch_avg_name']}\n"
                f"Плотность: {s['notes_per_second']} нот в секунду\n\n"
                f"Отчёты лежат здесь: {self.analysis_dir}"
            )

        # --- Проверка качества звука ---
        elif ttype is TaskType.ANALYZE:
            name = result.get("name", "?")
            rms = result["levels"]["rms_db"]
            peak = result["levels"]["peak_db"]
            self.status.configure(text=f"Качество звука проверено: {name}")
            messagebox.showinfo(
                "Проверка звука готова",
                f"{name}\n\n"
                f"Средняя громкость: {rms} дБ (0 — предел, −6 — вдвое тише)\n"
                f"Самый громкий момент: {peak} дБ\n"
                f"Если у самого громкого места 0 дБ — звук «упирается в потолок» "
                f"и может трещать.\n\n"
                f"Отчёты лежат здесь: {self.analysis_dir}")

        # --- Ноты готовы (piano / transcribe) ---
        elif ttype in (TaskType.TRANSCRIBE, TaskType.PIANO) and isinstance(result, Path):
            if "_piano" in result.stem:
                self.last_piano_path = result
            self.ui_state.midi = result
            # Если разбирали минус «одноголосным» способом — напоминаем,
            # что для аккордов есть «Минус как пианино»
            hint = ""
            if ttype is TaskType.TRANSCRIBE and "_instrumental" in result.stem:
                hint = " — если нот мало, попробуйте «Минус как пианино»"
            self.status.configure(text=f"Ноты готовы: {result.name}{hint}")
            self._refresh_actions()
            messagebox.showinfo("Ноты готовы",
                                f"Файл: {result}\n\n"
                                f"Теперь можно нажать «Проверить ноты» "
                                f"на вкладке «Качество».")

        else:
            self.status.configure(text="Готово")
            self._refresh_actions()
        self.progress.set(1.0)
