"""
Кастомные виджеты: DropZone (зона файла), FileCard (карточка файла),
WaveformView (волна), SeekBar и PlayerBar (плеер).

Что изменилось в оформлении и почему:
  * DropZone — теперь настоящая «зона»: пунктирная рамка акцентом, крупная
    иконка и подпись, что сюда можно бросить файл. Раньше это был серый
    прямоугольник с текстом, и догадаться о его назначении было нельзя;
  * после загрузки зона превращается в компактную карточку файла: имя, формат,
    длительность и кнопка «Заменить файл». Занимает 64 пикселя вместо 140;
  * волна без файла — 80 пикселей заглушки, с файлом — 120 пикселей и заливка
    с градиентом (сверху ярче, снизу прозрачнее). Оси и подписи убраны: они
    ничего не объясняли, а место занимали;
  * плеер — карточка с круглыми кнопками и толстой полосой перемотки: на
    плеер похоже, а не на набор виджетов.

Чего здесь НЕТ: перетаскивания файла из проводника. Приложение ждёт
tkinterdnd2, который в requirements уже стоит, но не подключён — это отдельная
работа. Пока файл выбирается кликом по зоне, а метод `set_drag_active()` готов
к подключению: когда перетаскивание появится, рамка начнёт «бежать».
"""
from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
# Бэкенд matplotlib здесь НЕ переключаем: график волны строится явно через
# Figure + FigureCanvasTkAgg и в глобальном бэкенде не нуждается. А вот анализам
# (core/analyzer.py, core/super_analyzer.py) нужен Agg — они рисуют в фоновом
# потоке, и TkAgg там приводит к предупреждениям и падениям.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import ImageTk
import soundfile as sf

from assets.svg import dashed_polyline, rounded_rect_path
from ui.button_styles import icon_button, icon_image, recolor, secondary_button
from ui.card import Card
from ui.icons_catalog import EMOJI_FALLBACK
from ui.theme import (
    DROPZONE_HEIGHT,
    FONT_BODY,
    FONT_CAPTION,
    PLAYER_HEIGHT,
    RADIUS_ROUND,
    WAVE_HEIGHT,
    WAVE_PLACEHOLDER_HEIGHT,
    font_spec,
    get_theme,
    mono_family,
)

logger = logging.getLogger(__name__)

# Пунктир зоны файла: длина штриха и пропуска в пикселях
DASH_ON = 7
DASH_OFF = 5
DASH_INSET = 2          # отступ рамки от края зоны
DASH_SPEED_MS = 60      # как часто сдвигается пунктир при перетаскивании
DASH_PHASE_STEP = 2     # на сколько пикселей сдвигать за кадр

# Сколько секунд показывать в подписи волны
WAVE_MIN_SECONDS = 0.05

# Высота карточки загруженного файла (зона ужимается до неё после выбора)
FILE_CARD_HEIGHT = 72


def _hex_rgb(color: str) -> tuple[float, float, float]:
    """'#4A9EFF' → (0.29, 0.62, 1.0) — matplotlib хочет числа 0..1."""
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _fmt_time(seconds: float) -> str:
    """Секунды → «3:07». Для длинных треков — «1:02:03»."""
    total = int(max(0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _fmt_duration(seconds: float) -> str:
    """«3 мин 07 с» — человеческая запись длительности для карточки файла."""
    total = int(round(max(0.0, seconds)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} мин {secs:02d} с"
    return f"{secs} с"


def _channels_word(channels: int) -> str:
    """Число каналов словами: 1 — моно, 2 — стерео, больше — «N канала»."""
    if channels == 1:
        return "моно"
    if channels == 2:
        return "стерео"
    return f"{channels} канала"


# =============================================================
#  DropZone — зона Drag&Drop
# =============================================================
class DropZone(tk.Canvas):
    """
    Зона «перетащите файл сюда»: пунктирная рамка, иконка, подписи.

    Всё нарисовано на одном Canvas, а не собрано из виджетов: только так можно
    нарисовать пунктирную рамку со скруглёнными углами, которая не «рассыпается»
    при изменении размера окна.

    Наружу: `set_drag_active(True/False)` — рамка начинает бежать пунктиром
    (готово к подключению перетаскивания), `show_file(path, info)` и
    `reset()` — переключение между зоной и карточкой файла.
    """

    def __init__(self, master, on_file: Callable[[str], None],
                 theme: dict | None = None, on_replace: Callable[[], None] | None = None,
                 **kw):
        self.theme = theme or get_theme("dark")
        # Высота — параметр со значением по умолчанию: окно может передать свою
        kw.setdefault("height", DROPZONE_HEIGHT)
        super().__init__(
            master,
            bg=self.theme["bg"],
            highlightthickness=0,
            bd=0,
            **kw,
        )
        self.on_file = on_file
        self._on_replace = on_replace or (lambda: self._pick())

        self._hover = False
        self._drag_active = False
        self._dash_phase = 0.0
        self._dash_job: str | None = None
        self._icon_photo = None
        self._current_file: Path | None = None
        self._info: dict = {}

        self._file = FileCard(self, theme=self.theme, on_replace=self._on_replace)
        self._file.bind("<Button-1>", lambda e: self._pick(), add="+")

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", lambda e: self._pick())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.after(60, self._redraw)

    # ---------- внешнее API ----------
    def show_file(self, path: str | Path, info: dict | None = None) -> None:
        """Файл выбран: вместо зоны показываем компактную карточку файла."""
        self._current_file = Path(path)
        self._info = dict(info or {})
        self._stop_dash()
        self.delete("all")
        self._file.set_file(self._current_file, self._info)
        self._file.place(relx=0, rely=0, relwidth=1, relheight=1)
        # DECISION: зона ужимается до высоты карточки. Иначе после загрузки файла
        # на экране оставалась пустая серая полоса на 140 пикселей — ровно то
        # «много пустого места», из-за которого интерфейс и переделывали.
        self._set_height(FILE_CARD_HEIGHT)

    def reset(self) -> None:
        """Снова показываем зону (например, после ошибки загрузки)."""
        self._current_file = None
        self._info = {}
        self._file.place_forget()
        self._set_height(DROPZONE_HEIGHT)
        self._redraw()

    def _set_height(self, height: int) -> None:
        """Меняет высоту зоны/карточки файла."""
        try:
            self.configure(height=height)
            if self.winfo_manager():
                self.pack_configure(fill="x")
        except Exception:
            logger.debug("Высоту зоны файла не удалось поменять", exc_info=True)

    def set_drag_active(self, active: bool) -> None:
        """Файл тащат над зоной: рамка бежит пунктиром."""
        if active == self._drag_active:
            return
        self._drag_active = active
        if active:
            self._start_dash()
        else:
            self._stop_dash()
            self._redraw()

    def apply_theme(self, theme: dict) -> None:
        """Перекрашивает зону под новую тему."""
        self.theme = theme
        self.configure(bg=theme["bg"])
        self._file.apply_theme(theme)
        self._icon_photo = None
        self._redraw()

    @property
    def current_file(self) -> Path | None:
        """Какой файл сейчас показан (нужно окну и проверкам)."""
        return self._current_file

    # ---------- события ----------
    def _pick(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудио", "*.mp3 *.wav *.flac *.m4a *.ogg"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self.on_file(path)

    def _on_enter(self, _event=None) -> None:
        self._hover = True
        self.configure(cursor="hand2")
        self._redraw()

    def _on_leave(self, _event=None) -> None:
        self._hover = False
        self.configure(cursor="")
        self._redraw()

    # ---------- пунктир ----------
    def _start_dash(self) -> None:
        if self._dash_job is None:
            self._tick_dash()

    def _stop_dash(self) -> None:
        if self._dash_job is not None:
            try:
                self.after_cancel(self._dash_job)
            except Exception:
                logger.debug("Таймер пунктира уже снят", exc_info=True)
            self._dash_job = None

    def _tick_dash(self) -> None:
        self._dash_phase = (self._dash_phase + DASH_PHASE_STEP) % (DASH_ON + DASH_OFF)
        # Перерисовываем только рамку: содержимое зоны не менялось.
        # (Раньше здесь стоял вызов несуществующего метода — «бегущий» пунктир
        # падал с AttributeError, но заметить это можно было только с
        # подключённым перетаскиванием.)
        self.delete("frame")
        self._draw_frame(self.winfo_width(), self.winfo_height())
        self._dash_job = self.after(DASH_SPEED_MS, self._tick_dash)

    # ---------- отрисовка ----------
    def _redraw(self) -> None:
        """Рисует зону целиком: рамку, иконку и подписи по центру."""
        self.delete("all")
        if self._current_file is not None:
            return
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return

        self._draw_frame(width, height)
        self._draw_content(width, height)

    def _draw_frame(self, width: int, height: int) -> None:
        """Пунктирная рамка со скруглёнными углами (тег frame — для перерисовки)."""
        if width <= 1 or height <= 1:
            return
        theme = self.theme
        color = theme["accent"] if (self._hover or self._drag_active) else theme["accent_dim"]
        path = rounded_rect_path(DASH_INSET + 1, DASH_INSET + 1,
                                 width - 2 * DASH_INSET - 2,
                                 height - 2 * DASH_INSET - 2, radius=14)
        for piece in dashed_polyline(path, DASH_ON, DASH_OFF, self._dash_phase):
            if len(piece) < 2:
                continue
            self.create_line(*[c for point in piece for c in point],
                             fill=color, width=2, capstyle="round",
                             joinstyle="round", tags="frame")

    def _draw_content(self, width: int, height: int) -> None:
        theme = self.theme
        center = width // 2
        icon_size = 32
        icon_top = DASH_INSET + 18

        img = icon_image("upload-cloud", icon_size,
                         theme["accent"] if self._hover else theme["accent_dim"],
                         master=self)
        if img is not None:
            # CTkImage нельзя положить на Canvas — берём обычную картинку Tk
            # и держим ссылку: без неё Tk соберёт её сборщиком мусора.
            self._icon_photo = ImageTk.PhotoImage(img._light_image)
            self.create_image(center, icon_top + icon_size // 2,
                              image=self._icon_photo)

        title_y = icon_top + icon_size + 26
        self.create_text(
            center, title_y, text="Перетащите файл сюда",
            fill=theme["text"], font=font_spec(15, "bold"),
        )
        self.create_text(
            center, title_y + 22, text="или нажмите, чтобы выбрать",
            fill=theme["text_dim"], font=font_spec(FONT_BODY),
        )
        self.create_text(
            center, title_y + 44,
            text="MP3, WAV, FLAC, M4A, OGG — файл никуда не отправляется",
            fill=theme["text_muted"], font=font_spec(FONT_CAPTION),
        )


# =============================================================
#  FileCard — компактная карточка загруженного файла
# =============================================================
class FileCard(ctk.CTkFrame):
    """
    Загруженный файл в две строки: имя и его параметры.

    Что показывает: имя, «3 мин 07 с · 44.1 кГц · стерео», кнопку «Заменить».
    Технические подписи вроде «Гц» и «кан.» убраны: человеку без музыкального
    образования они ничего не говорят.
    """

    def __init__(self, master, theme: dict | None = None,
                 on_replace: Callable[[], None] | None = None, **kw):
        self.theme = theme or get_theme("dark")
        super().__init__(master, fg_color=self.theme["panel"],
                         corner_radius=10, border_width=1,
                         border_color=self.theme["border"], **kw)
        self._on_replace = on_replace or (lambda: None)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=12)

        self.icon_label = ctk.CTkLabel(row, text="", width=32)
        self.icon_label.pack(side="left", padx=(0, 10))

        texts = ctk.CTkFrame(row, fg_color="transparent")
        texts.pack(side="left", fill="x", expand=True)
        self.name_label = ctk.CTkLabel(
            texts, text="", anchor="w", font=font_spec(14, "bold"),
            text_color=self.theme["text"])
        self.name_label.pack(fill="x")
        self.meta_label = ctk.CTkLabel(
            texts, text="", anchor="w", font=font_spec(FONT_CAPTION),
            text_color=self.theme["text_dim"])
        self.meta_label.pack(fill="x")
        self.texts = texts

        # DECISION: ширину задаём явно. С width=0 («по размеру текста») CTkButton
        # сжимался до 16 пикселей и подпись не помещалась — кнопка выглядела
        # пустой синей рамкой.
        self.replace_btn = secondary_button(
            row, "Заменить файл", self._on_replace, self.theme,
            icon=None, width=150, height=32, font_size=FONT_CAPTION)
        self.replace_btn.pack(side="right", padx=(10, 0))

        self._set_icon()

    def _set_icon(self) -> None:
        img = icon_image("music", 24, self.theme["accent"], master=self.icon_label)
        if img is not None:
            self.icon_label.configure(image=img)
        else:
            self.icon_label.configure(text=EMOJI_FALLBACK["music"],
                                      text_color=self.theme["accent"])

    def set_file(self, path: Path, info: dict | None = None) -> None:
        """Показывает файл: имя и то, что о нём известно."""
        info = dict(info or {})
        self.name_label.configure(text=Path(path).name)
        self.meta_label.configure(text=self._meta_text(info))

    def _meta_text(self, info: dict) -> str:
        parts: list[str] = []
        duration = info.get("duration_sec")
        if duration:
            parts.append(_fmt_duration(float(duration)))
        rate = info.get("sample_rate")
        if rate:
            parts.append(f"{int(rate) / 1000:.1f} кГц".replace(".0 ", " "))
        channels = info.get("channels")
        if channels:
            parts.append(_channels_word(int(channels)))
        return "  ·  ".join(parts) if parts else "Файл выбран"

    def apply_theme(self, theme: dict) -> None:
        self.theme = theme
        try:
            self.configure(fg_color=theme["panel"], border_color=theme["border"])
        except Exception:
            logger.debug("Карточку файла не удалось перекрасить", exc_info=True)
        self.name_label.configure(text_color=theme["text"])
        self.meta_label.configure(text_color=theme["text_dim"])
        # Кнопку перекрашиваем через её роль: цвет фона, обводки и иконки
        # зависят от темы, и «забыть» её — самая частая ошибка при перекраске.
        recolor(self.replace_btn, "secondary", theme)
        self._set_icon()


# =============================================================
#  WaveformView — волновая форма
# =============================================================
class WaveformView(ctk.CTkFrame):
    """
    Волна трека: заглушка 80 px без файла, 120 px с градиентной заливкой.

    Раньше график занимал 170 пикселей всегда — даже пустой, с осями 0.0–1.0,
    которые ничего не объясняли. Теперь:
      * нет файла — невысокая заглушка с картинкой-иконкой и подсказкой;
      * есть файл — заливка от центра: сверху ярче, снизу прозрачнее;
      * под волной подпись: «Вокал: song_vocals.wav — 3:07».
    """

    def __init__(self, master, theme: dict | None = None, **kw):
        self.theme = theme or get_theme("dark")
        # DECISION: фон рамки задаём явно. Без этого CustomTkinter красит её
        # своим дефолтным gray17/gray86, и под волной оставалась серая полоса —
        # она не совпадала ни с фоном окна, ни с панелью.
        super().__init__(master, fg_color=self.theme["panel"], **kw)
        self._last_path: str | Path | None = None
        self._last_caption: str = ""
        self._has_plot = False
        # Высоту задаём сами: заглушка ниже, волна выше
        height = kw.get("height") or WAVE_PLACEHOLDER_HEIGHT

        self.fig = Figure(figsize=(8, height / 100), dpi=100,
                          facecolor=self.theme["panel"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.theme["panel"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(height=height, bg=self.theme["panel"])
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Подпись под волной: чей файл нарисован. Без неё легко перепутать, что
        # сейчас видно — исходный трек, вокал или минус.
        self.caption = ctk.CTkLabel(
            self, text="", anchor="w", font=font_spec(FONT_CAPTION),
            text_color=self.theme["text_dim"],
        )
        self.caption.pack(fill="x", padx=4, pady=(2, 0))
        self._style_axes()

    # ---------- внешнее API ----------
    def apply_theme(self, theme: dict) -> None:
        """Перекрашивает график под новую тему и перерисовывает волну."""
        self.theme = theme
        self.fig.set_facecolor(theme["panel"])
        self.ax.set_facecolor(theme["panel"])
        try:
            self.configure(fg_color=theme["panel"])
            self.canvas.get_tk_widget().configure(bg=theme["panel"])
            self.caption.configure(text_color=theme["text_dim"])
        except Exception:
            logger.debug("Не удалось перекрасить подпись волны", exc_info=True)
        if self._last_path:
            self.plot_file(self._last_path, caption=self._last_caption)
        else:
            self.canvas.draw()

    def plot_file(self, path: str | Path, caption: str | None = None) -> None:
        """
        Рисует волну файла. caption — подпись под графиком (по умолчанию имя файла).

        Работает с любым WAV, который читает soundfile: моно, стерео, 16/24 бита.
        """
        self._last_path = path
        self._last_caption = caption if caption is not None else Path(path).name
        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = y.mean(axis=1)
            # Заглушка выключала оси — возвращаем: у волны они есть (без подписей)
            self.ax.set_axis_on()
            self._draw_wave(mono, len(y) / sr if sr else 0.0)
            self._has_plot = True
            self._set_height(WAVE_HEIGHT)
            self._set_caption(self._last_caption)
        except Exception:
            logger.exception("Не удалось отрисовать волну")
            self._set_caption(f"{EMOJI_FALLBACK['alert-triangle']} "
                              f"Не удалось показать волну: {Path(path).name}")

    def clear(self, message: str = "Волна не выбрана") -> None:
        """Стирает график: слушать нечего — и показывать нечего."""
        self._last_path = None
        self._last_caption = ""
        self._has_plot = False
        try:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.set_facecolor(self.theme["panel"])
            self.canvas.draw()
        except Exception:
            logger.exception("Не удалось очистить волну")
        self._set_height(WAVE_PLACEHOLDER_HEIGHT)
        self._set_caption(message)

    def show_placeholder(self, text: str = "Загрузите файл, чтобы увидеть волну") -> None:
        """
        Заглушка вместо пустой волны: текст по центру, без осей 0.0–1.0.

        Зачем: пустой график с осями выглядел как «сломанный». Пока файла нет,
        честнее показать, что тут будет и как его получить.
        """
        self._last_path = None
        self._last_caption = ""
        self._has_plot = False
        try:
            self.ax.clear()
            # Оси и рамка не нужны: рисовать нечего, а «0.0–1.0» только сбивает
            self.ax.set_axis_off()
            self.ax.set_facecolor(self.theme["panel"])
            self.ax.text(
                0.5, 0.5, text, transform=self.ax.transAxes,
                ha="center", va="center", fontsize=11, wrap=True,
                color=self.theme["text_dim"],
            )
            self.canvas.draw()
        except Exception:
            logger.exception("Не удалось нарисовать заглушку волны")
        self._set_height(WAVE_PLACEHOLDER_HEIGHT)
        # Подпись НЕ дублируем: текст уже нарисован в центре волны.
        self._set_caption("")

    @property
    def has_plot(self) -> bool:
        """Показана настоящая волна или заглушка."""
        return self._has_plot

    # ---------- внутреннее ----------
    def _set_height(self, height: int) -> None:
        """Меняет высоту волны: 80 пикселей для заглушки, 120 для графика."""
        try:
            self.canvas.get_tk_widget().configure(height=height)
            if self.winfo_manager():
                self.pack_configure(fill="x")
        except Exception:
            logger.debug("Высоту волны не удалось поменять", exc_info=True)

    def _style_axes(self) -> None:
        """Волна без осей и рамок: только сама форма сигнала."""
        theme = self.theme
        self.ax.set_axis_off()
        self.ax.set_facecolor(theme["panel"])
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def _draw_wave(self, mono: np.ndarray, duration: float) -> None:
        """
        Волна-заливка от центра: сверху ярче, ниже прозрачнее.

        DECISION: градиент рисуем не «картинкой с обрезкой по контуру» (clip
        path в matplotlib капризен и на части версий просто не применяется), а
        столбиками: на каждый пиксель ширины — вертикальная линия со своим
        цветом и прозрачностью. Выглядит так же, а работает всегда.
        """
        theme = self.theme
        self.ax.clear()
        self._style_axes()

        if mono.size == 0:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-1, 1)
            self.canvas.draw()
            return

        # Огибающая: на каждый столбик — минимум и максимум сигнала.
        # Так волна выглядит как в аудиоредакторе, а не как «ёлочка» из отсчётов.
        columns = 700
        step = max(1, mono.size // columns)
        trimmed = mono[: (mono.size // step) * step].reshape(-1, step)
        if trimmed.size == 0:
            trimmed = mono.reshape(1, -1)
        low = trimmed.min(axis=1)
        high = trimmed.max(axis=1)

        # Тихий трек подтягиваем, чтобы волна была видна; громкий не трогаем —
        # иначе по картинке нельзя будет судить, где в треке громче.
        peak = float(max(np.max(np.abs(high)), np.max(np.abs(low)), 1e-6))
        scale = min(1.0, 0.95 / peak)
        low = np.clip(low * scale, -1.0, 1.0)
        high = np.clip(high * scale, -1.0, 1.0)

        rgb = np.array(_hex_rgb(theme["accent"]))
        x = np.linspace(0, max(duration, WAVE_MIN_SECONDS), low.size)

        # Столбик высотой 32 точки: сверху светлее, к низу прозрачнее — это и
        # есть градиент. Прозрачность столбика зависит и от его высоты: тихие
        # места бледнее, громкие плотнее.
        rows = 32
        level = np.linspace(0.0, 1.0, rows)                  # 0 — низ, 1 — верх
        amplitude = np.clip((high - low) / 2, 0.02, 1.0)     # высота каждого столбика
        alpha_col = (0.35 + 0.6 * np.sqrt(amplitude))[None, :]        # (1, columns)
        brightness = (0.60 + 0.40 * level).reshape(rows, 1, 1)        # (rows, 1, 1)
        alpha = alpha_col * (0.10 + 0.90 * level ** 1.4).reshape(rows, 1)

        colors = np.zeros((rows, low.size, 4))
        colors[:, :, :3] = np.clip(rgb.reshape(1, 1, 3) * brightness, 0.0, 1.0)
        colors[:, :, 3] = np.clip(alpha, 0.0, 1.0)

        # Картинку с градиентом обрезаем по форме волны — иначе она залила бы
        # весь прямоугольник. Сначала рисуем саму форму (она даёт контур для
        # обрезки), потом кладём на неё градиент.
        envelope = self.ax.fill_between(x, low, high, color="none", linewidth=0,
                                        zorder=2)
        gradient = self.ax.imshow(
            colors, aspect="auto", origin="lower", interpolation="bilinear",
            extent=[0, max(duration, WAVE_MIN_SECONDS), -1.0, 1.0], zorder=3,
        )
        try:
            gradient.set_clip_path(envelope.get_paths()[0],
                                   transform=self.ax.transData)
        except Exception:
            # Обрезка не сработала (редкая сборка matplotlib) — оставляем
            # градиент во весь рост: волна на месте, только без контура.
            logger.debug("Обрезка градиента волны не удалась", exc_info=True)

        self.ax.plot(x, high, color=rgb, linewidth=0.6, alpha=0.75, zorder=4)
        self.ax.plot(x, low, color=rgb, linewidth=0.6, alpha=0.45, zorder=4)
        self.ax.set_xlim(0, max(duration, WAVE_MIN_SECONDS))
        self.ax.set_ylim(-1, 1)
        self.canvas.draw()

    def _set_caption(self, text: str) -> None:
        try:
            self.caption.configure(text=text)
        except Exception:
            logger.debug("Не удалось обновить подпись волны", exc_info=True)


# =============================================================
#  SeekBar — собственный слайдер перемотки на Canvas
# =============================================================
class SeekBar(tk.Canvas):
    """
    Собственный seek-слайдер.
    - Клик по треку — мгновенный seek в это место.
    - Перетаскивание ползунка — seek при отпускании.
    - set_position() — программное обновление без вызова on_seek.
    """
    # DECISION: высота 22 вместо 16. Полоса стала толще (4 px), и для неё
    # нужен запас: ползунок не должен упираться в края области.
    H = 22
    PAD = 8
    TRACK = 4

    def __init__(self, master, on_seek: Callable[[float], None],
                 theme: dict | None = None, **kw):
        self.theme = theme or get_theme("dark")
        super().__init__(
            master,
            height=self.H,
            bg=self.theme["panel"],
            highlightthickness=0,
            bd=0,
            **kw,
        )
        self._on_seek = on_seek
        self._duration = 0.0
        self._pos = 0.0          # текущая позиция в секундах
        self._dragging = False
        self._drag_pos = 0.0

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        # первые значения
        self.after(50, self._redraw)

    def apply_theme(self, theme: dict) -> None:
        """Меняет фон и цвета дорожки под новую тему."""
        self.theme = theme
        self.configure(bg=theme["panel"])
        self._redraw()

    # ---------- внешнее API ----------
    def set_duration(self, seconds: float) -> None:
        self._duration = max(0.0, float(seconds))
        self._pos = 0.0
        self._redraw()

    def set_position(self, seconds: float) -> None:
        """Программное обновление — НЕ вызывает on_seek."""
        if self._dragging:
            return
        self._pos = max(0.0, min(float(seconds), self._duration))
        self._redraw()

    # ---------- отрисовка ----------
    def _redraw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or h <= 1:
            return

        # фон-трек
        cy = h // 2
        track_x0 = self.PAD
        track_x1 = w - self.PAD
        self.create_line(track_x0, cy, track_x1, cy,
                         fill=self.theme["border"], width=self.TRACK,
                         capstyle="round")

        # заполнение до текущей позиции
        if self._duration > 0:
            frac = max(0.0, min(1.0, self._pos / self._duration))
        else:
            frac = 0.0
        fill_x = track_x0 + (track_x1 - track_x0) * frac
        if frac > 0:
            self.create_line(track_x0, cy, fill_x, cy,
                             fill=self.theme["accent"], width=self.TRACK,
                             capstyle="round")

        # ползунок: круг, чуть темнее акцента, с обводкой цветом панели
        r = 6
        self.create_oval(
            fill_x - r, cy - r, fill_x + r, cy + r,
            fill=self.theme["accent"], outline=self.theme["panel"], width=2,
        )

    # ---------- события мыши ----------
    def _x_to_seconds(self, x: int) -> float:
        w = self.winfo_width()
        track_x0 = self.PAD
        track_x1 = w - self.PAD
        if track_x1 <= track_x0:
            return 0.0
        frac = (x - track_x0) / (track_x1 - track_x0)
        frac = max(0.0, min(1.0, frac))
        return frac * self._duration

    def _on_click(self, event) -> None:
        if self._duration <= 0:
            return
        # Перетаскиваем или кликаем — в обоих случаях начинаем drag
        self._dragging = True
        self._drag_pos = self._x_to_seconds(event.x)
        self._pos = self._drag_pos
        self._redraw()

    def _on_drag(self, event) -> None:
        if not self._dragging or self._duration <= 0:
            return
        self._drag_pos = self._x_to_seconds(event.x)
        self._pos = self._drag_pos
        self._redraw()

    def _on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        # финальная позиция
        self._drag_pos = self._x_to_seconds(event.x)
        self._pos = self._drag_pos
        self._redraw()
        logger.info("SeekBar -> seek(%.2f)", self._drag_pos)
        try:
            self._on_seek(self._drag_pos)
        except Exception:
            logger.exception("Ошибка on_seek")


# =============================================================
#  PlayerBar — панель плеера
# =============================================================
class PlayerBar(Card):
    """
    Плеер-карточка: круглые кнопки, толстая полоса перемотки, «Слушать:».

    Как было: три серые кнопки с эмодзи и тонкая полоска — на плеер не похоже.
    Как стало: карточка с заголовком «Готово», круглые кнопки (главная —
    акцентная), моноширинный таймер и выбор источника в той же строке.
    Высота строки управления — 56 пикселей (PLAYER_HEIGHT).

    Seek — через собственный SeekBar (Canvas), а не CTkSlider: нужен мгновенный
    переход по клику в любую точку трека.
    """

    def __init__(
        self,
        master,
        on_play,
        on_pause,
        on_stop,
        on_volume,
        on_seek,
        on_listen=None,
        listen_values: tuple[str, ...] = ("—", "Исходный трек", "Вокал", "Минус"),
        theme: dict | None = None,
        **kw,
    ):
        theme = theme or get_theme("dark")
        super().__init__(master, title="Прослушать результат", theme=theme,
                         icon="play", **kw)
        self.theme = theme
        self._duration = 0.0
        self._listen_values = listen_values

        # --- Одна строка: кнопки, таймер, громкость, «Слушать» ---
        row = ctk.CTkFrame(self.body, fg_color="transparent",
                           height=PLAYER_HEIGHT)
        row.pack(fill="x")

        self.btn_play = icon_button(row, "play", on_play, theme,
                                    size=RADIUS_ROUND * 2, icon_size=20,
                                    round_=True)
        self.btn_pause = icon_button(row, "pause", on_pause, theme,
                                     size=RADIUS_ROUND * 2, icon_size=20,
                                     round_=True)
        self.btn_stop = icon_button(row, "square", on_stop, theme,
                                    size=RADIUS_ROUND * 2, icon_size=20,
                                    round_=True)
        self.btn_play.pack(side="left", padx=(0, 4))
        self.btn_pause.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)
        self.set_playing(False)

        # Время — в той же строке: отдельная строка под счётчик не нужна.
        # Шрифт моноширинный: цифры не «прыгают» при смене секунд.
        self.time_label = ctk.CTkLabel(
            row, text="0:00 / 0:00", width=96, anchor="w",
            font=(mono_family(), 12), text_color=theme["text_dim"])
        self.time_label.pack(side="left", padx=(10, 8))

        self.vol_icon = ctk.CTkLabel(row, text="", width=18)
        self.vol_icon.pack(side="left", padx=(4, 0))
        self._set_volume_icon()

        # Громкость: подпись с процентом рядом со слайдером. Одного значка мало —
        # непонятно, насколько тихо играет, а цифра отвечает на этот вопрос.
        self.vol_value = ctk.CTkLabel(
            row, text="80%", width=38, anchor="w",
            font=(mono_family(), 11), text_color=theme["text_dim"])
        self.vol_value.pack(side="left", padx=(2, 2))

        self._on_volume = on_volume

        def _volume(value, callback=on_volume):
            self.vol_value.configure(text=f"{int(round(float(value) * 100))}%")
            callback(value)

        self.vol = ctk.CTkSlider(
            row, from_=0, to=1, command=_volume, width=130,
            height=16, button_length=0, corner_radius=8,
            progress_color=theme["accent"], button_color=theme["accent"],
            button_hover_color=theme["accent_hover"], fg_color=theme["border"])
        self.vol.set(0.8)
        self.vol.pack(side="left")

        # «Слушать:» — выбор источника звука. Живёт здесь, а не отдельной строкой:
        # это часть плеера, и порознь они только занимали два ряда.
        # Подпись пакуется по правилу pack(side="right"): кто вызван позже —
        # тот левее, поэтому порядок вызовов здесь обратный.
        self.combo_listen = None
        if on_listen is not None:
            self.combo_listen = ctk.CTkOptionMenu(
                row, values=list(listen_values), command=on_listen, width=132,
                height=32, corner_radius=8, font=font_spec(FONT_CAPTION),
                fg_color=theme["panel_hover"], button_color=theme["panel_hover"],
                button_hover_color=theme["border"], text_color=theme["text"],
                dropdown_fg_color=theme["panel"],
                dropdown_hover_color=theme["panel_hover"],
                dropdown_text_color=theme["text"])
            self.combo_listen.set(listen_values[0])
            self.combo_listen.pack(side="right")
            self.listen_label = ctk.CTkLabel(
                row, text="Слушать:", font=font_spec(FONT_CAPTION),
                text_color=theme["text_dim"])
            self.listen_label.pack(side="right", padx=(6, 4))

        # --- Толстая полоса перемотки под строкой ---
        self.seek = SeekBar(self.body, on_seek=on_seek, theme=theme)
        self.seek.pack(fill="x", pady=(2, 0))

    # ---------- внешнее API ----------
    def apply_theme(self, theme: dict) -> None:
        """Прокидывает новую тему в полосу перемотки и подписи плеера."""
        super().apply_theme(theme)
        self.theme = theme
        self.seek.apply_theme(theme)
        self.time_label.configure(text_color=theme["text_dim"])
        self.listen_label.configure(text_color=theme["text_dim"])
        self.vol_value.configure(text_color=theme["text_dim"])
        self.vol.configure(progress_color=theme["accent"],
                           button_color=theme["accent"],
                           button_hover_color=theme["accent_hover"],
                           fg_color=theme["border"])
        if self.combo_listen is not None:
            self.combo_listen.configure(
                fg_color=theme["panel_hover"], button_color=theme["panel_hover"],
                button_hover_color=theme["border"], text_color=theme["text"],
                dropdown_fg_color=theme["panel"],
                dropdown_hover_color=theme["panel_hover"],
                dropdown_text_color=theme["text"])
        self._set_volume_icon()
        self.set_playing(self._playing)

    def set_playing(self, playing: bool) -> None:
        """
        Показывает, что происходит сейчас: играет или стоит.

        Акцентная заливка у активной кнопки — то же правило, что и для кнопок
        действий: главное на экране видно сразу.
        """
        self._playing = bool(playing)
        try:
            play_fg = self.theme["panel_hover"] if self._playing else self.theme["accent"]
            play_hover = self.theme["border"] if self._playing else self.theme["accent_hover"]
            self.btn_play.configure(fg_color=play_fg, hover_color=play_hover)
            self.btn_pause.configure(
                fg_color=self.theme["accent"] if self._playing else "transparent")
        except Exception:
            logger.debug("Состояние плеера не отрисовалось", exc_info=True)

    def set_listen_state(self, playing: bool = False) -> None:
        """
        Приводит плеер в согласованный вид: подпись в заголовке и выпадающий
        список показывают ОДНО И ТО ЖЕ.

        `OptionMenu.set()` сам команду не вызывает — поэтому окно может
        переключить источник программно (например, сразу после разделения),
        и плеер обязан это показать; раньше список оставался на старом
        значении, пока не выберешь вручную.
        """
        value = self.combo_listen.get() if self.combo_listen else "—"
        self.set_subtitle("" if value == "—" else value)

    def select_listen(self, value: str) -> None:
        """Программный выбор источника: список, подпись и волна — в один шаг."""
        if self.combo_listen is None:
            return
        try:
            self.combo_listen.set(value)
        except Exception:
            logger.debug("Значение %r не встало в список «Слушать»", value,
                         exc_info=True)
        self.set_listen_state()

    # --- обновление из главного окна ---
    def set_duration(self, seconds: float) -> None:
        self._duration = max(0.0, float(seconds))
        self.seek.set_duration(self._duration)
        self._update_time_label(0.0)

    def set_position(self, seconds: float) -> None:
        self.seek.set_position(seconds)
        self._update_time_label(seconds)

    def _update_time_label(self, pos: float) -> None:
        self.time_label.configure(text=f"{_fmt_time(pos)} / {_fmt_time(self._duration)}")

    def _set_volume_icon(self) -> None:
        img = icon_image("volume-2", 16, self.theme["text_dim"], master=self.vol_icon)
        if img is not None:
            self.vol_icon.configure(image=img, text="")
        else:
            self.vol_icon.configure(image=None, text=EMOJI_FALLBACK["volume-2"],
                                    text_color=self.theme["text_dim"])


__all__ = ["DropZone", "FileCard", "WaveformView", "SeekBar", "PlayerBar"]
