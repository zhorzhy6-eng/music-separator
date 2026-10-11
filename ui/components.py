"""
Кастомные виджеты: WaveformView, PlayerBar (со своим seek-слайдером), DropZone.
"""
from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# =============================================================
#  DropZone — зона Drag&Drop
# =============================================================
class DropZone(ctk.CTkFrame):
    def __init__(self, master, on_file: Callable[[str], None], **kw):
        super().__init__(master, **kw)
        self.on_file = on_file
        self.label = ctk.CTkLabel(
            self,
            text="Перетащите аудиофайл сюда\nили нажмите, чтобы выбрать",
            font=ctk.CTkFont(size=14),
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=20, pady=30)
        self.label.bind("<Button-1>", lambda e: self._pick())
        self.bind("<Button-1>", lambda e: self._pick())

    def _pick(self):
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


# =============================================================
#  WaveformView — волновая форма
# =============================================================
class WaveformView(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.fig = Figure(figsize=(8, 2), dpi=80, facecolor="#242424")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#242424")
        self.ax.tick_params(colors="#9a9a9a")
        for spine in self.ax.spines.values():
            spine.set_color("#444")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_file(self, path: str | Path) -> None:
        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = y.mean(axis=1)
            step = max(1, len(mono) // 4000)
            mono = mono[::step]
            t = np.linspace(0, len(y) / sr, len(mono))
            self.ax.clear()
            self.ax.plot(t, mono, color="#3b8ed0", linewidth=0.6)
            self.ax.set_xlabel("сек", color="#9a9a9a")
            self.canvas.draw()
        except Exception:
            logger.exception("Не удалось отрисовать волну")


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
    H = 24
    PAD = 8

    def __init__(self, master, on_seek: Callable[[float], None], **kw):
        super().__init__(
            master,
            height=self.H,
            bg="#242424",
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
        self.create_line(track_x0, cy, track_x1, cy, fill="#3a3a3a", width=4)

        # заполнение до текущей позиции
        if self._duration > 0:
            frac = max(0.0, min(1.0, self._pos / self._duration))
        else:
            frac = 0.0
        fill_x = track_x0 + (track_x1 - track_x0) * frac
        self.create_line(track_x0, cy, fill_x, cy, fill="#3b8ed0", width=4)

        # ползунок
        r = 8
        self.create_oval(
            fill_x - r, cy - r, fill_x + r, cy + r,
            fill="#3b8ed0", outline="#e8e8e8", width=1,
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
class PlayerBar(ctk.CTkFrame):
    """
    Панель плеера: Play / Pause / Stop + SeekBar + таймер + громкость.
    Seek — через собственный SeekBar (Canvas), а не CTkSlider.
    """
    def __init__(
        self,
        master,
        on_play,
        on_pause,
        on_stop,
        on_volume,
        on_seek,
        **kw,
    ):
        super().__init__(master, **kw)

        # --- Верхний ряд ---
        row1 = ctk.CTkFrame(self, fg_color="transparent")
        row1.pack(fill="x", padx=4, pady=(6, 2))

        self.btn_play = ctk.CTkButton(row1, text="▶", width=40, command=on_play)
        self.btn_pause = ctk.CTkButton(row1, text="⏸", width=40, command=on_pause)
        self.btn_stop = ctk.CTkButton(row1, text="⏹", width=40, command=on_stop)
        self.btn_play.pack(side="left", padx=4)
        self.btn_pause.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)

        self.time_label = ctk.CTkLabel(row1, text="0:00 / 0:00", width=110)
        self.time_label.pack(side="left", padx=(12, 4))

        ctk.CTkLabel(row1, text="🔊").pack(side="left", padx=(12, 4))
        self.vol = ctk.CTkSlider(row1, from_=0, to=1, command=on_volume, width=150)
        self.vol.set(0.8)
        self.vol.pack(side="left")

        # --- Нижний ряд: SeekBar ---
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", padx=4, pady=(0, 6))

        self.seek = SeekBar(row2, on_seek=on_seek)
        self.seek.pack(fill="x", expand=True, padx=4, pady=4)

        self._duration = 0.0

    # --- обновление из главного окна ---
    def set_duration(self, seconds: float) -> None:
        self._duration = max(0.0, float(seconds))
        self.seek.set_duration(self._duration)
        self._update_time_label(0.0)

    def set_position(self, seconds: float) -> None:
        self.seek.set_position(seconds)
        self._update_time_label(seconds)

    def _update_time_label(self, pos: float) -> None:
        def fmt(s: float) -> str:
            s = int(max(0, s))
            return f"{s // 60}:{s % 60:02d}"
        self.time_label.configure(text=f"{fmt(pos)} / {fmt(self._duration)}")