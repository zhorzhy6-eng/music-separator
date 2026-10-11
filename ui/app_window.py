"""
Главное окно приложения.
Горячие клавиши:
  Space — Play/Pause, → / ← — ±5 сек, Shift+→ / Shift+← — ±30 сек,
  Home — в начало, End — в конец.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from core.audio_utils import get_audio_info, AudioError
from core.player import AudioPlayer
from core.worker import Worker, ProgressEvent, DoneEvent, ErrorEvent, TaskType
from ui.components import DropZone, WaveformView, PlayerBar
from ui.theme import get_theme

logger = logging.getLogger(__name__)

SEEK_SMALL = 5
SEEK_LARGE = 30

# ---------- Demucs ----------
DEMUCS_MODELS = {
    "Demucs: htdemucs (база)":     "htdemucs",
    "Demucs: htdemucs_ft ✅ +SDR": "htdemucs_ft",
}

# ---------- BS-Roformer / Mel-Band Roformer ----------
BS_MODELS = {
    # Базовые (универсальные)
    "BS-Roformer: SW (базовая)":          "BS-Roformer-SW.ckpt",
    "BS-Roformer: Viperx-1297 🏆 инстр":  "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "BS-Roformer: Viperx-1296 🏆 вокал":  "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "BS-Roformer: MelBand-Inst-V2":       "melband_roformer_inst_v2.ckpt",
    "BS-Roformer: Duality-V1 (баланс)":   "melband_roformer_instvoc_duality_v1.ckpt",
    "BS-Roformer: Duality-V2 ✅ баланс":  "melband_roformer_instvox_duality_v2.ckpt",
    # Becruily
    "BS-Roformer: Becruily 🎤 karaoke":   "mel_band_roformer_karaoke_becruily.ckpt",
    "BS-Roformer: Becruily 🎤 vocals":    "mel_band_roformer_vocals_becruily.ckpt",
    "BS-Roformer: Becruily 🎸 inst":      "mel_band_roformer_instrumental_becruily.ckpt",
    # Karaoke
    "BS-Roformer: Karaoke-Aufr33":        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "BS-Roformer: Karaoke-Gabox":         "mel_band_roformer_karaoke_gabox.ckpt",
    "BS-Roformer: Karaoke-Gabox-V2":      "mel_band_roformer_karaoke_gabox_v2.ckpt",
    # Gabox (bleedless — без протечек)
    "BS-Roformer: Gabox 🎸 bleedless v1": "mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt",
    "BS-Roformer: Gabox 🎸 bleedless v2": "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    "BS-Roformer: Gabox 🎸 bleedless v3": "mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
    "BS-Roformer: Gabox 🎸 fullness v1":  "mel_band_roformer_instrumental_fullness_v1_gabox.ckpt",
    "BS-Roformer: Gabox 🎸 fullness v3":  "mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
    # Big (максимум качества, медленно)
    "BS-Roformer: Big Beta 4":            "melband_roformer_big_beta4.ckpt",
    "BS-Roformer: Big Beta 5e":           "melband_roformer_big_beta5e.ckpt",
    # Unwa
    "BS-Roformer: Unwa FT":               "mel_band_roformer_kim_ft_unwa.ckpt",
    "BS-Roformer: Unwa FT2 bleedless":    "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
}

# ---------- Гибриды ----------
HYBRID_ENGINES = {
    "Hybrid: Demucs_ft + Viperx-1297 🏆 инстр": {
        "demucs": "htdemucs_ft",
        "bs": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    "Hybrid: Demucs_ft + Duality-V2 ✅ баланс": {
        "demucs": "htdemucs_ft",
        "bs": "melband_roformer_instvox_duality_v2.ckpt",
    },
    "Hybrid: Demucs_ft + Becruily 🎤 karaoke": {
        "demucs": "htdemucs_ft",
        "bs": "mel_band_roformer_karaoke_becruily.ckpt",
    },
}

ALL_ENGINES = (
    list(DEMUCS_MODELS.keys())
    + list(BS_MODELS.keys())
    + list(HYBRID_ENGINES.keys())
)


class AppWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Audio Separator & Transcriber")
        self.geometry("1000x700")
        self.minsize(900, 600)

        self.theme_name = "dark"
        self.theme = get_theme(self.theme_name)
        self.current_file: str | None = None
        self.output_dir: Path = Path.cwd() / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_dir: Path = Path.cwd() / "logs" / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.vocals_path: Path | None = None
        self.instrumental_path: Path | None = None
        self.last_piano_path: Path | None = None

        self.worker = Worker()
        self.player = AudioPlayer()

        self._build_ui()
        self._bind_hotkeys()
        self._poll_events()
        self._tick_position()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(top, text="🎵 Audio Separator & Transcriber",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(top, text="🌗 Тема", width=100,
                      command=self._toggle_theme).pack(side="right", padx=8)

        self.drop = DropZone(self, on_file=self._load_file, height=100)
        self.drop.pack(fill="x", padx=10, pady=10)

        self.info_label = ctk.CTkLabel(self, text="Файл не выбран", anchor="w")
        self.info_label.pack(fill="x", padx=15)

        self.wave = WaveformView(self, height=180)
        self.wave.pack(fill="x", padx=10, pady=8)

        # Ряд 1: движок + разделить
        actions = ctk.CTkFrame(self)
        actions.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(actions, text="Движок:").pack(side="left", padx=(6, 2), pady=8)
        self.engine_combo = ctk.CTkOptionMenu(
            actions, values=ALL_ENGINES, width=340,
        )
        self.engine_combo.set("BS-Roformer: Duality-V2 ✅ баланс")
        self.engine_combo.pack(side="left", padx=(0, 8), pady=8)

        self.btn_separate = ctk.CTkButton(actions, text="🔀 Разделить",
                                          command=self._on_separate, state="disabled")
        self.btn_separate.pack(side="left", padx=6, pady=8)

        self.btn_open = ctk.CTkButton(actions, text="📂 Папка",
                                      command=self._open_output, state="disabled")
        self.btn_open.pack(side="right", padx=6)

        # Ряд 2: дальнейшие действия
        actions2 = ctk.CTkFrame(self)
        actions2.pack(fill="x", padx=10, pady=6)

        self.btn_voc_midi = ctk.CTkButton(actions2, text="🎼 Вокал → MIDI",
                                          command=lambda: self._on_transcribe("vocals"),
                                          state="disabled")
        self.btn_voc_midi.pack(side="left", padx=6, pady=8)

        self.btn_inst_midi = ctk.CTkButton(actions2, text="🎼 Минус → MIDI",
                                           command=lambda: self._on_transcribe("instrumental"),
                                           state="disabled")
        self.btn_inst_midi.pack(side="left", padx=6)

        self.btn_piano = ctk.CTkButton(actions2, text="🎹 Минус → Пианино",
                                       command=self._on_piano, state="disabled")
        self.btn_piano.pack(side="left", padx=6)

        # Ряд 3: анализы
        actions3 = ctk.CTkFrame(self)
        actions3.pack(fill="x", padx=10, pady=6)

        self.btn_analyze = ctk.CTkButton(actions3, text="📊 Анализ аудио",
                                         command=self._on_analyze_current,
                                         state="disabled")
        self.btn_analyze.pack(side="left", padx=6, pady=8)

        self.btn_analyze_midi = ctk.CTkButton(actions3, text="📊 MIDI-анализ",
                                              command=self._on_analyze_midi,
                                              state="disabled")
        self.btn_analyze_midi.pack(side="left", padx=6)

        self.btn_super_pair = ctk.CTkButton(
            actions3, text="🧠 СУПЕР-АНАЛИЗ v4 ВОКАЛ+МИНУС",
            command=self._on_super_pair_analysis,
            state="disabled",
            fg_color="#8e44ad", hover_color="#732d91",
        )
        self.btn_super_pair.pack(side="left", padx=6)

        self.btn_open_analysis = ctk.CTkButton(actions3, text="📈 Отчёты",
                                               command=self._open_analysis)
        self.btn_open_analysis.pack(side="right", padx=6)

        self.progress = ctk.CTkProgressBar(self)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=10, pady=(0, 4))
        self.status = ctk.CTkLabel(self, text="Готов", anchor="w")
        self.status.pack(fill="x", padx=15)

        self.player_bar = PlayerBar(self,
                                    on_play=self._player_play,
                                    on_pause=self._player_pause,
                                    on_stop=self._player_stop,
                                    on_volume=self._player_volume,
                                    on_seek=self._player_seek)
        self.player_bar.pack(fill="x", padx=10, pady=(4, 10))

        listen = ctk.CTkFrame(self)
        listen.pack(fill="x", padx=10)
        ctk.CTkLabel(listen, text="Слушать:").pack(side="left", padx=6)
        self.combo_listen = ctk.CTkOptionMenu(
            listen, values=["—", "Исходный трек", "Вокал", "Минус"],
            command=self._on_listen_change)
        self.combo_listen.set("—")
        self.combo_listen.pack(side="left", padx=6, pady=6)

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
            pass
        return "break"

    def _seek_end(self):
        try:
            end = self.player.duration
            self.player.seek(end)
            self.player_bar.set_position(end)
        except Exception:
            pass
        return "break"

    def _toggle_play_pause(self):
        try:
            if self.player.is_playing:
                self.player.pause()
            else:
                if self.combo_listen.get() == "—" and self.current_file:
                    self.combo_listen.set("Исходный трек")
                    self.player.load(self.current_file)
                    self.player_bar.set_duration(self.player.duration)
                    self.player_bar.set_position(0.0)
                self.player.play()
        except Exception:
            logger.exception("toggle_play_pause")
        return "break"

    # ---------- Обработчики ----------
    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        ctk.set_appearance_mode(self.theme_name)
        self.theme = get_theme(self.theme_name)

    def _load_file(self, path):
        try:
            info = get_audio_info(path)
        except AudioError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.current_file = path
        self.info_label.configure(
            text=f"📄 {info['name']} — {info['duration_sec']} с, "
                 f"{info['sample_rate']} Гц, {info['channels']} кан.")
        self.wave.plot_file(path)
        self.btn_separate.configure(state="normal")
        self.btn_voc_midi.configure(state="disabled")
        self.btn_inst_midi.configure(state="disabled")
        self.btn_piano.configure(state="disabled")
        self.btn_analyze.configure(state="normal")
        self.btn_analyze_midi.configure(state="disabled")
        self.btn_super_pair.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.status.configure(text="Файл загружен")
        self.progress.set(0)
        try:
            self.player.load(path)
            self.combo_listen.set("Исходный трек")
            self.player_bar.set_duration(self.player.duration)
            self.player_bar.set_position(0.0)
        except Exception:
            logger.exception("загрузка в плеер")

    def _on_separate(self):
        if not self.current_file:
            return
        self.btn_separate.configure(state="disabled")
        self.btn_voc_midi.configure(state="disabled")
        self.btn_inst_midi.configure(state="disabled")
        self.btn_piano.configure(state="disabled")
        self.btn_super_pair.configure(state="disabled")

        engine = self.engine_combo.get()
        self.status.configure(text=f"Разделение: {engine}")

        if engine in DEMUCS_MODELS:
            self.worker.submit_separation(self.current_file, str(self.output_dir),
                                          model_name=DEMUCS_MODELS[engine])
        elif engine in BS_MODELS:
            self.worker.submit_separation_bs(self.current_file, str(self.output_dir),
                                             bs_model=BS_MODELS[engine])
        elif engine in HYBRID_ENGINES:
            cfg = HYBRID_ENGINES[engine]
            self.worker.submit_separation_hybrid(
                self.current_file, str(self.output_dir),
                demucs_model=cfg["demucs"], bs_model=cfg["bs"],
            )
        else:
            messagebox.showerror("Ошибка", f"Неизвестный движок: {engine}")
            self.btn_separate.configure(state="normal")

    def _on_transcribe(self, kind):
        src = self.vocals_path if kind == "vocals" else self.instrumental_path
        if not src or not Path(src).exists():
            messagebox.showwarning("Нет файла", f"Нет файла для {kind}")
            return
        out = self.output_dir / f"{Path(self.current_file).stem}_{kind}.mid"
        self.status.configure(text=f"Транскрипция {kind} → MIDI...")
        self.worker.submit_transcription(str(src), str(out))

    def _on_piano(self):
        if not self.instrumental_path or not Path(self.instrumental_path).exists():
            messagebox.showwarning("Нет файла", "Сначала разделите трек")
            return
        out = self.output_dir / f"{Path(self.current_file).stem}_piano.mid"
        self.btn_piano.configure(state="disabled")
        self.status.configure(text="Транскрипция минуса в пианино (MuScriptor)...")
        self.worker.submit_piano(str(self.instrumental_path), str(out))

    def _on_analyze_current(self):
        src = self.player.current_file
        if src is None:
            messagebox.showwarning("Нет файла", "Плеер пуст")
            return
        self.status.configure(text=f"Анализ аудио: {Path(src).name}...")
        self.worker.submit_analysis(str(src), str(self.analysis_dir))

    def _on_analyze_midi(self):
        candidates = []
        if self.last_piano_path and self.last_piano_path.exists():
            candidates.append(self.last_piano_path)
        if self.current_file:
            stem = Path(self.current_file).stem
            for suffix in ("_piano.mid", "_vocals.mid", "_instrumental.mid"):
                p = self.output_dir / f"{stem}{suffix}"
                if p.exists():
                    candidates.append(p)

        if not candidates:
            messagebox.showwarning("Нет MIDI",
                                   "Сначала сделайте транскрипцию (🎹 или 🎼)")
            return

        midi_path = candidates[0]
        self.status.configure(text=f"Анализ MIDI: {midi_path.name}...")
        self.worker.submit_midi_analysis(str(midi_path), str(self.analysis_dir))

    def _on_super_pair_analysis(self):
        if not self.vocals_path or not self.instrumental_path:
            messagebox.showwarning("Нет пары",
                                   "Сначала разделите трек (вокал + минус)")
            return
        if not Path(self.vocals_path).exists() or not Path(self.instrumental_path).exists():
            messagebox.showwarning("Файлы не найдены",
                                   "Проверьте output/: нужны *_vocals.wav и *_instrumental.wav")
            return
        self.btn_super_pair.configure(state="disabled")
        self.status.configure(text="🧠 СУПЕР-АНАЛИЗ v4 ВОКАЛ+МИНУС...")
        self.worker.submit_pair_analysis(
            str(self.vocals_path), str(self.instrumental_path),
            str(self.analysis_dir),
        )

    def _open_output(self):
        try:
            p = str(self.output_dir)
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception:
            logger.exception("open_output")

    def _open_analysis(self):
        try:
            p = str(self.analysis_dir)
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception:
            logger.exception("open_analysis")

    # ---------- Плеер ----------
    def _on_listen_change(self, value):
        try:
            was_playing = self.player.is_playing
            if value == "Исходный трек" and self.current_file:
                self.player.load(self.current_file)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
                if was_playing:
                    self.player.play()
            elif value == "Вокал" and self.vocals_path:
                self.player.load(self.vocals_path)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
                if was_playing:
                    self.player.play()
            elif value == "Минус" and self.instrumental_path:
                self.player.load(self.instrumental_path)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
                if was_playing:
                    self.player.play()
            else:
                self.player.stop()
                self.player_bar.set_position(0.0)
        except Exception:
            logger.exception("listen_change")

    def _player_play(self):
        try:
            if self.combo_listen.get() == "—" and self.current_file:
                self.combo_listen.set("Исходный трек")
                self.player.load(self.current_file)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
            self.player.play()
        except Exception:
            logger.exception("player_play")

    def _player_pause(self):
        try:
            self.player.pause()
        except Exception:
            pass

    def _player_stop(self):
        try:
            self.player.stop()
            self.player_bar.set_position(0.0)
        except Exception:
            pass

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
            pass
        self.after(500, self._tick_position)

    # ---------- Опрос событий воркера ----------
    def _poll_events(self):
        try:
            while True:
                ev = self.worker.events.get_nowait()
                if isinstance(ev, ProgressEvent):
                    self.progress.set(ev.progress)
                    self.status.configure(text=f"{ev.stage} — {int(ev.progress*100)}%")
                elif isinstance(ev, DoneEvent):
                    self._on_done(ev)
                elif isinstance(ev, ErrorEvent):
                    self.progress.set(0)
                    self.status.configure(text=f"Ошибка: {ev.message}")
                    messagebox.showerror("Ошибка", ev.message)
                    self.btn_separate.configure(state="normal")
                    self.btn_piano.configure(state="normal")
                    self.btn_analyze.configure(state="normal")
                    self.btn_analyze_midi.configure(state="normal")
                    self.btn_super_pair.configure(state="normal")
        except Exception:
            pass
        self.after(100, self._poll_events)

    def _on_done(self, ev):
        result = ev.result
        ttype = ev.task_type  # аудит P0.3: маршрутизация по типу задачи, а не по форме результата

        # --- Разделение (Demucs / BS-Roformer / Hybrid) ---
        if ttype in (TaskType.SEPARATE, TaskType.SEPARATE_BS, TaskType.SEPARATE_HYBRID):
            self.vocals_path = result["vocals"]
            self.instrumental_path = result["instrumental"]
            self.status.configure(text="Разделение готово")
            self.btn_voc_midi.configure(state="normal")
            self.btn_inst_midi.configure(state="normal")
            self.btn_piano.configure(state="normal")
            self.btn_open.configure(state="normal")
            self.btn_analyze_midi.configure(state="normal")
            self.btn_super_pair.configure(state="normal")
            self.combo_listen.set("Вокал")
            try:
                self.player.load(self.vocals_path)
                self.player_bar.set_duration(self.player.duration)
                self.player_bar.set_position(0.0)
            except Exception:
                logger.exception("загрузка вокала")

        # --- СУПЕР-АНАЛИЗ v4 ---
        elif ttype is TaskType.ANALYZE_PAIR:
            v = result.get("verdict", {})
            kid = v.get("kid", "")
            pro = v.get("pro", "")
            phase = result.get("level_8_phase_smearing", {})
            tilt = result.get("level_9_spectral_tilt", {})
            transient = result.get("level_10_transient_loss", {})
            # аудит P0.3: реальные ключи отчёта — level_12_real_artifacts
            # и level_13_instrumental_events (раньше читался несуществующий
            # level_12_suspicious_seconds, из-за чего всегда было 0)
            real = result.get("level_12_real_artifacts", {})
            events = result.get("level_13_instrumental_events", {})
            real_count = real.get("count", 0)
            events_count = events.get("count", 0)
            b = result.get("level_5_bleed", {})
            vpi = result.get("level_6_vpi", {})

            self.status.configure(text="🧠 СУПЕР-АНАЛИЗ v4 готов")
            messagebox.showinfo(
                "🧠 СУПЕР-АНАЛИЗ v4",
                f"👶 {kid}\n\n"
                f"🩸 Bleed: {b.get('bleed_pct', '?')}%\n"
                f"🎤 VPI: {vpi.get('vpi_db', '?')} dB\n"
                f"🌀 Phase Smearing: {phase.get('phase_smearing_db', '?')} dB\n"
                f"🌑 Spectral Tilt: {tilt.get('spectral_tilt_pct', '?')}%\n"
                f"💥 Transient Loss: {transient.get('transient_loss_pct', '?')}%\n"
                f"📍 Реальных артефактов: {real_count}, "
                f"музыкальных событий: {events_count}\n\n"
                f"Отчёты: {self.analysis_dir}"
            )
            self.btn_super_pair.configure(state="normal")

        # --- MIDI-анализ ---
        elif ttype is TaskType.MIDI_ANALYZE:
            s = result["summary"]
            self.status.configure(
                text=f"MIDI: {s['total_notes']} нот, "
                     f"диапазон {s['pitch_min_name']}–{s['pitch_max_name']}, "
                     f"средняя {s['pitch_avg_name']}"
            )
            messagebox.showinfo(
                "MIDI-анализ готов",
                f"{result['name']}\n\n"
                f"Нот: {s['total_notes']}\n"
                f"Диапазон: {s['pitch_min_name']}–{s['pitch_max_name']} "
                f"({s['pitch_range_semitones']} полутонов)\n"
                f"Средняя нота: {s['pitch_avg_name']}\n"
                f"Плотность: {s['notes_per_second']} нот/сек\n\n"
                f"Отчёты: {self.analysis_dir}"
            )

        # --- Анализ аудио ---
        elif ttype is TaskType.ANALYZE:
            name = result.get("name", "?")
            rms = result["levels"]["rms_db"]
            peak = result["levels"]["peak_db"]
            self.status.configure(text=f"Анализ аудио: {name} | RMS {rms} dBFS")
            messagebox.showinfo("Анализ аудио готов",
                                f"{name}\nRMS: {rms} dBFS\nPeak: {peak} dBFS\n"
                                f"Отчёты: {self.analysis_dir}")

        # --- MIDI готов (piano / transcribe) ---
        elif ttype in (TaskType.TRANSCRIBE, TaskType.PIANO) and isinstance(result, Path):
            if "_piano" in result.stem:
                self.last_piano_path = result
            self.status.configure(text=f"MIDI готов: {result.name}")
            self.btn_piano.configure(state="normal")
            self.btn_analyze_midi.configure(state="normal")
            messagebox.showinfo("MIDI готов",
                                f"Файл: {result}\n\n"
                                f"Теперь нажми «📊 MIDI-анализ» для отчёта.")

        else:
            self.status.configure(text=f"Готово: {result}")
            self.btn_piano.configure(state="normal")
            self.btn_analyze_midi.configure(state="normal")
            self.btn_super_pair.configure(state="normal")
        self.progress.set(1.0)