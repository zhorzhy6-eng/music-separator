"""
Разделение аудио на вокал и инструментал через Demucs 4.x.
Использует официальный API demucs.api.Separator.

Постобработка:
  - Вокал: High-Pass Filter 100 Гц (2-й порядок Баттерворта).
  - Минус: Notch (Band EQ) на 2500 Гц, Q = 1.5.
  - Нормализация пика до -0.3 dBFS.

⚠️ Про глубину notch (аудит P0.1, вариант B — физика оставлена намеренно):
INSTR_NOTCH_GAIN_DB — параметр смешивания, а НЕ глубина. Фактическая глубина
на центре = 20*log10(1 - 10**(gain_db/20)); для текущего -3.0 это ≈ -10.7 dB.
Инверсия: чем БЛИЖЕ gain_db к 0, тем ГЛУБЖЕ notch, и наоборот — чем
отрицательнее, тем МЯГЧЕ (таблица соответствий в audio_utils.measure_notch_depth).
Фактическая глубина пишется в лог один раз за процесс (ctx_gain_db_measured).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, iirnotch, tf2sos

from core.audio_utils import ensure_wav, measure_notch_depth

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra"]
TARGET_PEAK = 10.0 ** (-0.3 / 20.0)  # ≈ 0.9660

# ---------- Параметры постобработки ----------
VOCALS_HPF_HZ = 100.0          # срез HPF для вокала
VOCALS_HPF_ORDER = 2           # порядок фильтра Баттерворта

INSTR_NOTCH_HZ = 2500.0        # центр notch для минуса
INSTR_NOTCH_Q = 1.5            # добротность
INSTR_NOTCH_GAIN_DB = -3.0     # параметр смешивания, НЕ глубина: фактически ≈ -10.7 dB (см. docstring)


# Глубину ночта измеряем один раз на процесс (см. аудит P0.1)
_NOTCH_DEPTH_REPORTED = False


def _report_notch_depth_once(sr: int, center_hz: float, q: float, gain_db: float) -> None:
    """
    Один раз за процесс измеряет и логирует фактическую глубину notch-фильтра.

    Нужно, чтобы в логе были честные цифры, а не только заявленные константы.
    Сама формула фильтра не меняется — только диагностика.
    """
    global _NOTCH_DEPTH_REPORTED
    if _NOTCH_DEPTH_REPORTED:
        return
    _NOTCH_DEPTH_REPORTED = True
    try:
        depth_db = measure_notch_depth(sr, center_hz, q, gain_db)
    except Exception as e:
        logger.warning("Не удалось измерить глубину notch: %s", e)
        return
    logger.info(
        "Notch: заявлено %s dB, фактическая глубина на центре %.2f dB",
        gain_db, depth_db,
        extra={"ctx_center_hz": center_hz, "ctx_q": q,
               "ctx_gain_db_declared": gain_db,
               "ctx_gain_db_measured": round(depth_db, 2)},
    )


class SeparationError(Exception):
    """Ошибка разделения."""


class AudioSeparator:
    def __init__(self, model_name: str = "htdemucs") -> None:
        if model_name not in AVAILABLE_MODELS:
            raise SeparationError(f"Неизвестная модель: {model_name}")
        self.model_name = model_name
        self._sep = None
        logger.info("AudioSeparator init", extra={"ctx_model": model_name})

    def _ensure_model(self):
        if self._sep is not None:
            return
        try:
            from demucs.api import Separator
            logger.info("Загружаю модель Demucs: %s", self.model_name)
            self._sep = Separator(model=self.model_name, progress=True)
            logger.info("Модель Demucs загружена")
        except Exception as e:
            logger.exception("Не удалось загрузить модель %s", self.model_name)
            raise SeparationError(f"Не удалось загрузить модель: {e}") from e

    def separate(self, input_path, output_dir, progress_cb=None) -> dict[str, Path]:
        self._ensure_model()
        in_path = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_path.exists():
            raise SeparationError(f"Файл не найден: {in_path}")

        try:
            if progress_cb:
                progress_cb(0.03, "Подготовка аудио")
            try:
                wav_path = ensure_wav(in_path, target_sr=44100)
            except Exception as e:
                raise SeparationError(f"Не удалось подготовить аудио: {e}") from e

            if progress_cb:
                progress_cb(0.10, "Разделение нейросетью (3-5 минут)")

            _, separated = self._sep.separate_audio_file(str(wav_path))
            base = in_path.stem

            def to_numpy(t):
                if hasattr(t, "cpu"):
                    t = t.cpu().numpy()
                return t

            # ---------- Вокал ----------
            if progress_cb:
                progress_cb(0.75, "Обработка вокала")
            vocals = to_numpy(separated.get("vocals"))
            if vocals is None:
                raise SeparationError("Модель не вернула вокал")
            vocals = self._apply_hpf(vocals, 44100, VOCALS_HPF_HZ, VOCALS_HPF_ORDER)
            vocals_path = out_dir / f"{base}_vocals.wav"
            self._save(vocals, vocals_path)

            # ---------- Инструментал ----------
            if progress_cb:
                progress_cb(0.88, "Обработка минуса")
            inst_stems = [
                to_numpy(separated[k])
                for k in ("drums", "bass", "other")
                if k in separated
            ]
            if not inst_stems:
                inst_stems = [to_numpy(v) for k, v in separated.items() if k != "vocals"]
            instrumental = np.sum(inst_stems, axis=0)
            instrumental = self._apply_notch(
                instrumental, 44100, INSTR_NOTCH_HZ, INSTR_NOTCH_Q, INSTR_NOTCH_GAIN_DB
            )
            instrumental_path = out_dir / f"{base}_instrumental.wav"
            self._save(instrumental, instrumental_path)

            if progress_cb:
                progress_cb(1.0, "Готово")

            logger.info("Разделение завершено",
                        extra={"ctx_input": str(in_path), "ctx_out_dir": str(out_dir)})
            return {"vocals": vocals_path, "instrumental": instrumental_path}

        except SeparationError:
            raise
        except Exception as e:
            logger.exception("Ошибка разделения")
            raise SeparationError(f"Ошибка разделения: {e}") from e

    # ---------- Фильтры ----------
    @staticmethod
    def _apply_hpf(arr: np.ndarray, sr: int, cutoff_hz: float, order: int) -> np.ndarray:
        """
        High-Pass Filter Баттерворта.
        arr: (channels, samples) или (samples,).
        """
        nyq = 0.5 * sr
        if cutoff_hz <= 0 or cutoff_hz >= nyq:
            return arr
        sos = butter(order, cutoff_hz / nyq, btype="highpass", output="sos")
        if arr.ndim == 1:
            return sosfilt(sos, arr).astype(np.float32)
        # arr: (channels, samples) -> фильтруем каждый канал отдельно
        out = np.empty_like(arr)
        for ch in range(arr.shape[0]):
            out[ch] = sosfilt(sos, arr[ch])
        logger.info("HPF применён", extra={"ctx_cutoff_hz": cutoff_hz, "ctx_order": order})
        return out.astype(np.float32)

    @staticmethod
    def _apply_notch(
        arr: np.ndarray, sr: int, center_hz: float, q: float, gain_db: float
    ) -> np.ndarray:
        """
        Notch (band-stop) фильтр с заданным ослаблением в дБ.
        arr: (channels, samples) или (samples,).
        """
        nyq = 0.5 * sr
        if center_hz <= 0 or center_hz >= nyq:
            return arr
        # iirnotch возвращает коэффициент для полного подавления.
        # Чтобы получить частичное ослабление, комбинируем исходный и фильтрованный сигнал.
        #
        # ⚠️ АУДИТ P0.1 (вариант B — решено оставить физику как есть):
        # коэффициенты ниже дают НЕ gain_db, а глубину 20*log10(1 - gain_lin)
        # на центре (там filtered ≈ 0). Таблица соответствий:
        #   gain_db =  -1.5 → фактически ≈ -16.0 dB
        #   gain_db =  -3.0 → фактически ≈ -10.7 dB   (текущее значение)
        #   gain_db =  -6.0 → фактически ≈  -6.0 dB
        #   gain_db = -10.0 → фактически ≈  -3.3 dB
        #   gain_db = -16.0 → фактически ≈  -1.5 dB
        # ПРАВИЛО: ближе к 0 — ГЛУБЖЕ; отрицательнее — МЯГЧЕ.
        # Меняя INSTR_NOTCH_GAIN_DB, сверяйся с таблицей, а не со знаком «dB».
        b, a = iirnotch(center_hz / nyq, q)
        sos = tf2sos(b, a)

        # Линейная амплитуда для gain_db (gain_db < 0 — ослабление)
        gain_lin = 10.0 ** (gain_db / 20.0)
        dry = 1.0 - gain_lin  # сколько оставить от исходного

        def _process(sig: np.ndarray) -> np.ndarray:
            filtered = sosfilt(sos, sig)
            # Смешиваем: filtered * gain_lin + original * (1 - gain_lin)
            return (filtered * gain_lin + sig * dry).astype(np.float32)

        # Диагностика фактической глубины (одна строка в логе на процесс)
        _report_notch_depth_once(sr, center_hz, q, gain_db)

        if arr.ndim == 1:
            out = _process(arr)
        else:
            out = np.empty_like(arr)
            for ch in range(arr.shape[0]):
                out[ch] = _process(arr[ch])

        logger.info("Notch применён",
                    extra={"ctx_center_hz": center_hz, "ctx_q": q, "ctx_gain_db": gain_db})
        return out.astype(np.float32)

    # ---------- Сохранение ----------
    def _save(self, arr: np.ndarray, path: Path, sample_rate: int = 44100) -> None:
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 2:
            arr = arr.T
        arr = np.clip(arr, -1.0, 1.0)

        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1e-6:
            arr = arr * (TARGET_PEAK / peak)
            logger.info("Нормализация",
                        extra={"ctx_file": str(path),
                               "ctx_peak_before": round(peak, 4),
                               "ctx_peak_after": round(TARGET_PEAK, 4)})

        sf.write(str(path), arr, sample_rate, subtype="PCM_16")
        logger.info("Записан WAV", extra={"ctx_file": str(path)})