"""
Разделение аудио через BS-Roformer / Mel-Band Roformer (через audio-separator CLI).

Поддерживает 2-stem и 6-stem модели.
Постобработка минуса:
  - HPF 25 Гц (2-й порядок) — убирает инфразвуковой гул.
  - Notch 2.5 кГц (Q=1.0) — убирает голосовой отпечаток.
  - LPF 19 кГц (2-й порядок) — почти не режет верх.

⚠️ Про глубину notch (аудит P0.1, вариант B — физика оставлена намеренно):
INSTR_NOTCH_GAIN_DB — параметр смешивания, а НЕ глубина. Фактическая глубина
на центре = 20*log10(1 - 10**(gain_db/20)); для текущего -1.5 это ≈ -16 dB,
то есть ГЛУБЖЕ, чем ≈ -10.7 dB у Demucs (несмотря на «мягкие» комментарии).
Инверсия: чем БЛИЖЕ gain_db к 0, тем ГЛУБЖЕ notch, и наоборот.
Фактическая глубина пишется в лог один раз за процесс (ctx_gain_db_measured).

⚠️ Побочный эффект: провал ≈ -16 dB на 2.5 кГц опускает полосу 1-4 кГц,
поэтому SPECTRAL TILT / darkening в супер-анализе может быть следствием
именно ночта, а не LPF 19 кГц.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from core.audio_utils import measure_notch_depth

logger = logging.getLogger(__name__)

# ============================================================
#  ВСЕ доступные модели (совпадают с audio-separator --list_models)
# ============================================================
AVAILABLE_BS_MODELS = [
    # Базовые BS-Roformer
    "BS-Roformer-SW.ckpt",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
    # MelBand Roformer
    "melband_roformer_inst_v1.ckpt",
    "melband_roformer_inst_v2.ckpt",
    "melband_roformer_inst_v1e.ckpt",
    "melband_roformer_instvoc_duality_v1.ckpt",
    "melband_roformer_instvox_duality_v2.ckpt",
    "mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    # Becruily
    "mel_band_roformer_karaoke_becruily.ckpt",
    "mel_band_roformer_vocals_becruily.ckpt",
    "mel_band_roformer_instrumental_becruily.ckpt",
    "bs_roformer_karaoke_frazer_becruily.ckpt",
    # Karaoke
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "mel_band_roformer_karaoke_gabox.ckpt",
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "bs_roformer_karaoke_anvuew.ckpt",
    # Gabox
    "mel_band_roformer_vocals_gabox.ckpt",
    "mel_band_roformer_instrumental_gabox.ckpt",
    "mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt",
    "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    "mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
    "mel_band_roformer_instrumental_fullness_v1_gabox.ckpt",
    "mel_band_roformer_instrumental_fullness_v2_gabox.ckpt",
    "mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
    # Big Beta
    "melband_roformer_big_beta4.ckpt",
    "melband_roformer_big_beta5e.ckpt",
    # Unwa
    "mel_band_roformer_kim_ft_unwa.ckpt",
    "mel_band_roformer_kim_ft2_unwa.ckpt",
    "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
]

# ============================================================
#  Постобработка минуса
# ============================================================
INSTR_HPF_HZ = 25.0          # было 30 — ниже, ещё мягче
INSTR_HPF_ORDER = 2

# ⚠️ INSTR_NOTCH_GAIN_DB — параметр смешивания, НЕ глубина (аудит P0.1, вариант B).
# Фактическая глубина на 2.5 кГц = 20*log10(1 - 10**(gain_db/20)):
#   -1.5  → ≈ -16.0 dB   (текущее значение; глубже, чем у Demucs)
#   -3.0  → ≈ -10.7 dB   (значение в separator.py)
#   -6.0  → ≈  -6.0 dB   (мягче)
#   -10.0 → ≈  -3.3 dB   (ещё мягче)
# ПРАВИЛО: ближе к 0 — ГЛУБЖЕ; отрицательнее — МЯГЧЕ.
INSTR_NOTCH_HZ = 2500.0
INSTR_NOTCH_Q = 1.0          # было 2.0 — шире (мягче по полосе)
INSTR_NOTCH_GAIN_DB = -1.5   # параметр смешивания: фактическая глубина ≈ -16 dB

INSTR_LPF_HZ = 19000.0       # было 16000 — почти не режет
INSTR_LPF_ORDER = 2          # было 4 — мягче

TARGET_PEAK = 10.0 ** (-0.3 / 20.0)  # ≈ 0.9660


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


class BSSeparationError(Exception):
    """Ошибка разделения через BS-Roformer."""


class BSAudioSeparator:
    def __init__(self, model_name: str = "BS-Roformer-SW.ckpt") -> None:
        if model_name not in AVAILABLE_BS_MODELS:
            raise BSSeparationError(f"Неизвестная BS-модель: {model_name}")
        self.model_name = model_name
        logger.info("BSAudioSeparator init", extra={"ctx_model": model_name})

    # ---------- публичное API ----------
    def separate(self, input_path, output_dir, progress_cb=None) -> dict[str, Path]:
        in_path = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise BSSeparationError(f"Файл не найден: {in_path}")

        tmp_dir = out_dir / "_bs_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for old in tmp_dir.glob("*.flac"):
            try:
                old.unlink()
            except Exception:
                pass

        if progress_cb:
            progress_cb(0.05, f"BS-Roformer ({self.model_name})")

        audio_sep_exe = self._find_audio_separator_exe()
        cmd = [
            str(audio_sep_exe),
            str(in_path),
            "--model_filename", self.model_name,
            "--output_dir", str(tmp_dir),
        ]
        logger.info("Запуск audio-separator", extra={"ctx_cmd": " ".join(cmd)})

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except FileNotFoundError:
            raise BSSeparationError(
                f"audio-separator не найден: {audio_sep_exe}"
            )

        last_lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            logger.info("[audio-separator] %s", line)
            last_lines.append(line)
            if len(last_lines) > 60:
                last_lines.pop(0)

        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise BSSeparationError("audio-separator превысил таймаут (30 минут)")

        if proc.returncode != 0:
            raise BSSeparationError(
                f"audio-separator упал с кодом {proc.returncode}\n"
                + "\n".join(last_lines[-20:])
            )

        if progress_cb:
            progress_cb(0.85, "Конвертация FLAC → WAV")

        flac_files = sorted(tmp_dir.glob("*.flac"))
        if not flac_files:
            raise BSSeparationError(
                f"audio-separator не выдал FLAC-файлов в {tmp_dir}\n"
                + "\n".join(last_lines[-10:])
            )

        vocals_src = None
        instrumental_single = None
        instrumental_parts = []

        for f in flac_files:
            name_low = f.stem.lower()
            if "(vocals)" in name_low:
                vocals_src = f
            elif "(instrumental)" in name_low:
                instrumental_single = f
            else:
                instrumental_parts.append(f)

        if vocals_src is None:
            raise BSSeparationError(
                f"Не найден стем vocals. Файлы: {[f.name for f in flac_files]}"
            )

        base = in_path.stem
        vocals_wav = out_dir / f"{base}_vocals.wav"
        instr_wav = out_dir / f"{base}_instrumental.wav"

        try:
            # Вокал — конвертим
            self._flac_to_wav(vocals_src, vocals_wav)

            # Инструментал — конвертим или собираем из частей
            if instrumental_single is not None:
                self._flac_to_wav(instrumental_single, instr_wav)
            elif instrumental_parts:
                self._sum_flacs_to_wav(instrumental_parts, instr_wav)
            else:
                raise BSSeparationError("Не найдены стемы для инструментала")

            # Постобработка минуса
            if progress_cb:
                progress_cb(0.92, "Постобработка минуса")
            self._postprocess_instrumental(instr_wav)

            # Нормализация пиков
            self._normalize_wav(vocals_wav)
            self._normalize_wav(instr_wav)

        except BSSeparationError:
            raise
        except Exception as e:
            raise BSSeparationError(f"Ошибка конвертации: {e}") from e

        if progress_cb:
            progress_cb(1.0, "Готово")

        logger.info(
            "BS-Roformer разделение завершено",
            extra={"ctx_vocals": str(vocals_wav), "ctx_instr": str(instr_wav)},
        )
        return {"vocals": vocals_wav, "instrumental": instr_wav}

    # ---------- поиск audio-separator.exe ----------
    @staticmethod
    def _find_audio_separator_exe() -> Path:
        candidates = []
        py_dir = Path(sys.executable).parent
        candidates.append(py_dir / "audio-separator.exe")
        candidates.append(py_dir / "audio-separator")

        from shutil import which
        path_exe = which("audio-separator")
        if path_exe:
            candidates.append(Path(path_exe))

        for c in candidates:
            if c.exists():
                logger.info("Найден audio-separator", extra={"ctx_path": str(c)})
                return c

        raise BSSeparationError(
            f"Не найден audio-separator.exe в {py_dir}\n"
            "Установи: pip install audio-separator[gpu]"
        )

    # ---------- конвертация ----------
    @staticmethod
    def _flac_to_wav(src: Path, dst: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise BSSeparationError(
                f"ffmpeg не смог сконвертировать {src.name}: {result.stderr[-500:]}"
            )

    @staticmethod
    def _sum_flacs_to_wav(sources: list[Path], dst: Path) -> None:
        if not sources:
            raise BSSeparationError("Нет файлов для суммирования")

        cmd = ["ffmpeg", "-y", "-v", "error"]
        for src in sources:
            cmd += ["-i", str(src)]
        n = len(sources)
        cmd += [
            "-filter_complex", f"amix=inputs={n}:normalize=0",
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise BSSeparationError(
                f"ffmpeg amix не смог собрать инструментал: {result.stderr[-500:]}"
            )

    # ---------- ПОСТОБРАБОТКА МИНУСА ----------
    @staticmethod
    def _postprocess_instrumental(path: Path) -> None:
        """
        Постобработка минуса:
          1. HPF 25 Гц (2-й порядок) — убирает инфразвук.
          2. Notch 2.5 кГц (Q=1.0) — убирает VPI. ⚠️ Фактическая глубина
             ≈ -16 dB (параметр -1.5 dB — это НЕ глубина, см. docstring модуля).
          3. LPF 19 кГц (2-й порядок) — почти не режет верх.
        """
        try:
            from scipy.signal import butter, sosfilt, iirnotch, tf2sos
        except ImportError:
            logger.warning("scipy не установлен — пропускаю постобработку минуса")
            return

        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            logger.warning("Не удалось прочитать WAV: %s", e)
            return

        nyq = 0.5 * sr

        # ---------- 1) HPF 25 Гц ----------
        if INSTR_HPF_HZ < nyq:
            sos_hp = butter(INSTR_HPF_ORDER, INSTR_HPF_HZ / nyq,
                            btype="highpass", output="sos")
            if data.ndim == 1:
                data = sosfilt(sos_hp, data).astype(np.float32)
            else:
                for ch in range(data.shape[1]):
                    data[:, ch] = sosfilt(sos_hp, data[:, ch]).astype(np.float32)
            logger.info("HPF применён",
                        extra={"ctx_cutoff_hz": INSTR_HPF_HZ,
                               "ctx_order": INSTR_HPF_ORDER})

        # ---------- 2) Notch 2.5 кГц ----------
        if INSTR_NOTCH_HZ < nyq:
            b, a = iirnotch(INSTR_NOTCH_HZ / nyq, INSTR_NOTCH_Q)
            sos_notch = tf2sos(b, a)

            # ⚠️ АУДИТ P0.1 (вариант B — физика оставлена намеренно):
            # на центре filtered ≈ 0, поэтому итог ≈ sig * (1 - gain_lin),
            # то есть фактическая глубина 20*log10(1 - gain_lin) ≈ -16 dB,
            # а НЕ -1.5 dB. ПРАВИЛО: ближе к 0 — ГЛУБЖЕ, отрицательнее — МЯГЧЕ
            # (таблица соответствий — в audio_utils.measure_notch_depth).
            gain_lin = 10.0 ** (INSTR_NOTCH_GAIN_DB / 20.0)
            dry = 1.0 - gain_lin

            def apply_notch(sig):
                filtered = sosfilt(sos_notch, sig)
                return (filtered * gain_lin + sig * dry).astype(np.float32)

            if data.ndim == 1:
                data = apply_notch(data)
            else:
                for ch in range(data.shape[1]):
                    data[:, ch] = apply_notch(data[:, ch])

            # Диагностика фактической глубины (одна строка в логе на процесс)
            _report_notch_depth_once(sr, INSTR_NOTCH_HZ, INSTR_NOTCH_Q,
                                     INSTR_NOTCH_GAIN_DB)

            logger.info("Notch применён",
                        extra={"ctx_center_hz": INSTR_NOTCH_HZ,
                               "ctx_q": INSTR_NOTCH_Q,
                               "ctx_gain_db": INSTR_NOTCH_GAIN_DB})

        # ---------- 3) LPF 19 кГц ----------
        if INSTR_LPF_HZ < nyq:
            sos_lp = butter(INSTR_LPF_ORDER, INSTR_LPF_HZ / nyq,
                            btype="lowpass", output="sos")
            if data.ndim == 1:
                data = sosfilt(sos_lp, data).astype(np.float32)
            else:
                for ch in range(data.shape[1]):
                    data[:, ch] = sosfilt(sos_lp, data[:, ch]).astype(np.float32)
            logger.info("LPF применён",
                        extra={"ctx_cutoff_hz": INSTR_LPF_HZ,
                               "ctx_order": INSTR_LPF_ORDER})

        # Обрезаем пики на всякий случай
        data = np.clip(data, -1.0, 1.0)

        sf.write(str(path), data, sr, subtype="PCM_16")
        logger.info("Постобработка минуса завершена",
                    extra={"ctx_file": str(path)})

    # ---------- НОРМАЛИЗАЦИЯ ----------
    @staticmethod
    def _normalize_wav(path: Path) -> None:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            logger.warning("Не удалось прочитать WAV для нормализации: %s", e)
            return

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak < 1e-6:
            return

        data = data * (TARGET_PEAK / peak)
        data = np.clip(data, -1.0, 1.0)

        sf.write(str(path), data, sr, subtype="PCM_16")
        logger.info(
            "Нормализация WAV",
            extra={
                "ctx_file": str(path),
                "ctx_peak_before": round(peak, 4),
                "ctx_peak_after": round(TARGET_PEAK, 4),
            },
        )