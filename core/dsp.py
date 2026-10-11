"""
Общая точка DSP: целевой пик, нормализация пары стемов, эквалайзер и фильтры.

Раньше эти константы и формулы дублировались в separator.py и separator_bs.py,
причём в двух вариантах с разной физикой. Теперь физика одна.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, tf2sos

from core.config import get_config

logger = logging.getLogger(__name__)

# Целевой пик для сохранённых файлов: 0.891 ≈ -1.0 dBFS.
# Один и тот же для одиночного стема и для пары вокал+минус.
# Значение по умолчанию; фактическое берётся из config/settings.json (target_peak).
TARGET_PEAK = 0.891


def get_target_peak() -> float:
    """
    Целевой пик из настроек.

    Некорректное значение (вне 0.1..1.0) не ломает тракт: пишем предупреждение
    и работаем на значении по умолчанию.
    """
    try:
        value = float(get_config().target_peak)
    except Exception:
        logger.warning("Не удалось прочитать target_peak — беру %s", TARGET_PEAK)
        return TARGET_PEAK
    if not (0.1 <= value <= 1.0):
        logger.warning("target_peak=%s вне допустимого диапазона 0.1..1.0 — беру %s",
                       value, TARGET_PEAK)
        return TARGET_PEAK
    return value


# ============================================================
#  Нормализация
# ============================================================
def pair_peak(stems: Mapping[str, np.ndarray]) -> float:
    """Максимальный пик по всем стемам пары."""
    peaks = [float(np.max(np.abs(x))) for x in stems.values() if x.size]
    return max(peaks) if peaks else 0.0


def normalize_pair(
    stems: Mapping[str, np.ndarray],
    target_peak: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Нормализует пару стемов ОДНИМ общим коэффициентом.

    Зачем: если нормировать вокал и минус по отдельности, их взаимный уровень
    ломается — сумма перестаёт соответствовать оригиналу, а аналитика уровней
    начинает измерять нашу же обработку. Общий коэффициент сохраняет баланс.

    target_peak=None — взять значение из настроек (core.config).
    """
    target = get_target_peak() if target_peak is None else target_peak
    peak = pair_peak(stems)
    if peak < 1e-9:
        logger.warning("Нормализация пары пропущена: пик ниже -180 dBFS")
        return {k: np.asarray(v, dtype=np.float32) for k, v in stems.items()}

    g = target / peak
    logger.info(
        "Нормализация пары",
        extra={
            "ctx_peak_before": round(peak, 4),
            "ctx_target_peak": round(target, 4),
            "ctx_gain_db": round(20.0 * math.log10(g), 2),
        },
    )
    # clip здесь — страховка: после общего коэффициента пик ровно target_peak < 1.0,
    # срабатывать не должен
    return {
        k: np.clip(np.asarray(v, dtype=np.float64) * g, -1.0, 1.0).astype(np.float32)
        for k, v in stems.items()
    }


def prevent_clipping_pair(
    stems: Mapping[str, np.ndarray],
    limit: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Понижает уровень пары ОДНИМ коэффициентом, но только если пик выше limit.

    Нужна там, где нормализация пары отключена (гибрид выставляет уровень сам,
    уже на финальной паре), но записывать файл с перегрузом всё равно нельзя.
    В отличие от normalize_pair ничего не поднимает: только опускает.
    """
    peak = pair_peak(stems)
    if peak <= limit or peak < 1e-9:
        return {k: np.asarray(v, dtype=np.float32) for k, v in stems.items()}

    g = (limit * 0.99) / peak
    logger.info(
        "Предохранительное понижение уровня пары",
        extra={"ctx_peak_before": round(peak, 4),
               "ctx_gain_db": round(20.0 * math.log10(g), 2)},
    )
    return {
        k: (np.asarray(v, dtype=np.float64) * g).astype(np.float32)
        for k, v in stems.items()
    }


def normalize_pair_files(
    vocals_path: str | Path,
    instr_path: str | Path,
    target_peak: float | None = None,
) -> float | None:
    """
    Нормализует ДВА WAV-файла одним общим коэффициентом и перезаписывает их.

    Нужна там, где вокал и минус приходят из разных движков (гибрид): тогда
    нормировать надо финальную пару, а не каждую ветку по отдельности.
    Возвращает применённый коэффициент в дБ (None — если нормализация не делалась).
    """
    v_p = Path(vocals_path)
    i_p = Path(instr_path)
    try:
        vocals, sr_v = sf.read(str(v_p), dtype="float32", always_2d=True)
        instr, sr_i = sf.read(str(i_p), dtype="float32", always_2d=True)
    except Exception as e:
        logger.warning("Не удалось прочитать стемы для нормализации пары: %s", e)
        return None

    if sr_v != sr_i:
        logger.warning("Sample rate стемов не совпадает: %s vs %s — нормализация пропущена",
                       sr_v, sr_i)
        return None

    target = get_target_peak() if target_peak is None else target_peak
    before = pair_peak({"vocals": vocals.T, "instrumental": instr.T})

    normalized = normalize_pair(
        {"vocals": vocals.T, "instrumental": instr.T}, target_peak=target)

    sf.write(str(v_p), normalized["vocals"].T, sr_v, subtype="PCM_16")
    sf.write(str(i_p), normalized["instrumental"].T, sr_i, subtype="PCM_16")

    gain_db = 20.0 * math.log10(target / before) if before > 1e-9 else 0.0
    logger.info(
        "Пара нормализована общим коэффициентом",
        extra={"ctx_vocals": str(v_p), "ctx_instr": str(i_p),
               "ctx_gain_db": round(gain_db, 2)},
    )
    return gain_db


# ============================================================
#  Фильтры
# ============================================================
def apply_per_channel(arr: np.ndarray, process: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Прогоняет фильтр по каждому каналу отдельно.
    arr: (channels, samples) или (samples,).
    """
    if arr.ndim == 1:
        return process(arr).astype(np.float32)
    out = np.empty_like(arr)
    for ch in range(arr.shape[0]):
        out[ch] = process(arr[ch])
    return out.astype(np.float32)


def butter_sos(sr: int, cutoff_hz: float, order: int, btype: str) -> np.ndarray | None:
    """Коэффициенты фильтра Баттерворта. None — если частота среза вне диапазона."""
    nyq = 0.5 * sr
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        logger.warning("Фильтр %s пропущен: частота %s Гц вне диапазона (sr=%s)",
                       btype, cutoff_hz, sr)
        return None
    return butter(order, cutoff_hz / nyq, btype=btype, output="sos")


def high_pass(arr: np.ndarray, sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """High-Pass Filter Баттерворта."""
    sos = butter_sos(sr, cutoff_hz, order, "highpass")
    if sos is None:
        return arr
    out = apply_per_channel(arr, lambda ch: sosfilt(sos, ch))
    logger.info("HPF применён", extra={"ctx_cutoff_hz": cutoff_hz, "ctx_order": order})
    return out


def low_pass(arr: np.ndarray, sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Low-Pass Filter Баттерворта."""
    sos = butter_sos(sr, cutoff_hz, order, "lowpass")
    if sos is None:
        return arr
    out = apply_per_channel(arr, lambda ch: sosfilt(sos, ch))
    logger.info("LPF применён", extra={"ctx_cutoff_hz": cutoff_hz, "ctx_order": order})
    return out


def peaking_eq_sos(sr: int, f0: float, q: float, gain_db: float) -> np.ndarray:
    """
    Peaking EQ (RBJ biquad) в виде sos-коэффициентов.

    gain_db — РЕАЛЬНАЯ глубина/подъём на центральной частоте f0.
    Отрицательный gain_db режет полосу, положительный — поднимает.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b = np.array([1.0 + alpha * A, -2.0 * cos_w0, 1.0 - alpha * A])
    a = np.array([1.0 + alpha / A, -2.0 * cos_w0, 1.0 - alpha / A])
    return tf2sos(b / a[0], a / a[0])


def peaking_eq(
    arr: np.ndarray,
    sr: int,
    f0: float,
    q: float,
    gain_db: float,
) -> np.ndarray:
    """
    Применяет peaking EQ. gain_db — честная глубина в дБ.

    Заменяет прежнюю самодельную смесь filtered*gain + sig*(1-gain), у которой
    фактическая глубина была 20*log10(1 - 10**(gain_db/20)), то есть −3.0 дБ
    в константе давали провал −10.7 дБ, а −1.5 дБ — целых −16 дБ.
    """
    nyq = 0.5 * sr
    if f0 <= 0 or f0 >= nyq:
        logger.warning("Peaking EQ пропущен: частота %s Гц вне диапазона (sr=%s)", f0, sr)
        return arr
    if abs(gain_db) < 1e-6:
        return arr

    sos = peaking_eq_sos(sr, f0, q, gain_db)
    out = apply_per_channel(arr, lambda ch: sosfilt(sos, ch))
    logger.info(
        "Peaking EQ применён",
        extra={"ctx_center_hz": f0, "ctx_q": q, "ctx_gain_db": gain_db},
    )
    return out


def frequency_response_db(
    sos: np.ndarray, sr: int, freqs: Iterable[float]
) -> list[float]:
    """
    АЧХ фильтра в дБ на заданных частотах.
    Нужна для проверки: «заявленная глубина = фактическая».
    """
    out = []
    for f in freqs:
        w = 2.0 * math.pi * f / sr
        z = complex(math.cos(w), -math.sin(w))
        num = sos[:, :3] @ np.array([1.0, z, z * z])
        den = sos[:, 3:] @ np.array([1.0, z, z * z])
        h = complex(np.prod(num) / np.prod(den))
        out.append(20.0 * math.log10(max(abs(h), 1e-12)))
    return out
