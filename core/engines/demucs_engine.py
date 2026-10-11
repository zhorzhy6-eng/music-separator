"""
Demucs — движок разделения (переехал из core/separator.py без изменения логики).
Использует официальный API demucs.api.Separator.

Постобработка:
  - Вокал: High-Pass Filter 100 Гц (2-й порядок Баттерворта).
  - Минус: peaking EQ 2.5 кГц — ОПЦИОНАЛЬНО и по умолчанию ВЫКЛЮЧЕН
    (ключ notch_enabled в config/settings.json).
  - Нормализация: ОДИН общий коэффициент на пару вокал+минус (core.dsp.normalize_pair).

Прочие общие вещи (целевой пик, фильтры, нормализация пары) живут в core/dsp.py.
Список доступных моделей — в реестре core/engines/__init__.py (ENGINES).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from core.audio_utils import ensure_wav, get_output_sample_rate, resample_audio
from core.config import get_config
from core.dsp import (
    get_target_peak,
    high_pass,
    normalize_pair,
    pair_peak,
    peaking_eq,
    prevent_clipping_pair,
)
from core.engines.base import EngineCancelled, ProgressCb, Separator
from core.stems import StemSet

logger = logging.getLogger(__name__)

# Частота, на которой работает модель Demucs
MODEL_SR = 44100

# ---------- Параметры постобработки ----------
VOCALS_HPF_HZ = 100.0          # срез HPF для вокала
VOCALS_HPF_ORDER = 2           # порядок фильтра Баттерворта

# Notch (peaking EQ) на минусе. По умолчанию ВЫКЛЮЧЕН (notch_enabled в
# config/settings.json): даже мягкий провал на 2.5 кГц слышен в самой разборчивой
# зоне (гитары, струнные, атака малого).
INSTR_NOTCH_HZ = 2500.0        # центр фильтра
INSTR_NOTCH_Q = 1.5            # добротность
# Честная глубина в дБ на центральной частоте (раньше параметр означал
# коэффициент смешивания: -3.0 давало фактически -10.7 dB).
INSTR_NOTCH_GAIN_DB_REAL = -3.0


class SeparationError(Exception):
    """Ошибка разделения."""


class DemucsEngine:
    """
    Движок Demucs. Реализует договор Separator (см. core/engines/base.py).

    Старое имя класса — AudioSeparator (оставлено ниже как совместимость).
    """

    name = "Demucs"

    def __init__(self, model_name: str = "htdemucs",
                 normalize_pair_output: bool = True) -> None:
        # DECISION: модель больше не сверяется со списком внутри движка — за это
        # отвечает реестр (в реестре только те модели, что мы показываем в UI).
        # Проверка нужна лишь на пустое значение: без модели Demucs не стартует.
        if not model_name:
            raise SeparationError("Не указана модель Demucs")
        self.model_name = model_name
        # False — уровень пары выставит тот, кто собирает финальный результат
        # (гибрид нормирует уже готовую пару вокал+минус). Иначе каждая ветка
        # тянула бы уровень на себя, и взаимный баланс ломался бы.
        self.normalize_pair_output = normalize_pair_output
        self._sep = None
        # Demucs считает внутри нашего же процесса, поэтому «отмена» здесь —
        # это флаг: считаться перестаём между шагами (см. cancel).
        self._cancelled = False
        logger.info("AudioSeparator init", extra={"ctx_model": model_name})

    # ---------- отмена ----------
    def cancel(self) -> bool:
        """
        Просит остановиться. False — прерывать нечего (ещё не начали/уже готово).

        Полностью прервать вычисление внутри нейросети нельзя: считающий вызов
        не умеет останавливаться на середине. Поэтому останавливаемся между
        шагами — на фильтрах, сохранении и загрузке модели.
        """
        self._cancelled = True
        return self._sep is not None

    def _throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise EngineCancelled("Разделение отменено")

    def _ensure_model(self):
        if self._sep is not None:
            return
        cfg = get_config()
        device = self._resolve_device(cfg.demucs_device)
        try:
            from demucs.api import Separator
            logger.info("Загружаю модель Demucs: %s", self.model_name,
                        extra={"ctx_device": device,
                               "ctx_shifts": cfg.demucs_shifts,
                               "ctx_overlap": cfg.demucs_overlap})
            self._sep = Separator(
                model=self.model_name,
                device=device,
                shifts=cfg.demucs_shifts,
                overlap=cfg.demucs_overlap,
                progress=True,
            )
            logger.info("Модель Demucs загружена")
        except Exception as e:
            logger.exception("Не удалось загрузить модель %s", self.model_name)
            raise SeparationError(f"Не удалось загрузить модель: {e}") from e

    @staticmethod
    def _resolve_device(device: str | None) -> str | None:
        """
        Устройство из настроек. Если просили CUDA, а её нет — работаем на CPU
        и говорим об этом: иначе Demucs падает с невнятной ошибкой.
        """
        dev = (device or "").strip()
        if dev.lower().startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return dev
            except Exception:
                pass
            logger.warning(
                "В настройках demucs_device=%r, но CUDA недоступна — работаю на CPU", dev)
            return "cpu"
        return dev or None

    def separate(self, input_path, output_dir, progress_cb=None) -> StemSet:
        self._cancelled = False        # новая задача — прошлая отмена не считается
        self._throw_if_cancelled()
        self._ensure_model()
        self._throw_if_cancelled()
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
            self._throw_if_cancelled()

            if progress_cb:
                progress_cb(0.10, "Разделение нейросетью (3-5 минут)")

            _, separated = self._sep.separate_audio_file(str(wav_path))
            base = in_path.stem
            # Считали долго — проверяем, не отменили ли, пока считали
            self._throw_if_cancelled()

            def to_numpy(t):
                if hasattr(t, "cpu"):
                    t = t.cpu().numpy()
                return t

            # ---------- Вокал: только фильтр, без нормализации ----------
            if progress_cb:
                progress_cb(0.75, "Обработка вокала")
            vocals = to_numpy(separated.get("vocals"))
            if vocals is None:
                raise SeparationError("Модель не вернула вокал")
            vocals = high_pass(vocals, 44100, VOCALS_HPF_HZ, VOCALS_HPF_ORDER)

            # ---------- Минус: только фильтр, без нормализации ----------
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
            if get_config().notch_enabled:
                instrumental = peaking_eq(
                    instrumental, 44100,
                    INSTR_NOTCH_HZ, INSTR_NOTCH_Q, INSTR_NOTCH_GAIN_DB_REAL,
                )
            else:
                logger.info("Notch выключен (notch_enabled=false в settings.json) — "
                            "минус не фильтруется")

            # ---------- ОДИН общий коэффициент на пару ----------
            if self.normalize_pair_output:
                norm = normalize_pair({"vocals": vocals, "instrumental": instrumental})
                vocals = norm["vocals"]
                instrumental = norm["instrumental"]
            else:
                # Уровень выставит сборщик финальной пары (гибрид).
                # Здесь только не даём сигналу уйти выше 0 dBFS.
                safe = prevent_clipping_pair(
                    {"vocals": vocals, "instrumental": instrumental})
                vocals = safe["vocals"]
                instrumental = safe["instrumental"]

            vocals_path = out_dir / f"{base}_vocals.wav"
            instr_path = out_dir / f"{base}_instrumental.wav"
            self._save(vocals, vocals_path)
            self._save(instrumental, instr_path)

            if progress_cb:
                progress_cb(1.0, "Готово")

            logger.info("Разделение завершено",
                        extra={"ctx_input": str(in_path), "ctx_out_dir": str(out_dir)})
            return StemSet(vocals=vocals_path, instrumental=instr_path)

        except SeparationError:
            raise
        except Exception as e:
            logger.exception("Ошибка разделения")
            raise SeparationError(f"Ошибка разделения: {e}") from e

    # ---------- Сохранение ----------
    def _save(self, arr: np.ndarray, path: Path, sample_rate: int = MODEL_SR) -> None:
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 2:
            arr = arr.T

        # Частота вывода — из настроек (output_sample_rate). Модель всегда отдаёт
        # 44.1 кГц, поэтому при другом значении честно пересчитываем.
        target_sr = get_output_sample_rate()
        if target_sr != sample_rate:
            arr = resample_audio(arr, sample_rate, target_sr)
            sample_rate = target_sr

        # Страховка от переполнения. Срабатывать не должна: нормализация пары
        # уже привела пик к target_peak (< 1.0). Если сработала — сигнал битый,
        # и это надо видеть в логе.
        peak = pair_peak({"arr": arr})
        if peak > 1.0:
            clipped = int(np.sum(np.abs(arr) > 1.0))
            logger.warning(
                "Clip-страховка сработала: пик %.4f, обрезано сэмплов %s",
                peak, clipped,
                extra={"ctx_file": str(path), "ctx_peak": round(peak, 4),
                       "ctx_clipped_samples": clipped},
            )
        arr = np.clip(arr, -1.0, 1.0)

        sf.write(str(path), arr, sample_rate, subtype="PCM_16")
        logger.info(
            "Записан WAV",
            extra={"ctx_file": str(path), "ctx_peak": round(peak, 4),
                   "ctx_sample_rate": sample_rate,
                   "ctx_target_peak": round(get_target_peak(), 4)},
        )


# ---------- Совместимость со старыми именами ----------
# Раньше модуль назывался core/separator.py, а класс — AudioSeparator.
# Код, написанный до переезда, продолжает работать через эти имена.
AudioSeparator = DemucsEngine

__all__ = [
    "DemucsEngine", "AudioSeparator", "SeparationError",
    "MODEL_SR", "VOCALS_HPF_HZ", "VOCALS_HPF_ORDER",
    "INSTR_NOTCH_HZ", "INSTR_NOTCH_Q", "INSTR_NOTCH_GAIN_DB_REAL",
]
