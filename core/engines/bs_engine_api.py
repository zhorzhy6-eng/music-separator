"""
BS-Roformer / Mel-Band Roformer через Python-API пакета audio_separator
(без запуска audio-separator.exe отдельным процессом).

Зачем ещё одна реализация рядом с core/engines/bs_engine.py:
  * модель грузится ОДИН раз и живёт в памяти, а не перезагружается на каждый
    трек (у subprocess-версии это ~10 с на трек);
  * прогресс приходит из самого инференса, а не вылавливается из stdout;
  * логи структурные, без полосок tqdm;
  * есть контроль точности (fp16) и размера батча.

Что специально совпадает с subprocess-версией, чтобы результат не «поехал»:
  * DSP-шаги и их порядок — чужим кодом не дублируются, а берутся из bs_engine
    (BSEngine._postprocess_instrumental) и core/dsp (normalize_pair_files);
  * встроенная нормализация библиотеки отключена (normalization_threshold=1.0) —
    иначе вокал и минус перестают быть комплементарными;
  * на выходе те же файлы: <трек>_vocals.wav и <трек>_instrumental.wav, PCM_16.

Чего не умеет: мгновенной отмены, как убийство процесса. Инференс прерывается
на ближайшем шаге батча (см. cancel) — на слух это те же доли секунды.

Формат входа (баг с MP3, исправлено):
  * audio-separator берёт ФОРМАТ ЗАПИСИ из формата входа: он читает subtype
    входного файла через soundfile и подставляет его же при записи стемов.
    У MP3 subtype = MPEG_LAYER_III, а стемы пишутся в WAV — libsndfile отвечает
    «Supported file format but unsupported encoding», и разделение падало;
  * поэтому ЛЮБОЙ вход кроме WAV (MP3, FLAC, M4A, OGG) сначала перекодируется
    в WAV в рабочей папке (см. _prepare_input_wav), и уже этот WAV уходит в
    библиотеку. Формат на выходе задан явно: output_format="WAV";
  * FLAC случайно «везло»: у него subtype PCM_16, и libsndfile принимал его для
    WAV. Это совпадение, а не гарантия — поэтому перекодируем всё, что не WAV.

Ключевые факты про API audio-separator 0.47 (проверено по исходникам пакета):
  * Separator(output_dir=..., output_format=..., normalization_threshold=...,
    sample_rate=..., model_file_dir=..., use_native_fp16=..., use_autocast=...,
    mdxc_params={...}); use_autocast и use_native_fp16 — ВЗАИМОИСКЛЮЧАЮЩИЕ
    режимы (одновременно нельзя: ValueError);
  * load_model(model_filename=...) — повторный вызов с тем же именем
    переиспользует уже загруженный экземпляр;
  * separate(path, custom_output_names=...) → список путей к файлам стемов;
    файлы пишутся в output_dir, который «запоминается» на момент load_model;
  * колбэка прогресса в API нет: полоски рисует tqdm внутри
    architectures/mdxc_separator.py — мы подменяем там tqdm своим (см. _tqdm_hook).
"""
from __future__ import annotations

import gc
import logging
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from core.audio_utils import get_output_sample_rate, resample_audio
from core.config import get_config
from core.dsp import normalize_pair_files
from core.engines.base import EngineCancelled, ProgressCb
from core.engines.bs_engine import (
    CLIP_SAFETY_PEAK,
    SUM_LIMITER_LIMIT,
    BSEngine,
    BSSeparationError,
)
from core.paths import MODELS_DIR
from core.stems import StemSet
from core.utils import cleanup_temp_dirs

logger = logging.getLogger(__name__)

# Модели BS-Roformer обучены на 44.1 кГц. Считать на другой частоте нельзя:
# модель «слышит» другой звук и разделяет хуже. Поэтому инференс всегда на 44100,
# а к частоте из настроек (output_sample_rate) приводим результат — как это делал
# ffmpeg в subprocess-версии.
MODEL_SAMPLE_RATE = 44100

# Рабочая папка внутри каталога вывода: сюда библиотека пишет свои WAV,
# отсюда мы забираем их на постобработку и нормализацию.
SCRATCH_DIR_NAME = "_bs_api_tmp"

# Формат, в котором библиотека обязана писать стемы. Задан явно и один раз:
# см. раздел «Формат входа» в шапке модуля — из-за неявного формата и падал MP3.
OUTPUT_FORMAT = "WAV"

# Форматы, которые библиотека принимает без перекодирования. Только WAV:
# у него subtype начинается с PCM_/FLOAT_ и годится для записи WAV-стемов.
# (PCM_24 тоже WAV — расширение важнее подтипа, лишний проход FFmpeg не нужен.)
NATIVE_INPUT_EXTENSIONS = frozenset({".wav"})

# Сколько секунд ждём FFmpeg на перекодирование входа. Минута аудио на слабой
# машине — секунды, так что 15 минут с запасом покрывают и часовой микс.
FFMPEG_TIMEOUT_SEC = 900

# Сколько загруженных моделей держим в памяти. Одна: каждая весит 0.6–1.7 ГБ
# и на 6 ГБ видеопамяти две модели рядом — верный путь к OOM.
MODEL_CACHE_SIZE = 1

# Порог для сообщения о перегрузке суммы частей (как у лимитера в bs_engine)
SUM_PARTS_PEAK_LIMIT = SUM_LIMITER_LIMIT

# Как называть работу в строке состояния. Строка СТАБИЛЬНАЯ — без имени модели и
# без процентов: по ней ui/app_window.py подбирает понятную фразу («работает
# модель BS-Roformer»), а имя модели и так видно в подписи карточки и в логе.
# Раньше сюда подставлялось имя файла модели — в окне это выглядело как
# «Разделяем: bS-Roformer (API): model_bs_roformer_ep_317_sdr_12.9755.ckpt — 38%… 38%».
STAGE_NAME = "Разделение нейросетью (BS-Roformer)"


class BSApiError(BSSeparationError):
    """Ошибка разделения через Python-API audio-separator.

    Наследник BSSeparationError: код, который ловил ошибки BS-движка раньше,
    продолжает их ловить и с новой реализацией.
    """


# ============================================================
#  Кэш загруженных моделей (главный выигрыш по времени)
# ============================================================
# Ключ: (каталог моделей, имя модели, fp16). Значение: готовый Separator.
_model_cache: dict[tuple[str, str, bool], object] = {}
_cache_lock = threading.Lock()

# Инференс одной модели за раз: GPU-очередь воркера и так пускает задачи по одной,
# но движок может дёрнуть и CLI, и тест — пусть будет явная защита.
_run_lock = threading.Lock()


def cache_info() -> dict:
    """Что сейчас лежит в кэше моделей (для логов, CLI и тестов)."""
    with _cache_lock:
        return {
            "size": len(_model_cache),
            "models": [{"model": k[1], "fp16": k[2]} for k in _model_cache],
        }


def clear_model_cache(free_gpu: bool = True) -> None:
    """
    Выгружает загруженные модели и освобождает память.

    Нужна как «аварийный выход»: если видеопамяти не хватает перед запуском
    Demucs, модель BS можно выгрузить, не перезапуская приложение.
    """
    with _cache_lock:
        dropped = list(_model_cache)
        _model_cache.clear()
    if dropped:
        logger.info("Кэш BS-моделей очищен", extra={"ctx_models": [k[1] for k in dropped]})
    gc.collect()
    if free_gpu:
        try:                                  # torch есть не всегда (например, в тестах)
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(
                    "Память GPU после очистки кэша",
                    extra={"ctx_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2)},
                )
        except Exception:
            logger.debug("Не удалось освободить кэш CUDA", exc_info=True)


# ============================================================
#  Разбор имён стемов
# ============================================================
_PARENS_RE = re.compile(r"\(([^)]+)\)")
_VOCAL_TOKENS = ("vocal", "vox", "karaoke", "voice", "acapella")
_INSTR_TOKENS = ("instrument", "inst", "accompaniment", "no_vocal", "other", "music")


def classify_stem(filename: str) -> str:
    """
    Что это за стем: "vocals", "instrumental" или "part" (неизвестная часть).

    Понимает два вида имён:
      * наши имена из custom_output_names — "<база>_vocals" / "<база>_instrumental"
        (проверяем по хвосту: в имени трека может встретиться слово vocals);
      * имена по умолчанию у библиотеки — "<база>_(Vocals)_<модель>"
        (берём то, что стоит в скобках, иначе имя модели сбило бы разбор).
    """
    stem = Path(filename).stem.lower()

    if stem.endswith(("_vocals", "_vocal", "_vox")):
        return "vocals"
    if stem.endswith(("_instrumental", "_inst", "_accompaniment")):
        return "instrumental"

    match = _PARENS_RE.search(stem)
    token = match.group(1) if match else stem
    if any(t in token for t in _VOCAL_TOKENS):
        return "vocals"
    if any(t in token for t in _INSTR_TOKENS):
        return "instrumental"
    return "part"


# ============================================================
#  Подготовка входа: любой формат → WAV
# ============================================================
def _convert_to_wav(src: Path, dst: Path) -> None:
    """
    Перекодирует файл в WAV (PCM 16 бит, стерео, 44.1 кГц) через FFmpeg.

    Частота 44100 выбрана не случайно: BS-модели обучены на ней, и библиотека
    всё равно приводит вход к ней. Если сделать это здесь, лишнего ресемплинга
    внутри не будет.
    """
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-ar", str(MODEL_SAMPLE_RATE),
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=FFMPEG_TIMEOUT_SEC)
    except FileNotFoundError as e:
        raise BSApiError(
            "Для разделения не-WAV файлов нужен FFmpeg, а его нет в системе.\n"
            "Установите: winget install Gyan.FFmpeg — и перезапустите приложение."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise BSApiError(
            f"FFmpeg не успел перекодировать {src.name} за {FFMPEG_TIMEOUT_SEC} с"
        ) from e

    if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        message = (result.stderr or "").strip()[:500]
        raise BSApiError(
            f"Не удалось подготовить {src.name} к разделению.\n"
            f"FFmpeg сообщил: {message or 'файл не читается'}"
        )


# ============================================================
#  Прогресс: подмена tqdm внутри audio_separator
# ============================================================
# В API нет колбэка прогресса, но инференс идёт циклом `for batch in tqdm(batches)`
# внутри architectures/mdxc_separator.py. Мы подставляем туда свой объект: он
# ведёт себя как tqdm для библиотеки и сообщает долю работы нам.
_hook_state = threading.local()


class _ProgressTqdm:
    """Замена tqdm: тот же итератор, но прогресс уходит в наш колбэк."""

    def __init__(self, iterable=None, total=None, *args, **kwargs):
        self._iterable = () if iterable is None else iterable
        try:
            self._total = total if total is not None else len(self._iterable)
        except TypeError:                     # генератор без len()
            self._total = None
        self._done = 0
        self._hook = getattr(_hook_state, "fn", None)

    def __iter__(self):
        it = iter(self._iterable)
        while True:
            try:
                item = next(it)
            except StopIteration:
                return
            yield item
            self._done += 1
            if self._hook is not None:
                self._hook(self._done, self._total)

    # Дальше — самая скромная совместимость с tqdm: библиотека эти методы
    # не зовёт, но если однажды позовёт, ничего не сломается.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n: int = 1) -> None:
        self._done += n

    def close(self) -> None:
        pass

    def set_description(self, *args, **kwargs) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def write(*args, **kwargs) -> None:
        pass


@contextmanager
def _tqdm_hook(on_step):
    """
    На время разделения подменяет tqdm в модуле mdxc_separator.

    Если модуль переименуют в новой версии пакета — просто не подменяем
    (прогресс будет ступенчатым, но разделение не сломается).
    """
    module = None
    prev = None
    try:
        from audio_separator.separator.architectures import mdxc_separator as module

        prev = getattr(module, "tqdm", None)
        if prev is not None:
            module.tqdm = _ProgressTqdm
            _hook_state.fn = on_step
    except Exception:
        logger.debug("Подмена tqdm не удалась — прогресс будет ступенчатым", exc_info=True)
        module = None

    try:
        yield
    finally:
        _hook_state.fn = None
        if module is not None and prev is not None:
            module.tqdm = prev


# ============================================================
#  Движок
# ============================================================
class BSEngineAPI:
    """
    BS-Roformer через Python-API. Реализует договор Separator
    (см. core/engines/base.py).

    Дополнительно к договору:
      * use_fp16 — половинная точность на GPU;
      * batch_size — размер батча (0 = как в конфиге модели);
      * keep_loaded — держать модель в памяти между треками.
    """

    # Диапазон прогресса, который отдаём под инференс (остальное — загрузка
    # модели, постобработка и нормализация).
    _INFER_LO = 0.10
    _INFER_HI = 0.85

    def __init__(
        self,
        model_name: str,
        model_dir: Path | None = None,
        use_fp16: bool = False,
        batch_size: int = 0,
        keep_loaded: bool = True,
        normalize_pair_output: bool = True,
    ) -> None:
        if not model_name:
            raise BSApiError("Не указана модель BS-Roformer")
        self.model_name = model_name
        self.name = f"BS-Roformer (API): {model_name}"
        self._model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self._use_fp16 = bool(use_fp16)
        self._batch_size = max(0, int(batch_size or 0))
        self._keep_loaded = bool(keep_loaded)
        # False — уровень выставит сборщик финальной пары (гибрид)
        self.normalize_pair_output = normalize_pair_output

        self._separator = None
        self._cancelled = False
        self._running = False
        self._progress_cb: ProgressCb | None = None
        self._last_progress = 0.0
        self._counter_lock = threading.Lock()
        logger.info(
            "BSEngineAPI init",
            extra={"ctx_model": model_name, "ctx_fp16": self._use_fp16,
                   "ctx_batch_size": self._batch_size,
                   "ctx_keep_loaded": self._keep_loaded},
        )

    # ---------- отмена ----------
    def cancel(self) -> bool:
        """
        Просит разделение остановиться. True — работа шла и будет прервана.

        Инференс Python-API нельзя убить мгновенно, как процесс: сигнал
        проверяется в начале каждого батча (см. _on_step) и между шагами.
        Для трека длиной 4 минуты это обычно меньше секунды ожидания.
        """
        with self._counter_lock:
            self._cancelled = True
            running = self._running
        if running:
            logger.warning("Отмена: прерываю BS-API на ближайшем шаге (%s)", self.model_name)
        return running

    def is_running(self) -> bool:
        """Идёт ли сейчас разделение."""
        with self._counter_lock:
            return self._running

    def _throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise EngineCancelled(f"Разделение отменено: {self.model_name}")

    def _report(self, progress: float, stage: str) -> None:
        """Прогресс наружу. Ошибка колбэка не должна ронять разделение."""
        self._last_progress = progress
        if self._progress_cb is None:
            return
        try:
            self._progress_cb(progress, stage)
        except Exception:
            logger.exception("Колбэк прогресса упал — продолжаю разделение")

    def _on_step(self, done: int, total: int | None) -> None:
        """
        Шаг инференса (вызывается из _ProgressTqdm после каждого батча).

        Здесь же живёт отмена: поднять исключение внутри цикла библиотеки —
        единственный способ прервать счёт, не дожидаясь конца трека.
        """
        self._throw_if_cancelled()
        if self._progress_cb is None or not total:
            return
        frac = min(done / total, 1.0)
        progress = self._INFER_LO + (self._INFER_HI - self._INFER_LO) * frac
        if progress - self._last_progress < 0.01:
            return                            # не чаще 1 % — не забиваем очередь UI
        self._report(progress, STAGE_NAME)

    # ---------- загрузка модели ----------
    def _ensure_model(self, scratch_dir: Path):
        """
        Возвращает готовый Separator. Модель грузится один раз за сессию.

        Кэш живёт на уровне модуля, потому что воркер собирает движок заново
        на каждую задачу (core/worker.py::_do_separate) — без общего кэша
        «один раз за сессию» не получилось бы.
        """
        key = (str(self._model_dir), self.model_name, self._use_fp16)
        if self._keep_loaded:
            with _cache_lock:
                cached = _model_cache.get(key)
            if cached is not None:
                logger.info("BS-модель уже в памяти: %s", self.model_name,
                            extra={"ctx_model": self.model_name})
                self._separator = cached
                return cached

        self._warn_if_model_missing()

        try:
            from audio_separator.separator import Separator as AudioSeparator
        except ImportError as e:
            raise BSApiError(
                "Пакет audio_separator не установлен.\n"
                "Установи: pip install audio-separator[gpu]"
            ) from e

        # Логи библиотеки — в общий лог проекта (у него уже есть обработчики,
        # свой StreamHandler он не добавит: см. Logger.hasHandlers).
        logging.getLogger("audio_separator").setLevel(logging.INFO)

        mdxc_params = {
            "segment_size": 256,
            "override_model_segment_size": False,
            # 0 — взять размер батча из конфига модели (обычно 1)
            "batch_size": self._batch_size or None,
            "overlap": None,                  # None — как в конфиге модели
            "pitch_shift": 0,
        }

        logger.info(
            "Загружаю BS-модель через API: %s (fp16=%s, batch=%s)",
            self.model_name, self._use_fp16, self._batch_size or "как в модели",
            extra={"ctx_model": self.model_name},
        )
        started = time.perf_counter()
        try:
            separator = AudioSeparator(
                output_dir=str(scratch_dir),
                output_format=OUTPUT_FORMAT,
                # 1.0 — встроенная нормализация выключена: уровень выставляем сами,
                # одним коэффициентом на пару (core.dsp.normalize_pair_files).
                normalization_threshold=1.0,
                sample_rate=MODEL_SAMPLE_RATE,
                model_file_dir=str(self._model_dir),
                use_soundfile=True,           # без ffmpeg и без int16-конвертации в памяти
                # DECISION: точность задаём ОДНИМ режимом. use_autocast и
                # use_native_fp16 взаимоисключающие — вместе библиотека падает.
                use_native_fp16=self._use_fp16,
                use_autocast=False,
                mdxc_params=mdxc_params,
            )
            separator.load_model(model_filename=self.model_name)
        except Exception as e:
            logger.exception("Не удалось загрузить BS-модель через API")
            raise BSApiError(
                f"Не удалось загрузить модель {self.model_name} через API.\n"
                f"Каталог моделей: {self._model_dir}\n"
                f"Ошибка: {e}"
            ) from e

        self._pin_output_format(separator)

        load_sec = time.perf_counter() - started
        logger.info("BS-модель загружена через API за %.1f с", load_sec,
                    extra={"ctx_model": self.model_name,
                           "ctx_load_sec": round(load_sec, 1)})

        self._separator = separator
        if self._keep_loaded:
            self._store_in_cache(key, separator)
        return separator

    def _store_in_cache(self, key, separator) -> None:
        """Кладёт модель в кэш и вытесняет лишние (память дороже времени)."""
        with _cache_lock:
            evicted = [k for k in _model_cache if k != key]
            for k in evicted:
                _model_cache.pop(k, None)
            _model_cache[key] = separator
        if evicted:
            logger.info(
                "Из кэша выгружены прежние модели (держим не больше %d)",
                MODEL_CACHE_SIZE, extra={"ctx_models": [k[1] for k in evicted]},
            )
            # Модель выкинули из словаря, но память освободится только после
            # сборщика мусора и очистки кэша CUDA — иначе следующая модель
            # упрётся в нехватку видеопамяти.
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("Не удалось очистить кэш CUDA", exc_info=True)

    def _warn_if_model_missing(self) -> None:
        """Предупреждаем заранее: библиотека скачает модель, это долго."""
        model_path = self._model_dir / self.model_name
        if not model_path.exists():
            logger.warning(
                "Файла модели нет локально — библиотека скачает его в %s "
                "(это может занять несколько минут)",
                self._model_dir, extra={"ctx_model": self.model_name},
            )

    @staticmethod
    def _pin_output_format(separator) -> None:
        """
        Ещё раз, уже на загруженной модели, фиксирует WAV и запись через soundfile.

        Зачем поверх параметра в конструкторе: формат вывода библиотека хранит
        в ДВУХ местах — у объекта Separator (по нему строятся имена файлов:
        <имя>.<output_format>) и у загруженного экземпляра модели (по нему
        реально пишется файл). Если значения разойдутся, получится тот же баг,
        что был с MP3, только с другой стороны: имя .wav, а внутри MP3.
        """
        for holder in (separator, getattr(separator, "model_instance", None)):
            if holder is None:
                continue
            try:
                holder.output_format = OUTPUT_FORMAT
                holder.use_soundfile = True     # pydub (ffmpeg) не нужен: пишем сами
            except Exception:
                logger.debug("Не удалось зафиксировать формат вывода у %r",
                             holder, exc_info=True)

    def _prepare_input_wav(self, in_path: Path, scratch_dir: Path) -> tuple[Path, bool]:
        """
        Готовит вход для библиотеки. Возвращает (путь, создали_ли_времянку).

        WAV отдаём как есть. Всё остальное (MP3, FLAC, M4A, OGG) перекодируем
        в WAV: причина — в шапке модуля. Времянка лежит в рабочей папке и
        удаляется сразу после разделения, чтобы не занимать гигабайты.
        """
        if in_path.suffix.lower() in NATIVE_INPUT_EXTENSIONS:
            return in_path, False

        converted = scratch_dir / f"{in_path.stem}__bsapi_input.wav"
        logger.info(
            "Вход не WAV (%s) — готовлю WAV через FFmpeg: %s",
            in_path.suffix.lower() or "без расширения", converted.name,
            extra={"ctx_src": str(in_path)},
        )
        _convert_to_wav(in_path, converted)
        return converted, True

    def _bind_output_dir(self, scratch_dir: Path) -> None:
        """
        Направляет вывод библиотеки в нашу рабочую папку.

        output_dir запоминается моделью в момент load_model, поэтому у модели
        из кэша его нужно переставить руками. Если внутренности библиотеки
        изменятся, мы просто останемся на прежней папке — чтение результатов
        идёт по путям, которые вернул separate(), а не по этой папке.
        """
        for holder in (self._separator, getattr(self._separator, "model_instance", None)):
            if holder is None:
                continue
            try:
                holder.output_dir = str(scratch_dir)
            except Exception:
                logger.debug("Не удалось переставить output_dir у %r", holder, exc_info=True)

    # ---------- публичное API ----------
    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_cb: ProgressCb | None = None,
    ) -> StemSet:
        self._cancelled = False
        self._progress_cb = progress_cb
        self._last_progress = 0.0

        in_path = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_path.exists():
            raise BSApiError(f"Файл не найден: {in_path}")

        scratch_dir = out_dir / SCRATCH_DIR_NAME
        scratch_dir.mkdir(parents=True, exist_ok=True)
        base = in_path.stem

        with _run_lock:
            with self._counter_lock:
                self._running = True
            prepared_wav: Path | None = None
            try:
                started = time.perf_counter()
                self._report(0.02, STAGE_NAME)
                separator = self._ensure_model(scratch_dir)
                self._throw_if_cancelled()

                self._clean_scratch(scratch_dir)
                self._bind_output_dir(scratch_dir)

                # MP3/FLAC/M4A/OGG приводим к WAV: иначе библиотека попробует
                # записать стемы кодеком входа и упадёт (см. шапку модуля).
                self._report(0.05, "Подготовка аудио")
                model_input, created = self._prepare_input_wav(in_path, scratch_dir)
                prepared_wav = model_input if created else None
                self._throw_if_cancelled()

                self._report(self._INFER_LO, STAGE_NAME)

                stems = self._run_separation(separator, model_input, scratch_dir, base)
                self._report(0.86, "Постобработка минуса")
                result = self._assemble(stems, out_dir, base)
                self._report(1.0, "Готово")

                logger.info(
                    "BS-API разделение завершено за %.1f с",
                    time.perf_counter() - started,
                    extra={"ctx_vocals": str(result.vocals),
                           "ctx_instr": str(result.instrumental),
                           "ctx_model": self.model_name,
                           "ctx_sec": round(time.perf_counter() - started, 1)},
                )
                self._log_gpu_memory()
                return result
            finally:
                with self._counter_lock:
                    self._running = False
                self._progress_cb = None
                # Временный WAV входа убираем всегда: он больше не нужен, а весит
                # как исходный трек. При keep_temp_dirs остальные файлы прогона
                # оставляем — их просили сохранить, а этот дубликат не просили.
                self._drop_prepared_input(prepared_wav)
                if not get_config().keep_temp_dirs:
                    cleanup_temp_dirs([scratch_dir])

    @staticmethod
    def _drop_prepared_input(prepared_wav: Path | None) -> None:
        """Удаляет временный WAV входа, если он создавался."""
        if prepared_wav is None:
            return
        try:
            prepared_wav.unlink(missing_ok=True)
            logger.debug("Временный WAV входа удалён: %s", prepared_wav.name)
        except Exception:
            logger.debug("Не удалось удалить временный WAV входа %s",
                         prepared_wav, exc_info=True)

    # ---------- шаги ----------
    def _run_separation(self, separator, in_path: Path, scratch_dir: Path, base: str) -> dict:
        """
        Сам инференс. Возвращает {"vocals": Path, "instrumental": Path, "parts": [Path]}.
        """
        custom_names = {
            "vocals": f"{base}__bsapi_vocals",
            "instrumental": f"{base}__bsapi_instrumental",
        }
        try:
            with _tqdm_hook(self._on_step):
                outputs = separator.separate(str(in_path), custom_output_names=custom_names)
        except EngineCancelled:
            raise
        except Exception as e:
            logger.exception("BS-API: разделение упало")
            raise BSApiError(f"BS-API не справился с {in_path.name}: {e}") from e

        self._throw_if_cancelled()

        stems: dict[str, object] = {"vocals": None, "instrumental": None, "parts": []}
        for item in outputs or []:
            path = Path(item)
            if not path.is_absolute():
                path = scratch_dir / path
            if not path.exists():
                logger.warning("BS-API вернул путь, которого нет: %s", path)
                continue
            kind = classify_stem(path.name)
            if kind == "part":
                stems["parts"].append(path)
            elif stems[kind] is None:
                stems[kind] = path
            else:
                logger.warning("BS-API вернул два стема «%s»: %s", kind, path)

        if stems["vocals"] is None:
            raise BSApiError(
                f"BS-API не выдал вокал. Файлы: {[Path(o).name for o in outputs or []]}"
            )
        return stems

    def _assemble(self, stems: dict, out_dir: Path, base: str) -> StemSet:
        """
        Читает стемы из рабочей папки, пишет финальные файлы и доводит их до
        того же вида, что subprocess-версия: HPF/LPF минуса и одна общая
        нормализация пары.
        """
        out_sr = get_output_sample_rate()

        vocals_final = out_dir / f"{base}_vocals.wav"
        instr_final = out_dir / f"{base}_instrumental.wav"

        v_data, v_sr = sf.read(str(stems["vocals"]), dtype="float32", always_2d=True)
        i_data, i_sr = self._instrumental_data(stems)

        if v_sr != out_sr or i_sr != out_sr:
            logger.info("Привожу стемы к частоте из настроек: %s/%s → %s Гц",
                        v_sr, i_sr, out_sr)
            v_data = resample_audio(v_data, v_sr, out_sr)
            i_data = resample_audio(i_data, i_sr, out_sr)
            v_sr = i_sr = out_sr

        sf.write(str(vocals_final), v_data, v_sr, subtype="PCM_16")
        sf.write(str(instr_final), i_data, i_sr, subtype="PCM_16")

        # Постобработка минуса — тот же код, что у subprocess-версии
        BSEngine._postprocess_instrumental(instr_final)

        if self.normalize_pair_output:
            self._report(0.95, "Нормализация пары")
            normalize_pair_files(vocals_final, instr_final)
        else:
            # Уровень выставит сборщик финальной пары (гибрид)
            logger.info("Нормализация пары отложена: уровень выставит сборщик пары")

        if not vocals_final.exists() or not instr_final.exists():
            raise BSApiError("BS-API не создал финальные файлы стемов")
        return StemSet(vocals=vocals_final, instrumental=instr_final)

    @staticmethod
    def _instrumental_data(stems: dict) -> tuple[np.ndarray, int]:
        """
        Минус как массив (samples, channels) + его частота.

        Если модель отдала минус одной дорожкой — читаем её. Если модель
        многостемовая и минуса нет, а есть части (drums/bass/other) — складываем
        их в один минус: так же делала subprocess-версия через ffmpeg amix.
        """
        path = stems.get("instrumental")
        if path is not None:
            return sf.read(str(path), dtype="float32", always_2d=True)

        parts = list(stems.get("parts") or [])
        if not parts:
            raise BSApiError("BS-API не выдал ни минус, ни части для его сборки")

        data = None
        sr = None
        for part in parts:
            chunk, part_sr = sf.read(str(part), dtype="float32", always_2d=True)
            if data is None:
                data, sr = chunk, part_sr
            elif part_sr == sr and chunk.shape == data.shape:
                data = data + chunk
            else:
                logger.warning("Часть минуса %s пропущена: не совпала по форме", part.name)

        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > SUM_PARTS_PEAK_LIMIT:
            # Как лимитер в subprocess-версии: опускаем только перегруз,
            # форму сигнала не портим.
            gain = SUM_PARTS_PEAK_LIMIT / peak
            logger.warning("Сумма частей минуса перегружена (%.3f) — опускаю на %.2f dB",
                           peak, 20.0 * np.log10(gain))
            data = data * gain
        data = np.clip(data, -CLIP_SAFETY_PEAK, CLIP_SAFETY_PEAK)
        logger.info("Минус собран из %d частей", len(parts))
        return data, sr

    @staticmethod
    def _clean_scratch(scratch_dir: Path) -> None:
        """Убирает файлы прошлого прогона, чтобы не копить гигабайты."""
        removed = 0
        for old in scratch_dir.glob("*"):
            if not old.is_file():
                continue
            try:
                old.unlink()
                removed += 1
            except Exception:
                logger.exception("Не удалось удалить %s", old)
        if removed:
            logger.debug("Рабочая папка BS-API очищена (%d файлов)", removed)

    @staticmethod
    def _log_gpu_memory() -> None:
        """Сколько занимает GPU после прогона — важно при 6 ГБ видеопамяти."""
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(
                    "Память GPU после BS-API",
                    extra={"ctx_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
                           "ctx_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2)},
                )
        except Exception:
            logger.debug("Не удалось прочитать память GPU", exc_info=True)


__all__ = [
    "BSEngineAPI", "BSApiError", "classify_stem",
    "clear_model_cache", "cache_info",
    "MODEL_SAMPLE_RATE", "SCRATCH_DIR_NAME", "MODEL_CACHE_SIZE",
    "OUTPUT_FORMAT", "NATIVE_INPUT_EXTENSIONS",
]
