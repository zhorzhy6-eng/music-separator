"""
Воркер («менеджер»): решает, какое блюдо когда готовить.

Что здесь есть:
  * ДВЕ очереди. GPU-задачи (разделение, транскрипция, пиано) идут строго по
    одной: на видеокарте с 6 ГБ две модели одновременно не помещаются — раньше
    каждый запрос запускал свой поток, и при двух нажатиях подряд драйвер
    уходил в своп или падал с OOM. Задачи без GPU (анализ) идут по две сразу.
  * Отмена (cancel): помечает задачу отменённой и, если она уже работает,
    просит движок прерваться (BS-движок гасит свой audio-separator.exe).
  * shutdown(): закрывает очереди и убивает дочерние процессы — при закрытии
    окна ничего не должно остаться висеть в диспетчере задач.

О ходе дела воркер сообщает через очередь self.events:
ProgressEvent / DoneEvent / ErrorEvent. UI читает её и рисует.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from core.engines import make_separator
from core.engines.base import EngineCancelled
from core.transcriber import MidiTranscriber
from core.analyzer import analyze_audio
from core.piano_transcriber import PianoTranscriber
from core.midi_analyzer import analyze_midi

logger = logging.getLogger(__name__)


class TaskType(Enum):
    SEPARATE = auto()
    TRANSCRIBE = auto()
    ANALYZE = auto()
    ANALYZE_PAIR = auto()
    PIANO = auto()
    MIDI_ANALYZE = auto()

    # DECISION: имена SEPARATE_BS и SEPARATE_HYBRID оставлены как псевдонимы того же
    # значения. Старый код, который сравнивает task_type is TaskType.SEPARATE_BS,
    # продолжает работать, а ветвление по типам задач больше не нужно: вид движка
    # лежит внутри EngineSpec, и его разбирает фабрика make_separator.
    SEPARATE_BS = SEPARATE
    SEPARATE_HYBRID = SEPARATE


# Какие задачи занимают видеокарту (идут по одной), какие — только процессор
GPU_TASKS = frozenset({TaskType.SEPARATE, TaskType.TRANSCRIBE, TaskType.PIANO})
CPU_TASKS = frozenset({TaskType.ANALYZE, TaskType.ANALYZE_PAIR, TaskType.MIDI_ANALYZE})

# Сколько задач одного вида пускать одновременно.
# GPU: одна — иначе две модели не помещаются в 6 ГБ видеопамяти.
# CPU: две — анализ ест процессор, но их можно вести параллельно.
GPU_WORKERS = 1
CPU_WORKERS = 2

CANCEL_MESSAGE = "⏹ Отменено"


@dataclass
class ProgressEvent:
    task_id: int
    progress: float
    stage: str


@dataclass
class DoneEvent:
    task_id: int
    task_type: TaskType
    result: Any


@dataclass
class ErrorEvent:
    task_id: int
    message: str
    cancelled: bool = False        # True — отменил пользователь, а не сбой


@dataclass
class _Task:
    """Заявка в очереди."""

    task_id: int
    task_type: TaskType
    fn: Callable[["_Task", Callable[[float, str], None]], Any]
    args: tuple = field(default_factory=tuple)


class Worker:
    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self._counter = 0
        self._lock = threading.Lock()

        # Что сейчас реально работает: номера задач и объекты движков.
        # Нужно, чтобы знать «занят ли воркер» и кого просить прерваться.
        self._active: set = set()

        self._cancelled: set[int] = set()
        self._shutdown = False

        # DECISION: только ThreadPoolExecutor (без multiprocessing) — как требует ТЗ.
        self._gpu_pool = ThreadPoolExecutor(
            max_workers=GPU_WORKERS, thread_name_prefix="ast-gpu")
        self._cpu_pool = ThreadPoolExecutor(
            max_workers=CPU_WORKERS, thread_name_prefix="ast-cpu")

    # ---------- заявки ----------
    def submit_separation(self, spec, input_path, output_dir) -> int:
        """
        Разделение по описанию движка из реестра (core.engines.EngineSpec).

        Раньше воркер знал про каждый движок отдельно (три метода и ветка if/elif
        по типу задачи). Теперь на вход идёт описание, а какой это движок —
        решает реестр, а собирает его фабрика make_separator.
        """
        from core.engines import ENGINES, EngineSpec
        if not isinstance(spec, EngineSpec):
            # Обратная совместимость: раньше третьим аргументом шло имя модели
            # Demucs ("htdemucs"). Понимаем и его, и id движка из реестра.
            key = str(spec)
            spec = ENGINES.get(key) or ENGINES.get("demucs:" + key)
            if spec is None:
                raise KeyError(f"Неизвестный движок: {key}")
        return self._submit(TaskType.SEPARATE,
                            lambda task, progress: self._do_separate(
                                spec, input_path, output_dir, progress))

    def submit_separation_bs(self, input_path, output_dir,
                             bs_model="BS-Roformer-SW.ckpt") -> int:
        """Старый вызов: разделение BS-моделью."""
        from core.engines import EngineSpec
        rec = EngineSpec(id="bs:" + bs_model, label=bs_model, kind="bs", model=bs_model)
        return self.submit_separation(rec, input_path, output_dir)

    def submit_separation_hybrid(self, input_path, output_dir,
                                 demucs_model="htdemucs_ft",
                                 bs_model="BS-Roformer-SW.ckpt") -> int:
        """Старый вызов: гибрид (вокал от Demucs, минус от BS)."""
        from core.engines import EngineSpec
        rec = EngineSpec(
            id=f"hybrid:{demucs_model}+{bs_model}", label=f"{demucs_model} + {bs_model}",
            kind="hybrid", model=bs_model, companion=demucs_model,
        )
        return self.submit_separation(rec, input_path, output_dir)

    def submit_transcription(self, input_audio, output_midi,
                             instrument="Фортепиано") -> int:
        def job(task, progress):
            tr = MidiTranscriber()
            return tr.transcribe(input_audio, output_midi, instrument,
                                 progress_cb=progress)
        return self._submit(TaskType.TRANSCRIBE, job, input_audio, output_midi)

    def submit_analysis(self, input_audio, report_dir=None) -> int:
        def job(task, progress):
            return analyze_audio(input_audio, report_dir, progress_cb=progress)
        return self._submit(TaskType.ANALYZE, job, input_audio, report_dir)

    def submit_pair_analysis(self, vocals_path, instr_path, report_dir=None) -> int:
        def job(task, progress):
            from core.super_analyzer import super_analyze_pair
            return super_analyze_pair(vocals_path, instr_path,
                                      report_dir, progress_cb=progress)
        return self._submit(TaskType.ANALYZE_PAIR, job, vocals_path, instr_path,
                            report_dir)

    def submit_piano(self, input_audio, output_midi, model_size="small") -> int:
        def job(task, progress):
            pt = PianoTranscriber(model_size=model_size)
            return pt.transcribe(input_audio, output_midi, progress_cb=progress)
        return self._submit(TaskType.PIANO, job, input_audio, output_midi)

    def submit_midi_analysis(self, midi_path, report_dir=None) -> int:
        def job(task, progress):
            return analyze_midi(midi_path, report_dir)
        return self._submit(TaskType.MIDI_ANALYZE, job, midi_path, report_dir)

    # ---------- отмена и остановка ----------
    def cancel(self, task_id: int | None) -> bool:
        """
        Просит отменить задачу. True — задача была и её отменили.

        Кто ещё не начал работу — до работы не дойдёт (проверка в начале запуска).
        Кто уже работает — получает сигнал движку: BS-движок гасит свой
        audio-separator.exe вместе с дочерними процессами.
        """
        if task_id is None:
            return False
        with self._lock:
            if not (1 <= task_id <= self._counter):
                return False
            self._cancelled.add(task_id)
        logger.warning("Запрошена отмена задачи", extra={"ctx_task_id": task_id})
        self._signal_cancel()
        return True

    def cancel_all(self) -> None:
        """Отменяет всю текущую работу (используется при закрытии окна)."""
        self._signal_cancel()

    def _signal_cancel(self) -> None:
        """Дёргает cancel() у всех работающих движков (если он у них есть)."""
        for engine in self._active_snapshot():
            stop = getattr(engine, "cancel", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    logger.exception("Движок не смог прерваться")

    def _active_snapshot(self) -> list:
        with self._lock:
            return list(self._active)

    def shutdown(self, wait: bool = False) -> None:
        """
        Закрывает воркер: отменяет текущую работу и гасит дочерние процессы.

        Вызывается при закрытии окна. Без этого audio-separator.exe продолжал
        работать после выхода из приложения.
        """
        if self._shutdown:
            return
        self._shutdown = True
        logger.info("Остановка воркера")
        self.cancel_all()
        self._gpu_pool.shutdown(wait=wait, cancel_futures=True)
        self._cpu_pool.shutdown(wait=wait, cancel_futures=True)

    @property
    def is_busy(self) -> bool:
        """Есть ли работа прямо сейчас (или в очереди)."""
        return bool(self._active)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """
        Ждёт, пока все задачи закончатся. True — дождались.

        Нужна CLI: без окна никто не следит за событиями, и процесс мог бы
        завершиться раньше, чем доделается разделение.
        """
        start = time.monotonic()
        while True:
            if not self._active:
                return True
            if timeout is not None and (time.monotonic() - start) > timeout:
                return False
            time.sleep(0.05)

    # ---------- внутреннее ----------
    def _next_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _is_cancelled(self, task_id: int) -> bool:
        with self._lock:
            return task_id in self._cancelled

    def _submit(self, task_type: TaskType, fn, *args) -> int:
        task_id = self._next_id()
        task = _Task(task_id=task_id, task_type=task_type, fn=fn, args=args)
        pool = self._gpu_pool if task_type in GPU_TASKS else self._cpu_pool
        # DECISION: после закрытия смены новые заявки не принимаем. Пул,
        # закрытый через shutdown(cancel_futures=True), молча ВЫБРАСЫВАЕТ задачу —
        # событие о ней не придёт никогда, и ожидающий её код завис бы навсегда.
        if self._shutdown:
            logger.warning("Задача отклонена: воркер уже остановлен",
                           extra={"ctx_task_id": task_id, "ctx_type": task_type.name})
            self.events.put(ErrorEvent(task_id, "Приложение закрывается"))
            return task_id
        try:
            pool.submit(self._run, task)
        except RuntimeError:
            # Пул закрылся в этот самый момент — тоже сообщаем честно
            logger.warning("Задача не принята: очередь уже закрыта",
                           extra={"ctx_task_id": task_id})
            self.events.put(ErrorEvent(task_id, "Приложение закрывается"))
            return task_id
        logger.info("Задача поставлена",
                    extra={"ctx_task_id": task_id, "ctx_type": task_type.name,
                           "ctx_queue": "gpu" if pool is self._gpu_pool else "cpu"})
        return task_id

    def _run(self, task: _Task) -> None:
        task_id = task.task_id

        def progress(p, stage):
            self.events.put(ProgressEvent(task_id, p, stage))

        if self._is_cancelled(task_id):
            logger.info("Задача отменена до старта", extra={"ctx_task_id": task_id})
            self.events.put(ErrorEvent(task_id, CANCEL_MESSAGE, cancelled=True))
            return

        self._running_add(task_id)
        try:
            result = task.fn(task, progress)
            if self._is_cancelled(task_id):
                # Отменили, пока считали: результат уже не нужен
                self.events.put(ErrorEvent(task_id, CANCEL_MESSAGE, cancelled=True))
            else:
                self.events.put(DoneEvent(task_id, task.task_type, result))
        except EngineCancelled:
            logger.info("Задача отменена пользователем", extra={"ctx_task_id": task_id})
            self.events.put(ErrorEvent(task_id, CANCEL_MESSAGE, cancelled=True))
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Задача упала",
                         extra={"ctx_task_id": task_id, "ctx_tb": tb})
            self.events.put(ErrorEvent(task_id, str(e)))
        finally:
            self._running_discard(task_id)

    def _running_add(self, item) -> None:
        with self._lock:
            self._active.add(item)

    def _running_discard(self, item) -> None:
        with self._lock:
            self._active.discard(item)

    # ---------- сами работы ----------
    def _do_separate(self, spec, input_path, output_dir, progress):
        """
        Разделение: движок собирает фабрика, воркер про его внутренности не знает.

        Заодно держит ссылку на движок, чтобы отмена могла до него дотянуться.
        """
        # BS-модель, посчитанная через Python-API, остаётся в видеопамяти между
        # треками — именно это и даёт экономию ~5–10 с на трек. Но если дальше
        # запускают Demucs или гибрид, держать её незачем: на 6 ГБ видеопамяти
        # две модели рядом — это OOM, а не экономия.
        if spec.kind != "bs":
            self._release_bs_cache()
        sep = make_separator(spec)
        self._running_add(sep)
        try:
            return sep.separate(input_path, output_dir, progress_cb=progress)
        finally:
            self._running_discard(sep)

    @staticmethod
    def _release_bs_cache() -> None:
        """Освобождает видеопамять от закэшированной BS-модели (если она есть)."""
        try:
            from core.engines.bs_engine_api import cache_info, clear_model_cache

            if cache_info()["size"]:
                logger.info("Освобождаю кэш BS-моделей перед другим движком",
                            extra={"ctx_engine_kind": "не bs"})
                clear_model_cache()
        except Exception:
            # Вопрос памяти не должен ронять саму задачу
            logger.debug("Не удалось освободить кэш BS-моделей", exc_info=True)
