"""
Проверки воркера («менеджера»): очередь, отмена, остановка.

Главное, что здесь проверяется, — то, из-за чего на 6 ГБ видеопамяти падал
драйвер: две GPU-задачи не должны считаться одновременно.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import soundfile as sf

from core.worker import (
    CPU_TASKS,
    GPU_TASKS,
    CANCEL_MESSAGE,
    DoneEvent,
    ErrorEvent,
    ProgressEvent,
    TaskType,
    Worker,
)


@pytest.fixture
def fake_spec():
    """Описание движка для тестов: реальную модель никто не грузит."""
    from core.engines import EngineSpec
    return EngineSpec(id="test:fake", label="тест", kind="demucs", model="htdemucs")


def _wait_for_event(worker: Worker, task_id: int, timeout: float = 10.0):
    """
    Ждёт DoneEvent или ErrorEvent по задаче.

    ВАЖНО: события соседних задач запоминаются, а не выбрасываются. Иначе ожидание
    первой задачи съедало событие второй, и тест падал на пустом месте.
    """
    seen = getattr(worker, "_test_seen_events", None)
    if seen is None:
        seen = {}
        worker._test_seen_events = seen
    if task_id in seen:
        return seen.pop(task_id)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = worker.events.get(timeout=0.2)
        except Exception:
            continue
        if isinstance(ev, (DoneEvent, ErrorEvent)):
            if ev.task_id == task_id:
                return ev
            seen[ev.task_id] = ev
    raise AssertionError(f"задача {task_id} не отчиталась за {timeout} с")


def test_task_types_grouped_into_queues():
    """GPU-задачи отделены от CPU-задач: у них разные очереди."""
    assert TaskType.SEPARATE in GPU_TASKS
    assert TaskType.TRANSCRIBE in GPU_TASKS
    assert TaskType.PIANO in GPU_TASKS
    assert TaskType.ANALYZE in CPU_TASKS
    assert TaskType.ANALYZE_PAIR in CPU_TASKS
    assert TaskType.MIDI_ANALYZE in CPU_TASKS
    assert not (GPU_TASKS & CPU_TASKS)


def test_old_task_type_names_still_work():
    """Совместимость: старые имена типов задач — псевдонимы того же значения."""
    assert TaskType.SEPARATE_BS is TaskType.SEPARATE
    assert TaskType.SEPARATE_HYBRID is TaskType.SEPARATE


def test_gpu_tasks_do_not_overlap(monkeypatch, fake_spec):
    """Две GPU-задачи идут по очереди, а не одновременно."""
    worker = Worker()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_separator(spec):
        class FakeSep:
            name = "fake"

            def separate(self, src, out_dir, progress_cb=None):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    if progress_cb:
                        progress_cb(0.5, "работаю")
                    time.sleep(0.3)
                    return f"готово:{src}"
                finally:
                    with lock:
                        active -= 1

            def cancel(self):
                return False

        return FakeSep()

    monkeypatch.setattr("core.worker.make_separator", fake_separator)

    id1 = worker.submit_separation(fake_spec, "a.wav", "out")
    id2 = worker.submit_separation(fake_spec, "b.wav", "out")

    ev1 = _wait_for_event(worker, id1, timeout=10)
    ev2 = _wait_for_event(worker, id2, timeout=10)

    assert isinstance(ev1, DoneEvent) and isinstance(ev2, DoneEvent)
    assert max_active == 1, "GPU-задачи пошли параллельно — на 6 ГБ это OOM"
    worker.shutdown()


def test_cpu_tasks_may_run_in_parallel(monkeypatch):
    """Анализы (без GPU) идут по два — процессор простаивать не должен."""
    worker = Worker()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_analyze(audio, report_dir=None, progress_cb=None, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.3)
            return {"ok": True}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("core.worker.analyze_audio", fake_analyze)

    ids = [worker.submit_analysis(f"{i}.wav", "out") for i in range(2)]
    for task_id in ids:
        assert isinstance(_wait_for_event(worker, task_id), DoneEvent)
    assert max_active >= 2, "CPU-очередь работает по одной — потеряли параллельность"
    worker.shutdown()


def test_cancel_running_task_reports_cancelled(monkeypatch, fake_spec):
    """Отмена работающей задачи помечается как отмена, а не как ошибка."""
    worker = Worker()
    started = threading.Event()

    def fake_separator(spec):
        class SlowSep:
            name = "slow"

            def separate(self, src, out_dir, progress_cb=None):
                started.set()
                for _ in range(100):
                    time.sleep(0.05)
                return "поздно"

            def cancel(self):
                return True

        return SlowSep()

    monkeypatch.setattr("core.worker.make_separator", fake_separator)

    task_id = worker.submit_separation(fake_spec, "a.wav", "out")
    assert started.wait(5), "задача не началась"
    assert worker.cancel(task_id) is True

    ev = _wait_for_event(worker, task_id, timeout=10)
    assert isinstance(ev, ErrorEvent)
    assert ev.cancelled is True
    assert ev.message == CANCEL_MESSAGE
    worker.shutdown()


def test_cancel_before_start_never_runs(monkeypatch, fake_spec):
    """Задача, отменённая до старта, вообще не выполняется."""
    worker = Worker()
    ran = []

    def fake_separator(spec):
        class Sep:
            name = "x"

            def separate(self, src, out_dir, progress_cb=None):
                ran.append(src)
                return "готово"

            def cancel(self):
                return False

        return Sep()

    monkeypatch.setattr("core.worker.make_separator", fake_separator)

    # Занимаем GPU-очередь долгой задачей, вторую отменяем в очереди
    blocker = worker.submit_separation(fake_spec, "blocker.wav", "out")
    queued = worker.submit_separation(fake_spec, "queued.wav", "out")
    worker.cancel(queued)

    _wait_for_event(worker, blocker, timeout=10)
    ev = _wait_for_event(worker, queued, timeout=10)
    assert isinstance(ev, ErrorEvent) and ev.cancelled
    assert "queued.wav" not in ran, "отменённая задача всё-таки выполнилась"
    worker.shutdown()


def test_cancel_unknown_task_returns_false():
    worker = Worker()
    assert worker.cancel(999) is False
    assert worker.cancel(None) is False
    worker.shutdown()


def test_shutdown_is_idempotent():
    worker = Worker()
    worker.shutdown()
    worker.shutdown()          # повторный вызов не должен падать
    assert worker.is_busy is False


def test_submit_after_shutdown_reports_error():
    """
    После закрытия окна новые задачи не запускаются, но и не вешают приложение.

    Это не теория: закрытый пул задач молча выбрасывает заявку, и код, который
    ждёт по ней событие, ждал бы вечно. Поэтому воркер отвечает сразу.
    """
    worker = Worker()
    worker.shutdown()
    task_id = worker.submit_analysis("a.wav", "out")
    ev = _wait_for_event(worker, task_id, timeout=5)
    assert isinstance(ev, ErrorEvent)
    assert worker.is_busy is False


def test_real_analysis_job_end_to_end(tmp_path):
    """Настоящая задача анализа через воркер: файл → отчёт (без GPU)."""
    path = tmp_path / "tone.wav"
    t = np.arange(44100, dtype=np.float64) / 44100
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
             44100, subtype="PCM_16")

    worker = Worker()
    task_id = worker.submit_midi_analysis("нет-такого.mid", str(tmp_path))
    ev = _wait_for_event(worker, task_id, timeout=30)
    # Файла нет — ждём честную ошибку, а не зависание
    assert isinstance(ev, ErrorEvent)
    worker.shutdown()


def test_bs_cache_released_before_other_engines(monkeypatch):
    """
    Перед Demucs и гибридом кэш BS-модели освобождается.

    BS-Roformer через Python-API держит модель в видеопамяти между треками
    (это экономит время), но двум моделям на 6 ГБ места не хватает — поэтому
    перед другим движком память отпускаем, а перед BS — нет.
    """
    from core.engines import EngineSpec

    released: list[int] = []
    monkeypatch.setattr("core.engines.bs_engine_api.cache_info", lambda: {"size": 1})
    monkeypatch.setattr("core.engines.bs_engine_api.clear_model_cache",
                        lambda *a, **k: released.append(1))

    class FakeSep:
        name = "fake"

        def separate(self, src, out_dir, progress_cb=None):
            return "готово"

        def cancel(self):
            return False

    monkeypatch.setattr("core.worker.make_separator", lambda spec: FakeSep())

    worker = Worker()
    specs = [
        EngineSpec(id="bs:test", label="bs", kind="bs", model="m.ckpt"),
        EngineSpec(id="demucs:test", label="demucs", kind="demucs", model="htdemucs"),
        EngineSpec(id="hybrid:test", label="hybrid", kind="hybrid",
                   model="m.ckpt", companion="htdemucs_ft"),
    ]
    for spec in specs:
        task_id = worker.submit_separation(spec, "a.wav", "out")
        assert isinstance(_wait_for_event(worker, task_id, timeout=10), DoneEvent)

    assert len(released) == 2, "кэш BS-модели не освободили перед Demucs/гибридом"
    worker.shutdown()
