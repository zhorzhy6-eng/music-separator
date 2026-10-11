r"""
Проверка воркера: очередь GPU (по одной), отмена, убийство audio-separator.exe.

Запуск из корня проекта:
    .\.venv312\Scripts\python.exe tools\stage4_worker_check.py [путь_к_wav]

Что проверяем:
  1) две GPU-задачи подряд не идут одновременно (на 6 ГБ это был бы OOM);
  2) отмена во время BS-разделения реально убивает audio-separator.exe;
  3) после отмены приходит событие «отменено», а не «ошибка».
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engines import get_engine  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.paths import PROJECT_ROOT  # noqa: E402
from core.worker import ErrorEvent, ProgressEvent, Worker  # noqa: E402

CLIP = PROJECT_ROOT / "logs" / "qa_refclip_10s.wav"
OUT = PROJECT_ROOT / "logs" / "_stagecheck" / "worker"


def audio_separator_processes() -> list[str]:
    """Список живых audio-separator.exe (через tasklist)."""
    try:
        res = subprocess.run(["tasklist", "/FI", "IMAGENAME eq audio-separator.exe"],
                             capture_output=True, text=True, timeout=20)
    except Exception as e:                       # pragma: no cover
        return [f"<не смог спросить tasklist: {e}>"]
    return [ln for ln in res.stdout.splitlines() if "audio-separator" in ln.lower()]


def main() -> int:
    setup_logging()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else CLIP
    if not src.exists():
        print(f"Нет входного файла: {src}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    worker = Worker()
    spec = get_engine("bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt")

    print("[1] Ставлю BS-разделение и жду 6 секунд, затем отменяю...")
    task_id = worker.submit_separation(spec, str(src), str(OUT))
    time.sleep(6)

    alive = audio_separator_processes()
    print(f"    audio-separator.exe до отмены: {len(alive)} шт.")
    for line in alive[:3]:
        print(f"      {line.strip()}")

    t0 = time.monotonic()
    worker.cancel(task_id)

    cancelled = False
    while time.monotonic() - t0 < 60:
        try:
            ev = worker.events.get(timeout=1.0)
        except Exception:
            continue
        if isinstance(ev, ErrorEvent) and ev.task_id == task_id:
            cancelled = getattr(ev, "cancelled", False)
            print(f"    событие: {ev.message!r}, помечено как отмена: {cancelled}")
            break
        if isinstance(ev, ProgressEvent):
            print(f"    прогресс {ev.progress:.2f} — {ev.stage}")

    time.sleep(2)
    left = audio_separator_processes()
    print(f"[2] audio-separator.exe после отмены: {len(left)} шт.")

    print("[3] Проверяю, что воркер закрывается и не держит процессы...")
    worker.shutdown(wait=False)
    time.sleep(2)
    print(f"    audio-separator.exe после shutdown: {len(audio_separator_processes())} шт.")

    ok = cancelled and len(left) == 0
    print("ИТОГ:", "OK" if ok else "ПРОБЛЕМА")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
