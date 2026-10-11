"""Замер: BS через Python-API против subprocess (audio-separator.exe).

Не инструмент проекта, а разовая проверка улучшения №1: время и совпадение
результата. Запускается вручную из корня проекта.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engines import ENGINES, make_separator          # noqa: E402
from core.engines.bs_engine import BSEngine               # noqa: E402
from core.logging_config import setup_logging             # noqa: E402

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
SRC = Path("logs/qa_refclip_10s.wav")
OUT = Path("output/_timing")


def run_api(tag: str) -> float:
    """Новый движок на каждый прогон — как делает воркер."""
    sep = make_separator(ENGINES[f"bs:{MODEL}"])
    t0 = time.perf_counter()
    sep.separate(SRC, OUT / tag)
    dt = time.perf_counter() - t0
    print(f"API  {tag}: {dt:6.1f} с  ({sep.name})")
    return dt


def run_proc(tag: str, engine: BSEngine) -> float:
    t0 = time.perf_counter()
    engine.separate(SRC, OUT / tag)
    dt = time.perf_counter() - t0
    print(f"PROC {tag}: {dt:6.1f} с")
    return dt


def compare(a_dir: Path, b_dir: Path) -> None:
    """Насколько совпали стемы двух реализаций."""
    base = SRC.stem
    for kind in ("vocals", "instrumental"):
        a, sr_a = sf.read(str(a_dir / f"{base}_{kind}.wav"), dtype="float32", always_2d=True)
        b, sr_b = sf.read(str(b_dir / f"{base}_{kind}.wav"), dtype="float32", always_2d=True)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        peak = float(np.max(np.abs(a))) or 1.0
        diff = float(np.max(np.abs(a - b))) / peak
        corr = float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
        print(f"{kind:14s} sr {sr_a}/{sr_b}  макс. расхождение {diff*100:.3f} % пика  "
              f"корреляция {corr:.5f}")


def main() -> None:
    setup_logging()
    api_cold = run_api("api_1")
    api_warm = run_api("api_2")

    engine = BSEngine(model_name=MODEL)
    proc_cold = run_proc("proc_1", engine)
    proc_warm = run_proc("proc_2", engine)

    print()
    print(f"API : первый {api_cold:6.1f} с, второй {api_warm:6.1f} с "
          f"(экономия {api_cold - api_warm:5.1f} с)")
    print(f"PROC: первый {proc_cold:6.1f} с, второй {proc_warm:6.1f} с "
          f"(экономия {proc_cold - proc_warm:5.1f} с)")
    print()
    compare(OUT / "api_1", OUT / "proc_1")


if __name__ == "__main__":
    main()
