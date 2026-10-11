"""
Генератор эталонных данных для бенчмарка (вариант C из плана: синтетика).

Создаёт папки вида:
    benchmark_data/track_01/mix.wav
    benchmark_data/track_01/vocals.wav
    benchmark_data/track_01/instrumental.wav

Зачем синтетика: чтобы измерить SDR/SIR/SAR, нужны треки, где известно
«правильное» разделение. Своих эталонов у проекта нет, а MUSDB18 — это гигабайты,
которые нельзя качать молча. Синтетика честно проверяет весь тракт бенчмарка и
даёт сравнимые между моделями числа, но НЕ заменяет музыкальный тест: модели
обучены на живой музыке и на искусственном сигнале ведут себя иначе.

Как получить настоящие числа: положите в benchmark_data/ свои папки с
mix.wav / vocals.wav / instrumental.wav (стемы должны быть ровно теми же
по длине и частоте, что и микс) и запустите tools/benchmark.py заново.

Запуск:
    .\\.venv312\\Scripts\\python.exe tools\\make_benchmark_data.py
    .\\.venv312\\Scripts\\python.exe tools\\make_benchmark_data.py --tracks 3 --seconds 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

SR = 44100
DEFAULT_OUT = PROJECT_ROOT / "benchmark_data"

# Целевой пик микса: как у файлов проекта (0.891 ≈ -1 dBFS)
MIX_PEAK = 0.891

README = """\
Синтетические эталоны для бенчмарка AST (tools/benchmark.py).

Что внутри каждого track_NN:
  mix.wav           — сумма вокала и минуса (то, что подаём на вход)
  vocals.wav        — «правильный» вокал (эталон для SDR)
  instrumental.wav  — «правильный» минус (эталон для SDR)

Сгенерировано tools/make_benchmark_data.py: у вокала вибрато и слоговая
огибающая, у минуса — аккорды, бас и барабаны. Сигналы синтетические, поэтому
числа SDR сравнимы МЕЖДУ МОДЕЛЯМИ, но не равны тому, что модель показывает на
живой музыке.

Чтобы измерить на настоящей музыке: положите сюда свои папки с теми же тремя
файлами (микс и стемы одной длины, 44100 Гц, стерео или моно) и запустите
tools/benchmark.py — генератор их не тронет.
"""


def _envelope(n: int, rng: np.random.Generator, syllable_sec: float = 0.35) -> np.ndarray:
    """Слоговая огибающая: вокал то звучит, то замолкает (как в песне)."""
    env = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        length = int(syllable_sec * SR * rng.uniform(0.5, 1.6))
        length = max(length, int(0.05 * SR))
        end = min(n, pos + length)
        seg = np.linspace(0, 1, end - pos)
        # мягкая атака и спад внутри слога
        env[pos:end] = np.sin(np.pi * seg) ** 1.5
        pos = end + int(rng.uniform(0.02, 0.12) * SR)
    return env


def make_vocals(n: int, rng: np.random.Generator, f0: float) -> np.ndarray:
    """Вокал: основной тон с вибрато + гармоники + формантный окрас."""
    t = np.arange(n, dtype=np.float64) / SR
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vibrato) / SR

    voice = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate((1.0, 0.5, 0.3, 0.15, 0.08), start=1):
        voice += amp * np.sin(k * phase)

    # «Форманты»: подчёркиваем 700 Гц и 2.5 кГц — там живёт разборчивость
    from scipy.signal import butter, sosfilt

    for center, width, gain in ((700.0, 120.0, 1.6), (2500.0, 400.0, 1.2)):
        lo = max(20.0, center - width)
        hi = min(SR / 2 - 100.0, center + width)
        sos = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
        voice = voice + gain * sosfilt(sos, voice)

    voice *= _envelope(n, rng)
    voice /= max(np.max(np.abs(voice)), 1e-9)

    # Стерео: голос стоит по центру, но чуть «дышит» по панораме
    pan = 0.5 + 0.15 * np.sin(2 * np.pi * 0.3 * t)
    left = voice * np.sqrt(pan)
    right = voice * np.sqrt(1.0 - pan)
    return np.stack([left, right], axis=1)


def _add_at(buf: np.ndarray, sample: np.ndarray, pos: int, gain: float = 1.0) -> None:
    """Подмешивает короткий сэмпл в буфер с позиции pos (без создания копий)."""
    if pos >= len(buf):
        return
    end = min(len(buf), pos + len(sample))
    buf[pos:end] += sample[: end - pos] * gain


def _kick() -> np.ndarray:
    """Бочка: короткий тон с падающей частотой."""
    length = int(0.18 * SR)
    t = np.arange(length, dtype=np.float64) / SR
    freq = 120.0 * np.exp(-t * 18.0) + 45.0
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 12.0)


def _snare(rng: np.random.Generator) -> np.ndarray:
    """Малый: щелчок шумом с быстрым спадом."""
    length = int(0.15 * SR)
    t = np.arange(length, dtype=np.float64) / SR
    noise = rng.normal(0.0, 1.0, length)
    from scipy.signal import butter, sosfilt

    sos = butter(2, 1800.0 / (SR / 2), btype="high", output="sos")
    body = sosfilt(sos, noise) * np.exp(-t * 25.0)
    tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 30.0) * 0.4
    return body + tone


def _hat(rng: np.random.Generator) -> np.ndarray:
    """Хэт: очень короткий шумовой всплеск."""
    length = int(0.05 * SR)
    t = np.arange(length, dtype=np.float64) / SR
    return rng.normal(0.0, 1.0, length) * np.exp(-t * 90.0)


def make_instrumental(n: int, rng: np.random.Generator, root: float) -> np.ndarray:
    """Минус: аккорды, басовая линия и барабаны."""
    t = np.arange(n, dtype=np.float64) / SR
    music = np.zeros(n, dtype=np.float64)

    # Аккорды: мажорные трезвучия по кругу (I–V–vi–IV в одной тональности)
    ratios = ((1.0, 1.25, 1.5), (1.5, 1.875, 2.25),
              (1.667, 2.0, 2.5), (1.333, 1.667, 2.0))
    bar = 2.0                                        # секунд на аккорд
    for idx, start in enumerate(np.arange(0, n / SR, bar)):
        a = int(start * SR)
        b = min(n, a + int(bar * SR))
        if b <= a:
            break
        chord = np.zeros(b - a, dtype=np.float64)
        seg_t = np.arange(b - a, dtype=np.float64) / SR
        for ratio in ratios[idx % len(ratios)]:
            freq = root * 2 * ratio
            for harm in (1.0, 0.4, 0.2):
                chord += harm * np.sin(2 * np.pi * freq * harm * seg_t)
        # мягкая атака, чтобы не было щелчков на стыках
        fade = np.minimum(1.0, seg_t / 0.05) * np.minimum(1.0, (bar - seg_t) / 0.05)
        music[a:b] += chord * fade * 0.25

    # Бас: по одной ноте на долю
    beat = 0.5
    for idx, start in enumerate(np.arange(0, n / SR, beat)):
        a = int(start * SR)
        b = min(n, a + int(beat * SR * 0.9))
        if b <= a:
            break
        seg_t = np.arange(b - a, dtype=np.float64) / SR
        freq = root * ratios[(idx // 4) % len(ratios)][0] * 0.5
        note = (np.sin(2 * np.pi * freq * seg_t)
                + 0.3 * np.sin(2 * np.pi * freq * 2 * seg_t))
        music[a:b] += note * np.exp(-seg_t * 3.0) * 0.4

    # Барабаны: бочка на 1 и 3, малый на 2 и 4, хэты восьмыми
    kick, snare = _kick(), _snare(rng)
    for beat_idx, start in enumerate(np.arange(0, n / SR, 0.5)):
        pos = int(start * SR)
        if beat_idx % 2 == 0:
            _add_at(music, kick, pos, gain=0.5)
        else:
            _add_at(music, snare, pos, gain=0.35)
    for start in np.arange(0, n / SR, 0.25):
        _add_at(music, _hat(rng), int(start * SR), gain=0.08)

    music /= max(np.max(np.abs(music)), 1e-9)
    # Лёгкое стерео: барабаны чуть шире, аккорды по центру
    width = 0.08 * np.roll(music, int(0.002 * SR))
    left = music + width
    right = music - width
    return np.stack([left, right], axis=1)


def make_track(index: int, seconds: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Один трек: (микс, вокал, минус). Микс — ровно сумма, без клиппинга."""
    rng = np.random.default_rng(seed + index)
    n = int(seconds * SR)

    f0 = float(rng.uniform(180.0, 330.0))            # тесситура «вокалиста»
    root = float(rng.uniform(90.0, 140.0))           # тоника минуса

    vocals = make_vocals(n, rng, f0)
    instr = make_instrumental(n, rng, root)

    # Вокал в миксе — на переднем плане, минус — на 6 дБ тише (как в поп-музыке)
    vocals *= 0.55
    instr *= 0.28

    mix = vocals + instr
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * MIX_PEAK
        vocals = vocals / peak * MIX_PEAK
        instr = instr / peak * MIX_PEAK
    return mix.astype(np.float32), vocals.astype(np.float32), instr.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Синтетические эталоны для бенчмарка AST")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="куда положить данные")
    parser.add_argument("--tracks", type=int, default=5, help="сколько треков (по умолчанию 5)")
    parser.add_argument("--seconds", type=float, default=12.0, help="длина трека, с")
    parser.add_argument("--seed", type=int, default=20261011, help="зерно генератора")
    parser.add_argument("--force", action="store_true",
                        help="перезаписать существующие треки")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.tracks + 1):
        track_dir = out_dir / f"track_{i:02d}"
        if track_dir.exists() and not args.force:
            print(f"Пропускаю {track_dir.name}: уже есть (--force чтобы перезаписать)")
            continue
        track_dir.mkdir(parents=True, exist_ok=True)
        mix, vocals, instr = make_track(i, args.seconds, args.seed)
        for name, data in (("mix", mix), ("vocals", vocals), ("instrumental", instr)):
            sf.write(str(track_dir / f"{name}.wav"), data, SR, subtype="PCM_16")
        print(f"{track_dir.name}: {args.seconds:.0f} с, {SR} Гц, пик микса "
              f"{float(np.max(np.abs(mix))):.3f}")

    (out_dir / "README.txt").write_text(README, encoding="utf-8")
    print(f"\nГотово: {out_dir}")
    print("Дальше: .\\.venv312\\Scripts\\python.exe tools\\benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
