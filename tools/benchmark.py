"""
Бенчмарк моделей разделения: SDR / SIR / SAR на эталонных треках.

Зачем: до сих пор модель выбиралась на слух. SDR (сколько сигнала дошло до
результата) даёт числовую опору: две модели можно сравнить на одних и тех же
треках и увидеть разницу в децибелах, а не «вроде эта почище».

Формат данных (папка на трек):
    benchmark_data/
      track_01/
        mix.wav           — то, что подаём на вход
        vocals.wav        — эталонный вокал
        instrumental.wav  — эталонный минус
      track_02/...

Эталоны можно взять свои, а можно сгенерировать синтетические:
    .\\.venv312\\Scripts\\python.exe tools\\make_benchmark_data.py

Запуск:
    .\\.venv312\\Scripts\\python.exe tools\\benchmark.py                 # пресет из 5 моделей
    .\\.venv312\\Scripts\\python.exe tools\\benchmark.py --engines bs:duality-v2 demucs:htdemucs_ft
    .\\.venv312\\Scripts\\python.exe tools\\benchmark.py --list          # что вообще есть

Что получается на выходе:
    logs/benchmark/summary.md               — таблица «модель × метрика»
    logs/benchmark/benchmark_<модель>.json  — подробности по трекам
    logs/benchmark/separated/<модель>/...   — сами стемы (послушать глазами ушами)

Про метрики простыми словами:
    SDR — насколько результат похож на эталон (больше — лучше);
    SIR — насколько чужой источник задавлен (больше — лучше);
    SAR — сколько артефактов добавила модель (больше — лучше).
Для идеального разделения SDR упирается в потолок ~30-40 дБ: дальше мешает
сама разница между «эталоном» и тем, что физически можно вытащить из микса.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_utils import resample_audio  # noqa: E402
from core.engines import ENGINES, get_engine, make_separator, resolve_engine_id  # noqa: E402
from core.paths import LOGS_DIR, MODELS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# Частота эталонов и моделей. Другая частота у файла — пересчитываем.
BASE_SR = 44100

# Пресет из плана: два BS-движка, один «тяжёлый» BS, Demucs и гибрид
DEFAULT_ENGINES = [
    "bs:melband_roformer_instvox_duality_v2.ckpt",
    "bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "bs:mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    "demucs:htdemucs_ft",
    "hybrid:htdemucs_ft+model_bs_roformer_ep_317_sdr_12.9755.ckpt",
]

STEMS = ("vocals", "instrumental")


# ============================================================
#  Аудио
# ============================================================
def load_audio(path: Path, label: str = "") -> tuple[np.ndarray, int]:
    """
    Читает аудио как (samples, channels) float32 и приводит к 44100 Гц.

    Частота приводится, а не считается ошибкой: микс, стемы и результат
    разделения обязаны сравниваться на одной частоте, иначе метрики бессмысленны.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != BASE_SR:
        logger.info("%s: частота %s Гц → привожу к %s", label or path.name, sr, BASE_SR)
        data = resample_audio(data, sr, BASE_SR)
        sr = BASE_SR
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    """Моно-дорожка: метрики считаем по сумме каналов (описано в README)."""
    return data.mean(axis=1) if data.ndim == 2 else data


# ============================================================
#  Метрики
# ============================================================
def _approx_metrics(ref: np.ndarray, est: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """
    SDR/SIR/SAR без mir_eval: разложение проекцией (упрощённый BSS-Eval).

    Отличие от классики: нет подбора 512-отводного фильтра на каждую долю
    секунды, поэтому числа чуть оптимистичнее и не сравнимы напрямую с
    публикациями. Для сравнения МОДЕЛЕЙ между собой этого достаточно.

    Возвращает три списка — по значению на каждый стем.
    """
    eps = 1e-12
    n = min(ref.shape[1], est.shape[1])
    ref, est = ref[:, :n], est[:, :n]

    def db(num: float, den: float) -> float:
        return float(10.0 * np.log10((num + eps) / (den + eps)))

    sdrs, sirs, sars = [], [], []
    for i in range(ref.shape[0]):
        estimate = est[i]
        own = ref[i] * (np.dot(estimate, ref[i]) / (np.dot(ref[i], ref[i]) + eps))
        foreign = np.zeros(n)
        for j in range(ref.shape[0]):
            if i == j:
                continue
            foreign += ref[j] * (np.dot(estimate, ref[j]) / (np.dot(ref[j], ref[j]) + eps))
        artifacts = estimate - own - foreign
        noise = estimate - own

        energy = float(np.sum(own ** 2))
        sdrs.append(db(energy, float(np.sum(noise ** 2))))
        sirs.append(db(energy, float(np.sum(foreign ** 2))))
        sars.append(db(energy, float(np.sum(artifacts ** 2))))
    return sdrs, sirs, sars


def compute_metrics(
    ref_vocals: np.ndarray,
    ref_instr: np.ndarray,
    est_vocals: np.ndarray,
    est_instr: np.ndarray,
) -> dict:
    """
    SDR/SIR/SAR для пары стемов + корреляция как простая проверка на сдвиг.

    Порядок стемов фиксирован (вокал, минус), перестановку не ищем: движок AST
    всегда кладёт вокал в vocals, а минус в instrumental.
    """
    ref = np.vstack([to_mono(ref_vocals), to_mono(ref_instr)])
    est = np.vstack([to_mono(est_vocals), to_mono(est_instr)])

    n = min(ref.shape[1], est.shape[1])
    if abs(ref.shape[1] - est.shape[1]) > int(0.01 * n):
        logger.warning("Длины эталона и результата разошлись: %s vs %s — режу по короткой",
                       ref.shape[1], est.shape[1])
    ref, est = ref[:, :n], est[:, :n]

    result: dict = {"metrics_source": "mir_eval"}
    try:
        import warnings

        from mir_eval.separation import bss_eval_sources

        with warnings.catch_warnings():
            # Функция помечена устаревшей на будущее, но пока это единственная
            # реализация классического BSS-Eval с фильтрами — берём её.
            warnings.simplefilter("ignore")
            sdr, sir, sar, _ = bss_eval_sources(ref, est, compute_permutation=False)
        for idx, stem in enumerate(STEMS):
            result[f"sdr_{stem}"] = round(float(sdr[idx]), 2)
            result[f"sir_{stem}"] = round(float(sir[idx]), 2)
            result[f"sar_{stem}"] = round(float(sar[idx]), 2)
    except Exception as e:
        logger.warning("mir_eval недоступен (%s) — считаю упрощённые метрики", e)
        result["metrics_source"] = "approx"
        result["note"] = ("mir_eval не установлен: метрики упрощённые, "
                          "сравнимы только между собой. Установка: "
                          "pip install -r requirements-dev.txt")
        sdrs, sirs, sars = _approx_metrics(ref, est)
        for idx, stem in enumerate(STEMS):
            result[f"sdr_{stem}"] = round(sdrs[idx], 2)
            result[f"sir_{stem}"] = round(sirs[idx], 2)
            result[f"sar_{stem}"] = round(sars[idx], 2)

    for idx, stem in enumerate(STEMS):
        a, b = ref[idx], est[idx]
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            result[f"corr_{stem}"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
    return result


# ============================================================
#  Прогон
# ============================================================
def list_tracks(data_dir: Path) -> list[Path]:
    """Папки треков: там, где есть микс и оба эталона."""
    tracks: list[Path] = []
    if not data_dir.exists():
        return tracks
    for path in sorted(data_dir.iterdir()):
        if not path.is_dir():
            continue
        if all((path / f"{name}.wav").exists() for name in ("mix", "vocals", "instrumental")):
            tracks.append(path)
        else:
            logger.warning("Пропускаю %s: нужны mix.wav, vocals.wav, instrumental.wav", path)
    return tracks


def engine_model_present(engine_id: str) -> tuple[bool, str]:
    """
    Есть ли модель на диске. Нужна, чтобы бенчмарк не тянул гигабайты молча:
    отсутствующую модель пропускаем с понятным сообщением.
    """
    try:
        spec = get_engine(engine_id)
    except KeyError:
        return False, "неизвестный движок"

    if spec.kind == "bs":
        model = MODELS_DIR / spec.model
        if not model.exists():
            return False, f"нет файла модели {model}"
    if spec.kind == "hybrid":
        bs_model = MODELS_DIR / spec.model
        if not bs_model.exists():
            return False, f"нет файла модели {bs_model}"
    return True, ""


def run_engine(
    engine_id: str,
    tracks: list[Path],
    out_dir: Path,
    keep_stems: bool,
    limit_sec: float,
) -> dict | None:
    """Прогоняет один движок по всем трекам и возвращает его результаты."""
    spec = get_engine(engine_id)
    print(f"\n=== {spec.label} [{spec.id}] ===")

    try:
        separator = make_separator(spec)
    except Exception as e:
        print(f"  движок не собрался: {e}")
        return None

    rows = []
    for track_dir in tracks:
        mix, _ = load_audio(track_dir / "mix.wav", "микс")
        ref_v, _ = load_audio(track_dir / "vocals.wav", "эталон вокала")
        ref_i, _ = load_audio(track_dir / "instrumental.wav", "эталон минуса")

        if limit_sec:
            n = int(limit_sec * BASE_SR)
            mix, ref_v, ref_i = mix[:n], ref_v[:n], ref_i[:n]

        # Микс для движка — отдельный файл: движок читает его сам
        work_dir = out_dir / "separated" / spec.id.replace(":", "_").replace("+", "__")
        work_dir.mkdir(parents=True, exist_ok=True)
        mix_path = work_dir / f"{track_dir.name}_mix.wav"
        sf.write(str(mix_path), mix, BASE_SR, subtype="PCM_16")

        started = time.perf_counter()
        try:
            result = separator.separate(mix_path, work_dir)
        except Exception as e:
            logger.exception("Прогон %s на %s упал", spec.id, track_dir.name)
            print(f"  {track_dir.name}: ошибка — {e}")
            rows.append({"track": track_dir.name, "error": str(e)})
            continue
        elapsed = time.perf_counter() - started

        est_v, _ = load_audio(result["vocals"], "вокал модели")
        est_i, _ = load_audio(result["instrumental"], "минус модели")

        if limit_sec:
            n = int(limit_sec * BASE_SR)
            est_v, est_i = est_v[:n], est_i[:n]

        metrics = compute_metrics(ref_v, ref_i, est_v, est_i)
        metrics.update({"track": track_dir.name, "engine": spec.id,
                        "seconds": round(elapsed, 1),
                        "duration_sec": round(min(len(mix), len(ref_v)) / BASE_SR, 1)})
        rows.append(metrics)
        print(f"  {track_dir.name}: SDR вокал {metrics['sdr_vocals']:+6.2f} дБ, "
              f"минус {metrics['sdr_instrumental']:+6.2f} дБ, {elapsed:5.1f} с")

        if not keep_stems:
            for path in (mix_path, result["vocals"], result["instrumental"]):
                try:
                    Path(path).unlink()
                except Exception:
                    logger.debug("Не удалось удалить %s", path)

    return {"engine": spec.id, "label": spec.label, "tracks": rows}


def summarize(results: list[dict], out_dir: Path) -> Path:
    """Сводная таблица по всем движкам: средние метрики и время."""
    lines = [
        "# Бенчмарк моделей разделения",
        "",
        "Метрики: **SDR** — похожесть на эталон, **SIR** — подавление чужого источника,",
        "**SAR** — сколько артефактов добавила модель. Везде больше = лучше, единица — дБ.",
        "Время — полное время разделения трека (включая загрузку модели).",
        "",
        "| Модель | SDR вокал | SDR минус | SIR вокал | SIR минус | SAR вокал | SAR минус | Время, с |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for res in results:
        rows = [r for r in res["tracks"] if "error" not in r]

        def mean(key: str) -> float | None:
            values = [r[key] for r in rows if key in r]
            return sum(values) / len(values) if values else None

        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:+.2f}"

        seconds = mean("seconds")
        lines.append(
            f"| {res['label']} | {fmt(mean('sdr_vocals'))} | {fmt(mean('sdr_instrumental'))} "
            f"| {fmt(mean('sir_vocals'))} | {fmt(mean('sir_instrumental'))} "
            f"| {fmt(mean('sar_vocals'))} | {fmt(mean('sar_instrumental'))} "
            f"| {'—' if seconds is None else f'{seconds:.1f}'} |"
        )

    sources = {r.get("metrics_source") for res in results for r in res["tracks"]}
    lines += [
        "",
        f"Источник метрик: {', '.join(sorted(s for s in sources if s)) or '—'}",
        f"Треков в прогоне: {max((len(r['tracks']) for r in results), default=0)}",
        "",
    ]
    if "approx" in sources:
        lines += [
            "> ⚠️ mir_eval не установлен — метрики упрощённые (без подбора фильтров).",
            "> Установите: `.\\\\.venv312\\\\Scripts\\\\python.exe -m pip install -r requirements-dev.txt`",
            "",
        ]

    lines += [
        "## Что важно понимать",
        "",
        "* На синтетических эталонах (`tools/make_benchmark_data.py`) числа показывают,",
        "  как модели справляются с искусственным сигналом, а не с музыкой.",
        "  Для честного сравнения положите в `benchmark_data/` свои треки с известными",
        "  стемами — формат тот же: `mix.wav`, `vocals.wav`, `instrumental.wav`.",
        "* SDR выше ~15 дБ на слух уже почти не отличается, а разница 0.2 дБ — это шум",
        "  измерения: смотрите на разницу от 1 дБ.",
        "* Разделение вокала и минуса не бывает идеальным: часть вокала всегда остаётся",
        "  в минусе (bleed), и наоборот.",
        "",
    ]

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SDR/SIR/SAR-бенчмарк движков AST")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "benchmark_data"),
                        help="папка с эталонными треками")
    parser.add_argument("--engines", nargs="*", default=None,
                        help="id движков или подписи из интерфейса (по умолчанию пресет)")
    parser.add_argument("--out", default=str(LOGS_DIR / "benchmark"), help="куда писать отчёты")
    parser.add_argument("--limit", type=float, default=0.0,
                        help="брать первые N секунд каждого трека (0 — весь трек)")
    parser.add_argument("--no-keep", action="store_true",
                        help="не сохранять разделённые стемы (только метрики)")
    parser.add_argument("--list", action="store_true", help="показать движки и выйти")
    args = parser.parse_args(argv)

    from core.logging_config import setup_logging
    setup_logging()

    if args.list:
        print(f"Всего движков: {len(ENGINES)}\n")
        for spec in ENGINES.values():
            ok, why = engine_model_present(spec.id)
            mark = "✓" if ok else "—"
            print(f" {mark} {spec.id}\n     {spec.label}" + ("" if ok else f"  ({why})"))
        return 0

    data_dir = Path(args.data)
    tracks = list_tracks(data_dir)
    if not tracks:
        print(f"Нет эталонных треков в {data_dir}")
        print("Сгенерировать синтетические: "
              ".\\.venv312\\Scripts\\python.exe tools\\make_benchmark_data.py")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Треков: {len(tracks)} ({data_dir})")
    print(f"Отчёты: {out_dir}")

    requested = args.engines or DEFAULT_ENGINES
    results: list[dict] = []
    for engine_id in requested:
        try:
            resolved = resolve_engine_id(engine_id)
        except KeyError as e:
            print(f"\n=== {engine_id}: {e} — пропускаю ===")
            continue

        ok, why = engine_model_present(resolved)
        if not ok:
            print(f"\n=== {get_engine(resolved).label}: {why} — пропускаю ===")
            continue

        result = run_engine(resolved, tracks, out_dir, keep_stems=not args.no_keep,
                            limit_sec=args.limit)
        if result is None:
            continue
        results.append(result)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        (out_dir / f"benchmark_{resolved.replace(':', '_').replace('+', '__')}.json"
         ).write_text(payload, encoding="utf-8")

    if not results:
        print("\nНи один движок не отработал — смотреть logs/ast.log")
        return 1

    summary = summarize(results, out_dir)
    print(f"\nСводка: {summary}")
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
