"""
BS-Roformer / Mel-Band Roformer — движок разделения через CLI audio-separator
(переехал из core/separator_bs.py без изменения логики).

Поддерживает 2-stem и 6-stem модели. Список доступных моделей — в реестре
core/engines/__init__.py (ENGINES).

Ключевое про нормализацию: CLI audio-separator по умолчанию сам приводит КАЖДЫЙ
стем к пику 0.9 (--normalization 0.9), из-за чего вокал и минус перестают быть
комплементарными. Поэтому мы запускаем CLI с --normalization 1.0 (встроенная
нормализация фактически отключена), а уровень выставляем сами — ОДНИМ общим
коэффициентом на пару (core.dsp.normalize_pair).

Постобработка минуса:
  - HPF 25 Гц (2-й порядок) — убирает инфразвуковой гул.
  - Peaking EQ 2.5 кГц — ОПЦИОНАЛЬНО и по умолчанию ВЫКЛЮЧЕН (notch_enabled).
  - LPF 19 кГц (2-й порядок) — почти не режет верх.
"""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from core.audio_utils import get_output_sample_rate
from core.config import get_config
from core.engines.base import EngineCancelled, ProgressCb, Separator
from core.paths import MODELS_DIR
from core.dsp import (
    high_pass,
    low_pass,
    normalize_pair_files,
    pair_peak,
    peaking_eq,
)
from core.utils import cleanup_temp_dirs
from core.stems import StemSet

logger = logging.getLogger(__name__)

# ============================================================
#  Постобработка минуса
# ============================================================
INSTR_HPF_HZ = 25.0
INSTR_HPF_ORDER = 2

# Notch (peaking EQ) на минусе. По умолчанию ВЫКЛЮЧЕН (notch_enabled в
# config/settings.json): провал в 2-3 кГц — это гитары, струнные и атака малого,
# то есть самая слышимая зона.
INSTR_NOTCH_HZ = 2500.0
INSTR_NOTCH_Q = 1.0
# Честная глубина в дБ на 2.5 кГц. Раньше параметр означал коэффициент
# смешивания, и -1.5 давало фактический провал около -16 dB.
INSTR_NOTCH_GAIN_DB_REAL = -1.5

INSTR_LPF_HZ = 19000.0
INSTR_LPF_ORDER = 2

# До какого пика опускать минус, если фильтры дали перерегулирование.
# 0.99 вместо 1.0 — маленький запас, чтобы округление в PCM_16 не создало
# полношкальные сэмплы.
CLIP_SAFETY_PEAK = 0.99

# Встроенная нормализация CLI отключена: 1.0 — максимально допустимое значение,
# при нём стемы не подтягиваются к 0.9 и остаются комплементарными.
CLI_NORMALIZATION = 1.0

# При суммировании нескольких стемов в один минус сумма может превысить 1.0,
# поэтому после amix стоит лимитер.
#
# ⚠️ level=disabled обязателен: по умолчанию alimiter ВКЛЮЧАЕТ авто-нормализацию
# выхода и возвращает пик обратно к 1.0 — то есть клиппинг остаётся, только
# «подтянутый». С level=disabled лимитер делает ровно то, что нужно: не пускает
# сигнал выше limit. Проверено на сумме 4 стемов: было 7800 обрезанных сэмплов,
# стало 0 при пике 0.95.
SUM_LIMITER_LIMIT = 0.95
SUM_LIMITER_ATTACK_MS = 5
SUM_LIMITER_RELEASE_MS = 50


def _render_cli_failure(exe: Path, code: int, lines: list[str], cmd: list[str]) -> str:
    """
    Понятный текст ошибки запуска CLI: что запускали, чем кончилось, что делать.

    Раньше при пустом выводе пользователь видел только «упал с кодом 1» и пустую
    строку — то есть не видел вообще ничего.
    """
    parts = [
        f"audio-separator завершился с кодом {code}.",
        "",
        f"Программа: {exe}",
        f"Команда:   {' '.join(cmd)}",
        "",
    ]
    if lines:
        tail = lines[-20:]
        parts.append(f"Последние {len(tail)} строк вывода (stdout и stderr вместе):")
        parts.extend(f"  {ln}" for ln in tail)
    else:
        parts.extend([
            "Вывод пустой: программа не успела напечатать ни строки.",
            "",
            "Возможные причины:",
            "  1) .venv312 переехал и .exe-обёртки сломаны — см. docs\\FIX_VENV.md;",
            "  2) audio-separator не установлен или стоит в другом окружении;",
            "  3) не загрузилась модель или отвалился провайдер ONNX Runtime (CUDA).",
            "",
            "Проверить вручную (PowerShell, из папки проекта):",
            f'  & "{exe}" --help',
        ])
    return "\n".join(parts)


class BSSeparationError(Exception):
    """Ошибка разделения через BS-Roformer."""


# .exe-обёртки консольных скриптов хранят ВНУТРИ зашитый путь к python'у на момент
# установки. Если проект или .venv312 переехали, путь ведёт в никуда и обёртка молча
# падает с кодом 1, не выводя ни строки. Достаём путь из файла и проверяем его.
_SHEBANG_RE = re.compile(rb"#!([^\r\n\x00]{1,260}?\.exe)", re.IGNORECASE)

# Куда отправлять человека за инструкцией, если обёртки мертвы
FIX_VENV_HINT = (
    "Похоже, .venv312 переехал вместе с проектом, и обёртки .exe остались\n"
    "от старого пути. Пошаговая инструкция: docs\\FIX_VENV.md"
)


def find_dead_shebang(exe: Path) -> str | None:
    """
    Возвращает несуществующий путь к python'у, зашитый в .exe-обёртку.

    None — если обёртка в порядке или это не обёртка (обычная программа).
    """
    try:
        blob = exe.read_bytes()
    except Exception:
        return None  # прочитать не вышло — решим по факту запуска
    match = _SHEBANG_RE.search(blob)
    if not match:
        return None
    target = match.group(1).decode("utf-8", errors="replace")
    return None if Path(target).exists() else target


# Признак для создания процесса в отдельной группе (Windows): тогда сигнал
# завершения уходит именно ему, а не всем консольным программам сразу.
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _kill_process_tree(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """
    Гасит дочерний процесс ВМЕСТЕ с его потомками.

    Одного proc.terminate() мало: audio-separator внутри себя запускает ffmpeg и
    python-воркеры, и они остаются висеть в диспетчере задач, продолжая держать
    GPU. На Windows для дерева есть штатная команда taskkill /T.
    """
    if proc.poll() is not None:
        return

    if sys.platform.startswith("win"):
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, text=True, timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            logger.exception("taskkill не сработал — завершаю процесс напрямую")

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Процесс не отозвался на terminate — убиваю принудительно")
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.error("Процесс не удалось завершить даже через kill")
        return

    try:
        proc.wait(timeout=timeout)
    except Exception:
        pass


def assert_launcher_alive(exe: Path) -> None:
    """
    Заранее проверяет, что .exe вообще способен запуститься.

    Лучше показать человеку понятную причину до запуска, чем потом гадать,
    почему «упал с кодом 1» и в выводе пусто.
    """
    dead = find_dead_shebang(exe)
    if dead is None:
        return
    logger.error(
        "Обёртка audio-separator указывает на несуществующий python",
        extra={"ctx_exe": str(exe), "ctx_dead_target": dead},
    )
    raise BSSeparationError(
        f"audio-separator.exe не может запуститься.\n\n"
        f"Внутри файла зашит путь:\n  {dead}\n"
        f"а такого файла больше нет.\n\n"
        f"{FIX_VENV_HINT}"
    )


class BSEngine:
    """
    Движок BS-Roformer. Реализует договор Separator (см. core/engines/base.py).

    Старое имя класса — BSAudioSeparator (оставлено ниже как совместимость).
    """

    name = "BS-Roformer"

    def __init__(self, model_name: str = "BS-Roformer-SW.ckpt",
                 normalize_pair_output: bool = True) -> None:
        # DECISION: список моделей больше не хранится внутри движка — за него
        # отвечает реестр (core/engines). Проверяем только пустое значение.
        if not model_name:
            raise BSSeparationError("Не указана модель BS-Roformer")
        self.model_name = model_name
        # False — уровень выставит сборщик финальной пары (гибрид)
        self.normalize_pair_output = normalize_pair_output
        # Процесс CLI, пока он работает. Нужен, чтобы уметь его прервать (cancel).
        self._current_proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._cancelled = False
        logger.info("BSEngine init", extra={"ctx_model": model_name})

    # ---------- отмена ----------
    def cancel(self) -> bool:
        """
        Прерывает работу CLI, если он запущен. True — процесс был и его погасили.

        Без этого «Отмена» останавливала только ожидание: сам audio-separator.exe
        продолжал грузить GPU до конца, а при закрытии окна висел в диспетчере
        задач. Убиваем ДЕРЕВО процессов: CLI запускает внутри себя ffmpeg и
        python-воркеры, и без этого они остались бы висеть.
        """
        self._cancelled = True
        with self._proc_lock:
            proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return False
        logger.warning("Отмена: завершаю audio-separator.exe (pid=%s)", proc.pid)
        try:
            _kill_process_tree(proc)
        except Exception:
            logger.exception("Не удалось завершить audio-separator.exe")
            return False
        finally:
            with self._proc_lock:
                self._current_proc = None
        return True

    def is_running(self) -> bool:
        """Работает ли сейчас дочерний CLI."""
        with self._proc_lock:
            proc = self._current_proc
        return proc is not None and proc.poll() is None

    def _throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise EngineCancelled("Разделение отменено")

    # ---------- публичное API ----------
    def separate(self, input_path, output_dir, progress_cb=None) -> StemSet:
        self._cancelled = False        # новая задача — прошлая отмена не считается
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
                logger.exception("Не удалось удалить старый FLAC: %s", old)

        if progress_cb:
            progress_cb(0.05, f"BS-Roformer ({self.model_name})")

        self._throw_if_cancelled()
        audio_sep_exe = self._find_audio_separator_exe()
        models_dir = self._ensure_models_dir()
        cmd = [
            str(audio_sep_exe),
            str(in_path),
            "--model_filename", self.model_name,
            "--output_dir", str(tmp_dir),
            "--normalization", str(CLI_NORMALIZATION),
            # Модели кладём в models/ проекта. Без этого CLI берёт свой каталог
            # по умолчанию (C:\tmp\audio-separator-models или /tmp на другом диске),
            # который зависит от текущего диска и легко теряется.
            "--model_file_dir", str(models_dir),
        ]
        logger.info("Запуск audio-separator",
                    extra={"ctx_cmd": " ".join(cmd), "ctx_cmd_list": cmd})

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
                # Отдельная группа процессов: при отмене гасим дерево целиком
                creationflags=_CREATE_NEW_PROCESS_GROUP,
            )
        except FileNotFoundError:
            raise BSSeparationError(
                f"Не удалось запустить audio-separator:\n  {audio_sep_exe}\n\n"
                "Файла нет или он не запускается. Проверь установку:\n"
                "  python -m pip install audio-separator[gpu]\n"
                "Если проект переезжал — см. docs\\FIX_VENV.md"
            )

        # Запоминаем процесс: пока он жив, его можно прервать (см. cancel)
        with self._proc_lock:
            self._current_proc = proc

        last_lines = []
        try:
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
        finally:
            with self._proc_lock:
                self._current_proc = None

        # Пользователь нажал «Отмена» — это не ошибка, а намеренное действие
        if self._cancelled:
            raise EngineCancelled(f"Разделение отменено: {self.model_name}")

        if proc.returncode != 0:
            logger.error(
                "audio-separator упал",
                extra={"ctx_code": proc.returncode, "ctx_exe": str(audio_sep_exe),
                       "ctx_output_lines": len(last_lines)},
            )
            raise BSSeparationError(_render_cli_failure(
                exe=audio_sep_exe, code=proc.returncode, lines=last_lines, cmd=cmd,
            ))

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

            # Постобработка минуса (фильтры, без изменения уровня)
            if progress_cb:
                progress_cb(0.92, "Постобработка минуса")
            self._postprocess_instrumental(instr_wav)

            # Нормализация — ОДИН общий коэффициент на пару вокал+минус
            if self.normalize_pair_output:
                normalize_pair_files(vocals_wav, instr_wav)
            else:
                # Уровень выставит сборщик финальной пары (гибрид).
                # Перегруза тут нет: вокал приходит из ffmpeg в PCM_16,
                # а минус после постобработки не превышает 1.0.
                logger.info(
                    "Нормализация пары отложена: уровень выставит сборщик финальной пары")

        except BSSeparationError:
            raise
        except Exception as e:
            raise BSSeparationError(f"Ошибка конвертации: {e}") from e

        if progress_cb:
            progress_cb(1.0, "Готово")

        # Убираем за собой FLAC-и: они уже сконвертированы и только занимают место.
        # При ошибке (выше) папку НЕ трогаем — по ней удобно разбираться.
        if get_config().keep_temp_dirs:
            logger.info("keep_temp_dirs=True — папка %s оставлена для отладки", tmp_dir)
        else:
            cleanup_temp_dirs([tmp_dir])

        logger.info(
            "BS-Roformer разделение завершено",
            extra={"ctx_vocals": str(vocals_wav), "ctx_instr": str(instr_wav)},
        )
        return StemSet(vocals=vocals_wav, instrumental=instr_wav)

    # ---------- поиск audio-separator.exe ----------
    @staticmethod
    def _ensure_models_dir() -> Path:
        """
        Каталог моделей BS-Roformer — models/ в корне проекта.

        Раньше CLI складывал модели в свой каталог по умолчанию
        (/tmp/audio-separator-models, а на Windows — относительно текущего диска),
        из-за чего после переезда проекта модели качались заново.
        """
        models_dir = MODELS_DIR
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("Не удалось создать каталог моделей %s", models_dir)
        logger.info("Модели BS: %s", models_dir, extra={"ctx_models_dir": str(models_dir)})
        return models_dir

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
                # Проверяем ДО запуска: жива ли обёртка (см. docs/FIX_VENV.md)
                assert_launcher_alive(c)
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
            "-ar", str(get_output_sample_rate()),
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise BSSeparationError(
                f"ffmpeg не смог сконвертировать {src.name}: {result.stderr[-500:]}"
            )

    @staticmethod
    def _sum_flacs_to_wav(sources: list[Path], dst: Path) -> None:
        """
        Складывает несколько стемов в один минус.
        amix с normalize=0 не уменьшает громкость, поэтому сумма может выйти
        за 1.0 — сразу за ним стоит лимитер, иначе запись в pcm_s16le клиппует.
        """
        if not sources:
            raise BSSeparationError("Нет файлов для суммирования")

        cmd = ["ffmpeg", "-y", "-v", "error"]
        for src in sources:
            cmd += ["-i", str(src)]
        n = len(sources)
        # level=disabled — выключаем авто-нормализацию лимитера (см. константы выше)
        limiter = (
            f"alimiter=level=disabled:limit={SUM_LIMITER_LIMIT}"
            f":attack={SUM_LIMITER_ATTACK_MS}:release={SUM_LIMITER_RELEASE_MS}"
        )
        cmd += [
            "-filter_complex", f"amix=inputs={n}:normalize=0,{limiter}",
            "-c:a", "pcm_s16le",
            "-ar", str(get_output_sample_rate()),
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
        Постобработка минуса (уровень не трогает — им заведует пара-нормализация):
          1. HPF 25 Гц (2-й порядок) — убирает инфразвук.
          2. Peaking EQ 2.5 кГц — только если notch_enabled в settings.json.
          3. LPF 19 кГц (2-й порядок) — почти не режет верх.
        """
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            logger.warning("Не удалось прочитать WAV: %s", e)
            return

        # Работаем в форме (channels, samples) — так удобнее общим фильтрам
        ch_major = data.T

        # ---------- 1) HPF 25 Гц ----------
        ch_major = high_pass(ch_major, sr, INSTR_HPF_HZ, INSTR_HPF_ORDER)

        # ---------- 2) Peaking EQ 2.5 кГц (по умолчанию выключен) ----------
        if get_config().notch_enabled:
            ch_major = peaking_eq(
                ch_major, sr,
                INSTR_NOTCH_HZ, INSTR_NOTCH_Q, INSTR_NOTCH_GAIN_DB_REAL,
            )
        else:
            logger.info("Notch выключен (notch_enabled=false в settings.json) — "
                        "минус не фильтруется")

        # ---------- 3) LPF 19 кГц ----------
        ch_major = low_pass(ch_major, sr, INSTR_LPF_HZ, INSTR_LPF_ORDER)

        # Фильтры дают перерегулирование: если стем пришёл с пиком ровно 0 dBFS
        # (CLI с --normalization 1.0), после фильтров он уходит выше 1.0.
        # Раньше это резалось np.clip — плоские верхушки в самых громких местах.
        # Теперь опускаем ВЕСЬ сигнал одним коэффициентом: форма не портится.
        peak = pair_peak({"data": ch_major})
        if peak > 1.0:
            g = CLIP_SAFETY_PEAK / peak
            over = int(np.sum(np.abs(ch_major) > 1.0))
            logger.warning(
                "Перегруз фильтров %.4f — масштабирую до %.2f (%.2f dB), "
                "сэмплов выше 1.0 было %d",
                peak, CLIP_SAFETY_PEAK, 20.0 * math.log10(g), over,
                extra={"ctx_peak": round(peak, 4),
                       "ctx_gain_db": round(20.0 * math.log10(g), 2),
                       "ctx_samples_over": over},
            )
            ch_major = ch_major * g
        ch_major = np.clip(ch_major, -1.0, 1.0)   # финальная страховка, обычно no-op

        sf.write(str(path), ch_major.T, sr, subtype="PCM_16")
        logger.info("Постобработка минуса завершена",
                    extra={"ctx_file": str(path), "ctx_peak": round(peak, 4)})


# ---------- Совместимость со старыми именами ----------
# Раньше модуль назывался core/separator_bs.py, а класс — BSAudioSeparator.
BSAudioSeparator = BSEngine

__all__ = [
    "BSEngine", "BSAudioSeparator", "BSSeparationError",
    "find_dead_shebang", "assert_launcher_alive", "FIX_VENV_HINT",
    "INSTR_HPF_HZ", "INSTR_HPF_ORDER", "INSTR_NOTCH_HZ", "INSTR_NOTCH_Q",
    "INSTR_NOTCH_GAIN_DB_REAL", "INSTR_LPF_HZ", "INSTR_LPF_ORDER",
    "CLIP_SAFETY_PEAK", "CLI_NORMALIZATION", "SUM_LIMITER_LIMIT",
    "SUM_LIMITER_ATTACK_MS", "SUM_LIMITER_RELEASE_MS",
]
