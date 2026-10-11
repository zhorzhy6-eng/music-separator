r"""
Полный QA-прогон AST: все вкладки на реальных песнях.

Запуск из корня проекта:
    .\.venv312\Scripts\python.exe tools\qa_full.py                 # весь прогон
    .\.venv312\Scripts\python.exe tools\qa_full.py --stage env     # только окружение
    .\.venv312\Scripts\python.exe tools\qa_full.py --stage s1      # одна песня
    .\.venv312\Scripts\python.exe tools\qa_full.py --stage edge    # краевые случаи
    .\.venv312\Scripts\python.exe tools\qa_full.py --no-shots      # без скриншотов

Этапы (--stage):
    env   — окружение, песни, состояние до прогона
    s1    — песня 1: пресет «минус как в оригинале» + все вкладки, плеер, тема
    s2    — песня 2: пресет «убрать вокал (для караоке)»
    s3    — песня 3: пресет «быстро и просто» (Demucs)
    edge  — краевые случаи: WAV/FLAC вход, отмена, смена трека
    close — закрытие окна во время задачи (уничтожает окно, поэтому последним)
    all   — всё подряд

Как это работает (и почему так):
  * поднимается НАСТОЯЩЕЕ окно `ui.app_window.AppWindow` — то же, что видит человек;
  * нажимаются те же кнопки: у CTkButton вызывается `invoke()` (он сам смотрит,
    не выключена ли кнопка), у пресетов моделей — тот же обработчик, что висит на
    клике по карточке, у выпадающего списка «Слушать» — тот же колбэк, что и при
    выборе пункта. Мышь не двигается: робот этого не умеет;
  * все `messagebox` перехватываются в журнал: иначе прогон встал бы на первом же
    «Готово» и ждал нажатия «ОК»;
  * события воркера обрабатываются штатным циклом окна (`app.update()`), поэтому
    проверяется связка «воркер → интерфейс», а не только ядро;
  * каждое состояние снимается скриншотом в logs/qa_full_<дата>/screenshots/.

Результаты: logs/qa_full_<дата>/report.md (человеку), results.json (машине),
progress.md (промежуточный ход), _stdout.log (сырой вывод — пишет вызывающий).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SONGS_DIR = PROJECT_ROOT / "Песни для тестов"


def find_song(name: str) -> Path:
    """
    Ищет песню в папке, сравнивая имена в одной форме Unicode.

    Зачем: на диске «царь зверей -.mp3» записано с комбинируемой краткой (NFD),
    а в коде имя набрано готовой буквой «й» (NFC). Выглядят они одинаково, но
    это разные строки — и обычное сравнение путей даёт «файла нет».
    """
    import unicodedata

    target = unicodedata.normalize("NFC", name)
    if SONGS_DIR.exists():
        for candidate in SONGS_DIR.glob("*.mp3"):
            if unicodedata.normalize("NFC", candidate.name) == target:
                return candidate
    return SONGS_DIR / name


# Песни для прогона. Выбор не случаен:
#   * Skyfall (минусовка) — пресет «минус как в оригинале»: вокала в треке нет,
#     зато плотная музыка, и видно, не «съедает» ли модель инструменты;
#   * Funeral Song — пресет «убрать вокал (караоке)»: настоящий вокал, который надо убрать;
#   * Дед Максим — пресет «быстро и просто» (Demucs): короткая песня, проверяем,
#     что путь работает и укладывается в минуту.
SONG_1 = find_song("Adele_-_Skyfall_minus_76810390.mp3")
SONG_2 = find_song("Mono_Inc_-_Funeral_Song_57628408.mp3")
SONG_3 = find_song("Красная Плесень - Дед Максим.mp3")
# Песня, которую НИКОГДА не разделяли: нужна для проверки «загрузили новый трек —
# кнопки выключились». У остальных в output/ уже лежат стемы от прошлых прогонов.
SONG_CLEAN = find_song("царь зверей -.mp3")

# Пресеты из реестра (core.engines.PRESETS) — по названию, чтобы прогон не
# разъехался, если порядок пресетов когда-нибудь поменяют.
PRESET_INSTRUMENTAL = "Получить минус как в оригинале"
PRESET_KARAOKE = "Убрать вокал (для караоке)"
PRESET_FAST = "Быстро и просто"

# Какие движки ждём от пресетов (см. PRESETS в core/engines/__init__.py)
EXPECT_INSTRUMENTAL = "bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt"   # Viperx-1297
EXPECT_KARAOKE = "bs:mel_band_roformer_karaoke_becruily.ckpt"          # Becruily karaoke
EXPECT_FAST = "demucs:htdemucs_ft"

# Замена для песни 2: модель из ТОЙ ЖЕ группы «🎤 Убрать вокал (для караоке)»,
# но уже лежащая в models/. Нужна потому, что Becruily karaoke отсутствует
# локально, а тянуть 1.6 ГБ автономный прогон не имеет права (> 100 МБ).
KARAOKE_LOCAL_FALLBACK = "bs:melband_roformer_instvox_duality_v2.ckpt"

# Длительность клипа для краевых случаев. Полное разделение WAV и FLAC — это ещё
# по 3–5 минут GPU на каждый, а проверяем мы ФОРМАТ ВХОДА, а не качество:
# ветки кода для WAV (без перекодирования) и FLAC (с перекодированием) от длины
# не зависят. 25 секунд — с запасом на «модель вообще стартовала».
EDGE_CLIP_SEC = 25.0

# Сколько ждём задачу. На GTX 1660 SUPER 4-минутный трек BS-моделью — 3–8 минут;
# 40 минут покрывают и первый запуск со скачиванием модели.
TIMEOUT_SEPARATE = 40 * 60
TIMEOUT_TRANSCRIBE = 15 * 60
TIMEOUT_PIANO = 20 * 60
TIMEOUT_ANALYZE = 10 * 60

# Допуск на длину стемов: файлы пишутся PCM_16, длина совпадает до кадра,
# но ресемплинг может дать расхождение в пару кадров.
FRAMES_TOLERANCE = 8

SEVERITY_CRIT = "🔴 критично"
SEVERITY_MID = "🟠 средне"
SEVERITY_MINOR = "🟡 мелко"

# Что мы вправе считать «правильным» пиком пары (target_peak из настроек)
EXPECT_PEAK = 0.891


# =============================================================
#  Баги, найденные этим прогоном
# =============================================================
# DECISION: список ведётся руками. Драйвер умеет нажимать кнопки и мерить, но
# «это баг, а не задумано» решает человек — автоматически такое не отличить от
# ожидаемого поведения. Список попадает в report.md отдельной таблицей, и туда
# же дописываются баги, найденные во время прогона (journal.bug).
KNOWN_BUGS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "BS-разделение при запуске без консоли",
        "main.py, core/logging_config.py",
        "При запуске через pythonw.exe (AST.bat) разделение падало с "
        "«'NoneType' object has no attribute 'write'»: у процесса без консоли "
        "sys.stdout и sys.stderr равны None, а в них пишет и логирование, и "
        "полоска прогресса внутри audio-separator.",
        SEVERITY_CRIT, "да, 100% (отсоединённый pythonw.exe)",
        "да — немые потоки подменяются файлами logs/stdout.log и logs/stderr.log",
    ),
    (
        "Окно теряло клавиатуру после списка моделей",
        "ui/engine_selector.py, ui/app_window.py, ui/widgets.py",
        "После открытия и закрытия списка моделей (или меню «⚙») окно оставалось "
        "без фокуса: focus_get() → None, и горячие клавиши плеера (→ ← Shift+→ "
        "Space Home End) перестали работать, пока не щёлкнешь по окну. "
        "Всплывающее окно забирало фокус, а после destroy() Tk его не возвращал.",
        SEVERITY_MID, "да, стабильно",
        "да — фокус возвращается окну (ui/widgets.restore_main_focus)",
    ),
    (
        "Супер-анализ пугал числами там, где измерять нечего",
        "core/super_analyzer.py, core/reports.py, ui/app_window.py",
        "На минусовке (в треке нет вокала) вердикт сообщал «Голосовой отпечаток "
        "заметен (VPI 139.8 dB)», «Минус темнеет (−65.1 %)» и «Подозрительных "
        "секунд: 88». Всё это посчитано от тишины: вокал-стем имел RMS −55.9 dBFS, "
        "а опорой сравнения служили 3 кадра из 5707, где и сам минус — нули.",
        SEVERITY_MID, "да, на любом треке без вокала",
        "да — «вокала нет» распознаётся отдельно, VPI не выдумывается",
    ),
    (
        "Строка состояния показывала техническое имя вместо понятного",
        "core/engines/bs_engine_api.py, ui/app_window.py",
        "Во время разделения было видно «Разделяем: bS-Roformer (API): "
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt — 38%… 38%»: имя файла модели "
        "вместо человеческих слов, первая буква опущена правилом «без кричащих "
        "заглавных», и процент напечатан дважды.",
        SEVERITY_MINOR, "да, всегда при BS-разделении",
        "да — «Разделяем: работает модель BS-Roformer… 38%»",
    ),
    (
        "Две шестерёнки в шапке",
        "ui/app_window.py",
        "Кнопка меню «⚙» показывала и загруженную иконку, и запасной символ «⚙» "
        "одновременно — они рисовались друг на друге. Причина: после icon_button "
        "текст кнопки переписывался на «⚙» безусловно, хотя icon_button сам "
        "подставляет символ только тогда, когда картинки нет.",
        SEVERITY_MINOR, "да, всегда (видно на скриншотах прогона)",
        "да — лишняя строка убрана, остаётся одна иконка",
    ),
    (
        "Проверка tools/ui_smoke.py падала на обычной консоли",
        "tools/ui_smoke.py",
        "Запуск из PowerShell/cmd без предварительного `chcp 65001` обрывался "
        "с UnicodeEncodeError: консоль Windows в cp1251, а в подписи движка есть "
        "«✅». Проверка падала раньше, чем что-либо проверяла.",
        SEVERITY_MINOR, "да, на консоли в cp1251",
        "да — вывод переведён в UTF-8 с заменой непередаваемых символов",
    ),
    (
        "Сохранение настроек меняет запись движка в файле",
        "ui/settings_dialog.py",
        "После «Сохранить» в config/settings.json вместо подписи "
        "«BS-Roformer: Duality-V2 ✅ баланс» лежит её id "
        "«bs:melband_roformer_instvox_duality_v2.ckpt», а также дописываются "
        "поля bs_* со значениями по умолчанию.",
        SEVERITY_MINOR, "да, всегда",
        "не чиним — безвредно: resolve_engine_id понимает обе формы, "
        "поведение не меняется",
    ),
)


# =============================================================
#  Журнал: проверки и баги
# =============================================================
@dataclass
class Check:
    stage: str
    name: str
    ok: bool | None          # True — прошло, False — нет, None — с замечанием
    note: str = ""


@dataclass
class Bug:
    title: str
    where: str
    symptom: str
    severity: str
    reproducible: str
    fixed: str = "нет"


@dataclass
class Journal:
    out_dir: Path
    checks: list[Check] = field(default_factory=list)
    bugs: list[Bug] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    section: str = "общее"
    _last_flush: float = 0.0

    def check(self, name: str, ok: bool | None, note: str = "") -> bool:
        self.checks.append(Check(self.section, name, ok, note))
        icon = {True: "✅", False: "❌", None: "⚠️"}[ok]
        line = f"  {icon} {name}"
        if note:
            line += f" — {note}"
        print(line, flush=True)
        self._flush()
        return bool(ok)

    def bug(self, title: str, where: str, symptom: str, severity: str,
            reproducible: str = "да", fixed: str = "нет") -> Bug:
        b = Bug(title, where, symptom, severity, reproducible, fixed)
        self.bugs.append(b)
        print(f"  🐞 БАГ [{severity}] {title} — {symptom}", flush=True)
        self._flush(force=True)
        return b

    def save(self, name: str, payload) -> None:
        self.data[name] = payload
        self._flush(force=True)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.ok is False)

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c.ok is None)

    def _flush(self, force: bool = False) -> None:
        """
        Промежуточные файлы: если прогон упадёт на середине, результат не потеряется.

        Пишем не на каждой проверке (их сотни, а results.json растёт), а раз в 3 с
        и всегда — после важных событий (баг, сохранение данных).
        """
        now = time.monotonic()
        if not force and now - self._last_flush < 3.0:
            return
        self._last_flush = now
        try:
            # progress.md — дешёвый, пишем часто: по нему видно ход прогона живьём.
            lines = ["# Ход QA-прогона AST", ""]
            for c in self.checks:
                icon = {True: "✅", False: "❌", None: "⚠️"}[c.ok]
                lines.append(f"- {icon} **[{c.stage}]** {c.name}"
                             + (f" — {c.note}" if c.note else ""))
            (self.out_dir / "progress.md").write_text("\n".join(lines), encoding="utf-8")

            # checks.json — машиночитаемая копия проверок. Нужна, чтобы собрать
            # отчёт заново, не гоняя прогон (см. --stage report).
            (self.out_dir / "checks.json").write_text(
                json.dumps([{"stage": c.stage, "name": c.name, "ok": c.ok,
                             "note": c.note} for c in self.checks],
                           ensure_ascii=False, indent=1),
                encoding="utf-8")

            # results.json — тяжёлый (внутри полные отчёты анализа), поэтому реже.
            if force or now - getattr(self, "_last_json", 0.0) >= 15.0:
                self._last_json = now
                (self.out_dir / "results.json").write_text(
                    json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
        except Exception:
            logging.getLogger(__name__).debug("Промежуточный отчёт не записался",
                                              exc_info=True)


# =============================================================
#  Перехват messagebox
# =============================================================
class MessageBoxSpy:
    """
    Заглушка вместо модальных окон.

    Без неё прогон встал бы на первом же «Готово» и ждал нажатия «ОК» — робот
    нажимать не умеет. Все окна записываются в журнал: по ним видно, что именно
    приложение сказало человеку.
    """

    def __init__(self) -> None:
        self.log: list[dict] = []

    def install(self) -> None:
        import tkinter.messagebox as mb

        def make(kind: str, answer=None):
            def spy(title="", message="", **kw):
                self.log.append({
                    "kind": kind, "title": str(title), "message": str(message),
                    "at": time.strftime("%H:%M:%S"),
                })
                return answer
            return spy

        mb.showinfo = make("info")
        mb.showwarning = make("warning")
        mb.showerror = make("error")
        mb.askyesno = make("askyesno", False)
        mb.askokcancel = make("askokcancel", False)
        mb.askquestion = make("askquestion", "no")

    @property
    def mark(self) -> int:
        return len(self.log)

    def since(self, mark: int) -> list[dict]:
        return self.log[mark:]

    def errors(self, mark: int = 0) -> list[dict]:
        return [m for m in self.log[mark:] if m["kind"] == "error"]

    def warnings(self, mark: int = 0) -> list[dict]:
        return [m for m in self.log[mark:] if m["kind"] == "warning"]

    def infos(self, mark: int = 0) -> list[dict]:
        return [m for m in self.log[mark:] if m["kind"] == "info"]


# =============================================================
#  Лог приложения
# =============================================================
class AppLog:
    """Чтение logs/ast.log (JSON-lines) с делением на «до/после метки»."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def mark(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def since(self, mark: int) -> list[dict]:
        """
        Записи, появившиеся после метки.

        Учитываем ротацию: если файл стал КОРОЧЕ метки — значит его перевернули,
        и читаем целиком (лучше лишние строки, чем потерянная ошибка).
        """
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        offset = mark if size >= mark else 0
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                raw = fh.read()
        except OSError:
            return []
        out: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def errors(entries: list[dict]) -> list[dict]:
        return [e for e in entries if e.get("level") in ("ERROR", "CRITICAL")]

    @staticmethod
    def warnings(entries: list[dict]) -> list[dict]:
        return [e for e in entries if e.get("level") == "WARNING"]


# =============================================================
#  Замеры
# =============================================================
def measure_wav(path: Path) -> dict:
    """Замеры аудиофайла: длина, частота, каналы, пик, RMS, клиппинг."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if data.size else 0.0
    info = sf.info(str(path))
    return {
        "path": str(path),
        "name": path.name,
        "size_mb": round(path.stat().st_size / 1048576, 2),
        "sample_rate": sr,
        "channels": int(data.shape[1]),
        "frames": int(data.shape[0]),
        "duration_sec": round(data.shape[0] / sr, 3) if sr else 0.0,
        "peak": round(peak, 6),
        "peak_db": round(20 * math.log10(peak), 2) if peak > 0 else None,
        "rms_db": round(20 * math.log10(rms), 2) if rms > 0 else None,
        "subtype": info.subtype,
        "clipped_samples": int(np.sum(np.abs(data) >= 0.999969)),
    }


def measure_midi(path: Path) -> dict:
    """Замеры MIDI: ноты, диапазон, полные дубли, перекрытия одной высоты."""
    import warnings

    import pretty_midi

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        midi = pretty_midi.PrettyMIDI(str(path))
    quirks = [str(w.message) for w in caught]

    notes = []
    for instr in midi.instruments:
        if instr.is_drum:
            continue
        notes.extend(instr.notes)

    keys = [(n.pitch, int(midi.time_to_tick(n.start)), int(midi.time_to_tick(n.end)))
            for n in notes]
    dupes = len(keys) - len(set(keys))

    by_pitch: dict[int, list[tuple[float, float]]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append((n.start, n.end))
    overlaps = 0
    for group in by_pitch.values():
        group.sort()
        for (_, prev_end), (start, _) in zip(group, group[1:]):
            if start < prev_end - 1e-6:
                overlaps += 1

    pitches = sorted({n.pitch for n in notes})
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "notes": len(notes),
        "instruments": len(midi.instruments),
        "programs": sorted({i.program for i in midi.instruments}),
        "duplicate_notes": dupes,
        "overlapping_same_pitch": overlaps,
        "unique_pitches": len(pitches),
        "pitch_min": pitches[0] if pitches else None,
        "pitch_max": pitches[-1] if pitches else None,
        "duration_sec": round(midi.get_end_time(), 2),
        "warnings": quirks[:5],
    }


def scan_json_bad(obj, path: str = "$") -> tuple[list[str], list[str]]:
    """Рекурсивно ищет NaN/Inf и null. Возвращает (nan_inf, nulls)."""
    bad: list[str] = []
    nulls: list[str] = []

    def walk(node, where: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{where}[{i}]")
        elif isinstance(node, float):
            if math.isnan(node) or math.isinf(node):
                bad.append(f"{where}={node}")
        elif node is None:
            nulls.append(where)

    walk(obj, path)
    return bad, nulls


def audio_separator_processes() -> list[str]:
    """Живые audio-separator.exe (их не должно оставаться после отмены/закрытия)."""
    try:
        res = subprocess.run(["tasklist", "/FI", "IMAGENAME eq audio-separator.exe"],
                             capture_output=True, text=True, timeout=30)
    except Exception as e:                     # noqa: BLE001 — диагностика, не проверка
        return [f"<tasklist не ответил: {e}>"]
    return [ln.strip() for ln in res.stdout.splitlines()
            if "audio-separator" in ln.lower() and ".exe" in ln.lower()]


def make_clip(src: Path, dst: Path, seconds: float, fmt: str) -> Path:
    """Короткий клип из середины трека — для проверок формата входа."""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec = {"wav": ["-c:a", "pcm_s16le"], "flac": ["-c:a", "flac"]}[fmt]
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", "30", "-t", str(seconds),
           "-i", str(src), "-ar", "44100", "-ac", "2", *codec, str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg не сделал клип: {result.stderr[:300]}")
    return dst


# =============================================================
#  Прогон
# =============================================================
class QaRun:
    def __init__(self, out_dir: Path, shots: bool = True) -> None:
        self.out = out_dir
        self.shots_dir = out_dir / "screenshots"
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(out_dir)
        self.mb = MessageBoxSpy()
        self.applog = AppLog(PROJECT_ROOT / "logs" / "ast.log")
        self.shots_enabled = shots
        self.app = None
        self.window_closed = False
        self._shot_no = 0

    # ---------- окно ----------
    def launch(self):
        """Поднимает настоящее окно приложения."""
        from ui.app_window import AppWindow

        self.mb.install()
        self.app = AppWindow()
        self.app.update()
        self._trap_window_close()
        return self.app

    def _trap_window_close(self) -> None:
        """
        Страховка: если окно закроется САМО, в отчёте будет видно, кто это сделал.

        Во время первого прогона окно закрылось на 25-й секунде разделения —
        задача осталась висеть, а прогон ждал её до таймаута. Причина была не
        видна: `_on_close` пишет в лог только «Остановка воркера». Теперь любое
        закрытие окна оставляет трассировку, и такой случай сразу объясним.
        """
        original = self.app._on_close

        def traced() -> None:
            stack = "".join(traceback.format_stack()[:-1])
            logging.getLogger("qa.driver").error(
                "Окно закрыто во время прогона. Кто вызвал _on_close:\n%s", stack)
            print("  ⚠️ окно закрылось само — трассировка в logs/ast.log", flush=True)
            self.window_closed = True
            return original()

        self.app._on_close = traced
        # protocol() хранит ссылку на метод, поставленную в __init__ — переставляем
        self.app.protocol("WM_DELETE_WINDOW", traced)

    def pump(self, seconds: float) -> None:
        """Крутит цикл окна указанное время: события Tk + события воркера."""
        end = time.monotonic() + max(0.0, seconds)
        while True:
            if self.window_closed:
                return
            self.app.update()
            self.app.update_idletasks()
            if time.monotonic() >= end:
                return
            time.sleep(0.02)

    def pump_until(self, predicate, timeout: float, tick=None, tick_every: float = 15.0):
        """Крутит цикл окна, пока predicate() не станет True. Возвращает (успех, сек)."""
        started = time.monotonic()
        last_tick = started
        while time.monotonic() - started < timeout:
            # Окно закрылось само — ждать больше нечего, иначе прогон повиснет
            # до таймаута задачи (40 минут) и всё это время ничего не делает.
            if self.window_closed:
                return False, time.monotonic() - started
            self.app.update()
            self.app.update_idletasks()
            if predicate():
                return True, time.monotonic() - started
            now = time.monotonic()
            if tick is not None and now - last_tick >= tick_every:
                tick(now - started)
                last_tick = now
            time.sleep(0.03)
        return False, time.monotonic() - started

    def log_mark(self) -> int:
        return self.applog.mark()

    def screenshot(self, name: str, full_screen: bool = False) -> str | None:
        """Скриншот окна (или всего экрана — для всплывающих списков)."""
        if not self.shots_enabled:
            return None
        try:
            from PIL import ImageGrab
        except Exception:
            return None
        self._shot_no += 1
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name)[:60]
        path = self.shots_dir / f"{self._shot_no:03d}_{safe}.png"
        try:
            if full_screen:
                img = ImageGrab.grab(all_screens=True)
            else:
                app = self.app
                app.update_idletasks()
                app.deiconify()
                app.lift()
                app.attributes("-topmost", True)
                app.focus_force()
                for _ in range(5):
                    app.update()
                    time.sleep(0.08)
                app.update_idletasks()
                x, y = app.winfo_rootx(), app.winfo_rooty()
                w, h = app.winfo_width(), app.winfo_height()
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
            img.save(path)
            print(f"    📸 {path.name}", flush=True)
            return str(path)
        except Exception:
            logging.getLogger(__name__).debug("Скриншот %s не получился", name,
                                              exc_info=True)
            return None
        finally:
            try:
                if self.app is not None and _safe_exists(self.app):
                    self.app.attributes("-topmost", False)
            except Exception:
                pass

    # ---------- задачи ----------
    def run_task(self, action, label: str, timeout: float, tick_every: float = 15.0):
        """
        Нажимает кнопку и ждёт конца задачи.

        action — то, что делает кнопка (её `invoke()` либо обработчик).
        """
        mb_mark = self.mb.mark
        before_id = self.app.current_task_id
        action()
        # DECISION: «стартовала ли задача» проверяем ДО первого update().
        # Задача ставится синхронно, но короткая (проверка нот — 0.07 с) успевает
        # завершиться внутри update(), и current_task_id обнуляется. В прогоне
        # 11.10.2026 из-за этого быстрая задача была записана как «не стартовала».
        started = (self.app.current_task_id is not None
                   and self.app.current_task_id != before_id)
        self.app.update()

        if not started:
            return {"started": False, "finished": False, "seconds": 0.0,
                    "messages": self.mb.since(mb_mark), "progress_log": [],
                    "status": self.app.status.cget("text")}

        progress_log: list[str] = []

        def tick(elapsed: float) -> None:
            text = self.app.status.cget("text")
            progress_log.append(f"{elapsed:7.1f} с | {self.app.progress.get():.2f} | {text}")
            print(f"    … {elapsed:7.1f} с | {self.app.progress.get():.2f} | {text}",
                  flush=True)

        ok, elapsed = self.pump_until(lambda: self.app.current_task_id is None,
                                      timeout, tick=tick, tick_every=tick_every)
        text = self.app.status.cget("text")
        progress_log.append(f"{elapsed:7.1f} с | {self.app.progress.get():.2f} | {text}")
        return {
            "started": True,
            "finished": ok,
            "seconds": round(elapsed, 1),
            "progress_log": progress_log,
            "messages": self.mb.since(mb_mark),
            "status": text,
            "busy": self.app.ui_state.busy,
        }

    # ---------- проверка лога ----------
    def log_scan(self, mark: int, stage: str) -> list[dict]:
        """Проверяет, что за этап в logs/ast.log не появилось ERROR."""
        entries = self.applog.since(mark)
        errors = AppLog.errors(entries)
        warnings = AppLog.warnings(entries)
        prev = self.journal.section
        self.journal.section = stage
        self.journal.check(
            "в логе нет ERROR", not errors,
            "чисто" if not errors else "; ".join(f"{e['msg'][:120]}" for e in errors[:3]))
        if warnings:
            self.journal.check(
                "предупреждения в логе", None,
                f"{len(warnings)} шт: " + "; ".join(w["msg"][:80] for w in warnings[:3]))
        self.journal.section = prev
        return entries

    def set_section(self, name: str) -> None:
        self.journal.section = name
        print(f"\n=== {name} ===", flush=True)


def _safe_exists(app) -> int | None:
    """winfo_exists после destroy() поднимает TclError — это и значит «окна нет»."""
    try:
        return app.winfo_exists()
    except Exception:
        return None


# =============================================================
#  ЭТАП 0. Окружение
# =============================================================
def stage_env(run: QaRun) -> None:
    j = run.journal
    run.set_section("Окружение")

    import platform

    info: dict = {
        "дата": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "платформа": platform.platform(),
        "интерпретатор": sys.executable,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_доступна"] = bool(torch.cuda.is_available())
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1048576)
    except Exception as e:                     # noqa: BLE001
        info["torch"] = f"ошибка: {e}"
    for module in ("demucs", "torchcrepe", "audio_separator", "soundfile", "scipy",
                   "pretty_midi", "customtkinter", "muscriptor", "sounddevice"):
        try:
            mod = __import__(module)
            info[module] = getattr(mod, "__version__", "есть")
        except Exception as e:                 # noqa: BLE001
            info[module] = f"НЕТ: {e}"
    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                             timeout=30)
        info["ffmpeg"] = res.stdout.splitlines()[0].split(" Copyright")[0] \
            if res.stdout else "нет вывода"
    except Exception as e:                     # noqa: BLE001
        info["ffmpeg"] = f"НЕТ: {e}"
    info["audio-separator.exe"] = (
        PROJECT_ROOT / ".venv312" / "Scripts" / "audio-separator.exe").exists()

    # Что уже лежит локально: от этого зависит, будет ли прогон тянуть модели
    from core.paths import MODELS_DIR
    info["модели_локально"] = sorted(p.name for p in MODELS_DIR.glob("*.ckpt"))

    songs = []
    for f in sorted(SONGS_DIR.glob("*.mp3")):
        try:
            with sf.SoundFile(str(f)) as fh:
                dur = len(fh) / fh.samplerate
        except Exception:
            dur = None
        songs.append({"имя": f.name, "мб": round(f.stat().st_size / 1048576, 2),
                      "секунд": round(dur, 1) if dur else None})
    info["песен_в_папке"] = len(songs)
    info["песни"] = songs

    j.save("окружение", info)
    for key, value in info.items():
        if key not in ("песни", "модели_локально"):
            print(f"    {key}: {value}", flush=True)
    print(f"    модели локально: {info['модели_локально']}", flush=True)

    j.check("окружение собрано", True,
            f"Python {info['python']}, torch {info.get('torch')}, "
            f"CUDA {info.get('cuda_доступна')}")
    j.check("папка «Песни для тестов» на месте", len(songs) >= 3, f"{len(songs)} mp3")
    for song in (SONG_1, SONG_2, SONG_3, SONG_CLEAN):
        j.check(f"песня найдена: {song.name}", song.exists())


# =============================================================
#  Вкладка «Разделение»
# =============================================================
def pick_preset(run: QaRun, title: str) -> str | None:
    """
    Нажимает кнопку-пресет в списке моделей — как это делает человек.

    Пресеты нарисованы не кнопками, а карточками с привязкой к клику, поэтому
    зовём ровно тот обработчик, который висит на карточке (_apply_preset).
    """
    from core.engines import PRESETS

    preset = next((p for p in PRESETS if p.title == title), None)
    if preset is None:
        run.journal.check(f"пресет «{title}» есть в реестре", False,
                          "не найден в core.engines.PRESETS")
        return None

    selector = run.app.engine_combo
    selector.open()
    run.pump(0.6)
    run.screenshot(f"selector_{title[:24]}", full_screen=True)
    selector._apply_preset(preset)
    run.pump(0.6)
    return selector.get()


def press_menu_item(run: QaRun, prefix: str) -> bool:
    """
    Нажимает пункт меню «⚙» по началу подписи — как это делает человек.

    Пункты меню — обычные кнопки во всплывающем окне, поэтому зовём их invoke()
    (он сам смотрит, не выключена ли кнопка). Возвращает True, если пункт нашли.
    """
    app = run.app
    if app._menu_popup is None:
        app._open_menu()
        run.pump(0.5)
    for button, _icon in getattr(app, "_menu_icon_widgets", []):
        if str(button.cget("text")).startswith(prefix):
            button.invoke()
            run.pump(0.5)
            return True
    app._close_menu()
    run.pump(0.3)
    return False


def load_song(run: QaRun, path: Path, shot_name: str | None = None) -> dict:
    """Загружает трек так, как это делает зона файла (DropZone → _load_file)."""
    j = run.journal
    run.app._load_file(str(path))
    run.pump(0.9)
    if shot_name:
        run.screenshot(shot_name)

    info = run.app._last_info
    j.check(f"файл «{path.name}» загружен", run.app.current_file is not None
            and Path(run.app.current_file).name == path.name)
    j.check("зона файла превратилась в карточку с именем и длительностью",
            run.app.drop.current_file is not None
            and run.app.drop._file.winfo_ismapped(),
            f"в карточке: {run.app.drop._file.name_label.cget('text')!r} / "
            f"{run.app.drop._file.meta_label.cget('text')!r}, "
            f"высота зоны {run.app.drop.cget('height')} px")
    j.check("параметры файла определены", bool(info.get("duration_sec")),
            f"{info.get('duration_sec')} с, {info.get('sample_rate')} Гц, "
            f"{info.get('channels')} кан.")
    j.check("волна нарисована", run.app.wave.has_plot,
            f"подпись: {run.app.wave.caption.cget('text')!r}")
    j.check("кнопка «Разделить» включилась",
            run.app.btn_separate.cget("state") == "normal")
    j.check("в плеер загружен исходный трек",
            str(run.app.player.current_file) == str(path),
            f"«Слушать»: {run.app.combo_listen.get()!r}")
    return info


def check_stems(run: QaRun, path: Path, stage: str,
                expect_peak: float = EXPECT_PEAK) -> dict:
    """Проверки пары вокал+минус: файлы, длины, частота, каналы, пик, клиппинг."""
    j = run.journal
    prev = j.section
    j.section = stage
    stem = path.stem
    vocals = run.app.output_dir / f"{stem}_vocals.wav"
    instr = run.app.output_dir / f"{stem}_instrumental.wav"

    if not vocals.exists() or not instr.exists():
        j.check("созданы *_vocals.wav и *_instrumental.wav", False,
                f"вокал: {vocals.exists()}, минус: {instr.exists()}")
        j.section = prev
        return {"stem": stem, "vocals_exists": vocals.exists(),
                "instrumental_exists": instr.exists()}

    v = measure_wav(vocals)
    i = measure_wav(instr)
    result = {"stem": stem, "vocals": v, "instrumental": i}

    j.check("созданы *_vocals.wav и *_instrumental.wav", True,
            f"вокал {v['size_mb']} МБ, минус {i['size_mb']} МБ")
    j.check("длины вокала и минуса совпадают",
            abs(v["frames"] - i["frames"]) <= FRAMES_TOLERANCE,
            f"{v['frames']} и {i['frames']} кадров "
            f"({v['duration_sec']} с / {i['duration_sec']} с)")
    j.check("частота дискретизации 44100 Гц",
            v["sample_rate"] == 44100 and i["sample_rate"] == 44100,
            f"{v['sample_rate']} / {i['sample_rate']}")
    j.check("оба файла стерео", v["channels"] == 2 and i["channels"] == 2,
            f"{v['channels']} / {i['channels']}")
    j.check(f"пик не выше {expect_peak} у обоих",
            v["peak"] <= expect_peak + 1e-4 and i["peak"] <= expect_peak + 1e-4,
            f"вокал {v['peak']} ({v['peak_db']} dBFS), "
            f"минус {i['peak']} ({i['peak_db']} dBFS)")
    j.check("полношкальных сэмплов (клиппинг) нет",
            v["clipped_samples"] == 0 and i["clipped_samples"] == 0,
            f"{v['clipped_samples']} / {i['clipped_samples']}")
    j.section = prev
    return result


def stage_separate(run: QaRun, path: Path, preset: str, stage: str,
                   expect_engine: str, shot_prefix: str) -> dict:
    """Общий путь: загрузка → пресет → «Разделить» → проверка файлов."""
    j = run.journal
    run.set_section(stage)

    load_song(run, path, shot_name=f"{shot_prefix}_loaded")

    engine_id = pick_preset(run, preset)
    run.screenshot(f"{shot_prefix}_preset")
    j.check(f"пресет «{preset}» выбрал ожидаемую модель",
            engine_id == expect_engine, f"выбрано: {engine_id}")
    j.check("на кнопке селектора видно понятное имя",
            bool(run.app.engine_combo.display_label()),
            f"{run.app.engine_combo.display_label()!r}")

    mark = run.log_mark()
    mb_mark = run.mb.mark
    print(f"    ⏳ разделяю «{path.name}»…", flush=True)
    res = run.run_task(run.app.btn_separate.invoke, stage, TIMEOUT_SEPARATE)
    j.save(f"{shot_prefix}_separate", res)

    j.check("задача стартовала", res.get("started"),
            "кнопка «Разделить» приняла нажатие")
    j.check("окно не закрылось само во время задачи", not run.window_closed)
    j.check("задача дошла до конца без таймаута", res.get("finished"),
            f"{res.get('seconds')} с")
    errs = [m for m in res.get("messages", []) if m["kind"] in ("error", "warning")]
    j.check("модальных окон с ошибкой не было", not errs,
            "чисто" if not errs else "; ".join(m["title"] for m in errs))
    done = [m for m in run.mb.since(mb_mark) if m["kind"] == "info"]
    del done
    run.screenshot(f"{shot_prefix}_done")

    stems = check_stems(run, path, stage)
    entries = run.log_scan(mark, stage)
    j.save(f"{shot_prefix}_log", {
        "всего": len(entries),
        "error": len(AppLog.errors(entries)),
        "warning": len(AppLog.warnings(entries)),
        "warning_тексты": [e["msg"][:200] for e in AppLog.warnings(entries)][:5],
        "время_сек": res.get("seconds"),
        "progress": res.get("progress_log"),
    })
    prev = j.section
    j.section = stage
    j.check("кнопки вернулись в рабочее состояние",
            run.app.btn_separate.cget("state") == "normal"
            and run.app.btn_cancel.cget("state") == "disabled")
    j.section = prev
    return stems


def separate_with_selected(run: QaRun, path: Path, stage: str, prefix: str) -> dict:
    """Разделение той моделью, что уже выбрана в селекторе (для песни 2)."""
    j = run.journal
    run.set_section(stage)
    load_song(run, path, shot_name=f"{prefix}_loaded")
    mark = run.log_mark()
    mb_mark = run.mb.mark
    res = run.run_task(run.app.btn_separate.invoke, stage, TIMEOUT_SEPARATE)
    j.save(f"{prefix}_separate", res)
    errs = [m for m in run.mb.since(mb_mark) if m["kind"] in ("error", "warning")]
    j.check("разделение прошло без ошибок",
            res.get("started") and res.get("finished") and not errs,
            f"{res.get('seconds')} с" + (f", окно: {errs[0]['message'][:160]}"
                                         if errs else ""))
    run.screenshot(f"{prefix}_done")
    stems = check_stems(run, path, stage)
    run.log_scan(mark, stage)
    return stems


# =============================================================
#  Вкладка «Ноты»
# =============================================================
def stage_transcribe(run: QaRun, path: Path, stage: str) -> dict:
    j = run.journal
    run.set_section(stage)
    app = run.app
    app.tabs.set("Ноты")
    run.pump(0.5)
    run.screenshot("notes_tab")

    stem = path.stem
    out: dict = {}

    j.check("кнопка «Вокал в ноты» включена",
            app.btn_voc_midi.cget("state") == "normal")
    j.check("кнопка «Минус в ноты» включена",
            app.btn_inst_midi.cget("state") == "normal")
    j.check("кнопка «Минус как пианино» включена",
            app.btn_piano.cget("state") == "normal")

    # --- вокал → ноты ---
    mark = run.log_mark()
    res = run.run_task(app.btn_voc_midi.invoke, "вокал→ноты", TIMEOUT_TRANSCRIBE)
    voca_mid = app.output_dir / f"{stem}_vocals.mid"
    exists = voca_mid.exists() and voca_mid.stat().st_size > 0
    j.check("«Вокал в ноты»: файл *_vocals.mid создан", exists,
            f"{voca_mid.name}, {voca_mid.stat().st_size if voca_mid.exists() else 0} байт, "
            f"{res.get('seconds')} с")
    if exists:
        m = measure_midi(voca_mid)
        out["vocals_midi"] = m
        j.check("в нотах вокала есть ноты", m["notes"] > 0,
                f"{m['notes']} нот, диапазон {m['pitch_min']}–{m['pitch_max']}")
        j.check("в нотах вокала нет полных дублей", m["duplicate_notes"] == 0,
                f"{m['duplicate_notes']} дублей")
    run.log_scan(mark, f"{stage}: вокал→ноты")
    run.screenshot("notes_vocals_midi")

    # --- минус → ноты ---
    mark = run.log_mark()
    res = run.run_task(app.btn_inst_midi.invoke, "минус→ноты", TIMEOUT_TRANSCRIBE)
    inst_mid = app.output_dir / f"{stem}_instrumental.mid"
    exists = inst_mid.exists() and inst_mid.stat().st_size > 0
    j.check("«Минус в ноты»: файл *_instrumental.mid создан", exists,
            f"{inst_mid.name}, {inst_mid.stat().st_size if inst_mid.exists() else 0} байт, "
            f"{res.get('seconds')} с")
    if exists:
        m = measure_midi(inst_mid)
        out["instrumental_midi"] = m
        density = m["notes"] / max(m["duration_sec"], 1e-6)
        j.check("«Минус в ноты» дал хоть какие-то ноты", m["notes"] > 0,
                f"{m['notes']} нот ({density:.2f} нот/с) — torchcrepe одноголосный, "
                f"для микса это ожидаемо мало")
    run.log_scan(mark, f"{stage}: минус→ноты")
    run.screenshot("notes_instrumental_midi")

    # --- минус как пианино ---
    mark = run.log_mark()
    res = run.run_task(app.btn_piano.invoke, "пианино", TIMEOUT_PIANO)
    piano_mid = app.output_dir / f"{stem}_piano.mid"
    exists = piano_mid.exists() and piano_mid.stat().st_size > 0
    j.check("«Минус как пианино»: файл *_piano.mid создан", exists,
            f"{piano_mid.name}, {piano_mid.stat().st_size if piano_mid.exists() else 0} "
            f"байт, {res.get('seconds')} с")
    if exists:
        m = measure_midi(piano_mid)
        out["piano_midi"] = m
        j.check("в пианино нет полных дублей нот",
                m["duplicate_notes"] == 0,
                f"{m['duplicate_notes']} дублей на {m['notes']} нот")
        j.check("в пианино нет наложений одной высоты",
                m["overlapping_same_pitch"] == 0,
                f"{m['overlapping_same_pitch']} наложений")
        j.check("в пианино достаточно разных высот", m["unique_pitches"] >= 5,
                f"{m['unique_pitches']} разных высот на {m['notes']} нот")
    run.log_scan(mark, f"{stage}: пианино")
    run.screenshot("notes_piano")

    j.check("после нот включилась «Проверить ноты»",
            app.btn_analyze_midi.cget("state") == "normal")
    return out


# =============================================================
#  Вкладка «Качество»
# =============================================================
def stage_analyze(run: QaRun, path: Path, stage: str) -> dict:
    j = run.journal
    run.set_section(stage)
    app = run.app
    app.tabs.set("Качество")
    run.pump(0.5)
    run.screenshot("quality_tab")

    out: dict = {}
    analysis_dir = app.analysis_dir

    # --- качество звука ---
    mark = run.log_mark()
    res = run.run_task(app.btn_analyze.invoke, "качество звука", TIMEOUT_ANALYZE)
    current = Path(app.player.current_file) if app.player.current_file else None
    base = current.stem if current else "?"
    out["analyzed_file"] = current.name if current else None
    json_path = analysis_dir / f"{base}_analysis.json"
    txt_path = analysis_dir / f"{base}_analysis.txt"
    wave_png = analysis_dir / f"{base}_waveform.png"
    spec_png = analysis_dir / f"{base}_spectrogram.png"

    j.check("создан *_analysis.json", json_path.exists(), json_path.name)
    j.check("создан *.txt-отчёт", txt_path.exists(), txt_path.name)
    j.check("создан *_waveform.png", wave_png.exists(), wave_png.name)
    j.check("создан *_spectrogram.png", spec_png.exists(), spec_png.name)

    if json_path.exists():
        report = json.loads(json_path.read_text(encoding="utf-8"))
        out["audio_report"] = report
        nan_bad, nulls = scan_json_bad(report)
        j.check("в JSON отчёта нет NaN/Inf", not nan_bad,
                "чисто" if not nan_bad else "; ".join(nan_bad[:5]))
        j.check("в JSON отчёта нет null", not nulls,
                "чисто" if not nulls else f"{len(nulls)} шт: " + ", ".join(nulls[:6]))
        j.check("в отчёте есть ключи levels/silence/stereo/f0_hz",
                all(k in report for k in ("levels", "silence", "stereo", "f0_hz")),
                ", ".join(sorted(report.keys()))[:170])
    for png in (wave_png, spec_png):
        if png.exists():
            head = png.read_bytes()[:8]
            j.check(f"{png.name} — настоящий PNG", head == b"\x89PNG\r\n\x1a\n",
                    f"{png.stat().st_size // 1024} КБ")
    run.log_scan(mark, f"{stage}: качество звука")
    run.screenshot("quality_sound")

    # --- проверка нот ---
    mark = run.log_mark()
    candidates = app._midi_candidates()
    chosen = candidates[0] if candidates else None
    res = run.run_task(app.btn_analyze_midi.invoke, "проверка нот", TIMEOUT_ANALYZE)
    j.check("«Проверить ноты» отработала",
            res.get("started") and res.get("finished"), f"{res.get('seconds')} с")
    midi_json = (analysis_dir / f"{chosen.stem}_midi_analysis.json") if chosen else None
    j.check("создан отчёт *_midi_analysis.json",
            bool(midi_json and midi_json.exists()),
            midi_json.name if midi_json else "нет нот")
    if midi_json and midi_json.exists():
        report = json.loads(midi_json.read_text(encoding="utf-8"))
        out["midi_report"] = report
        nan_bad, nulls = scan_json_bad(report)
        j.check("в отчёте по нотам нет NaN/Inf", not nan_bad,
                "чисто" if not nan_bad else "; ".join(nan_bad[:5]))
        summary = report.get("summary", {})
        j.check("в отчёте по нотам total_notes > 0", summary.get("total_notes", 0) > 0,
                f"{summary.get('total_notes')} нот, "
                f"{summary.get('pitch_min_name')}–{summary.get('pitch_max_name')}")
    run.log_scan(mark, f"{stage}: проверка нот")
    run.screenshot("quality_midi")

    # --- качество разделения ---
    mark = run.log_mark()
    res = run.run_task(app.btn_super_pair.invoke, "качество разделения", TIMEOUT_ANALYZE)
    pair_json = analysis_dir / f"{path.stem}_super_pair_analysis.json"
    j.check("создан *_super_pair_analysis.json", pair_json.exists(), pair_json.name)
    if pair_json.exists():
        report = json.loads(pair_json.read_text(encoding="utf-8"))
        out["pair_report"] = report
        j.check("в отчёте есть поле coverage", "coverage" in report,
                ", ".join(list(report.get("coverage", {}).keys()))[:130])
        nan_bad, nulls = scan_json_bad(report)
        j.check("в отчёте пары нет NaN/Inf", not nan_bad,
                "чисто" if not nan_bad else "; ".join(nan_bad[:5]))
        j.check("в отчёте пары нет null", not nulls,
                "чисто" if not nulls else f"{len(nulls)} шт: " + ", ".join(nulls[:6]))
        verdict = report.get("verdict", {})
        j.check("вердикт написан простыми словами",
                bool(verdict.get("kid") or verdict.get("kid_lines")),
                str(verdict.get("kid") or verdict.get("kid_lines"))[:170])
        cov = report.get("coverage", {})
        per_sec = cov.get("per_second", {})
        # Покрытие лежит во вложенном блоке coverage.vpi; разбираем так же, как
        # это делает core/reports.py (плоский вариант тоже понимаем).
        vpi_block = report.get("level_6_vpi", {}) or {}
        vpi_cov = cov.get("vpi", {}) if isinstance(cov.get("vpi"), dict) else {}
        coverage_pct = next((c for c in (cov.get("active_coverage_pct"),
                                         vpi_cov.get("active_coverage_pct"),
                                         vpi_block.get("active_coverage_pct"))
                             if c is not None), None)
        j.check("отчёт сообщает, сколько вокала реально проанализировано",
                coverage_pct is not None,
                f"покрытие {coverage_pct}% активного вокала, "
                f"посекундно {per_sec.get('seconds_analyzed')} из "
                f"{per_sec.get('seconds_total')} с")

        # Регрессия на баг, найденный 11.10.2026: на минусовке (вокала в треке
        # нет) вердикт выдавал «отпечаток голоса 139.8 дБ» и «минус темнеет
        # -65 %» — числа, посчитанные от тишины. Теперь про это надо сказать
        # прямо, а числа не показывать.
        level1 = report.get("level_1_overall", {}) or {}
        vocal_absent = (level1.get("vocals_rms_db") is not None
                        and level1["vocals_rms_db"] < -45.0
                        and level1["vocals_rms_db"]
                        < (level1.get("instrumental_rms_db") or 0.0) - 30.0)
        if vocal_absent:
            kid_text = str(verdict.get("kid", "")).lower()
            j.check("вокала в треке нет — вердикт говорит об этом прямо",
                    "вокала практически нет" in kid_text,
                    str(verdict.get("kid"))[:150])
            j.check("вокала в треке нет — VPI не выдуман",
                    "vpi_db" not in vpi_block,
                    vpi_block.get("vpi_note", "VPI посчитан"))
        else:
            j.check("VPI посчитан и физически осмыслен",
                    vpi_block.get("vpi_db") is not None
                    and -60.0 < float(vpi_block.get("vpi_db", 0.0)) < 60.0,
                    f"VPI {vpi_block.get('vpi_db')} dB")
    info_msgs = [m for m in res.get("messages", []) if m["kind"] == "info"]
    j.check("приложение показало вердикт человеку", bool(info_msgs),
            info_msgs[0]["title"] if info_msgs else "окна не было")
    run.log_scan(mark, f"{stage}: качество разделения")
    run.screenshot("quality_pair")
    return out


# =============================================================
#  Вкладка «Настройки»
# =============================================================
def stage_settings(run: QaRun, stage: str) -> dict:
    j = run.journal
    run.set_section(stage)
    app = run.app
    from core.config import SETTINGS_FILE
    from ui.settings_dialog import FIELDS

    app.tabs.set("Настройки")
    run.pump(0.7)
    run.screenshot("settings_tab")
    panel = app.settings_panel

    j.check("полей в панели ровно столько, сколько описано в FIELDS",
            len(FIELDS) == len(panel.vars) == len(panel._widgets),
            f"{len(FIELDS)} полей: " + ", ".join(f.name for f in FIELDS))
    missing = [f.name for f in FIELDS if f.name not in panel._widgets]
    j.check("все поля построены", not missing, "все на месте" if not missing else str(missing))
    not_packed = [f.name for f in FIELDS
                  if panel._widgets[f.name].winfo_manager() == ""]
    j.check("все поля размещены в окне", not not_packed,
            "все на месте" if not not_packed else str(not_packed))
    j.check("у всех полей есть подпись", len(panel._field_labels) == len(FIELDS),
            f"{len(panel._field_labels)} подписей")
    j.check("подписи полей читаемы (не пустые)",
            all(lbl.cget("text").strip() for lbl in panel._field_labels),
            "; ".join(lbl.cget("text") for lbl in panel._field_labels[:4]) + " …")

    before = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    j.save("settings_before", before)

    # --- меняем устройство на cpu ---
    mark = run.log_mark()
    panel.vars["demucs_device"].set("cpu")
    run.pump(0.3)
    panel.btn_save.invoke()
    run.pump(1.2)
    after = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    j.check("настройка demucs_device=cpu записалась в файл",
            after.get("demucs_device") == "cpu",
            f"в файле: {after.get('demucs_device')!r}")
    j.check("окно подхватило настройку без перезапуска",
            app.config_data.demucs_device == "cpu",
            f"config_data.demucs_device = {app.config_data.demucs_device!r}")
    log_entries = run.applog.since(mark)
    changed = [e for e in log_entries
               if "demucs_device" in json.dumps(e, ensure_ascii=False)]
    j.check("смена настройки попала в logs/ast.log", bool(changed),
            changed[0]["msg"][:110] if changed else "записи нет")
    run.screenshot("settings_cpu_saved")

    # --- возвращаем cuda ---
    panel.vars["demucs_device"].set("cuda")
    run.pump(0.3)
    panel.btn_save.invoke()
    run.pump(1.2)
    restored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    j.check("настройка вернулась к cuda", restored.get("demucs_device") == "cuda",
            f"в файле: {restored.get('demucs_device')!r}")

    # DECISION: сравниваем настройки ПО СМЫСЛУ, а не по строке. Сохранение из
    # окна дописывает в файл поля, которых там не было (bs_* появились позже), и
    # записывает движок его id вместо подписи («BS-Roformer: Duality-V2 …» →
    # «bs:melband_roformer_instvox_duality_v2.ckpt»). И то и другое — норма:
    # resolve_engine_id понимает обе формы. Строгое сравнение ловило бы это как
    # «настройки разъехались», хотя поведение не изменилось.
    from core.engines import resolve_engine_id

    def same_engine(before_value, after_value) -> bool:
        try:
            return resolve_engine_id(str(before_value)) == resolve_engine_id(str(after_value))
        except KeyError:
            return str(before_value) == str(after_value)

    diffs = []
    for key, was in before.items():
        now = restored.get(key)
        if key == "default_engine":
            if not same_engine(was, now):
                diffs.append(f"default_engine: {was!r} → {now!r}")
        elif was != now:
            diffs.append(f"{key}: {was!r} → {now!r}")
    j.check("настройки, которые не трогали, не изменились", not diffs,
            "все прежние значения на месте" if not diffs else "; ".join(diffs))
    added = sorted(set(restored) - set(before))
    if added:
        j.check("в файл дописаны поля, которых в нём не было", None,
                f"{', '.join(added)} — это значения по умолчанию из AppConfig, "
                f"поведение не меняется")
    j.check("кнопка «Сохранить» осталась рабочей",
            panel.btn_save.cget("state") == "normal")
    run.log_scan(mark, stage)
    run.screenshot("settings_final")
    return {"before": before, "after": restored}


# =============================================================
#  Плеер
# =============================================================
def stage_player(run: QaRun, path: Path, stage: str) -> dict:
    j = run.journal
    run.set_section(stage)
    app = run.app
    player = app.player
    bar = app.player_bar
    seek = bar.seek
    out: dict = {}
    mark = run.log_mark()

    def choose(value: str) -> None:
        """Выбор в «Слушать:» — ровно тот колбэк, что зовётся при выборе пункта."""
        app.combo_listen._dropdown_callback(value)
        run.pump(0.7)

    choose("Исходный трек")
    dur = player.duration
    j.check("плеер знает длительность трека", dur > 0,
            f"{dur:.2f} с, файл "
            f"{Path(player.current_file).name if player.current_file else '—'}")

    # --- play ---
    bar.btn_play.invoke()
    run.pump(2.0)
    pos = player.position
    j.check("▶ идёт воспроизведение", player.is_playing and pos > 0.3,
            f"за 2 с прошло {pos:.2f} с")
    out["play_2s"] = round(pos, 3)
    run.screenshot("player_playing")

    # --- pause / resume ---
    bar.btn_pause.invoke()
    p1 = player.position
    run.pump(0.9)
    p2 = player.position
    j.check("⏸ пауза останавливает позицию", (not player.is_playing)
            and abs(p2 - p1) < 0.05, f"{p1:.3f} → {p2:.3f} с")
    bar.btn_play.invoke()
    run.pump(1.2)
    j.check("▶ после паузы игра продолжается", player.is_playing
            and player.position > p2 + 0.2, f"позиция {player.position:.2f} с")

    # --- stop ---
    bar.btn_stop.invoke()
    run.pump(0.6)
    j.check("⏹ стоп обнуляет позицию",
            (not player.is_playing) and player.position < 0.05,
            f"позиция {player.position:.3f} с")

    # --- seek перетаскиванием (настоящие события мыши по Canvas) ---
    choose("Исходный трек")
    seek.update()
    width = seek.winfo_width()
    target_x = int(width * 0.5)
    for event in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>"):
        seek.event_generate(event, x=target_x, y=seek.H // 2)
    run.pump(0.7)
    span = max(1, (width - seek.PAD) - seek.PAD)
    expected = (target_x - seek.PAD) / span * dur
    got = player.position
    j.check("перемотка перетаскиванием ползунка",
            abs(got - expected) < max(1.0, dur * 0.05),
            f"клик в x={target_x} из {width} → {got:.2f} с "
            f"(ожидалось ~{expected:.2f} с)")
    out["seek_drag"] = {"expected": round(expected, 3), "got": round(got, 3)}

    # --- горячие клавиши ---
    # Сначала — регрессия на найденный баг: после открытия и закрытия списка
    # моделей окно теряло клавиатуру, и стрелки перестали двигать плеер.
    app.engine_combo.open()
    run.pump(0.5)
    app.engine_combo.close()
    run.pump(0.4)
    j.check("после закрытия списка моделей клавиатура осталась у окна",
            app.focus_get() is not None,
            f"focus_get() = {app.focus_get()!r}")

    def hotkey(spec: str, expect: float, label: str) -> None:
        """
        Шлёт окну нажатие клавиши и проверяет сдвиг позиции.

        Одна повторная попытка: синтетическая инжекция клавиш в Windows иногда
        «теряется» (окно на миг теряет активность). В прогоне 11.10.2026 так
        отвалились Shift+→ и всё после него, хотя в изоляции те же нажатия
        работают. Если со второй попытки проходит — это флак драйвера, а не баг
        приложения, и в отчёте так и написано.
        """
        for attempt in (1, 2):
            app.focus_force()
            app._player_seek(60.0)
            run.pump(0.3)
            before = player.position
            app.event_generate(spec, when="now")
            run.pump(0.35)
            shift = player.position - before
            if abs(shift - expect) < 0.5:
                j.check(label, True, f"сдвиг {shift:+.2f} с"
                        + ("" if attempt == 1 else " (со второй попытки — "
                           "инжекция клавиш нестабильна, не баг приложения)"))
                return
        j.check(label, False,
                f"сдвиг {shift:+.2f} с, ожидалось {expect:+.1f} с "
                f"(две попытки, фокус: {app.focus_get()!r})")

    hotkey("<Right>", +5.0, "→ перематывает вперёд на 5 с")
    hotkey("<Left>", -5.0, "← перематывает назад на 5 с")
    hotkey("<Shift-Right>", +30.0, "Shift+→ перематывает вперёд на 30 с")
    hotkey("<Shift-Left>", -30.0, "Shift+← перематывает назад на 30 с")

    app.focus_force()
    app._player_seek(50.0)
    run.pump(0.3)
    app.event_generate("<Home>", when="now")
    run.pump(0.4)
    j.check("Home — в начало", player.position < 0.05, f"позиция {player.position:.3f} с")
    app._player_seek(50.0)
    run.pump(0.3)
    app.event_generate("<End>", when="now")
    run.pump(0.5)
    j.check("End — в конец", abs(player.position - dur) < 0.5,
            f"позиция {player.position:.2f} из {dur:.2f} с")

    # --- Space ---
    app._player_seek(0.0)
    run.pump(0.3)
    app.event_generate("<space>", when="now")
    run.pump(1.2)
    playing = player.is_playing
    app.event_generate("<space>", when="now")
    run.pump(0.6)
    j.check("Space — play/pause", playing and not player.is_playing,
            f"играло: {playing}, после второго нажатия: {player.is_playing}")
    app._player_stop()
    run.pump(0.4)

    # --- «Слушать:» и волна ---
    vocals, instr = app._effective_stems()
    for value, expected_path in (("Вокал", vocals), ("Минус", instr),
                                 ("Исходный трек", path)):
        choose(value)
        got_file = Path(player.current_file) if player.current_file else None
        caption = app.wave.caption.cget("text")
        j.check(f"«Слушать: {value}» — в плеере нужный файл",
                got_file is not None and expected_path is not None
                and got_file.name == Path(expected_path).name,
                f"в плеере {got_file.name if got_file else '—'}")
        j.check(f"«Слушать: {value}» — волна перерисована под этот файл",
                app.wave.has_plot and Path(expected_path).name in caption,
                f"подпись волны: {caption!r}")
    run.screenshot("player_listen_source")

    # --- смена источника во время игры ---
    choose("Исходный трек")
    app._player_play()
    run.pump(1.2)
    choose("Вокал")
    j.check("смена «Слушать» во время игры не роняет плеер",
            player.is_playing and player.current_file is not None
            and "vocals" in player.current_file.name,
            f"играет: {player.is_playing}, файл "
            f"{player.current_file.name if player.current_file else '—'}")
    app._player_stop()
    run.pump(0.4)

    # --- «Слушать: —» ---
    choose("—")
    j.check("«Слушать: —» очищает плеер и волну",
            not player.is_playing and not app.wave.has_plot,
            f"подпись волны: {app.wave.caption.cget('text')!r}")
    choose("Вокал")
    run.log_scan(mark, stage)
    return out


# =============================================================
#  Тема
# =============================================================
def stage_theme(run: QaRun, stage: str) -> dict:
    j = run.journal
    run.set_section(stage)
    app = run.app
    out: dict = {}

    def snapshot(name: str) -> None:
        theme = app.theme
        window_bg = app.cget("fg_color")
        wave_bg = app.wave.fig.get_facecolor()
        expected_wave = tuple(int(theme["panel"].lstrip("#")[i:i + 2], 16) / 255
                              for i in (0, 2, 4))
        j.check(f"тема «{name}»: фон окна перекрасился",
                window_bg == theme["bg"], f"{window_bg} (ожидалось {theme['bg']})")
        j.check(f"тема «{name}»: фон волны перекрасился",
                all(abs(a - b) < 0.02 for a, b in zip(wave_bg[:3], expected_wave)),
                f"{tuple(round(float(v), 3) for v in wave_bg[:3])} (ожидалось "
                f"{tuple(round(v, 3) for v in expected_wave)})")
        j.check(f"тема «{name}»: главная кнопка в цвете акцента",
                app.btn_separate.cget("fg_color") == theme["accent"],
                f"{app.btn_separate.cget('fg_color')}")
        j.check(f"тема «{name}»: полоса перемотки перекрасилась",
                app.player_bar.seek.theme["panel"] == theme["panel"],
                f"{app.player_bar.seek.theme['panel']}")
        j.check(f"тема «{name}»: волна не потерялась", app.wave.has_plot,
                f"подпись: {app.wave.caption.cget('text')!r}")
        out[name] = {"bg": window_bg, "panel": theme["panel"],
                     "accent": theme["accent"],
                     "wave_face": [round(float(v), 4) for v in wave_bg]}

    j.check("стартуем с тёмной темы", app.theme_name == "dark", app.theme_name)

    # Меню «⚙» в тёмной теме. Тему переключаем ИМЕННО через меню, как просили
    # в задании, а не вызовом внутреннего метода: так проверяется и пункт меню.
    app._open_menu()
    run.pump(0.5)
    j.check("меню «⚙» открылось", app._menu_popup is not None)
    run.screenshot("theme_dark_menu", full_screen=True)
    app._close_menu()
    run.pump(0.4)
    run.screenshot("theme_dark")
    snapshot("dark")

    # --- меню: «Светлая тема» ---
    j.check("в меню есть пункт «Светлая тема»",
            press_menu_item(run, "Светлая тема"))
    j.check("тема переключилась на светлую через меню",
            app.theme_name == "light", app.theme_name)
    snapshot("light")
    app.tabs.set("Разделение")
    run.pump(0.4)
    run.screenshot("theme_light")

    # меню в светлой теме
    app._open_menu()
    run.pump(0.5)
    j.check("меню «⚙» открылось в светлой теме", app._menu_popup is not None,
            f"фон меню {app._menu_popup.cget('fg_color')}")
    run.screenshot("theme_light_menu", full_screen=True)
    app._close_menu()
    run.pump(0.4)
    j.check("после закрытия меню «⚙» клавиатура осталась у окна",
            app.focus_get() is not None, f"focus_get() = {app.focus_get()!r}")

    # --- меню: «Тёмная тема» ---
    j.check("в меню есть пункт «Тёмная тема»",
            press_menu_item(run, "Тёмная тема"))
    j.check("тема вернулась к тёмной через меню",
            app.theme_name == "dark", app.theme_name)
    run.screenshot("theme_back_to_dark")

    # --- прочие пункты меню: проверяем, что они вообще работают ---
    j.check("меню: «Настройки…» открывает вкладку настроек",
            press_menu_item(run, "Настройки") and app.tabs.get() == "Настройки",
            f"активная вкладка: {app.tabs.get()!r}")
    app.tabs.set("Разделение")
    run.pump(0.4)
    mb_mark = run.mb.mark
    j.check("меню: «О программе» показывает окно",
            press_menu_item(run, "О программе")
            and any(m["kind"] == "info" for m in run.mb.since(mb_mark)),
            (run.mb.since(mb_mark) or [{"title": "окна не было"}])[0]["title"])
    return out


# =============================================================
#  Краевые случаи
# =============================================================
def stage_edge(run: QaRun, stage: str) -> dict:
    j = run.journal
    run.set_section(stage)
    app = run.app
    work = run.out / "clips"
    work.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # --- WAV и FLAC вход (BS-API) ---
    for fmt in ("wav", "flac"):
        clip = work / f"_qa_edge_{fmt}.{fmt}"
        try:
            make_clip(SONG_1, clip, EDGE_CLIP_SEC, fmt)
        except Exception as e:                 # noqa: BLE001
            j.check(f"клип {fmt.upper()} сделан", False, str(e))
            continue
        j.check(f"клип {fmt.upper()} сделан", True,
                f"{clip.name}, {clip.stat().st_size // 1024} КБ, {EDGE_CLIP_SEC} с")

        engine_id = pick_preset(run, PRESET_INSTRUMENTAL)
        j.check(f"вход {fmt.upper()}: пресет выбрал модель",
                engine_id == EXPECT_INSTRUMENTAL, str(engine_id))
        load_song(run, clip)
        mark = run.log_mark()
        mb_mark = run.mb.mark
        res = run.run_task(app.btn_separate.invoke, f"BS-разделение {fmt}",
                           TIMEOUT_SEPARATE)
        errs = [m for m in run.mb.since(mb_mark) if m["kind"] in ("error", "warning")]
        j.check(f"вход {fmt.upper()}: разделение не упало",
                res.get("started") and res.get("finished") and not errs,
                f"{res.get('seconds')} с" + (f", окно: {errs[0]['message'][:140]}"
                                             if errs else ""))
        out[fmt] = {"seconds": res.get("seconds"), "started": res.get("started"),
                    "message": errs[0]["message"][:200] if errs else None}
        check_stems(run, clip, f"{stage}: вход {fmt}")
        run.log_scan(mark, f"{stage}: вход {fmt}")
        run.screenshot(f"edge_{fmt}_done")

    # --- отмена ---
    run.set_section(f"{stage}: отмена")
    pick_preset(run, PRESET_INSTRUMENTAL)
    load_song(run, SONG_1)
    before = audio_separator_processes()
    mb_mark = run.mb.mark
    app.btn_separate.invoke()
    run.pump(6.0)
    j.check("задача идёт, когда нажимаем «Отмена»", app.current_task_id is not None,
            f"id={app.current_task_id}, статус {app.status.cget('text')!r}")
    j.check("кнопка «Отмена» стала доступной во время задачи",
            app.btn_cancel.cget("state") == "normal")
    run.screenshot("edge_cancel_before")
    cancel_started = time.monotonic()
    app.btn_cancel.invoke()
    stopped, _ = run.pump_until(lambda: app.current_task_id is None, 180)
    cancelled_sec = time.monotonic() - cancel_started
    j.check("задача прервалась после «Отмены»", stopped,
            f"остановилась через {cancelled_sec:.1f} с")
    msgs = run.mb.since(mb_mark)
    j.check("отмена не показала окно с ошибкой",
            not [m for m in msgs if m["kind"] == "error"],
            f"статус: {app.status.cget('text')!r}")
    run.pump(3.0)
    after = audio_separator_processes()
    j.check("после отмены не осталось audio-separator.exe", not after,
            f"до: {len(before)}, после: {len(after)}")
    j.check("кнопки вернулись: «Разделить» активна, «Отмена» выключена",
            app.btn_separate.cget("state") == "normal"
            and app.btn_cancel.cget("state") == "disabled")
    j.check("воркер свободен после отмены", not app.worker.is_busy)
    run.screenshot("edge_cancel_after")
    out["cancel"] = {"seconds_to_stop": round(cancelled_sec, 1),
                     "status": app.status.cget("text")}

    # --- смена трека ---
    run.set_section(f"{stage}: смена трека")
    load_song(run, SONG_1)
    j.check("у разделённого трека кнопки доступны",
            app.btn_piano.cget("state") == "normal"
            and app.btn_super_pair.cget("state") == "normal"
            and app.btn_voc_midi.cget("state") == "normal")
    load_song(run, SONG_CLEAN, shot_name="edge_new_song")
    j.check("после загрузки неразделённого трека «Пианино» выключено",
            app.btn_piano.cget("state") == "disabled",
            f"трек {SONG_CLEAN.name} в output/ не разделялся")
    j.check("после загрузки неразделённого трека «Проверить качество разделения» "
            "выключено", app.btn_super_pair.cget("state") == "disabled")
    j.check("после загрузки неразделённого трека «Вокал в ноты» выключено",
            app.btn_voc_midi.cget("state") == "disabled")
    j.check("старые пути стемов сброшены",
            app.vocals_path is None and app.instrumental_path is None
            and app.last_piano_path is None,
            f"vocals={app.vocals_path}, instrumental={app.instrumental_path}, "
            f"piano={app.last_piano_path}")
    j.check("проверка нот выключена для нового трека",
            app.btn_analyze_midi.cget("state") == "disabled")
    j.check("«Проверить качество звука» доступна для нового трека",
            app.btn_analyze.cget("state") == "normal")

    load_song(run, SONG_1)
    return out


def stage_close(run: QaRun, stage: str) -> dict:
    """Закрытие окна во время задачи. Уничтожает окно — поэтому последним."""
    j = run.journal
    run.set_section(stage)
    app = run.app

    pick_preset(run, PRESET_INSTRUMENTAL)
    load_song(run, SONG_2)
    app.btn_separate.invoke()
    run.pump(6.0)
    j.check("задача идёт в момент закрытия окна", app.current_task_id is not None,
            f"id={app.current_task_id}, статус {app.status.cget('text')!r}")
    run.screenshot("close_before")
    app._on_close()
    time.sleep(5.0)
    left = audio_separator_processes()
    j.check("после закрытия окна не осталось audio-separator.exe", not left,
            f"найдено: {len(left)}")
    exists = _safe_exists(app)
    j.check("окно действительно закрылось", exists in (0, None),
            f"winfo_exists={exists}")
    run.app = None
    return {"left_processes": len(left)}


# =============================================================
#  Отчёт
# =============================================================
def write_report(run: QaRun, stages_done: list[str]) -> Path:
    j = run.journal
    env = j.data.get("окружение", {})
    path = run.out / "report.md"
    lines: list[str] = []
    add = lines.append

    add(f"# Полный QA-прогон AST от {datetime.now().strftime('%Y-%m-%d')}")
    add("")
    add("Прогон всех вкладок приложения на реальных песнях из папки «Песни для тестов».")
    add("")
    add("Как нажимались кнопки: у настоящих кнопок вызывался `invoke()` (он сам "
        "смотрит, не выключена ли кнопка), у пресетов моделей — тот же обработчик, "
        "что висит на клике по карточке, у списка «Слушать» — тот же колбэк, что и "
        "при выборе пункта. Мышь не двигалась: робот этого не умеет. Модальные окна "
        "перехвачены в журнал, каждое состояние снято скриншотом, события воркера "
        "обрабатывались штатным циклом окна — то есть проверялась связка "
        "«воркер → интерфейс», а не только ядро.")
    add("")
    add(f"Этапы прогона: {', '.join(stages_done)}. "
        f"Длительность: {j.data.get('длительность_прогона_сек')} с.")
    add("")

    add("## Окружение")
    add("")
    add("| Параметр | Значение |")
    add("|---|---|")
    for key in ("python", "платформа", "torch", "cuda_доступна", "cuda", "gpu", "vram_mb",
                "demucs", "torchcrepe", "audio_separator", "muscriptor", "soundfile",
                "scipy", "pretty_midi", "customtkinter", "sounddevice", "ffmpeg",
                "audio-separator.exe"):
        if key in env:
            add(f"| {key} | {env[key]} |")
    add(f"| модели в models/ | {', '.join(env.get('модели_локально', []))} |")
    add("")

    songs = env.get("песни", [])
    if songs:
        add(f"### Песни для тестов (в папке {env.get('песен_в_папке')} mp3)")
        add("")
        add("| Трек | Размер | Длительность | Использован для |")
        add("|---|---|---|---|")
        usage = {
            SONG_1.name: "песня 1: пресет «минус как в оригинале» + все вкладки",
            SONG_2.name: "песня 2: пресет «для караоке»",
            SONG_3.name: "песня 3: пресет «быстро и просто» (Demucs)",
            SONG_CLEAN.name: "смена трека (кнопки должны выключиться)",
        }
        for song in songs:
            add(f"| {song['имя']} | {song['мб']} МБ | {song['секунд']} с | "
                f"{usage.get(song['имя'], '—')} |")
        add("")

    add("## Результаты по разделам")
    add("")
    sections: dict[str, list[Check]] = {}
    for check in j.checks:
        sections.setdefault(check.stage, []).append(check)
    for section, checks in sections.items():
        ok = sum(1 for c in checks if c.ok is True)
        bad = sum(1 for c in checks if c.ok is False)
        warn = sum(1 for c in checks if c.ok is None)
        status = "✅" if bad == 0 else "❌"
        if bad == 0 and warn:
            status = "⚠️"
        add(f"### {section} — {status} ({ok} ок, {bad} провал, {warn} замечаний)")
        add("")
        for check in checks:
            icon = {True: "✅", False: "❌", None: "⚠️"}[check.ok]
            add(f"- {icon} {check.name}" + (f" — {check.note}" if check.note else ""))
        add("")

    add("## Данные разделения")
    add("")
    for key, label in (("s1_stems", "Песня 1"), ("s2_stems", "Песня 2"),
                       ("s3_stems", "Песня 3")):
        stems = j.data.get(key)
        if not stems or "vocals" not in stems:
            continue
        v, i = stems["vocals"], stems["instrumental"]
        add(f"**{label}** — `{stems['stem']}`")
        add("")
        add("| Файл | Размер | Длительность | Частота | Каналы | Пик | RMS |")
        add("|---|---|---|---|---|---|---|")
        for meas in (v, i):
            add(f"| {meas['name']} | {meas['size_mb']} МБ | {meas['duration_sec']} с | "
                f"{meas['sample_rate']} Гц | {meas['channels']} | {meas['peak']} "
                f"({meas['peak_db']} dBFS) | {meas['rms_db']} dBFS |")
        add("")

    add("## Данные транскрипции (песня 1)")
    add("")
    tr = j.data.get("s1_transcribe") or {}
    if tr:
        add("| Файл | Размер | Нот | Разных высот | Диапазон | Дубли | Наложения |")
        add("|---|---|---|---|---|---|---|")
        for key, meas in tr.items():
            add(f"| {key} → {meas['name']} | {meas['size_bytes']} Б | {meas['notes']} | "
                f"{meas['unique_pitches']} | {meas['pitch_min']}–{meas['pitch_max']} | "
                f"{meas['duplicate_notes']} | {meas['overlapping_same_pitch']} |")
        add("")

    add("## Данные анализа (песня 1)")
    add("")
    an = j.data.get("s1_analyze") or {}
    audio = an.get("audio_report") or {}
    if audio:
        levels = audio.get("levels", {})
        add(f"- Проанализирован файл: `{an.get('analyzed_file')}`")
        add(f"- Средняя громкость (RMS): {levels.get('rms_db')} dBFS, "
            f"пик: {levels.get('peak_db')} dBFS, "
            f"клиппинг: {levels.get('clipped_samples')} сэмплов")
        add(f"- Тишина: {audio.get('silence', {}).get('silence_pct')}% времени")
    pair = an.get("pair_report") or {}
    if pair:
        cov = pair.get("coverage", {})
        vpi = pair.get("level_6_vpi", {})
        add(f"- Супер-анализ пары: покрытие активного вокала "
            f"{cov.get('active_coverage_pct')}% ({vpi.get('active_frames_used')} "
            f"кадров из {vpi.get('active_frames_total')})")
        add(f"- Вердикт: {str(pair.get('verdict', {}).get('kid'))[:300]}")
    add("")

    add("## Найденные баги")
    add("")
    add("| # | Раздел | Симптом | Файл | Воспроизводится | Починен |")
    add("|---|---|---|---|---|---|")
    for i, bug in enumerate(j.bugs, 1):
        add(f"| {i} | {bug.title} | {bug.symptom} | `{bug.where}` | "
            f"{bug.reproducible} | {bug.fixed} |")
    if not j.bugs:
        add("| — | — | багов не найдено | — | — | — |")
    add("")

    add("## Что нужно послушать ушами")
    add("")
    add("Цифры — не приговор качеству, а подсказка, куда направить ухо: слух")
    add("проверяет то, что микрофон-робот проверить не может.")
    add("")
    add("- **Минус песни 1** — не «съедены» ли инструменты, нет ли эха голоса.")
    add("- **Вокал песни 1** — не тише ли он минуса (парная нормализация выставляет "
        "один уровень на пару, и вокал может оказаться тише).")
    add("- **Пианино песни 1** — не «двоит» ли партия (дубли нот мы посчитали "
        "числами, но на слух это слышно иначе).")
    add("- **Минус песни 2** — ушёл ли вокал полностью (посчитать это можно только "
        "косвенно).")
    add("- **Минус песни 3 (Demucs)** — не «водянистый» ли звук: Demucs слабее "
        "BS-моделей, зато считается за минуту.")
    add("")

    add("## Проваленные проверки")
    add("")
    failed_checks = [c for c in j.checks if c.ok is False]
    if failed_checks:
        add("| Раздел | Проверка | Что не так |")
        add("|---|---|---|")
        for check in failed_checks:
            add(f"| {check.stage} | {check.name} | {check.note} |")
    else:
        add("Провалов нет.")
    add("")

    add("## Что осталось непроверенным")
    add("")
    add("- Пути, для которых нужна модель, отсутствующая локально (см. таблицу "
        "выше): скачивание больше 100 МБ автономный прогон не выполняет.")
    add("- Настоящие клики мышью и перетаскивание файла из проводника: драйвер "
        "зовёт те же обработчики, но сам курсор не двигает.")
    add("- Остальные движки реестра (26 моделей): прогонялись три пресета и "
        "локальная модель группы «караоке».")
    add("- Слуховое качество результата — см. раздел выше.")
    add("")

    total = len(j.checks)
    passed = sum(1 for c in j.checks if c.ok is True)
    failed = sum(1 for c in j.checks if c.ok is False)
    warned = sum(1 for c in j.checks if c.ok is None)
    add("## Итог")
    add("")
    add(f"- Проверок всего: **{total}**")
    add(f"- Пройдено: **{passed}** из {total}")
    add(f"- Провалено: **{failed}**")
    add(f"- С замечанием: **{warned}**")
    add(f"- Багов найдено: **{len(j.bugs)}**")
    add(f"- Багов починено: **{sum(1 for b in j.bugs if b.fixed.startswith('да'))}**")
    add(f"- Осталось незакрытых: **{failed + sum(1 for b in j.bugs if not b.fixed.startswith('да'))}**")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# =============================================================
#  Точка входа
# =============================================================
STAGES = ("env", "s1", "s2", "s3", "edge", "close",
          "theme", "player", "analyze", "report", "all")
ALL_STAGES = ("env", "s1", "s2", "s3", "edge", "close")
NEEDS_WINDOW = ("s1", "s2", "s3", "edge", "close", "theme", "player", "analyze")


# =============================================================
#  Пересборка отчёта по уже собранным результатам
# =============================================================
_PROGRESS_RE = re.compile(r"^- (?P<icon>✅|❌|⚠️) \*\*\[(?P<stage>.+?)\]\*\* (?P<rest>.*)$")


def load_checks(out_dir: Path) -> list[Check]:
    """
    Возвращает проверки прошлого прогона: сначала из checks.json, потом из progress.md.

    Зачем: прогон длится десятки минут и тратит GPU. Пересобрать отчёт (например,
    дополнить таблицу багов) должно быть можно без повторного прогона.
    """
    checks_file = out_dir / "checks.json"
    if checks_file.exists():
        try:
            return [Check(item["stage"], item["name"], item.get("ok"), item.get("note", ""))
                    for item in json.loads(checks_file.read_text(encoding="utf-8"))]
        except Exception:
            logging.getLogger(__name__).warning("checks.json не прочитался", exc_info=True)

    progress = out_dir / "progress.md"
    checks: list[Check] = []
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            match = _PROGRESS_RE.match(line.strip())
            if not match:
                continue
            ok = {"✅": True, "❌": False, "⚠️": None}[match["icon"]]
            rest = match["rest"]
            name, _, note = rest.partition(" — ")
            checks.append(Check(match["stage"], name, ok, note))
    return checks


def stage_report(out_dir: Path) -> None:
    """Собирает report.md из сохранённых результатов, ничего не запуская."""
    run = QaRun(out_dir, shots=False)
    run.journal.checks = load_checks(out_dir)
    data_file = out_dir / "results.json"
    if data_file.exists():
        run.journal.data = json.loads(data_file.read_text(encoding="utf-8"))
    for title, where, symptom, severity, reproducible, fixed in KNOWN_BUGS:
        run.journal.bug(title, where, symptom, severity, reproducible, fixed)
    stages = run.journal.data.get("этапы") or []
    report = write_report(run, stages)
    print(f"Отчёт пересобран: {report}")
    print(f"Проверок: {len(run.journal.checks)}, "
          f"провалов: {run.journal.failed}, замечаний: {run.journal.warned}, "
          f"багов в таблице: {len(run.journal.bugs)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Полный QA-прогон AST")
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--no-shots", action="store_true", help="без скриншотов")
    parser.add_argument("--out", default="", help="папка отчёта")
    args = parser.parse_args()

    # DECISION: консоль Windows по умолчанию в cp1251 и падает с UnicodeEncodeError
    # на первом же ✅ — а прогон длинный, терять его из-за значков нельзя.
    # Переводим вывод в UTF-8 с заменой непередаваемых символов.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from core.logging_config import setup_logging

    setup_logging()

    date = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "logs" / f"qa_full_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = QaRun(out_dir, shots=not args.no_shots)

    # Настройки — пользовательский файл: сохраняем копию и в конце вернём как было
    settings = PROJECT_ROOT / "config" / "settings.json"
    backup = out_dir / "settings_before.json"
    if settings.exists():
        shutil.copy2(settings, backup)

    stages = list(ALL_STAGES) if args.stage == "all" else [args.stage]
    done: list[str] = []
    started = time.time()

    if args.stage == "report":
        # Пересборка отчёта: ничего не запускаем, только складываем то, что уже есть.
        stage_report(out_dir)
        return 0

    try:
        if "env" in stages:
            stage_env(run)
            done.append("env")

        if any(s in stages for s in NEEDS_WINDOW):
            run.launch()
            run.journal.check("окно приложения поднялось", run.app is not None,
                              f"{run.app.winfo_width()}x{run.app.winfo_height()} px")
            # Регрессия: запасной символ «⚙» рисовался поверх загруженной иконки,
            # и в шапке было две шестерёнки одна на другой.
            run.journal.check(
                "в шапке одна иконка меню, без символа поверх картинки",
                run.app.btn_menu.cget("text") == ""
                or getattr(run.app.btn_menu, "_ast_icon", None) is None,
                f"текст кнопки-меню: {run.app.btn_menu.cget('text')!r}")
            run.screenshot("start")

        if "s1" in stages:
            run.journal.save("s1_stems", stage_separate(
                run, SONG_1, PRESET_INSTRUMENTAL, "Песня 1: разделение",
                EXPECT_INSTRUMENTAL, "s1"))
            run.journal.save("s1_transcribe", stage_transcribe(
                run, SONG_1, "Песня 1: вкладка «Ноты»"))
            run.journal.save("s1_analyze", stage_analyze(
                run, SONG_1, "Песня 1: вкладка «Качество»"))
            run.journal.save("s1_settings", stage_settings(
                run, "Песня 1: вкладка «Настройки»"))
            run.journal.save("s1_player", stage_player(
                run, SONG_1, "Песня 1: плеер"))
            run.journal.save("s1_theme", stage_theme(run, "Песня 1: тема"))
            done.append("s1")

        if "s2" in stages:
            run.set_section("Песня 2: караоке")
            engine = pick_preset(run, PRESET_KARAOKE)
            run.screenshot("s2_preset")
            run.journal.check("пресет «для караоке» ведёт на модель Becruily karaoke",
                              engine == EXPECT_KARAOKE, f"выбрано: {engine}")
            from core.paths import MODELS_DIR
            local = (MODELS_DIR / "mel_band_roformer_karaoke_becruily.ckpt").exists()
            # Это не проверка «работает / не работает», а отметка о том, что путь
            # проверен не полностью: модели нет локально, а тянуть 1.6 ГБ
            # автономный прогон не имеет права (правило задачи: > 100 МБ —
            # только с разрешения). Поэтому ⚠️, а не ❌.
            run.journal.check(
                "модель Becruily karaoke есть локально",
                True if local else None,
                "скачана" if local else
                "нет: нужен ~1.6 ГБ из интернета — путь проверен на модели той же группы")
            run.journal.save("s2_karaoke_model", {"expected": EXPECT_KARAOKE,
                                                  "got": engine, "local": local})
            if local:
                run.journal.save("s2_stems", separate_with_selected(
                    run, SONG_2, "Песня 2: разделение", "s2"))
            else:
                # DECISION: 1.6 ГБ в автономном прогоне не тянем — правило задачи
                # разрешает остановиться на скачивании больше 100 МБ. Чтобы проверить
                # не только строку в списке, а ВЕСЬ путь (интерфейс → движок → стемы →
                # вкладки), берём модель из ТОЙ ЖЕ группы «🎤 Убрать вокал (для караоке)»,
                # которая уже лежит в models/ — Duality-V2.
                run.journal.check(
                    "путь «караоке» прогнан на локальной модели той же группы",
                    True, f"Duality-V2 вместо Becruily (обе — «Убрать вокал»); "
                          f"скачать Becruily — 1.6 ГБ, это отдельное решение")
                run.journal.save("s2_substitution", KARAOKE_LOCAL_FALLBACK)
                run.app.engine_combo.set(KARAOKE_LOCAL_FALLBACK)
                run.app._on_engine_change(KARAOKE_LOCAL_FALLBACK)
                run.pump(0.4)
                run.journal.save("s2_stems", separate_with_selected(
                    run, SONG_2, "Песня 2: разделение (локальная модель)", "s2"))
            done.append("s2")

        if "s3" in stages:
            run.journal.save("s3_stems", stage_separate(
                run, SONG_3, PRESET_FAST, "Песня 3: быстро и просто",
                EXPECT_FAST, "s3"))
            elapsed = (run.journal.data.get("s3_separate") or {}).get("seconds")
            run.journal.section = "Песня 3: быстро и просто"
            run.journal.check("Demucs уложился в минуту",
                              elapsed is not None and elapsed < 60.0,
                              f"{elapsed} с" if elapsed is not None else "нет замера")
            done.append("s3")

        if "edge" in stages:
            run.journal.save("edge", stage_edge(run, "Краевые случаи"))
            done.append("edge")

        if "theme" in stages:
            # Отдельный этап: только тема и меню. Нужен, чтобы перепроверить
            # оформление, не гоняя заново разделение (это минуты GPU).
            load_song(run, SONG_1, shot_name="theme_source")
            run.journal.save("theme", stage_theme(run, "Тема и меню"))
            done.append("theme")

        if "player" in stages:
            # Отдельный этап: только плеер. Стемы песни 1 к этому моменту уже
            # лежат в output/, поэтому разделение не нужно.
            load_song(run, SONG_1, shot_name="player_source")
            run.journal.save("player", stage_player(run, SONG_1, "Плеер"))
            done.append("player")

        if "analyze" in stages:
            # Отдельный этап: только проверки качества на готовой паре.
            load_song(run, SONG_1, shot_name="analyze_source")
            run.app.player_bar.select_listen("Вокал")
            run.app._on_listen_change("Вокал")
            run.journal.save("s1_analyze", stage_analyze(
                run, SONG_1, "Проверки качества"))
            done.append("analyze")

        if "close" in stages:
            run.journal.save("close", stage_close(run, "Закрытие окна во время задачи"))
            done.append("close")

    except Exception:
        run.journal.check("прогон дошёл до конца без падения драйвера", False,
                          traceback.format_exc()[-900:])
        logging.getLogger(__name__).exception("QA-прогон упал")
    finally:
        try:
            if run.app is not None and _safe_exists(run.app):
                run.app.worker.shutdown(wait=False)
                run.app.destroy()
        except Exception:
            pass
        try:
            if backup.exists():
                shutil.copy2(backup, settings)
        except Exception:
            logging.getLogger(__name__).exception("Не удалось вернуть settings.json")

        run.journal.save("модальные окна", run.mb.log)
        run.journal.save("этапы", done)
        run.journal.save("длительность_прогона_сек", round(time.time() - started, 1))
        for title, where, symptom, severity, reproducible, fixed in KNOWN_BUGS:
            run.journal.bug(title, where, symptom, severity, reproducible, fixed)
        report = write_report(run, done)
        print(f"\n{'=' * 70}")
        print(f"Отчёт:     {report}")
        print(f"Прогресс:  {run.out / 'progress.md'}")
        print(f"Данные:    {run.out / 'results.json'}")
        print(f"Скриншоты: {run.shots_dir}")
        print(f"{'=' * 70}")

    return 0 if run.journal.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
