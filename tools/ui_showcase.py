r"""
Все состояния интерфейса на снимках — для проверки оформления глазами.

Запуск (из корня проекта):

    .\.venv312\Scripts\python.exe tools\ui_showcase.py

Что делает: собирает окно, подкладывает временные файлы (исходник + вокал +
минус + MIDI) и снимает каждый экран: разделение, ноты, качество, настройки,
список моделей. Складывает PNG в logs/showcase/<тема>_<экран>.png.

Файлы создаются во ВРЕМЕННОЙ папке — настоящий output/ не трогается.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                      # noqa: E402
import soundfile as sf                  # noqa: E402
from PIL import ImageGrab               # noqa: E402

SR = 44100


def _write(path: Path, freq: float, seconds: float = 30.0) -> Path:
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    env = 0.4 + 0.35 * np.sin(2 * np.pi * 0.4 * t)
    mono = (0.45 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
    sf.write(str(path), np.repeat(mono[:, None], 2, axis=1), SR, subtype="PCM_16")
    return path


def _capture(app, out: Path, name: str) -> None:
    app.update_idletasks()
    app.lift()
    app.attributes("-topmost", True)
    app.focus_force()
    for _ in range(4):
        app.update()
        time.sleep(0.08)
    time.sleep(0.25)
    app.update()
    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(out / name)
    print(f"  {name}")


def main() -> int:
    from ui.app_window import AppWindow, TAB_ANALYZE, TAB_SETTINGS, TAB_TRANSCRIBE

    tmp = Path(tempfile.mkdtemp(prefix="ast_showcase_"))
    out = PROJECT_ROOT / "logs" / "showcase"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Временная папка: {tmp}")
    print(f"Снимки: {out}")

    mix = _write(tmp / "song.wav", 440.0)
    _write(tmp / "song_vocals.wav", 660.0)
    _write(tmp / "song_instrumental.wav", 220.0)
    (tmp / "song_vocals.mid").write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0")

    for theme in ("dark", "light"):
        app = AppWindow()
        if theme == "light":
            app._toggle_theme()
        app.output_dir = tmp
        app._load_file(str(mix))
        app.filetypes = getattr(app, "filetypes", None)

        _capture(app, out, f"{theme}_1_separate.png")

        # Состояние «разделено»: стемы находим на диске, кнопки включаются
        app._on_listen_change("Вокал")
        app._refresh_actions()
        app.update_idletasks()
        _capture(app, out, f"{theme}_2_result.png")

        app.tabs.set(TAB_TRANSCRIBE)
        _capture(app, out, f"{theme}_3_notes.png")

        app.tabs.set(TAB_ANALYZE)
        _capture(app, out, f"{theme}_4_quality.png")

        app.tabs.set(TAB_SETTINGS)
        _capture(app, out, f"{theme}_5_settings.png")

        app.tabs.set(TAB_TRANSCRIBE)
        app.engine_combo.open()
        _capture(app, out, f"{theme}_6_engines.png")
        app.engine_combo.close()

        app.destroy()

    print("Готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
