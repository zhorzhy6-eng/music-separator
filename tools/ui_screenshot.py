r"""
Снимок окна: собрать интерфейс, загрузить тестовый трек и сохранить скриншот.

Запуск (из корня проекта):

    .\.venv312\Scripts\python.exe tools\ui_screenshot.py --theme dark --state loaded
    .\.venv312\Scripts\python.exe tools\ui_screenshot.py --theme light --state empty

Флаги:
    --theme dark|light   какая тема (по умолчанию dark)
    --state empty|loaded какое состояние показать (пустое окно / файл загружен)
    --out путь.png       куда сохранить (по умолчанию logs/_ui_<тема>_<состояние>.png)

Зачем: оформление нельзя проверить тестами — его надо увидеть. Скриншот
складывается в logs/, чтобы можно было сравнить «до» и «после».
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                      # noqa: E402
import soundfile as sf                  # noqa: E402


def make_sample(path: Path, seconds: float = 42.0, sr: int = 44100) -> Path:
    """Осмысленный тестовый трек: тон с затиханиями и «битами» — волна видна."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    env = 0.35 + 0.3 * np.sin(2 * np.pi * 0.35 * t)
    vocal = 0.35 * np.sin(2 * np.pi * 440 * t) * env
    beat = np.zeros_like(t)
    beat[:: int(sr / 2)] = 1.0
    music = 0.25 * np.sin(2 * np.pi * 110 * t) + 0.2 * beat
    mix = (vocal + music).astype(np.float32)
    stereo = np.repeat(mix[:, None], 2, axis=1)
    sf.write(str(path), stereo, sr, subtype="PCM_16")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Снимок окна AST")
    parser.add_argument("--theme", default="dark", choices=("dark", "light"))
    parser.add_argument("--state", default="loaded", choices=("empty", "loaded"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from PIL import ImageGrab
    from ui.app_window import AppWindow

    app = AppWindow()
    if args.theme == "light":
        app._toggle_theme()

    if args.state == "loaded":
        wav = make_sample(PROJECT_ROOT / "logs" / "_ui_sample.wav")
        app._load_file(str(wav))

    app.update_idletasks()
    app.deiconify()
    # Окно должно быть действительно видно: ImageGrab снимает экран, а не окно.
    # Иначе в кадр попадёт то, что случайно оказалось сверху.
    app.lift()
    app.attributes("-topmost", True)
    app.focus_force()
    for _ in range(6):
        app.update()
        time.sleep(0.12)
        app.update_idletasks()

    # Небольшая пауза: Windows должен успеть перерисовать окно после подъёма
    time.sleep(0.4)
    app.update()

    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    shot = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "logs" / f"_ui_{args.theme}_{args.state}.png")
    shot.save(out)
    print(f"Снимок: {out} ({shot.width}x{shot.height})")
    app.attributes("-topmost", False)
    app.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
