r"""
Снимок витрины моделей: три пресета и полный список по задачам.

Запуск из корня проекта:

    .\.venv312\Scripts\python.exe tools\ui_selector_shot.py --theme dark
    .\.venv312\Scripts\python.exe tools\ui_selector_shot.py --theme dark --expanded

Зачем: выбор моделей — это то, что нельзя проверить тестами. Тесты сверяют,
что строки и группы на месте; а понять, «читается ли это человеку, который
не музыкант», можно только глазами. Снимок ложится в logs/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Снимок списка моделей AST")
    parser.add_argument("--theme", default="dark", choices=("dark", "light"))
    parser.add_argument("--expanded", action="store_true",
                        help="раскрыть полный список моделей")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from PIL import ImageGrab
    from ui.app_window import AppWindow

    app = AppWindow()
    if args.theme == "light":
        app._toggle_theme()

    app.update_idletasks()
    app.deiconify()
    app.lift()
    app.attributes("-topmost", True)
    app.focus_force()
    for _ in range(6):
        app.update()
        time.sleep(0.1)

    selector = app.engine_combo
    selector.open()
    if args.expanded:
        selector._toggle_list()
    app.update()
    time.sleep(0.6)
    app.update()
    app.update_idletasks()

    popup = selector._popup
    popup.update_idletasks()
    x, y = popup.winfo_rootx(), popup.winfo_rooty()
    w, h = popup.winfo_width(), popup.winfo_height()
    shot = ImageGrab.grab(bbox=(x - 2, y - 2, x + w + 2, y + h + 2), all_screens=True)

    name = f"_ui_selector_{args.theme}" + ("_expanded" if args.expanded else "")
    out = Path(args.out) if args.out else PROJECT_ROOT / "logs" / f"{name}.png"
    shot.save(out)
    print(f"Снимок: {out} ({shot.width}x{shot.height})")

    selector.close()
    app.attributes("-topmost", False)
    app.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
