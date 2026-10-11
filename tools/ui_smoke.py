r"""
Ручной смоук-тест окна: собрать UI и сразу закрыть.

Запуск (из корня проекта):
    .\.venv312\Scripts\python.exe tools\ui_smoke.py

Проверяет, что окно собирается, что в нём четыре вкладки и что список движков
в интерфейсе совпадает с реестром core.engines. Разделение при этом НЕ
запускается.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engines import all_labels, engines_by_group  # noqa: E402
from ui.app_window import (  # noqa: E402
    TAB_ANALYZE,
    TAB_SEPARATE,
    TAB_SETTINGS,
    TAB_TRANSCRIBE,
    AppWindow,
)
from ui.theme import DROPZONE_HEIGHT  # noqa: E402


def main() -> int:
    # DECISION: консоль Windows по умолчанию в cp1251 и падает с
    # UnicodeEncodeError на первом же символе из подписи движка (в названиях
    # моделей есть «✅»). Проверка при этом падала раньше, чем что-то проверяла.
    # Переводим вывод в UTF-8: значок лучше, чем потерянная проверка.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    app = AppWindow()
    try:
        app.withdraw()                     # окно не мешает
        # update(), а не update_idletasks(): нужно, чтобы виджеты получили
        # реальные размеры — от них зависит отрисовка зоны файла и волны
        app.update()

        tabs = list(app.tabs._tab_dict.keys())
        expected_tabs = [TAB_SEPARATE, TAB_TRANSCRIBE, TAB_ANALYZE, TAB_SETTINGS]
        print("вкладки:", tabs)

        in_ui = app.engine_combo.values()
        expected = all_labels()
        print(f"движков в UI: {len(in_ui)}, в реестре: {len(expected)}")
        print(f"групп движков: {len(engines_by_group())}")
        print(f"выбран по умолчанию: {app.engine_combo.label()!r}")

        problems = []
        if tabs != expected_tabs:
            problems.append(f"вкладки не совпали: {tabs}")
        if sorted(in_ui) != sorted(expected):
            problems.append("список движков UI разошёлся с реестром")
        if app.wave.ax.axison:
            problems.append("волна без файла показывает оси вместо заглушки")
        if app.progress_row.winfo_manager() != "":
            problems.append("прогресс-бар виден, хотя задача не идёт")
        if not app.engine_combo.label():
            problems.append("движок по умолчанию не подставился в список")
        if int(app.drop.cget("height")) != DROPZONE_HEIGHT:
            problems.append("зона файла не той высоты")
        if len(app.drop.find_withtag("frame")) < 10:
            problems.append("у зоны файла нет пунктирной рамки")
        if app.btn_separate.cget("fg_color") != app.theme["accent"]:
            problems.append("«Разделить» перестала быть главной кнопкой")
        if app.btn_cancel.cget("fg_color") != app.theme["danger"]:
            problems.append("«Отмена» перестала быть красной")

        # Иконки: без них кнопки останутся без картинок
        from tools.make_icons import check as icons_check
        if icons_check() != 0:
            problems.append("иконки не готовы (tools/make_icons.py)")

        if problems:
            for p in problems:
                print("ПРОБЛЕМА:", p)
            return 1
        print("OK: UI собирается, 4 вкладки, список движков — из реестра, "
              "зона файла с пунктиром, кнопки с ролями, иконки на месте")
        return 0
    finally:
        app.destroy()


if __name__ == "__main__":
    sys.exit(main())
