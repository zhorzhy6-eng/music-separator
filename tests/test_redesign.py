"""
Проверки оформления: палитра, кнопки, карточки, иконки, волна, плеер.

Смысл набора: редизайн легко «сломать» одной случайной правкой — снова вернуть
дефолтный серый цвет, потерять иконку или сделать кнопку неотличимой от фона.
Эти проверки ловят такие поломки без запуска приложения и без глаз.

Что НЕ проверяется: красиво ли это. Вкус тестом не измерить; здесь — только
«на месте ли то, что задумано»: цвета из темы, размеры, наличие иконок и
подсказок.
"""
from __future__ import annotations

import pytest

import customtkinter as ctk

from ui.theme import DARK, RADIUS_ROUND

tkinter = pytest.importorskip("tkinter")


def _make_window():
    try:
        from ui.app_window import AppWindow
        app = AppWindow()
    except Exception as e:                      # pragma: no cover
        pytest.skip(f"графика недоступна: {e}")
    app.withdraw()
    return app


# ============================================================
#  Палитра и типографика
# ============================================================
def test_theme_has_all_roles():
    """В обеих темах есть все цвета, на которые смотрит интерфейс."""
    from ui.theme import DARK, LIGHT

    required = ("bg", "panel", "panel_hover", "border", "accent", "accent_hover",
                "accent_dim", "accent_soft", "text", "text_dim", "text_muted",
                "success", "warning", "error", "danger", "danger_hover",
                "on_accent", "overlay")
    for name, theme in (("тёмная", DARK), ("светлая", LIGHT)):
        for key in required:
            assert key in theme, f"{name} тема: нет цвета {key}"
            assert str(theme[key]).startswith("#"), f"{name} тема: {key} не цвет"


def test_themes_are_not_default_customtkinter():
    """
    Цвета — подобранные, а не дефолт CustomTkinter.

    Дефолт (#1a1a1a / #242424 / #3b8ed0) — ровно то, из-за чего интерфейс
    выглядел «серой панелью инженера».
    """
    from ui.theme import DARK

    assert DARK["bg"].lower() != "#1a1a1a"
    assert DARK["panel"].lower() != "#242424"
    assert DARK["accent"].lower() != "#3b8ed0"


def test_light_theme_is_not_a_mirror_of_dark():
    """Светлая тема проработана отдельно, а не «инверсия» тёмной."""
    from ui.theme import DARK, LIGHT

    assert LIGHT["accent"] != DARK["accent"], "акцент одинаков — светлая не проработана"
    assert LIGHT["bg"].lower() == "#f8f9fa"
    assert LIGHT["panel"].lower() == "#ffffff"


def test_font_scale_is_ordered():
    """Заголовок крупнее кнопки, подпись мельче основного текста."""
    from ui.theme import FONT_BODY, FONT_BUTTON, FONT_CAPTION, FONT_TITLE

    assert FONT_TITLE > FONT_BUTTON >= FONT_BODY > FONT_CAPTION


def test_fonts_are_available_and_named():
    """Шрифт интерфейса выбран из списка и реально существует в системе."""
    from tkinter import font as tkfont

    from ui.theme import FONT_CANDIDATES, font_family, font_spec, mono_family

    app = _make_window()
    try:
        family = font_family(app)
        available = {name.lower() for name in tkfont.families(app)}
        assert family.lower() in available, f"шрифта {family!r} нет в системе"
        assert family in FONT_CANDIDATES or family == "Segoe UI"
        assert mono_family(app)
        spec = font_spec(13, "bold")
        assert spec[1] == 13 and spec[2] == "bold"
    finally:
        app.destroy()


# ============================================================
#  Кнопки: иерархия
# ============================================================
def test_primary_button_is_filled_with_accent():
    from ui.button_styles import primary_button
    from ui.theme import DARK

    root = ctk.CTk()
    root.withdraw()
    try:
        btn = primary_button(root, "Разделить", lambda: None, DARK)
        assert btn.cget("fg_color") == DARK["accent"]
        assert btn.cget("text_color") == DARK["on_accent"]
        assert btn.cget("height") == 40
        assert btn.cget("corner_radius") == 8
    finally:
        root.destroy()


def test_secondary_button_is_outlined():
    from ui.button_styles import secondary_button
    from ui.theme import DARK

    root = ctk.CTk()
    root.withdraw()
    try:
        btn = secondary_button(root, "Вокал в ноты", lambda: None, DARK)
        assert btn.cget("border_width") == 1
        assert btn.cget("border_color") == DARK["accent"]
        assert btn.cget("text_color") == DARK["accent"]
    finally:
        root.destroy()


def test_danger_button_is_red():
    """«Отмена» отличается от остальных кнопок — её нельзя нажать случайно."""
    from ui.button_styles import danger_button
    from ui.theme import DARK

    root = ctk.CTk()
    root.withdraw()
    try:
        btn = danger_button(root, "Отмена", lambda: None, DARK)
        assert btn.cget("fg_color") == DARK["danger"]
    finally:
        root.destroy()


def test_ghost_button_has_no_frame():
    from ui.button_styles import ghost_button
    from ui.theme import DARK

    root = ctk.CTk()
    root.withdraw()
    try:
        btn = ghost_button(root, "Открыть папку", lambda: None, DARK)
        assert btn.cget("border_width") == 0
        assert btn.cget("text_color") == DARK["text_dim"]
    finally:
        root.destroy()


def test_window_buttons_have_roles(tmp_path):
    """Главная кнопка — акцентная, отмена — красная, остальные — второго плана."""
    import numpy as np
    import soundfile as sf

    from ui.theme import DARK

    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app._load_file(str(wav))
        assert app.btn_separate.cget("fg_color") == DARK["accent"], \
            "«Разделить» перестала быть главной кнопкой"
        assert app.btn_cancel.cget("fg_color") == DARK["danger"]
        for name in ("btn_voc_midi", "btn_inst_midi", "btn_piano",
                     "btn_analyze", "btn_analyze_midi", "btn_super_pair"):
            btn = getattr(app, name)
            assert btn.cget("border_width") == 1, f"{name}: нет обводки второго плана"
        assert app.btn_open_output.cget("border_width") == 0
    finally:
        app.destroy()


def test_recolor_follows_theme(tmp_path):
    """Смена темы перекрашивает кнопки, а не оставляет их в старых цветах."""
    from ui.theme import DARK, LIGHT

    app = _make_window()
    try:
        assert app.btn_separate.cget("fg_color") == DARK["accent"]
        app._toggle_theme()
        assert app.btn_separate.cget("fg_color") == LIGHT["accent"]
        assert app.card_engine.cget("fg_color") == LIGHT["panel"]
        assert app.drop.cget("bg") == LIGHT["bg"]
        assert app.wave.cget("fg_color") == LIGHT["panel"]
    finally:
        app.destroy()


def test_no_widget_keeps_default_grey_after_theme_switch():
    """
    После переключения темы в окне не остаётся виджетов с дефолтным серым.

    Именно так выглядела ошибка «половина окна осталась тёмной»: CustomTkinter
    даёт своим виджетам собственный серый цвет, и забытый виджет выбивается.
    """
    app = _make_window()
    try:
        app._toggle_theme()                   # уходим в светлую тему
        app.update_idletasks()

        defaults = {"gray14", "gray17", "gray20", "gray24", "gray86", "gray90"}
        stale: list[str] = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    fg = str(child.cget("fg_color"))
                except Exception:
                    fg = ""
                if fg in defaults or fg.upper() in {"#1A1A1A", "#242424"}:
                    stale.append(f"{type(child).__name__}:{fg}")
                walk(child)

        walk(app)
        assert stale == [], f"остались цвета старой темы: {stale}"
    finally:
        app.destroy()


# ============================================================
#  Иконки
# ============================================================
def test_icons_are_generated():
    """Иконки Lucide лежат на диске — иначе кнопки останутся без картинок."""
    from pathlib import Path

    from ui.icons_catalog import ICON_NAMES, ICON_SIZES

    icons_dir = Path(__file__).resolve().parent.parent / "assets" / "icons"
    missing = [f"{name}_{size}.png" for name in ICON_NAMES for size in ICON_SIZES
               if not (icons_dir / f"{name}_{size}.png").exists()]
    assert missing == [], (
        f"нет иконок: {missing[:5]}… Запустите tools/make_icons.py")


def test_get_icon_returns_image_and_caches():
    from ui import icons as icon_lib

    root = ctk.CTk()
    root.withdraw()
    try:
        icon_lib.clear_cache()
        first = icon_lib.get_icon("scissors", 20, "#4A9EFF", master=root)
        assert first is not None, "иконка не загрузилась"
        assert first.cget("size") == (20, 20)
        assert icon_lib.get_icon("scissors", 20, "#4A9EFF", master=root) is first, \
            "иконка не кешируется"
        # Другой цвет — другая картинка (цвет «вшит» в изображение)
        assert icon_lib.get_icon("scissors", 20, "#E8E8E8", master=root) is not first
        assert icon_lib.get_icon("нет-такой-иконки", 20, "#fff", master=root) is None
    finally:
        root.destroy()


def test_icon_pixels_are_tinted():
    """Иконка действительно перекрашивается, а не остаётся белой."""
    from PIL import Image

    from ui.icons import ICONS_DIR, _tint

    with Image.open(ICONS_DIR / "music_32.png") as raw:
        tinted = _tint(raw.convert("RGBA"), "#FF0000")
    pixels = tinted.load()
    colored = [pixels[x, y] for y in range(tinted.height)
               for x in range(tinted.width) if pixels[x, y][3] > 200]
    assert colored, "в иконке нет плотных пикселей"
    assert all(px[0] > 200 and px[1] < 60 for px in colored), "цвет не применился"


def test_buttons_use_icons_not_emoji(tmp_path):
    """В подписях кнопок не осталось эмодзи: только текст и картинка-иконка."""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app._load_file(str(wav))
        emoji = ("🔀", "🎼", "📊", "🧠", "⏹", "📂", "📈", "💾", "🎹", "↩")
        for name in ("btn_separate", "btn_voc_midi", "btn_inst_midi", "btn_piano",
                     "btn_analyze", "btn_analyze_midi", "btn_super_pair",
                     "btn_open_output", "btn_open_analysis", "btn_cancel"):
            text = str(getattr(app, name).cget("text"))
            assert not any(ch in text for ch in emoji), f"{name}: эмодзи в подписи {text!r}"
    finally:
        app.destroy()


# ============================================================
#  Карточки и раскладка
# ============================================================
def test_tabs_are_split_into_cards():
    """Содержимое вкладок разложено по карточкам, а не висит на пустом фоне."""
    from ui.card import Card

    app = _make_window()
    try:
        assert isinstance(app.card_engine, Card)
        assert isinstance(app.card_transcribe, Card)
        assert isinstance(app.card_analyze, Card)
        for card in (app.card_engine, app.card_transcribe, app.card_analyze):
            assert card.cget("corner_radius") == 12
            assert card.body is not None
        # Кнопки лежат ВНУТРИ карточек своих вкладок
        assert app.card_transcribe.body in _ancestors(app.btn_voc_midi)
        assert app.card_analyze.body in _ancestors(app.btn_super_pair)
        # «Разделить» — главное действие вкладки: оно под карточкой способа
        # разделения, а не внутри неё (иначе теряется смысл «главной кнопки»)
        assert app.tab_separate in _ancestors(app.btn_separate)
        assert app.card_engine not in _ancestors(app.btn_separate)
    finally:
        app.destroy()


def _ancestors(widget):
    chain = []
    w = widget
    while w is not None:
        chain.append(w)
        w = getattr(w, "master", None)
    return chain


def test_card_title_and_theme_switch():
    from ui.card import Card
    from ui.theme import DARK, LIGHT

    root = ctk.CTk()
    root.withdraw()
    try:
        card = Card(root, "Заголовок", DARK, subtitle="подпись")
        assert card.title_label.cget("text") == "Заголовок"
        card.apply_theme(LIGHT)
        assert card.cget("fg_color") == LIGHT["panel"]
        card.set_subtitle("новое")
        assert card.subtitle_label.cget("text") == "новое"
    finally:
        root.destroy()


# ============================================================
#  Зона файла, волна, плеер
# ============================================================
def test_dropzone_shows_zone_then_file_card(tmp_path):
    """До загрузки — зона 140 px, после — компактная карточка файла."""
    import numpy as np
    import soundfile as sf

    from ui.components import FILE_CARD_HEIGHT

    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
    # Стерео-версия: в карточке файла должно быть написано «стерео», а не «моно»
    stereo = tmp_path / "song_stereo.wav"
    sf.write(str(stereo), np.zeros((44100, 2), dtype=np.float32), 44100,
             subtype="PCM_16")

    app = _make_window()
    try:
        assert app.drop.current_file is None
        assert int(app.drop.cget("height")) == 140, "зона не той высоты"

        app._load_file(str(stereo))

        assert app.drop.current_file == stereo
        assert int(app.drop.cget("height")) == FILE_CARD_HEIGHT, "зона не ужалась"
        assert app.drop._file.name_label.cget("text") == "song_stereo.wav"
        meta = app.drop._file.meta_label.cget("text")
        assert "1 с" in meta and "44.1 кГц" in meta and "стерео" in meta, meta

        # Моно тоже описывается словами, а не цифрой каналов
        app._load_file(str(wav))
        assert "моно" in app.drop._file.meta_label.cget("text")

        app.drop.reset()
        assert int(app.drop.cget("height")) == 140
    finally:
        app.destroy()


def test_dropzone_draws_dashed_border(tmp_path):
    """
    Зона файла рисует ПУНКТИРНУЮ рамку, а при перетаскивании — «бегущую».

    Рамка нарисована на Canvas: только так пунктир со скруглёнными углами
    остаётся ровным при любом размере окна.
    """
    app = _make_window()
    try:
        app.update_idletasks()
        app.drop._redraw()
        dashed = len(app.drop.find_withtag("frame"))
        assert dashed > 10, f"пунктир не нарисован (линий: {dashed})"

        was = app.drop._dash_phase
        app.drop.set_drag_active(True)
        app.drop._tick_dash()                 # один кадр анимации
        assert app.drop._dash_phase != was, "пунктир не двигается"
        app.drop.set_drag_active(False)
        assert app.drop._dash_job is None, "таймер пунктира не остановился"
    finally:
        app.destroy()


def test_dropzone_center_has_icon_and_text(tmp_path):
    """В зоне есть иконка загрузки и обе подписи: что делать и как ещё можно."""
    app = _make_window()
    try:
        app.update_idletasks()
        app.drop._redraw()
        texts = [app.drop.itemcget(item, "text")
                 for item in app.drop.find_all()
                 if app.drop.type(item) == "text"]
        images = [item for item in app.drop.find_all()
                  if app.drop.type(item) == "image"]
        assert any("Перетащите файл" in t for t in texts), texts
        assert any("нажмите" in t for t in texts), texts
        assert images, "в зоне нет иконки"
    finally:
        app.destroy()


def test_wave_height_switches_between_placeholder_and_plot(tmp_path):
    """Заглушка ниже (80), волна выше (120) — место не тратится впустую."""
    import numpy as np
    import soundfile as sf

    from ui.theme import WAVE_HEIGHT, WAVE_PLACEHOLDER_HEIGHT

    wav = tmp_path / "song.wav"
    t = np.linspace(0, 0.5, 22050)
    sf.write(str(wav), (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
             44100, subtype="PCM_16")

    app = _make_window()
    try:
        canvas = app.wave.canvas.get_tk_widget()
        assert int(canvas.cget("height")) == WAVE_PLACEHOLDER_HEIGHT
        assert app.wave.has_plot is False

        app._load_file(str(wav))

        assert int(canvas.cget("height")) == WAVE_HEIGHT
        assert app.wave.has_plot is True
        # Оси выключены: подписи «0.0–1.0» ничего не объясняли
        assert not app.wave.ax.axison
    finally:
        app.destroy()


def test_wave_caption_has_name_and_duration(tmp_path):
    """Под волной — имя файла и длительность, а не только имя."""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app._load_file(str(wav))
        caption = app.wave.caption.cget("text")
        assert "Исходный трек" in caption and "song.wav" in caption
        assert "0:01" in caption, f"в подписи нет длительности: {caption!r}"
    finally:
        app.destroy()


def test_player_is_a_card_with_round_buttons():
    """Плеер — карточка: скругление 12, кнопки-кружки, моноширинный таймер."""
    from ui.card import Card

    app = _make_window()
    try:
        bar = app.player_bar
        assert isinstance(bar, Card)
        assert bar.cget("corner_radius") == 12
        assert bar.btn_play.cget("corner_radius") == RADIUS_ROUND
        assert bar.btn_play.cget("width") == RADIUS_ROUND * 2
        assert bar.cget("fg_color") == DARK["panel"]

        bar.set_playing(True)
        assert bar.btn_pause.cget("fg_color") == DARK["accent"]
        bar.set_playing(False)
        assert bar.btn_play.cget("fg_color") == DARK["accent"]

        bar.set_duration(75)
        bar.set_position(35)
        assert bar.time_label.cget("text") == "0:35 / 1:15"
    finally:
        app.destroy()


def test_volume_shows_percent():
    """Рядом с громкостью видно число: непонятно «насколько тихо» без него."""
    from ui.components import PlayerBar
    from ui.theme import DARK

    root = ctk.CTk()
    root.withdraw()
    try:
        seen: list[float] = []
        bar = PlayerBar(root, on_play=lambda: None, on_pause=lambda: None,
                        on_stop=lambda: None, on_volume=seen.append,
                        on_seek=lambda s: None, theme=DARK)
        bar.vol.set(0.35)
        bar.vol._command(0.35)              # как будто человек подвинул ползунок
        assert bar.vol_value.cget("text") == "35%"
        assert seen and abs(seen[-1] - 0.35) < 1e-6, "значение не дошло до плеера"
    finally:
        root.destroy()


def test_listen_choice_updates_player_title():
    """Выбор «Слушать» виден и в списке, и в заголовке карточки плеера."""
    from ui.components import PlayerBar
    from ui.theme import DARK

    root = ctk.CTk()
    root.withdraw()
    try:
        bar = PlayerBar(root, on_play=lambda: None, on_pause=lambda: None,
                        on_stop=lambda: None, on_volume=lambda v: None,
                        on_seek=lambda s: None, on_listen=lambda v: None,
                        theme=DARK)
        assert bar.combo_listen is not None, "список «Слушать» пропал из плеера"
        bar.select_listen("Вокал")
        assert bar.combo_listen.get() == "Вокал"
        assert bar.subtitle_label.cget("text") == "Вокал"
        bar.select_listen("—")
        assert bar.subtitle_label.cget("text") == ""
    finally:
        root.destroy()


# ============================================================
#  Понятный язык
# ============================================================
def test_shortcuts_explain_what_to_do(tmp_path):
    """Подсказка-шаг говорит, что делать дальше, а не что произошло."""
    import numpy as np
    import soundfile as sf

    from ui.app_window import HINT_CHOOSE_FILE, HINT_LISTEN, HINT_PRESS_SPLIT

    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        assert app.hint.cget("text") == HINT_CHOOSE_FILE

        app._load_file(str(wav))
        assert app.hint.cget("text") == HINT_PRESS_SPLIT

        # Появляются стемы — подсказка переходит к прослушиванию
        for kind in ("vocals", "instrumental"):
            sf.write(str(tmp_path / f"song_{kind}.wav"),
                     np.zeros(1000, dtype=np.float32), 44100, subtype="PCM_16")
        app.output_dir = tmp_path
        app._refresh_actions()
        assert app.hint.cget("text") == HINT_LISTEN
    finally:
        app.destroy()


def test_progress_is_described_in_plain_words():
    """«Разделяем: обрабатываем голос… 75%» вместо «Demucs: разделение — 75%»."""
    from core.worker import ProgressEvent, TaskType

    app = _make_window()
    try:
        app._current_task_type = TaskType.SEPARATE
        app._handle_event(ProgressEvent(task_id=1, progress=0.75,
                                        stage="Обработка вокала"))
        text = app.status.cget("text")
        assert text == "Разделяем: обрабатываем голос… 75%", text

        app._current_task_type = TaskType.ANALYZE_PAIR
        app._handle_event(ProgressEvent(task_id=1, progress=0.45,
                                        stage="Phase Smearing"))
        text = app.status.cget("text")
        assert "смазанность атаки" in text, text
        assert "Phase" not in text, text
    finally:
        app.destroy()


def test_settings_labels_avoid_jargon():
    """В подписях настроек нет технических слов без объяснения."""
    from ui.settings_dialog import FIELDS

    jargon = ("shifts", "overlap", "fp16", "batch", "Notch", "dBFS", "SDR")
    for field in FIELDS:
        for word in jargon:
            assert word.lower() not in field.label.lower(), \
                f"{field.name}: жаргон в подписи «{field.label}»"


def test_plain_words_dictionary_covers_report_terms():
    """Термины из отчётов объяснены простыми словами."""
    from ui.theme import PLAIN_WORDS, plain

    for term in ("SDR", "Bleed", "VPI", "Phase Smearing", "Spectral Tilt",
                 "Transient Loss"):
        assert plain(term), f"нет объяснения для {term}"
        assert len(PLAIN_WORDS[term]) > 20
