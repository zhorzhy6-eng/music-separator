"""
Проверки волны под плеером: «Слушать: Вокал/Минус» перерисовывает график.

Смысл улучшения: раньше при выборе «Вокал» менялся только звук, а на графике
оставался исходный трек — глазами одно, ушами другое.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from ui.components import WaveformView

tkinter = pytest.importorskip("tkinter")

SR = 44100


def _write(path, freq: float = 440.0, seconds: float = 0.5, channels: int = 2):
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    mono = 0.5 * np.sin(2 * np.pi * freq * t)
    data = mono if channels == 1 else np.repeat(mono[:, None], channels, axis=1)
    sf.write(str(path), data.astype(np.float32), SR, subtype="PCM_16")
    return path


def _make_window():
    try:
        from ui.app_window import AppWindow
        app = AppWindow()
    except Exception as e:                      # pragma: no cover
        pytest.skip(f"графика недоступна: {e}")
    app.withdraw()
    return app


@pytest.fixture
def view():
    """Отдельная волна в скрытом окне."""
    app = _make_window()
    widget = WaveformView(app)
    widget.pack()
    yield widget
    try:
        app.destroy()
    except Exception:
        pass


# ============================================================
#  WaveformView
# ============================================================
def test_plot_file_shows_file_name_by_default(view, tmp_path):
    path = _write(tmp_path / "track.wav")

    view.plot_file(path)

    assert "track.wav" in view.caption.cget("text")
    assert view._last_path == path


def test_plot_file_uses_given_caption(view, tmp_path):
    path = _write(tmp_path / "track_vocals.wav")

    view.plot_file(path, caption="Вокал: track_vocals.wav")

    assert view.caption.cget("text") == "Вокал: track_vocals.wav"


def test_plot_file_handles_mono_and_stereo(view, tmp_path):
    """Любой WAV из output/ должен рисоваться: и моно, и стерео."""
    for name, channels in (("mono.wav", 1), ("stereo.wav", 2)):
        path = _write(tmp_path / name, channels=channels)
        view.plot_file(path)
        assert name in view.caption.cget("text")


def test_plot_file_reports_broken_file(view, tmp_path):
    """Битый файл: пишем в подпись, а не молчим с прошлой картинкой."""
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav at all")

    view.plot_file(broken, caption="Вокал: broken.wav")

    assert "Не удалось" in view.caption.cget("text")


def test_clear_resets_caption_and_path(view, tmp_path):
    view.plot_file(_write(tmp_path / "track.wav"))

    view.clear()

    assert view._last_path is None
    assert "не выбрана" in view.caption.cget("text").lower()


def test_show_placeholder_hides_axes(view):
    """Заглушка: текст по центру, осей 0.0–1.0 нет."""
    view.show_placeholder("Загрузите файл, чтобы увидеть волну")

    assert view._last_path is None
    assert not view.ax.axison, "оси остались под заглушкой"
    drawn = [t.get_text() for t in view.ax.texts]
    assert any("Загрузите файл" in t for t in drawn), drawn


def test_plot_file_after_placeholder_returns_axes(view, tmp_path):
    """
    После загрузки файла заглушка исчезает, а на её месте — настоящая волна.

    Оси у волны выключены намеренно (редизайн): подписи «0.0–1.0» ничего не
    объясняли и занимали место. Поэтому проверяем не `axison`, а то, что
    нарисована волна и текста-заглушки больше нет.
    """
    view.show_placeholder()

    path = _write(tmp_path / "track.wav")
    view.plot_file(path)

    assert view.has_plot, "волна не нарисовалась после загрузки файла"
    assert view._last_path == path
    assert view.ax.texts == [] or not any(
        "Загрузите файл" in t.get_text() for t in view.ax.texts)


def test_theme_change_keeps_caption(view, tmp_path):
    """Смена темы перерисовывает волну — подпись не должна потеряться."""
    from ui.theme import get_theme

    view.plot_file(_write(tmp_path / "track.wav"), caption="Минус: track.wav")

    view.apply_theme(get_theme("light"))

    assert view.caption.cget("text") == "Минус: track.wav"


# ============================================================
#  Связка с окном
# ============================================================
def test_load_file_captions_source(tmp_path):
    app = _make_window()
    try:
        wav = _write(tmp_path / "song.wav")
        app._load_file(str(wav))

        caption = app.wave.caption.cget("text")
        assert "Исходный трек" in caption and "song.wav" in caption
    finally:
        app.destroy()


def test_listen_switch_redraws_wave(tmp_path):
    """Главная проверка улучшения: переключение «Слушать» меняет картинку."""
    app = _make_window()
    try:
        stem = "song"
        wav = _write(tmp_path / f"{stem}.wav")
        # Стемы «как после разделения» — окно ищет их по имени текущего трека
        vocals = _write(tmp_path / f"{stem}_vocals.wav", freq=880.0)
        instr = _write(tmp_path / f"{stem}_instrumental.wav", freq=220.0)
        app.output_dir = tmp_path
        app._load_file(str(wav))

        app._on_listen_change("Вокал")
        assert "Вокал" in app.wave.caption.cget("text")
        assert "vocals" in app.wave.caption.cget("text")
        assert app.wave._last_path == vocals
        assert app.player.current_file == vocals

        app._on_listen_change("Минус")
        assert "Минус" in app.wave.caption.cget("text")
        assert app.wave._last_path == instr
        assert app.player.current_file == instr

        app._on_listen_change("Исходный трек")
        assert "Исходный трек" in app.wave.caption.cget("text")
        assert app.wave._last_path == wav
        assert app.player.current_file == wav
    finally:
        app.destroy()


def test_listen_switch_without_stems_warns(tmp_path):
    """Стемов ещё нет — говорим об этом и не рисуем чужой файл."""
    app = _make_window()
    try:
        wav = _write(tmp_path / "song.wav")
        app.output_dir = tmp_path
        app._load_file(str(wav))

        app._on_listen_change("Вокал")

        assert "ещё нет" in app.status.cget("text")
        assert app.combo_listen.get() == "—"
        assert app.wave._last_path is None
    finally:
        app.destroy()


def test_listen_dash_clears_wave(tmp_path):
    app = _make_window()
    try:
        wav = _write(tmp_path / "song.wav")
        app._load_file(str(wav))

        app._on_listen_change("—")

        assert app.wave._last_path is None
        assert app.player.is_playing is False
        assert app.player.position == 0.0
    finally:
        app.destroy()


def test_switch_does_not_play_foreign_stems(tmp_path):
    """
    В output/ лежат стемы ДРУГОГО трека — они не должны ни играть, ни рисоваться.
    """
    app = _make_window()
    try:
        wav = _write(tmp_path / "song.wav")
        _write(tmp_path / "other_vocals.wav", freq=880.0)   # чужой вокал
        app.output_dir = tmp_path
        app._load_file(str(wav))

        app._on_listen_change("Вокал")

        assert app.wave._last_path is None
        assert app.combo_listen.get() == "—"
    finally:
        app.destroy()
