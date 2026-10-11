"""
Проверки диалога настроек («⚙ Настройки»).

Две части:
  * чистые функции (validate_values / apply_values / config_to_values) — без окна;
  * сам диалог и его связка с главным окном — окно поднимается скрытым.

Правило то же, что и в остальных тестах: файл config/settings.json пользователя
не трогаем — подменяем путь на временный.
"""
from __future__ import annotations

import json

import pytest

from core.config import AppConfig
from ui.settings_dialog import (
    FIELDS,
    FIELDS_BY_NAME,
    SettingsDialog,
    apply_values,
    config_to_values,
    defaults_values,
    validate_values,
)

tkinter = pytest.importorskip("tkinter")


# ============================================================
#  Чистые функции
# ============================================================
def test_every_config_field_is_in_dialog():
    """
    Настройка без поля в диалоге — это настройка, которую нельзя изменить
    (именно так и было до этой задачи). Проверяем, что такого нет.
    """
    config_fields = set(AppConfig.__dataclass_fields__)
    dialog_fields = {f.name for f in FIELDS}
    assert config_fields == dialog_fields, (
        f"нет в диалоге: {config_fields - dialog_fields}; "
        f"лишние: {dialog_fields - config_fields}")


def test_defaults_match_appconfig():
    """Значения по умолчанию в диалоге — те же, что в AppConfig."""
    cfg = AppConfig()
    for name, value in defaults_values().items():
        assert value == getattr(cfg, name), f"{name}: {value} != {getattr(cfg, name)}"


def test_default_values_pass_validation():
    assert validate_values(defaults_values()) == []


def test_config_to_values_roundtrip():
    """Настройки → значения полей → настройки: ничего не теряется."""
    cfg = AppConfig(demucs_device="cpu", target_peak=0.75, bs_batch_size=4,
                    bs_use_fp16=True, notch_enabled=True)
    restored = apply_values(cfg, config_to_values(cfg))
    assert restored == cfg


@pytest.mark.parametrize("name,value,expected_part", [
    ("demucs_shifts", 0, "Точность разделения"),
    ("demucs_shifts", 50, "Точность разделения"),
    ("demucs_overlap", 1.5, "Сглаживание стыков"),
    ("demucs_overlap", -0.1, "Сглаживание стыков"),
    ("target_peak", 1.5, "Громкость результата"),
    ("target_peak", 0.0, "Громкость результата"),
    ("output_sample_rate", 100, "Частота"),
    ("bs_batch_size", 64, "кусков"),
    ("demucs_device", "tpu", "видеокарте"),
    ("default_engine", "такого движка нет", "моделей"),
])
def test_bad_values_are_rejected(name, value, expected_part):
    values = defaults_values()
    values[name] = value
    problems = validate_values(values)
    assert any(expected_part in p for p in problems), f"не поймали {name}={value}: {problems}"


def test_apply_values_changes_only_listed_fields():
    """Поля, которых нет во входном словаре, остаются как были."""
    cfg = AppConfig(demucs_shifts=3, keep_temp_dirs=True)
    updated = apply_values(cfg, {"target_peak": 0.7})
    assert updated.target_peak == 0.7
    assert updated.demucs_shifts == 3
    assert updated.keep_temp_dirs is True


@pytest.mark.parametrize("name,value,expected", [
    # Слайдер всегда отдаёт float, а в настройке лежит int
    ("demucs_shifts", 4.0, 4),
    ("bs_batch_size", "4", 4),         # OptionMenu отдаёт строку
    ("output_sample_rate", "48000", 48000),
    ("target_peak", 0.75, 0.75),
    ("notch_enabled", 1, True),        # Switch отдаёт 0/1
    ("bs_use_fp16", 0, False),
    ("demucs_device", "cpu", "cpu"),
])
def test_apply_values_coerces_types(name, value, expected):
    updated = apply_values(AppConfig(), {name: value})
    result = getattr(updated, name)
    assert result == expected
    assert isinstance(result, type(expected)), f"{name}: тип {type(result)}"


def test_field_choices_cover_config_defaults():
    """Значение по умолчанию обязано быть среди вариантов выбора."""
    for field in FIELDS:
        if field.kind != "choice":
            continue
        default = str(getattr(AppConfig(), field.name))
        assert default in field.choices, f"{field.name}: {default} нет в {field.choices}"


# ============================================================
#  Диалог
# ============================================================
def _make_window():
    try:
        from ui.app_window import AppWindow
        app = AppWindow()
    except Exception as e:                      # pragma: no cover
        pytest.skip(f"графика недоступна: {e}")
    app.withdraw()
    return app


def _safe_destroy(widget) -> None:
    """Закрывает окно, если оно ещё живо (после сохранения диалог уже закрыт)."""
    try:
        if widget is not None and widget.winfo_exists():
            widget.destroy()
    except Exception:
        pass


def test_dialog_shows_current_config():
    app = _make_window()
    dialog = None
    try:
        cfg = AppConfig(demucs_device="cpu", demucs_shifts=5, target_peak=0.7,
                        bs_batch_size=2, notch_enabled=True)
        dialog = SettingsDialog(app, cfg, on_save=lambda c: None)
        dialog.withdraw()

        values = dialog.collect()
        assert values["demucs_device"] == "cpu"
        assert values["demucs_shifts"] == 5
        assert float(values["target_peak"]) == pytest.approx(0.7)
        assert values["bs_batch_size"] == "2"
        assert bool(values["notch_enabled"]) is True
    finally:
        if dialog is not None:
            dialog.destroy()
        app.destroy()


def test_dialog_save_returns_new_config(monkeypatch):
    """«Сохранить» отдаёт готовый AppConfig — файл пишет уже главное окно."""
    app = _make_window()
    dialog = None
    saved = []
    try:
        monkeypatch.setattr("ui.settings_dialog.messagebox.showerror",
                            lambda *a, **k: None)
        dialog = SettingsDialog(app, AppConfig(), on_save=saved.append)
        dialog.withdraw()

        dialog.vars["demucs_device"].set("cpu")
        dialog.vars["demucs_shifts"].set(4.0)
        dialog.vars["bs_use_fp16"].set(True)
        dialog._save()

        assert len(saved) == 1
        assert saved[0].demucs_device == "cpu"
        assert saved[0].demucs_shifts == 4
        assert saved[0].bs_use_fp16 is True
        assert saved[0].target_peak == AppConfig().target_peak   # не тронуто
    finally:
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        app.destroy()


def test_dialog_does_not_save_broken_values(monkeypatch):
    """Некорректное значение: окно с ошибкой, файл не трогаем."""
    app = _make_window()
    dialog = None
    saved = []
    shown = []
    try:
        monkeypatch.setattr("ui.settings_dialog.messagebox.showerror",
                            lambda title, text, *a, **k: shown.append(text))
        dialog = SettingsDialog(app, AppConfig(), on_save=saved.append)
        dialog.withdraw()

        dialog.vars["demucs_device"].set("cpu")
        dialog.vars["target_peak"].set(1.5)      # вне допустимого
        dialog._save()

        assert saved == [], "сохранили заведомо неверные настройки"
        assert shown and "Громкость результата" in shown[0]
        assert dialog.winfo_exists(), "окно закрылось, хотя сохранить не вышло"
    finally:
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        app.destroy()


def test_dialog_reset_returns_defaults():
    app = _make_window()
    dialog = None
    try:
        dialog = SettingsDialog(app, AppConfig(demucs_device="cpu", demucs_shifts=7),
                                on_save=lambda c: None)
        dialog.withdraw()
        assert dialog.collect()["demucs_device"] == "cpu"

        dialog._reset()

        values = dialog.collect()
        assert values["demucs_device"] == AppConfig().demucs_device
        assert values["demucs_shifts"] == AppConfig().demucs_shifts
    finally:
        if dialog is not None:
            dialog.destroy()
        app.destroy()


# ============================================================
#  Связка с главным окном
# ============================================================
@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """settings.json — во временном каталоге, кэш настроек восстанавливается."""
    import core.config as config_module

    path = tmp_path / "settings.json"
    monkeypatch.setattr(config_module, "SETTINGS_FILE", path)
    saved_cache = config_module._cached_config
    yield path
    config_module._cached_config = saved_cache


def test_window_saves_settings_to_file(isolated_settings):
    """Кнопка «Сохранить» доводит дело до файла и до состояния окна."""
    app = _make_window()
    try:
        app._apply_settings(AppConfig(demucs_device="cpu", target_peak=0.8))

        written = json.loads(isolated_settings.read_text(encoding="utf-8"))
        assert written["demucs_device"] == "cpu"
        assert written["target_peak"] == 0.8
        assert app.config_data.demucs_device == "cpu"
        assert app.config_data.target_peak == 0.8
    finally:
        app.destroy()


def test_window_updates_engine_combo_after_save(isolated_settings):
    """Сменили движок по умолчанию — в списке он же, без перезапуска."""
    from core.engines import get_engine

    other = get_engine("demucs:htdemucs")
    app = _make_window()
    try:
        app._apply_settings(AppConfig(default_engine=other.id))
        assert app.engine_combo.get() == other.id
        assert app.settings_panel.collect()["default_engine"] == other.id
    finally:
        app.destroy()


def test_demucs_device_from_settings_reaches_engine(isolated_settings):
    """
    Настройка demucs_device действительно доходит до движка Demucs.

    Ровно этот сценарий проверялся руками: поставить в настройках cpu —
    Demucs должен собраться на процессоре (в логе device=cpu).
    """
    from core.config import get_config
    from core.engines.demucs_engine import DemucsEngine

    AppConfig(demucs_device="cpu").save()
    engine = DemucsEngine(model_name="htdemucs")

    assert engine._resolve_device(get_config(reload=True).demucs_device) == "cpu"

    AppConfig(demucs_device="cuda").save()
    assert engine._resolve_device(get_config(reload=True).demucs_device) == "cuda"


def test_settings_button_exists():
    """Меню «⚙» живёт в шапке одной кнопкой (а не двумя кнопками, как раньше)."""
    app = _make_window()
    try:
        header = app.winfo_children()[0]
        assert app.btn_menu in header.winfo_children()
        # Инвариант «иконка ИЛИ запасной символ»: раньше кнопка несла текст «⚙»
        # всегда и рисовала его поверх загруженной картинки.
        assert (app.btn_menu.cget("text") == ""
                or getattr(app.btn_menu, "_ast_icon", None) is None)

        app._go_to_settings()
        assert app.tabs.get() == "Настройки"
    finally:
        app.destroy()


# ============================================================
#  Вкладка «⚙ Настройки» (бывший диалог)
# ============================================================
def _make_panel(config: AppConfig | None = None, on_save=None):
    """Панель настроек в скрытом окне. Нет графики — тест пропускается."""
    from ui.settings_panel import SettingsPanel

    app = _make_window()
    panel = SettingsPanel(app, config or AppConfig(), on_save=on_save or (lambda c: None))
    panel.pack()
    app.withdraw()
    return app, panel


def test_panel_shows_current_config():
    app, panel = _make_panel(AppConfig(demucs_device="cpu", demucs_shifts=5,
                                       target_peak=0.7, bs_batch_size=2,
                                       notch_enabled=True))
    try:
        values = panel.collect()
        assert values["demucs_device"] == "cpu"
        assert values["demucs_shifts"] == 5
        assert float(values["target_peak"]) == pytest.approx(0.7)
        assert values["bs_batch_size"] == "2"
        assert bool(values["notch_enabled"]) is True
    finally:
        app.destroy()


def test_panel_save_returns_new_config(monkeypatch):
    """«Сохранить» отдаёт готовый AppConfig — файл пишет уже главное окно."""
    saved = []
    app, panel = _make_panel(AppConfig(), on_save=saved.append)
    try:
        monkeypatch.setattr("ui.settings_panel.messagebox.showerror",
                            lambda *a, **k: None)
        panel.vars["demucs_device"].set("cpu")
        panel.vars["demucs_shifts"].set(4.0)
        panel.vars["bs_use_fp16"].set(True)
        panel._save()

        assert len(saved) == 1
        assert saved[0].demucs_device == "cpu"
        assert saved[0].demucs_shifts == 4
        assert saved[0].bs_use_fp16 is True
        assert saved[0].target_peak == AppConfig().target_peak   # не тронуто
    finally:
        app.destroy()


def test_panel_does_not_save_broken_values(monkeypatch):
    """Некорректное значение: окно с ошибкой, файл не трогаем."""
    saved = []
    shown = []
    app, panel = _make_panel(AppConfig(), on_save=saved.append)
    try:
        monkeypatch.setattr("ui.settings_panel.messagebox.showerror",
                            lambda title, text, *a, **k: shown.append(text))
        panel.vars["demucs_device"].set("cpu")
        panel.vars["target_peak"].set(1.5)      # вне допустимого
        panel._save()

        assert saved == [], "сохранили заведомо неверные настройки"
        assert shown and "Громкость результата" in shown[0]
    finally:
        app.destroy()


def test_panel_reset_returns_defaults():
    app, panel = _make_panel(AppConfig(demucs_device="cpu", demucs_shifts=7))
    try:
        assert panel.collect()["demucs_device"] == "cpu"

        panel._reset()

        values = panel.collect()
        assert values["demucs_device"] == AppConfig().demucs_device
        assert values["demucs_shifts"] == AppConfig().demucs_shifts
    finally:
        app.destroy()


def test_panel_has_every_config_field():
    """На вкладке настроек есть ВСЕ поля AppConfig — иначе настройку не поменять."""
    app, panel = _make_panel()
    try:
        assert set(panel.collect()) == set(AppConfig.__dataclass_fields__)
    finally:
        app.destroy()


def test_panel_engine_choice_is_grouped_selector():
    """Движок по умолчанию выбирается тем же группированным списком."""
    from ui.engine_selector import EngineSelector
    from core.engines import ENGINES

    app, panel = _make_panel(AppConfig(default_engine="demucs:htdemucs_ft"))
    try:
        widget = panel._widgets["default_engine"]
        assert isinstance(widget, EngineSelector)
        assert sorted(widget.values()) == sorted(s.label for s in ENGINES.values())
        assert widget.get() == "demucs:htdemucs_ft"
    finally:
        app.destroy()


def test_settings_tab_saves_to_file(isolated_settings):
    """Сохранение с вкладки доводит дело до файла на диске."""
    app = _make_window()
    try:
        app._go_to_settings()
        app.settings_panel.vars["demucs_shifts"].set(6.0)
        app.settings_panel._save()

        written = json.loads(isolated_settings.read_text(encoding="utf-8"))
        assert written["demucs_shifts"] == 6
        assert app.config_data.demucs_shifts == 6
    finally:
        app.destroy()
