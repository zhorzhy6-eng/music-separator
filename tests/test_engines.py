"""
Проверки реестра движков — «меню» приложения.

Смысл: список моделей живёт в одном месте. Если реестр сломается, интерфейс,
CLI и фабрика сломаются вместе с ним — значит, реестр должен быть целым.
"""
from __future__ import annotations

import pytest

from core.engines import (
    DEFAULT_ENGINE_ID,
    ENGINES,
    EngineSpec,
    all_labels,
    engines_by_label,
    get_engine,
    make_separator,
    resolve_engine_id,
)


def test_registry_not_empty():
    assert len(ENGINES) > 0


def test_get_engine_unknown_id_raises():
    with pytest.raises(KeyError):
        get_engine("нет-такого-движка")


def test_get_engine_returns_spec():
    spec = get_engine("demucs:htdemucs")
    assert isinstance(spec, EngineSpec)
    assert spec.kind == "demucs"
    assert spec.model == "htdemucs"


def test_default_engine_exists():
    """Движок по умолчанию обязан быть в реестре, иначе UI упадёт на старте."""
    assert DEFAULT_ENGINE_ID in ENGINES


def test_ids_match_keys():
    """Ключ словаря и поле id не должны разъезжаться."""
    for key, spec in ENGINES.items():
        assert key == spec.id, f"ключ {key} не совпадает с id {spec.id}"


def test_labels_are_unique():
    """Две одинаковые подписи в выпадающем списке — это неоднозначный выбор."""
    labels = all_labels()
    assert len(labels) == len(set(labels))


def test_engines_by_label_covers_everything():
    by_label = engines_by_label()
    assert len(by_label) == len(ENGINES)
    for spec in ENGINES.values():
        assert by_label[spec.label] == spec.id


def test_hybrid_engines_have_companion():
    hybrids = [s for s in ENGINES.values() if s.kind == "hybrid"]
    assert hybrids, "гибридные движки должны быть в реестре"
    for spec in hybrids:
        assert spec.companion, f"у гибрида {spec.id} не указана модель вокала"


def test_kinds_are_known():
    assert {s.kind for s in ENGINES.values()} <= {"demucs", "bs", "hybrid"}


def test_resolve_engine_id_understands_label_and_id():
    """В settings.json лежит подпись, в CLI удобнее id — понимаем оба варианта."""
    spec = get_engine(DEFAULT_ENGINE_ID)
    assert resolve_engine_id(spec.id) == spec.id
    assert resolve_engine_id(spec.label) == spec.id


def test_resolve_engine_id_unknown_raises():
    with pytest.raises(KeyError):
        resolve_engine_id("такого движка нет")


@pytest.mark.parametrize("kind,model,companion,use_api,expected", [
    ("demucs", "htdemucs", None, True, "DemucsEngine"),
    ("bs", "BS-Roformer-SW.ckpt", None, True, "BSEngineAPI"),
    # Запасной путь: тот же движок через audio-separator.exe
    ("bs", "BS-Roformer-SW.ckpt", None, False, "BSEngine"),
    ("hybrid", "BS-Roformer-SW.ckpt", "htdemucs_ft", True, "HybridEngine"),
])
def test_factory_builds_every_kind(kind, model, companion, use_api, expected):
    """Фабрика собирает нужный движок для каждого вида, не загружая модели."""
    spec = EngineSpec(id=f"{kind}:test", label="тест", kind=kind,
                      model=model, companion=companion, use_api=use_api)
    sep = make_separator(spec)
    assert type(sep).__name__ == expected
    assert hasattr(sep, "separate")
    assert hasattr(sep, "cancel")


def test_bs_engines_use_python_api_by_default():
    """BS-движки из реестра идут через Python-API (модель грузится один раз)."""
    for spec in ENGINES.values():
        if spec.kind == "bs":
            assert spec.use_api, f"{spec.id}: ожидался Python-API"


def test_factory_unknown_kind_raises():
    spec = EngineSpec(id="x:test", label="тест", kind="что-то", model="m")
    with pytest.raises(ValueError):
        make_separator(spec)


def test_engines_satisfy_protocol():
    """Движки соответствуют общему договору Separator."""
    from core.engines import Separator
    sep = make_separator(get_engine("demucs:htdemucs"))
    assert isinstance(sep, Separator)


def test_old_import_names_still_work():
    """Обратная совместимость: старые имена классов и модулей живы."""
    from core.separator import AudioSeparator
    from core.separator_bs import BSAudioSeparator
    from core.separator_hybrid import HybridSeparator
    from core.engines.bs_engine import BSEngine
    from core.engines.demucs_engine import DemucsEngine
    from core.engines.hybrid_engine import HybridEngine

    assert AudioSeparator is DemucsEngine
    assert BSAudioSeparator is BSEngine
    assert HybridSeparator is HybridEngine


def test_unknown_model_no_longer_checked_inside_engine():
    """Список моделей — забота реестра, а не движка: пустое значение всё же ловим."""
    from core.engines.demucs_engine import DemucsEngine, SeparationError
    with pytest.raises(SeparationError):
        DemucsEngine(model_name="")
