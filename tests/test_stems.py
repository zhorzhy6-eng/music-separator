"""
Проверки StemSet — типизированного результата разделения.

StemSet заменил «голый» словарь, но обязан вести себя как словарь: иначе
сломается весь код, который писался до рефакторинга (воркер, UI, гибрид).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.stems import STEM_NAMES, StemSet


def test_as_dict_keeps_only_filled_stems():
    s = StemSet(vocals=Path("v.wav"), bass=Path("b.wav"))
    assert s.as_dict() == {"vocals": Path("v.wav"), "bass": Path("b.wav")}


def test_getitem_returns_path():
    s = StemSet(vocals=Path("v.wav"))
    assert s["vocals"] == Path("v.wav")


def test_getitem_missing_raises_keyerror():
    """Пустой стем — это ошибка, а не None: молча вернуть None нельзя."""
    s = StemSet()
    with pytest.raises(KeyError):
        _ = s["vocals"]


def test_contains():
    s = StemSet(vocals=Path("v.wav"))
    assert "vocals" in s
    assert "instrumental" not in s


def test_get_with_default():
    s = StemSet(vocals=Path("v.wav"))
    assert s.get("vocals") == Path("v.wav")
    assert s.get("drums") is None
    assert s.get("drums", "нет") == "нет"


def test_behaves_like_old_dict():
    """Совместимость: итерация, keys(), items(), len()."""
    s = StemSet(vocals=Path("v.wav"), instrumental=Path("i.wav"))
    assert sorted(s.keys()) == ["instrumental", "vocals"]
    assert dict(s.items())["vocals"] == Path("v.wav")
    assert sorted(iter(s)) == ["instrumental", "vocals"]
    assert len(s) == 2
    assert len(StemSet()) == 0


def test_unknown_stem_name_is_not_silently_accepted():
    """Опечатка в имени стема не должна выглядеть как «стема нет» молча."""
    s = StemSet(vocals=Path("v.wav"))
    assert s.get("vocal") is None          # get мягкий, как у словаря
    with pytest.raises(KeyError):
        _ = s["vocal"]                     # а по ключу — честная ошибка


def test_stem_names_cover_two_main_stems():
    assert "vocals" in STEM_NAMES
    assert "instrumental" in STEM_NAMES
