"""
Проверки BS-движка на Python-API (core/engines/bs_engine_api.py).

Здесь НЕ запускается настоящая модель: GPU-тесты в обычный прогон не годятся
(минуты на трек и нужна видеокарта). Зато проверяется всё, что вокруг модели:
разбор имён стемов, прогресс из tqdm-цикла, отмена, сборка финальных файлов,
нормализация пары и уборка за собой.

Настройки подменяются на значения по умолчанию: тесты не должны зависеть от
того, что лежит в config/settings.json у пользователя.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.config import AppConfig
from core.engines.base import EngineCancelled
from core.engines.bs_engine_api import (
    OUTPUT_FORMAT,
    BSApiError,
    BSEngineAPI,
    _ProgressTqdm,
    _convert_to_wav,
    _tqdm_hook,
    cache_info,
    classify_stem,
    clear_model_cache,
)

SR = 44100


@pytest.fixture(autouse=True)
def _default_config(monkeypatch):
    """Настройки по умолчанию вместо пользовательских + чистый кэш моделей."""
    cfg = AppConfig()
    monkeypatch.setattr("core.engines.bs_engine_api.get_config", lambda: cfg)
    monkeypatch.setattr("core.engines.bs_engine.get_config", lambda: cfg)
    monkeypatch.setattr("core.dsp.get_config", lambda: cfg)
    clear_model_cache(free_gpu=False)
    yield cfg
    clear_model_cache(free_gpu=False)


@pytest.fixture
def stereo_wav(tmp_path):
    """Короткий стерео-трек: 1 секунда, 440 Гц."""
    t = np.arange(SR, dtype=np.float64) / SR
    mono = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    data = np.repeat(mono[:, None], 2, axis=1).astype(np.float32)
    path = tmp_path / "track.wav"
    sf.write(str(path), data, SR, subtype="PCM_16")
    return path


# ============================================================
#  Разбор имён стемов
# ============================================================
@pytest.mark.parametrize("filename,expected", [
    ("track__bsapi_vocals.wav", "vocals"),
    ("track__bsapi_instrumental.wav", "instrumental"),
    # Имена по умолчанию у библиотеки: <база>_(Стем)_<модель>
    ("track_(Vocals)_mel_band_roformer_vocals_becruily.wav", "vocals"),
    ("track_(Instrumental)_mel_band_roformer_instrumental_bleedless_v2_gabox.wav",
     "instrumental"),
    ("track_(Karaoke)_mel_band_roformer_karaoke_gabox.wav", "vocals"),
    # Имя трека содержит слово vocals — не должно сбить разбор
    ("my_vocals_song__bsapi_instrumental.wav", "instrumental"),
    ("track_(Drums)_model.wav", "part"),
])
def test_classify_stem(filename, expected):
    assert classify_stem(filename) == expected


# ============================================================
#  Прогресс и отмена внутри цикла tqdm
# ============================================================
def test_progress_tqdm_iterates_and_reports():
    """Замена tqdm отдаёт ровно те же элементы и считает шаги."""
    steps = []
    with _tqdm_hook(lambda done, total: steps.append((done, total))):
        items = list(_ProgressTqdm([1, 2, 3]))

    assert items == [1, 2, 3]
    assert steps == [(1, 3), (2, 3), (3, 3)]


def test_progress_tqdm_survives_total_unknown():
    """Генератор без len() — не ошибка, просто без общего числа шагов."""
    steps = []
    with _tqdm_hook(lambda done, total: steps.append((done, total))):
        items = list(_ProgressTqdm(x for x in range(2)))

    assert items == [0, 1]
    assert steps == [(1, None), (2, None)]


def test_tqdm_hook_restores_original():
    """После разделения в библиотеке снова её собственный tqdm."""
    from audio_separator.separator.architectures import mdxc_separator

    original = mdxc_separator.tqdm
    with _tqdm_hook(lambda done, total: None):
        assert mdxc_separator.tqdm is _ProgressTqdm
    assert mdxc_separator.tqdm is original


def test_cancel_inside_inference_raises_engine_cancelled():
    """Отмена поднимает EngineCancelled прямо в цикле инференса."""
    engine = BSEngineAPI(model_name="test.ckpt")

    def hook(done, total):
        if done == 2:
            engine.cancel()
        engine._on_step(done, total)

    with _tqdm_hook(hook):
        with pytest.raises(EngineCancelled):
            for _ in _ProgressTqdm(range(5)):
                pass


def test_cancel_idle_returns_false():
    """Отменять нечего — честный False (кнопка «Отмена» на это опирается)."""
    engine = BSEngineAPI(model_name="test.ckpt")
    assert engine.cancel() is False


def test_cancel_during_run_returns_true():
    engine = BSEngineAPI(model_name="test.ckpt")
    engine._running = True
    assert engine.cancel() is True
    assert engine._cancelled is True


def test_progress_callback_error_does_not_break_run():
    """Упавший колбэк прогресса не должен ронять разделение."""
    engine = BSEngineAPI(model_name="test.ckpt")
    engine._progress_cb = lambda p, s: (_ for _ in ()).throw(RuntimeError("бум"))
    engine._report(0.5, "тест")          # не должно бросить исключение


# ============================================================
#  Сборка результата (без настоящей модели)
# ============================================================
class _FakeSeparator:
    """Заглушка audio_separator.Separator: пишет два стема и возвращает пути.

    Амплитуды ≤ 1.0: библиотека пишет стемы уже нормированными, в PCM_16,
    поэтому всё, что выше единицы, обрезалось бы — как обрезалось бы и у неё.
    """

    def __init__(self, scratch, vocals_amp=0.9, instr_amp=0.15):
        self.output_dir = str(scratch)
        self.model_instance = self
        self.calls = 0
        self._vocals_amp = vocals_amp
        self._instr_amp = instr_amp

    def separate(self, path, custom_output_names=None):
        self.calls += 1
        names = custom_output_names or {}
        out = []
        for key, amp in (("vocals", self._vocals_amp), ("instrumental", self._instr_amp)):
            name = names.get(key, key)
            target = f"{self.output_dir}/{name}.wav"
            n = SR
            t = np.arange(n, dtype=np.float64) / SR
            freq = 440.0 if key == "vocals" else 220.0
            # Плавное начало: у сигнала, который стартует «ступенькой», HPF даёт
            # короткий выброс — он есть и в жизни, но проверять на нём уровни
            # пары неудобно.
            fade = np.minimum(1.0, t / 0.02)
            mono = amp * np.sin(2 * np.pi * freq * t) * fade
            sf.write(target, np.repeat(mono[:, None], 2, axis=1).astype(np.float32),
                     SR, subtype="PCM_16")
            out.append(target)
        return out


def _patch_fake(monkeypatch, **kwargs):
    def fake_ensure(self, scratch_dir):
        if self._separator is None:
            self._separator = _FakeSeparator(scratch_dir, **kwargs)
        return self._separator

    monkeypatch.setattr(BSEngineAPI, "_ensure_model", fake_ensure)


def test_separate_writes_final_stems(monkeypatch, stereo_wav, tmp_path):
    """Полный проход: два файла на выходе, PCM_16, пик пары — на целевом уровне."""
    _patch_fake(monkeypatch)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    result = engine.separate(stereo_wav, out_dir)

    assert result.vocals == out_dir / "track_vocals.wav"
    assert result.instrumental == out_dir / "track_instrumental.wav"
    peaks = {}
    for name, path in (("vocals", result.vocals), ("instrumental", result.instrumental)):
        info = sf.info(str(path))
        assert info.subtype == "PCM_16"
        assert info.samplerate == SR
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        peaks[name] = float(np.max(np.abs(data)))
        assert peaks[name] <= 1.0, f"{name}: перегруз {peaks[name]}"

    # Нормализация пары выставляет ОДИН уровень: самый громкий стем = target_peak
    assert max(peaks.values()) == pytest.approx(0.891, abs=0.01)


def test_separate_normalizes_pair_with_one_gain(monkeypatch, stereo_wav, tmp_path):
    """
    Вокал и минус должны получить ОДИН общий коэффициент: иначе взаимный
    баланс уезжает (это правило core.dsp.normalize_pair).
    """
    _patch_fake(monkeypatch, vocals_amp=0.9, instr_amp=0.15)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    result = engine.separate(stereo_wav, out_dir)

    v, _ = sf.read(str(result.vocals), dtype="float32", always_2d=True)
    i, _ = sf.read(str(result.instrumental), dtype="float32", always_2d=True)
    # Громкость минуса к вокалу сохранилась: 0.15/0.9 = 1/6
    ratio = float(np.max(np.abs(i))) / float(np.max(np.abs(v)))
    assert 0.10 < ratio < 0.20, f"баланс пары уехал: {ratio}"


def test_separate_cleans_scratch_dir(monkeypatch, stereo_wav, tmp_path):
    """keep_temp_dirs=false — рабочая папка после прогона убрана."""
    _patch_fake(monkeypatch)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    engine.separate(stereo_wav, out_dir)

    assert not (out_dir / "_bs_api_tmp").exists()


def test_separate_keeps_scratch_when_asked(monkeypatch, stereo_wav, tmp_path):
    """keep_temp_dirs=true — папка остаётся для разбирательств."""
    monkeypatch.setattr("core.engines.bs_engine_api.get_config",
                        lambda: AppConfig(keep_temp_dirs=True))
    monkeypatch.setattr("core.engines.bs_engine.get_config",
                        lambda: AppConfig(keep_temp_dirs=True))
    _patch_fake(monkeypatch)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    engine.separate(stereo_wav, out_dir)

    assert (out_dir / "_bs_api_tmp").exists()


def test_separate_without_vocals_raises(monkeypatch, stereo_wav, tmp_path):
    """Нет вокала в ответе библиотеки — понятная ошибка, а не молчаливый ноль."""

    class _OnlyInstr(_FakeSeparator):
        def separate(self, path, custom_output_names=None):
            outputs = super().separate(path, custom_output_names)
            return [outputs[1]]

    def fake_ensure(self, scratch_dir):
        if self._separator is None:
            self._separator = _OnlyInstr(scratch_dir)
        return self._separator

    monkeypatch.setattr(BSEngineAPI, "_ensure_model", fake_ensure)
    engine = BSEngineAPI(model_name="test.ckpt")

    with pytest.raises(BSApiError):
        engine.separate(stereo_wav, tmp_path / "out")


def test_one_gain_for_pair_after_postprocess(monkeypatch, stereo_wav, tmp_path):
    """Постобработка минуса не сбивает общий уровень пары."""
    _patch_fake(monkeypatch, vocals_amp=0.9, instr_amp=0.9)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    result = engine.separate(stereo_wav, out_dir)

    peaks = {}
    for name, path in (("vocals", result.vocals), ("instrumental", result.instrumental)):
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        peaks[name] = float(np.max(np.abs(data)))
    # Оба стема были одинаковой громкости и оба пришли к целевому пику
    assert peaks["vocals"] == pytest.approx(0.891, abs=0.02)
    assert peaks["instrumental"] == pytest.approx(0.891, abs=0.02)


def test_missing_input_file_raises(tmp_path):
    engine = BSEngineAPI(model_name="test.ckpt")
    with pytest.raises(BSApiError):
        engine.separate(tmp_path / "нет-такого.wav", tmp_path / "out")


# ============================================================
#  Кэш моделей
# ============================================================
def test_cache_is_empty_after_clear():
    clear_model_cache(free_gpu=False)
    assert cache_info()["size"] == 0


def test_engine_is_not_bound_to_one_output_dir(monkeypatch, stereo_wav, tmp_path):
    """
    Модель из кэша должна уметь писать в новую папку: output_dir у неё
    переставляется перед каждым прогоном.
    """
    created = []

    def fake_ensure(self, scratch_dir):
        if self._separator is None:
            separator = _FakeSeparator(scratch_dir)
            created.append(separator)
            self._separator = separator
        return self._separator

    monkeypatch.setattr(BSEngineAPI, "_ensure_model", fake_ensure)
    engine = BSEngineAPI(model_name="test.ckpt")

    engine.separate(stereo_wav, tmp_path / "out1")
    engine.separate(stereo_wav, tmp_path / "out2")

    assert len(created) == 1, "модель пересоздана вместо переиспользования"
    assert created[0].output_dir.endswith("_bs_api_tmp")
    assert (tmp_path / "out2" / "track_vocals.wav").exists()


# ============================================================
#  MP3 на входе (баг «Supported file format but unsupported encoding»)
# ============================================================
class _LibraryLikeSeparator:
    """
    Заглушка, повторяющая ПОВЕДЕНИЕ библиотеки, а не только её интерфейс.

    Настоящий audio-separator копирует subtype входного файла в выходной
    (common_separator.write_audio_soundfile: output_subtype = input_subtype).
    У MP3 subtype = MPEG_LAYER_III, а стем пишется в WAV — libsndfile отвечает
    «Supported file format but unsupported encoding». Именно на этом падал
    движок, поэтому заглушка делает то же самое: если движок снова начнёт
    отдавать в библиотеку MP3, тест упадёт так же, как падал пользователь.
    """

    def __init__(self, scratch, received: list[str]):
        self.output_dir = str(scratch)
        self.model_instance = self
        self.output_format = OUTPUT_FORMAT
        self.use_soundfile = True
        self._received = received

    def load_model(self, model_filename=None):
        return None

    def separate(self, path, custom_output_names=None):
        path = Path(path)
        self._received.append(path.name)
        try:
            subtype = sf.info(str(path)).subtype       # ← как у библиотеки
        except Exception:
            subtype = "PCM_16"
        out = []
        for key, amp, freq in (("vocals", 0.9, 440.0), ("instrumental", 0.2, 220.0)):
            name = (custom_output_names or {}).get(key, key)
            target = f"{self.output_dir}/{name}.{self.output_format.lower()}"
            t = np.arange(SR, dtype=np.float64) / SR
            mono = amp * np.sin(2 * np.pi * freq * t) * np.minimum(1.0, t / 0.02)
            sf.write(target, np.repeat(mono[:, None], 2, axis=1).astype(np.float32),
                     SR, subtype=subtype)              # ← падает на MPEG_LAYER_III
            out.append(target)
        return out


@pytest.fixture
def mp3_input(tmp_path):
    """Настоящий MP3 (кодек MPEG_LAYER_III), а не переименованный WAV."""
    import subprocess

    wav = tmp_path / "probe_src.wav"
    t = np.arange(SR, dtype=np.float64) / SR
    mono = 0.4 * (np.sin(2 * np.pi * 440.0 * t) + np.sin(2 * np.pi * 110.0 * t))
    sf.write(str(wav), np.repeat(mono[:, None], 2, axis=1).astype(np.float32),
             SR, subtype="PCM_16")
    mp3 = tmp_path / "probe.mp3"
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                        "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                       check=True, capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        pytest.skip(f"FFmpeg недоступен: {e}")
    assert sf.info(str(mp3)).subtype == "MPEG_LAYER_III", "ожидался MP3-подтип"
    return mp3


def _patch_library_like(monkeypatch, received: list[str]):
    def fake_ensure(self, scratch_dir):
        if self._separator is None:
            self._separator = _LibraryLikeSeparator(scratch_dir, received)
        return self._separator

    monkeypatch.setattr(BSEngineAPI, "_ensure_model", fake_ensure)


def test_mp3_input_is_converted_before_library(monkeypatch, mp3_input, tmp_path):
    """
    Регрессия: MP3 обязан доехать до библиотеки уже WAV-ом.

    Проверяем и то, что имя исходного трека не потерялось: файлы на выходе
    называются по MP3, а не по временному WAV.
    """
    received: list[str] = []
    _patch_library_like(monkeypatch, received)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    result = engine.separate(mp3_input, out_dir)

    assert received and received[0].endswith(".wav"), \
        f"в библиотеку ушёл не WAV: {received}"
    assert Path(result.vocals).name == "probe_vocals.wav"
    assert Path(result.instrumental).name == "probe_instrumental.wav"
    assert Path(result.vocals).exists() and Path(result.instrumental).exists()


def test_mp3_converted_wav_is_removed_after_run(monkeypatch, mp3_input, tmp_path):
    """Временный WAV входа не копится: после прогона его нет даже при keep_temp_dirs."""
    monkeypatch.setattr("core.engines.bs_engine_api.get_config",
                        lambda: AppConfig(keep_temp_dirs=True))
    monkeypatch.setattr("core.engines.bs_engine.get_config",
                        lambda: AppConfig(keep_temp_dirs=True))
    received: list[str] = []
    _patch_library_like(monkeypatch, received)
    out_dir = tmp_path / "out"
    engine = BSEngineAPI(model_name="test.ckpt")

    engine.separate(mp3_input, out_dir)

    leftovers = list((out_dir / "_bs_api_tmp").glob("*__bsapi_input.wav"))
    assert leftovers == [], f"временный WAV остался: {leftovers}"


def test_wav_input_is_not_converted(monkeypatch, stereo_wav, tmp_path):
    """WAV идёт в библиотеку как есть — лишний проход FFmpeg не нужен."""
    received: list[str] = []
    _patch_library_like(monkeypatch, received)
    engine = BSEngineAPI(model_name="test.ckpt")

    engine.separate(stereo_wav, tmp_path / "out")

    assert received == ["track.wav"]


def test_prepare_input_wav_reports_whether_it_converted(tmp_path, monkeypatch):
    """_prepare_input_wav честно говорит, создал ли он времянку (её потом удаляют)."""
    engine = BSEngineAPI(model_name="test.ckpt")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros((100, 2), dtype=np.float32), SR)

    path, created = engine._prepare_input_wav(wav, scratch)
    assert path == wav and created is False

    fake = tmp_path / "b.mp3"
    fake.write_bytes(b"not really audio")
    converted = scratch / "b__bsapi_input.wav"
    monkeypatch.setattr("core.engines.bs_engine_api._convert_to_wav",
                        lambda src, dst: dst.write_bytes(b"RIFF"))
    path, created = engine._prepare_input_wav(fake, scratch)
    assert path == converted and created is True


# ============================================================
#  Формат вывода задан явно
# ============================================================
def test_output_format_is_pinned_on_loaded_model():
    """
    Формат вывода фиксируется в обоих местах, где его держит библиотека.

    У Separator он задаёт имена файлов, у загруженной модели — чем реально
    пишется файл. Разойдутся — получим файл .wav с содержимым другого формата.
    """
    class _Holder:
        def __init__(self):
            self.output_format = "MP3"
            self.use_soundfile = False

    outer, inner = _Holder(), _Holder()
    outer.model_instance = inner

    BSEngineAPI._pin_output_format(outer)

    for holder in (outer, inner):
        assert holder.output_format == OUTPUT_FORMAT
        assert holder.use_soundfile is True


def test_output_format_constant_is_wav():
    """Константа — часть контракта с библиотекой: звук пишем только в WAV."""
    assert OUTPUT_FORMAT == "WAV"


# ============================================================
#  Перекодирование входа (обёртка над FFmpeg)
# ============================================================
def test_convert_to_wav_makes_readable_file(tmp_path):
    """FFmpeg-обёртка отдаёт настоящий WAV нужной частоты и каналов."""
    import subprocess

    src = tmp_path / "source.wav"
    t = np.arange(SR, dtype=np.float64) / SR
    sf.write(str(src), np.repeat((0.3 * np.sin(2 * np.pi * 300 * t))[:, None], 2, axis=1),
             SR, subtype="PCM_16")
    dst = tmp_path / "converted.wav"
    try:
        _convert_to_wav(src, dst)
    except BSApiError as e:
        pytest.skip(f"FFmpeg недоступен: {e}")

    info = sf.info(str(dst))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "PCM_16"


def test_convert_to_wav_reports_broken_input(tmp_path):
    """Битый вход — понятная ошибка, а не пустой файл на выходе."""
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"definitely not audio")
    dst = tmp_path / "out.wav"

    with pytest.raises(BSApiError) as exc:
        _convert_to_wav(broken, dst)

    assert not dst.exists()
    assert "broken.mp3" in str(exc.value)
