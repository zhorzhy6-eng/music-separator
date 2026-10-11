# AST — Project Sheet

> Паспорт проекта: что это, из чего состоит, как запускать, где узкие места.
> Описывает **фактическое** состояние кода в `E:\Programmy\AST`, а не желаемое.
> Основано на сплошном чтении исходников; расхождения с `requirements.txt` отмечены отдельно.

---

## 1. Что это

**AST (Audio Separator & Transcriber)** — локальное десктопное приложение под Windows для работы с музыкой:

1. **Разделение трека на вокал и минус** тремя движками: Demucs, BS-Roformer/Mel-Band Roformer (через внешний `audio-separator`) и гибрид (Demucs → вокал, BS-Roformer → минус).
2. **Постобработка стемов** — HPF, notch 2.5 кГц, LPF, нормализация пика до −0.3 dBFS.
3. **Транскрипция в MIDI** — вокал и минус через `torchcrepe` (F0 → ноты → MIDI); отдельно «Минус → Пианино» через нейросеть MuScriptor.
4. **Анализ аудио** — уровни, клиппинг, тишина, октавный спектр, стерео, грубая F0 + графики.
5. **Супер-анализ пары «вокал + минус»** — 5 уровней качества, метрики артефактов разделения (phase smearing, spectral tilt, transient loss, bleed score, карта подозрительных секунд) и вердикт.
6. **Анализ MIDI** — диапазон, октавы, топ нот, плотность, velocity, темп.

Интерфейс — одно окно на CustomTkinter: выбор файла, осциллограмма, плеер, прогресс, статус.

**Статус:** рабочий прототип. `README.md` пустой, автотестов нет, системы контроля версий в папке нет. Это надо понимать до первой правки.

---

## 2. Стек и окружение

| Слой | Технологии (фактические версии из venv) |
|---|---|
| Язык | Python **3.12.10**, venv `.venv312` |
| GUI | customtkinter **6.0.0**, tkinterdnd2 0.6.3 (в коде не используется), Pillow 12.3.0 |
| Аудио I/O | soundfile 0.14.0, sounddevice 0.5.6, pydub 0.25.1, librosa 1.0.0 (в коде не используется) |
| DSP | numpy 2.5.2, scipy 1.18.1 |
| Разделение | demucs 4.1.0 (API `demucs.api.Separator`), audio-separator 0.47.0 (CLI, Roformer-модели), onnxruntime-gpu 1.30.0 |
| Транскрипция | torchcrepe 0.0.24, pretty_midi 0.2.11, mido 1.3.3, muscriptor 0.3.0, beat_this 1.1.0 |
| ML-рантайм | torch 2.5.1+cu121, torchaudio 2.5.1+cu121, torchvision 0.20.1+cu121 (CUDA 12.1) |
| Графики | matplotlib 3.11.2 (backend `Agg` в отчётах, `TkAgg` в окне) |
| Внешние exe | FFmpeg / ffprobe в `PATH`; `audio-separator.exe` в `.venv312\Scripts` |

**Дрейф `requirements.txt` от реального окружения** — файл устарел:

| Пакет | В requirements | Фактически |
|---|---|---|
| customtkinter | `==5.2.2` | 6.0.0 |
| tkinterdnd2 | `==0.4.3` | 0.6.3 |
| basic-pitch | `>=0.4.0` | **не установлен** (транскрипция идёт через torchcrepe) |
| torchcrepe, scipy, muscriptor, audio-separator, beat_this | отсутствуют | установлены и нужны коду |

Отдельная деталь: `__pycache__` содержит байткод и `cpython-312`, и `cpython-314` — проект запускали и на Python 3.14 (в `transcriber.py` это отражено комментарием «без basic-pitch»). Целевое окружение сейчас — 3.12.

---

## 3. Структура репозитория

```
AST/
├── main.py                     # точка входа: sys.path + логирование + запуск UI
├── requirements.txt            # устарел (см. §2)
├── README.md                   # ПУСТОЙ (0 байт)
├── PROJECT_SHEET.md            # этот документ
├── config/
│   └── settings.json           # ПУСТОЙ (0 байт) — кодом не читается
├── Токен/
│   └── Text Document.txt       # HuggingFace-токен открытым текстом — кодом не используется
├── core/                       # ядро: логика обработки
│   ├── __init__.py             # пустой
│   ├── audio_utils.py          # инфо о файле, FFmpeg-конвертация, измерение notch
│   ├── separator.py            # Demucs: вокал + минус
│   ├── separator_bs.py         # BS-Roformer / Mel-Band через audio-separator CLI
│   ├── separator_hybrid.py     # гибрид: Demucs-вокал + BS-минус
│   ├── transcriber.py          # аудио → MIDI (torchcrepe)
│   ├── piano_transcriber.py    # минус → фортепианный MIDI (MuScriptor)
│   ├── analyzer.py             # анализ аудио + PNG/JSON/TXT
│   ├── super_analyzer.py       # супер-анализ пары (v4.1) — рабочий
│   ├── midi_analyzer.py        # анализ MIDI
│   ├── player.py               # плеер на sounddevice
│   ├── worker.py               # фоновые задачи + очередь событий
│   ├── logging_config.py       # JSON-lines логи
│   ├── piano_reducer.py        # МЁРТВЫЙ КОД (никем не импортируется)
│   └── coresuper_analyzer.py   # МЁРТВЫЙ КОД (старая версия супер-анализатора)
├── ui/
│   ├── __init__.py             # пустой
│   ├── app_window.py           # главное окно (617 строк) — сценарии, горячие клавиши
│   ├── components.py           # DropZone, WaveformView, SeekBar, PlayerBar
│   └── theme.py                # DARK/LIGHT-словари, get_theme()
├── logs/                       # создаётся при запуске
│   ├── ast.log                 # JSON-lines, ротация 5 МБ × 3
│   └── analysis/               # отчёты анализа (json/txt/png)
├── output/                     # результаты: *_vocals.wav, *_instrumental.wav, *.mid
└── .venv312/                   # окружение (создавалось по пути E:\Python\AST — проект перенесён)
```

---

## 4. Архитектура

### 4.1 Принцип разделения

- **Ядро** (`core/`) — логика обработки. Не знает про виджеты, не импортирует `ui/`.
- **Край** (`ui/`) — окно, кнопки, плеер, отрисовка. Работа с ядром идёт **только** через `core/worker.py`.
- **Контракт между ядром и краем** — три типа событий плюс `progress_cb`. Это единственная точка стыковки; ломать её нельзя.

### 4.2 Поток данных

```
UI (ui/app_window.py)
  └─ submit_*()  →  Worker (core/worker.py)
                       └─ daemon-thread  →  модуль ядра
                                              ├─ progress_cb(p, "стадия") → ProgressEvent
                                              ├─ успех                    → DoneEvent(task_id, task_type, result)
                                              └─ исключение               → ErrorEvent(task_id, message)
  UI опрашивает worker.events каждые 100 мс и обновляет прогресс/статус/кнопки
```

### 4.3 Многопоточность: как есть и чего нет

- **Модель:** отдельный `threading.Thread(daemon=True)` на задачу (`Worker._submit`). Пул отсутствует, лимит одновременных задач не задан.
- **Обмен:** `queue.Queue` (`Worker.events`) с событиями `ProgressEvent` / `DoneEvent` / `ErrorEvent`.
- **Плеер:** отдельный поток PortAudio (`sounddevice.OutputStream` с callback) + `threading.Lock` на состояние.
- **Внешний процесс:** `audio-separator.exe` запускается через `subprocess.Popen` и читается построчно — GIL не мешает, но stdout парсится целиком в лог.
- **Чего нет:** отмены задач, приоритетов, ограничения параллелизма, `asyncio`/`multiprocessing`.
- **Практический вывод:** многопоточность здесь оправдана (UI не блокируется), но отсутствие отмены — главная боль: Demucs идёт 3–5 минут на трек, BS-Roformer — до 30 минут по таймауту, и остановить это из интерфейса нельзя.

---

## 5. Интерфейс

**Окно** `AppWindow(ctk.CTk)` — заголовок «Audio Separator & Transcriber», 1000×700 (minsize 900×600), тема dark + blue. Вкладок нет, вертикальный стек:

1. Шапка: заголовок + кнопка «🌗 Тема».
2. `DropZone` — зона загрузки файла.
3. Строка информации: имя, длительность, частота, каналы.
4. `WaveformView` — осциллограмма через matplotlib (`FigureCanvasTkAgg`, децимация до 4000 точек).
5. Ряд 1: `CTkOptionMenu` с движком (27 вариантов) + «🔀 Разделить» + «📂 Папка».
6. Ряд 2: «🎼 Вокал → MIDI», «🎼 Минус → MIDI», «🎹 Минус → Пианино».
7. Ряд 3: «📊 Анализ аудио», «📊 MIDI-анализ», «🧠 СУПЕР-АНАЛИЗ v4 ВОКАЛ+МИНУС» + «📈 Отчёты».
8. `CTkProgressBar`, строка статуса, `PlayerBar` (▶ ⏸ ⏹, кастомный `SeekBar` на `tk.Canvas`, громкость 0–1, по умолчанию 0.8).
9. Селектор «Слушать:» — исходный трек / вокал / минус.

Движок по умолчанию — «BS-Roformer: Duality-V2 ✅ баланс». Кнопки блокируются на время задачи. Опрос воркера — 100 мс, позиция плеера — 500 мс.

**Горячие клавиши** (`bind_all`, `ui/app_window.py:223-230`):

| Клавиши | Действие |
|---|---|
| `Space` | Play / Pause |
| `→` / `←` | ±5 секунд |
| `Shift+→` / `Shift+←` | ±30 секунд |
| `Home` / `End` | в начало / в конец |

**Классы компонентов** (`ui/components.py`): `DropZone._pick`; `WaveformView.plot_file`; `SeekBar` — `set_duration`, `set_position`, `_x_to_seconds`, `_on_click/_on_drag/_on_release`; `PlayerBar` — `set_duration`, `set_position`, `_update_time_label`.

**Тема** (`ui/theme.py`): словари `DARK` (bg `#1a1a1a`, panel `#242424`, accent `#3b8ed0`, accent_hover `#36719f`, text `#e8e8e8`, text_dim `#9a9a9a`, success `#4caf50`, error `#e53935`) и `LIGHT` (bg `#f5f5f5`, panel `#ffffff`, accent `#1f6aa5`, accent_hover `#144870`, text `#1a1a1a`, text_dim `#666666`, success `#2e7d32`, error `#c62828`), доступ через `get_theme(name="dark")`.

**Известные несоответствия UI (факт, не баг-репорт):**

- **Drag & drop заявлен, но не работает:** на `DropZone` написано «Перетащите аудиофайл сюда», однако `tkinterdnd2` не импортируется ни в одном файле — работает только клик и `filedialog.askopenfilename` (фильтр `*.mp3 *.wav *.flac *.m4a *.ogg`).
- **Тема фактически мертва:** цвета захардкожены в `components.py` (`#242424`, `#3b8ed0`, `#3a3a3a`, `#9a9a9a`), а `_toggle_theme` лишь переключает `ctk.set_appearance_mode` и перезаписывает `self.theme`, который нигде не читается. Шрифты заданы inline (`CTkFont(size=14/18)`).
- **Логи в окне не показываются:** только строка статуса и `messagebox`; полный лог — в `logs/ast.log`.
- **Ошибки глушатся:** `except Exception: pass` в `_poll_events`, `_tick_position`, `_seek_home/_seek_end`, `_player_pause/_player_stop`.
- **Списка файлов (плейлиста) нет:** один текущий трек.

---

## 6. Справочник по модулям ядра

### core/worker.py — диспетчер задач

Единственная точка входа в ядро для UI. `TaskType`: `SEPARATE`, `SEPARATE_BS`, `SEPARATE_HYBRID`, `TRANSCRIBE`, `ANALYZE`, `ANALYZE_PAIR`, `PIANO`, `MIDI_ANALYZE`. Все методы возвращают `task_id`.

| Метод | Что делает |
|---|---|
| `submit_separation(input_path, output_dir, model_name="htdemucs")` | Demucs: вокал + минус |
| `submit_separation_bs(input_path, output_dir, bs_model="BS-Roformer-SW.ckpt")` | BS-Roformer / Mel-Band |
| `submit_separation_hybrid(input_path, output_dir, demucs_model="htdemucs_ft", bs_model="BS-Roformer-SW.ckpt")` | Гибрид |
| `submit_transcription(input_audio, output_midi, instrument="Фортепиано")` | Аудио → MIDI (torchcrepe) |
| `submit_analysis(input_audio, report_dir=None)` | Анализ аудио |
| `submit_pair_analysis(vocals_path, instr_path, report_dir=None)` | Супер-анализ пары |
| `submit_piano(input_audio, output_midi, model_size="small")` | Минус → пианино (MuScriptor) |
| `submit_midi_analysis(midi_path, report_dir=None)` | Анализ MIDI |

### core/audio_utils.py — общие утилиты

- `SUPPORTED_EXTENSIONS = {.mp3, .wav, .flac, .m4a, .ogg}`, `AudioError`.
- `is_supported(path)`, `get_audio_info(path) -> dict` — сначала `soundfile`, при неудаче `ffprobe`.
- `ensure_wav(path, target_sr=44100) -> Path` — WAV отдаёт как есть, остальное перекодирует FFmpeg в `%TEMP%\ast_cache`.
- `load_mono(path, sr)`, `write_wav(path, y, sr)`.
- `measure_notch_depth(sr, center_hz, q, gain_db, duration_sec=1.0) -> float` — измеряет фактическую глубину частичного notch (см. §9.1).

### core/separator.py — Demucs

- `AVAILABLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra"]`, `SeparationError`, `AudioSeparator(model_name="htdemucs")`.
- `separate(input_path, output_dir, progress_cb=None) -> {"vocals": Path, "instrumental": Path}`.
- Цепочка: `ensure_wav(44100)` → `demucs.api.Separator(model=..., progress=True)` → `separate_audio_file()` → вокал: HPF 100 Гц (Баттерворт, порядок 2); минус: сумма `drums+bass+other` → notch 2500 Гц, Q=1.5, `gain_db=-3.0` → нормализация пика до −0.3 dBFS → запись `PCM_16`.
- Прогресс: 0.03 → 0.10 → 0.75 → 0.88 → 1.0.

### core/separator_bs.py — BS-Roformer / Mel-Band Roformer

- Реестр `AVAILABLE_BS_MODELS` (~35 имён `.ckpt`: `BS-Roformer-SW.ckpt`, `mel_band_roformer_*` от gabox/becruily/unwa/aufr33, karaoke-модели).
- `BSSeparationError`, `BSAudioSeparator(model_name="BS-Roformer-SW.ckpt")`, `separate(input_path, output_dir, progress_cb=None)`.
- Работает **через внешний CLI**: `_find_audio_separator_exe()` ищет `audio-separator.exe` рядом с `sys.executable`, затем в `PATH`; запуск `subprocess.Popen` с `--model_filename` и `--output_dir <out>/_bs_tmp`, stdout построчно в лог, таймаут 1800 с.
- Постобработка минуса: HPF 25 Гц → notch 2500 Гц (Q=1.0, `gain_db=-1.5`, фактическая глубина ≈ −16 dB) → LPF 19 кГц → нормализация. Вокал: только нормализация.
- Выход: `<base>_vocals.wav`, `<base>_instrumental.wav`; временные FLAC — в `<out>/_bs_tmp`.

### core/separator_hybrid.py — гибрид

- `HybridSeparator(demucs_model="htdemucs_ft", bs_model="BS-Roformer-SW.ckpt")`, `HybridSeparationError`.
- Шаг 1: Demucs → вокал (в `<out>/_hybrid_demucs`, прогресс 0.02→0.50). Шаг 2: BS-Roformer → минус (в `<out>/_hybrid_bs`, прогресс 0.52→0.98). Шаг 3: `shutil.copy2` в `<base>_vocals.wav` и `<base>_instrumental.wav`.
- Ошибка содержит указание, какой движок упал.

### core/transcriber.py — аудио → MIDI

- `TranscriptionError`, `MidiTranscriber`, `transcribe(input_audio, output_midi, instrument_name="Фортепиано", progress_cb=None) -> Path`.
- Алгоритм: чтение → `torchcrepe.predict` (F0 + confidence, `device = cuda if available else cpu`) → `_frames_to_notes` (`min_conf=0.3`, `min_dur=0.05 c`, разрыв при |Δpitch| > 0.5 полутона, velocity 90) → `pretty_midi`.
- `GM_INSTRUMENTS`: Фортепиано (0), Гитара нейлон (24), Скрипка (40), Саксофон (65), Флейта (73), Синт-лид (80).
- Прогресс: 0.1 → 0.3 → 0.8 → 1.0.

### core/piano_transcriber.py — минус → пианино (MuScriptor)

- `PianoTranscriberError`, `PianoTranscriber(model_size="small")`, `transcribe(input_audio, output_midi, progress_cb=None) -> Path`.
- `muscriptor` импортируется лениво (`TranscriptionModel.load_model`), модель кэшируется в объекте; `model.transcribe_and_postprocess(path)` возвращает MIDI в `bytes`.
- Промежуточный `<out_stem>_multi.mid` **остаётся на диске**.
- Регистр: `pitch < 48 → +24`, `48..59 → +12`, выше — clamp до 127; барабаны отбрасываются.

### core/analyzer.py — анализ аудио

- `AnalysisError`, `analyze_audio(audio_path, report_dir=None, progress_cb=None) -> dict`.
- Метрики: peak/RMS/LUFS(прибл.), клиппинг ≥ 0.99, доля тишины (< −60 dB, кадры 50 мс), средняя энергия по октавам `[63…8000]`, стерео (корреляция L/R, ширина), F0 (автокорреляция, кадры 100 мс, окно 30 с из середины).
- Прогресс: 0.05 → 0.20 → 0.35 → 0.50 → 0.65 → 0.80 → 0.90 → 1.0.
- Ограничения: FFT и спектрограмма — только первые 60 секунд; LUFS — эмпирическая поправка `rms_db − 3.0`, не ITU-R BS.1770.

### core/super_analyzer.py — супер-анализ пары (v4.1) — рабочий

- `SuperAnalysisError`, `super_analyze_pair(vocals_path, instrumental_path, report_dir=None, progress_cb=None)`.
- Метрики: `_compute_bleed_score` (протечка вокала в минус), `_compute_vpi` (vocal presence index), `_phase_smearing`, `_spectral_tilt`, `_transient_loss`, `_spectral_difference_per_second`, карта подозрительных секунд.
- Вердикт `_make_verdict_v4(...)`; графики `_save_waveforms`, `_save_envelopes`, `_save_suspicious`; текст — `_render_text_v4`.
- `_json_safe(obj)` приводит numpy-типы к питоновским.

### core/midi_analyzer.py — анализ MIDI

- `MidiAnalysisError`, `analyze_midi(midi_path, report_dir=None) -> dict`. **Прогресс-колбэка нет** (UI показывает только статус).
- Метрики: диапазон, распределение по октавам/нотам (`NOTE_NAMES`, `_note_name`, `_octave_label`), плотность, velocity, темп (`estimate_tempo()`, fallback 120.0), длительность.
- Без нот — исключение «В MIDI нет нот».

### core/player.py — плеер

- `AudioPlayer`: `load(path)`, `play()`, `pause()`, `stop()`, `seek(seconds)`, `set_volume(v)`; свойства `is_playing`, `current_file`, `position`, `duration`.
- Реализация: `sounddevice.OutputStream` с callback, `blocksize=2048`, состояние под `threading.Lock`, остановка через `raise sd.CallbackStop`.

### core/logging_config.py — логи

- `setup_logging(level=logging.INFO)`: файл `logs/ast.log` (`RotatingFileHandler`, 5 МБ × 3, utf-8) + консоль.
- Формат файла — **JSON-lines**: `ts`, `level`, `logger`, `module`, `line`, `msg`, `exc` (traceback), плюс любые `extra={"ctx_...": ...}` → ключ без префикса `ctx_`.
- Контракт: всё нужное для разбора полётов добавляй через `extra={"ctx_...": ...}` — иначе в JSON этого не будет.

### Мёртвый код (никем не импортируется)

- `core/piano_reducer.py` — сведение MIDI в пиано (`PianoReducer.reduce`); логика заменена кодом внутри `piano_transcriber.py`.
- `core/coresuper_analyzer.py` — старая версия супер-анализатора (дополнительно рисует `_super_pair_flatness.png`); `worker.py` использует `super_analyzer.py`.

Перед удалением стоит сверить, нет ли в них нужных формул и порогов — они служили справочником.

---

## 7. Настройки и конфигурация

- **`config/settings.json` пустой и кодом не читается** — ни `json.load`, ни обращения к `settings.json` в проекте нет. Все параметры зашиты константами в модулях: список движков, модели, частоты фильтров, пороги, громкость 0.8, размеры окна, пути `output/` и `logs/`.
- **Переменные окружения:** единственное использование — `os.environ.copy()` + `PYTHONUNBUFFERED=1` перед запуском `audio-separator` (`core/separator_bs.py`). `HF_TOKEN`, `HUGGINGFACE*`, `API_KEY` в коде **отсутствуют**.
- Токен HuggingFace лежит открытым текстом в `Токен/Text Document.txt` и кодом не используется — см. §9.4.

---

## 8. Куда что сохраняется

| Действие | Путь | Файлы |
|---|---|---|
| Разделение (любой движок) | `./output` (`Path.cwd()/"output"`) | `<base>_vocals.wav`, `<base>_instrumental.wav` (PCM_16, 44.1 кГц, пик −0.3 dBFS) |
| BS-разделение | `./output` | временные FLAC в `./output/_bs_tmp` |
| Гибрид | `./output` | временные `./output/_hybrid_demucs`, `./output/_hybrid_bs` |
| Транскрипция | `./output` | `<base>_vocals.mid`, `<base>_instrumental.mid` |
| Минус → пианино | `./output` | `<base>_piano.mid` + `<base>_piano_multi.mid` (остаётся) |
| Анализ аудио | `./logs/analysis` | `<base>_analysis.json`, `.txt`, `_waveform.png`, `_spectrogram.png` |
| Анализ MIDI | `./logs/analysis` | `<base>_midi_analysis.json`, `.txt` |
| Супер-анализ пары | `./logs/analysis` | `<base>_super_pair_analysis.json`, `.txt`, `_super_pair_waveforms.png`, `_envelopes.png`, `_suspicious.png` |
| Логи | `./logs` | `ast.log` (JSON-lines, ротация 5 МБ × 3) |

`<base>` — имя исходного файла без расширения. Пути зависят от рабочего каталога запуска (`Path.cwd()`), поэтому запускать приложение лучше из корня проекта. В коде ядра дефолт отчётов другой (`<папка файла>/analysis`) — UI всегда передаёт `./logs/analysis` явно.

---

## 9. Особые места, о которых надо знать до правок

### 9.1 Инверсия параметра notch (аудит P0.1, вариант B)

`INSTR_NOTCH_GAIN_DB` — **не глубина**, а параметр смешивания сухого и отфильтрованного сигнала. Фактическая глубина на центральной частоте: `20*log10(1 − 10**(gain_db/20))`.

| `gain_db` | Фактическая глубина |
|---|---|
| −1.5 (текущее в `separator_bs.py`) | ≈ −16.0 dB |
| −3.0 (текущее в `separator.py`) | ≈ −10.7 dB |
| −6.0 | ≈ −6.0 dB |
| −10.0 | ≈ −3.3 dB |
| −16.0 | ≈ −1.5 dB |

Правило: **ближе к нулю — глубже, отрицательнее — мягче.** Физика оставлена намеренно; фактическая глубина измеряется `measure_notch_depth` и пишется в лог один раз за процесс. При изменении константы сверяйся с таблицей, а не со знаком «dB».

### 9.2 `torchcrepe` и частота дискретизации

`hop_length=160` и `frame_dur = 160.0 / sr` зашиты в предположении 16 кГц, но ресемплинга к 16 кГц нет — `sr` берётся из файла. Для 44.1/48 кГц тайминги нот будут смещены. Известное ограничение.

### 9.3 Внешние зависимости рантайма

Без них приложение стартует, но часть сценариев падает:

- **FFmpeg / ffprobe** в `PATH` (`winget install Gyan.FFmpeg`) — MP3/M4A и конвертация FLAC→WAV.
- **`audio-separator.exe`** в `.venv312\Scripts` (`pip install audio-separator[gpu]`) — BS-Roformer и гибрид.
- **CUDA 12.1** — сборки torch `+cu121`. Без GPU код откатится на CPU, но время вырастет в разы.
- **Веса моделей** Demucs / MuScriptor / Roformer скачиваются при первом запуске (нужен интернет, для части Roformer-моделей — токен HuggingFace).

### 9.4 Секреты

`Токен/Text Document.txt` содержит HuggingFace-токен открытым текстом, код его не читает. Что сделать:

1. Считать токен скомпрометированным и **перевыпустить**, если папка куда-то копировалась или архивировалась.
2. Хранить секреты в `HF_TOKEN` (переменная окружения) или в файле вне репозитория.
3. Добавить `.gitignore`: `Токен/`, `.venv312/`, `output/`, `logs/`, `__pycache__/`.

### 9.5 Прочее

- Временные каталоги и файлы не удаляются: `_bs_tmp`, `_hybrid_demucs`, `_hybrid_bs`, `<base>_piano_multi.mid` — со временем копятся в `output/`.
- Ошибки в UI частично глушатся (`except Exception: pass`) — при отладке смотри `logs/ast.log`, а не окно.
- Явных `TODO/FIXME/HACK/NotImplementedError` в коде нет — незавершённые места обозначены иначе (см. §5 и §10).

---

## 10. Запуск

```powershell
cd E:\Programmy\AST
.\.venv312\Scripts\Activate.ps1
python main.py
```

Либо без активации: `.\.venv312\Scripts\python.exe main.py`.

Проверка окружения:

```powershell
.\.venv312\Scripts\python.exe -c "import torch, demucs, torchcrepe, muscriptor, customtkinter, scipy; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
Test-Path .\.venv312\Scripts\audio-separator.exe
```

Ожидаемое поведение: создаются `logs/` (с `ast.log`) и `output/`, открывается окно, движок по умолчанию — BS-Roformer Duality-V2. Первый прогон разделения скачивает веса — это долго и требует интернета. Дальше: выбрать файл (клик по зоне загрузки), «🔀 Разделить», затем транскрипция/анализ по кнопкам; результаты — в `output/`, отчёты — в `logs/analysis/`.

---

## 11. Технический долг и риски (по приоритету)

| # | Проблема | Риск | Что делать |
|---|---|---|---|
| 1 | Нет системы контроля версий в папке | любые правки необратимы | `git init` + `.gitignore` (`output/`, `logs/`, `.venv312/`, `__pycache__/`, `Токен/`) |
| 2 | Токен HF открытым текстом | утечка доступа к моделям | перевыпустить токен, перейти на `HF_TOKEN`, убрать файл из репозитория |
| 3 | `requirements.txt` не соответствует окружению | окружение не воспроизводится | пересобрать из `pip freeze`, вынести runtime/dev, убрать `basic-pitch` |
| 4 | Нет отмены задач и лимита параллелизма | долгие прогоны не остановить; несколько Demucs съедят VRAM | токен отмены (`threading.Event`) в `Worker` + проверки в циклах; лимит одновременных GPU-задач |
| 5 | Конфигурация зашита в код, `settings.json` мёртв | правка параметра = правка кода | вынести движки/фильтры/пути в JSON-конфиг с дефолтами и валидацией |
| 6 | Мёртвые модули `piano_reducer.py`, `coresuper_analyzer.py` | путаница при чтении | сверить формулы и удалить |
| 7 | `torchcrepe` без ресемплинга к 16 кГц | смещённые тайминги нот | ресемплинг входа либо `hop_length` из `sr` |
| 8 | Временные файлы не чистятся | мусор в `output/` | уборка временных каталогов после успешного прогона |
| 9 | Drag & drop не подключён | обещание в UI не выполняется | либо подключить `tkinterdnd2`, либо убрать текст |
| 10 | Тема не применяется к виджетам | переключение темы не работает | прокинуть цвета из `theme.py` в `components.py` |
| 11 | Нет автотестов | регрессии в DSP ловятся только на слух | тесты на чистых функциях (`_frames_to_notes`, `measure_notch_depth`, `_json_safe`) + smoke-тест `Worker` |
| 12 | Пустой README | вход в проект непонятен | README + ссылка на этот документ |
| 13 | Смешение языков: русские docstring/логи, английские идентификаторы | низкий | зафиксировать правило: идентификаторы английские, тексты для пользователя русские |

---

## 12. Как безопасно менять код

1. Сначала зафиксировать состояние: `git init` + первый коммит. Пока этого нет, любая ошибка необратима.
2. Ядро и края не смешивать: логика — в `core/`, виджеты — в `ui/`, связь только через `worker.py` и его события.
3. Новый сценарий: добавить `TaskType`, метод `submit_*`, ветку в `Worker._run`, обработку в UI. Контракт событий не менять.
4. Новые поля в логах — через `extra={"ctx_...": ...}`, иначе JSON-лог их потеряет.
5. Числовые параметры DSP менять по таблице §9.1 и сверять с `measure_notch_depth` в логе.
6. После правки: прогнать короткий трек (~30 сек) через затронутый сценарий и прочитать `logs/ast.log` — там каждый шаг с параметрами.
7. Проверять на Python 3.12 (`cpython-314` байткод в кэше — наследие других запусков, ориентироваться на него не надо).
