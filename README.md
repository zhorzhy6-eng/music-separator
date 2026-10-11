# AST — Audio Separator & Transcriber

Локальное десктопное приложение под Windows для работы с музыкой: разделение трека
на вокал и минус, транскрипция в MIDI и анализ качества результата.

Всё считается **локально на вашей машине** — аудио никуда не отправляется.

## Что умеет

- **Разделение на вокал и минус** тремя движками:
  - **Demucs** (`htdemucs`, `htdemucs_ft`) — быстро, ~3–5 минут на трек;
  - **BS-Roformer / Mel-Band Roformer** — 20+ моделей на выбор (Viperx, Duality,
    Becruily, Gabox, Unwa и др.), качество выше, но до 30 минут на трек;
  - **Гибрид** — вокал от Demucs, минус от BS-Roformer.
- **Постобработка стемов** — HPF, notch 2.5 кГц, LPF, нормализация пика до −0.3 dBFS.
- **Транскрипция в MIDI** — вокал и минус через `torchcrepe`; отдельно
  «минус → фортепиано» через нейросеть MuScriptor.
- **Анализ аудио** — уровни, клиппинг, тишина, октавный спектр, стерео, грубая F0 + графики.
- **Супер-анализ пары «вокал + минус»** — метрики артефактов разделения
  (phase smearing, spectral tilt, transient loss, bleed score), карта подозрительных
  секунд и итоговый вердикт.
- **Анализ MIDI** — диапазон, октавы, топ нот, плотность, velocity, темп.

## Требования

| Что | Версия / примечание |
|---|---|
| ОС | Windows 10/11 |
| Python | 3.12 |
| GPU | NVIDIA с CUDA 12.1 (можно и на CPU, но в разы медленнее) |
| FFmpeg | `ffmpeg` и `ffprobe` должны быть в `PATH` |
| Интернет | нужен один раз — для скачивания весов моделей |

## Установка

```powershell
git clone https://github.com/zhorzhy6-eng/music-separator.git
cd music-separator
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

Дальше — зависимости, двумя шагами. Первый файл ставит всё, кроме torch:

```powershell
pip install -r requirements.txt
```

Второй — сам torch под CUDA 12.1. Он вынесен отдельно, потому что сборки `+cu121`
живут не на PyPI, а на индексе PyTorch; индекс уже прописан внутри файла:

```powershell
pip install -r requirements-gpu.txt
```

Сборка `torch 2.5.1+cu121` требует драйвер NVIDIA от 525.60.13. Если GPU нет,
поставьте обычный torch с PyPI — на CPU всё работает, но в разы медленнее:

```powershell
pip install torch torchaudio
```

Про `audio-separator`: он даёт `.venv312\Scripts\audio-separator.exe` — именно его
вызывает приложение для BS-Roformer. Без него движки BS и «Гибрид» не работают.

## Токен Hugging Face

Часть моделей BS-Roformer лежит на Hugging Face в закрытых (gated) репозиториях —
для их скачивания нужен токен: https://huggingface.co/settings/tokens

Задайте его переменной окружения — код читать файлы с токеном не умеет:

```powershell
# только для текущей сессии
$env:HF_TOKEN = "hf_ваш_токен"

# либо постоянно, для текущего пользователя
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_ваш_токен", "User")
```

`audio-separator` подхватит переменную сам. Модели Demucs, torchcrepe и MuScriptor
скачиваются без токена.

## Запуск

```powershell
.\.venv312\Scripts\python.exe main.py
```

Проверка окружения перед первым запуском:

```powershell
.\.venv312\Scripts\python.exe -c "import torch, demucs, torchcrepe, customtkinter, scipy; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
Test-Path .\.venv312\Scripts\audio-separator.exe
```

Ожидаемое поведение: создаются каталоги `logs/` и `output/`, открывается окно.
**Первый прогон разделения скачивает веса моделей** — это долго и требует интернета.
Дальше: выбрать файл, нажать «🔀 Разделить», затем транскрипция и анализ по кнопкам.

## Результаты

- `output/` — `*_vocals.wav`, `*_instrumental.wav`, `*.mid`;
- `logs/ast.log` — основной лог (JSON-lines, ротация 5 МБ × 3);
- `logs/analysis/` — отчёты анализа: JSON, TXT и PNG с графиками.

Временные каталоги (`output/_bs_tmp`, `_hybrid_demucs`, `_hybrid_bs`) после работы
не удаляются — их можно чистить вручную.

## Структура проекта

```
main.py                      # точка входа
core/                        # ядро: разделение, транскрипция, анализ
  separator.py               #   Demucs
  separator_bs.py            #   BS-Roformer / Mel-Band Roformer (audio-separator CLI)
  separator_hybrid.py        #   гибрид Demucs + BS-Roformer
  transcriber.py             #   аудио → MIDI (torchcrepe)
  piano_transcriber.py       #   минус → фортепиано (MuScriptor)
  analyzer.py                #   анализ аудио
  super_analyzer.py          #   супер-анализ пары «вокал + минус»
  midi_analyzer.py           #   анализ MIDI
  player.py                  #   плеер (sounddevice)
  worker.py                  #   фоновые задачи + очередь событий
ui/                          # интерфейс: окно, компоненты, темы
config/settings.json         # пока не используется — настройки зашиты в код
```

Принцип: ядро (`core/`) не знает про виджеты, край (`ui/`) общается с ним только
через `core/worker.py` и его события.

## Ограничения

- Отмены долгих задач нет: запущенное разделение из интерфейса не остановить.
- Лимита параллельных задач нет — несколько прогонов на GPU съедят видеопамять.
- Автотестов нет.
- Настройки зашиты в код, `config/settings.json` не используется.
- Драг-энд-дроп в интерфейсе заявлен, но не подключён.
- `tkinterdnd2` установлен, но в коде не применяется.

