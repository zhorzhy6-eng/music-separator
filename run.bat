@echo off
chcp 65001 >nul
title AST - GPU (Python 3.12 + CUDA)
cd /d "%~dp0"

echo Запускаю AST на GPU (Python 3.12 + CUDA)...
echo.

call ".venv312\Scripts\activate.bat"

if errorlevel 1 (
    echo [ОШИБКА] Не удалось активировать .venv312
    echo Проверь, что папка .venv312 есть в корне проекта.
    pause
    exit /b 1
)

echo Проверка CUDA...
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет')"
echo.

python main.py

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Приложение завершилось с кодом %errorlevel%
    echo Смотри logs\ast.log
    pause
)