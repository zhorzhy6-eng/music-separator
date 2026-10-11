@echo off
chcp 65001 >nul
cd /d "%~dp0"

call ".venv312\Scripts\activate.bat"

if errorlevel 1 (
    echo [ОШИБКА] Не удалось активировать .venv312
    pause
    exit /b 1
)

start "" pythonw main.py