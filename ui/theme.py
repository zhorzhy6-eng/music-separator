"""Цветовые схемы для CustomTkinter."""
from __future__ import annotations


DARK = {
    "bg": "#1a1a1a",
    "panel": "#242424",
    "accent": "#3b8ed0",
    "accent_hover": "#36719f",
    "text": "#e8e8e8",
    "text_dim": "#9a9a9a",
    "success": "#4caf50",
    "error": "#e53935",
}

LIGHT = {
    "bg": "#f5f5f5",
    "panel": "#ffffff",
    "accent": "#1f6aa5",
    "accent_hover": "#144870",
    "text": "#1a1a1a",
    "text_dim": "#666666",
    "success": "#2e7d32",
    "error": "#c62828",
}


def get_theme(name: str = "dark") -> dict:
    return DARK if name == "dark" else LIGHT