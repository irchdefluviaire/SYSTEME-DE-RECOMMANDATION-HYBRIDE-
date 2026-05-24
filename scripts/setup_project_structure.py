#!/usr/bin/env python3
"""Créer les dossiers manquants pour organiser le projet."""

import os
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

# Dossiers à créer
DIRS = [
    BASE_DIR / "outputs" / "evaluation",
    BASE_DIR / "outputs" / "traces",
    BASE_DIR / "data" / "pdf_chunks",
]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)
    gitkeep = d / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    print(f"✓ {d.relative_to(BASE_DIR)}")

print("✅ Tous les dossiers ont été créés!")
