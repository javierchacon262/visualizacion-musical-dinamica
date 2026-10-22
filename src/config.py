"""
Rutas y constantes del proyecto.

Por defecto los datos viven en ./data dentro del repositorio, para que un
clon recién hecho funcione sin configurar nada. Si ya los tienes en otro
sitio, apunta la variable de entorno y evitas volver a descargar 1.3 GB:

    export DEAM_DIR=/ruta/a/deam
"""

from __future__ import annotations

import os
from pathlib import Path

# ── rutas ──────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DEAM_DIR = Path(os.environ.get("DEAM_DIR", RAIZ / "data"))
AUDIO_DIR = DEAM_DIR / "MEMD_audio"
ANOT_DIR = DEAM_DIR / "annotations" / "annotations averaged per song" / "dynamic (per second annotations)"
ANOT_ESTATICAS = (
    DEAM_DIR / "annotations" / "annotations averaged per song" / "song_level"
)
SALIDA_DIR = Path(os.environ.get("DEAM_OUT", DEAM_DIR / "procesado"))
MEL_DIR = SALIDA_DIR / "mel"

# ── audio ──────────────────────────────────────────────────────────────────
SR = 22050

# El análisis espectral se hace fino (50 Hz) y luego se agrega en bloques para
# casar con las etiquetas. Hacerlo directamente a 2 Hz desperdiciaría el 80 %
# del audio: con hop de 0.5 s, una FFT de 2048 muestras solo mira 93 ms de
# cada medio segundo.
HOP = 441                 # 20 ms  →  50 fotogramas por segundo
N_FFT = 2048
FPS_ANALISIS = SR / HOP   # 50.0
BLOQUE = 25               # 25 fotogramas de 20 ms = 0.5 s exactos

# ── anotaciones ────────────────────────────────────────────────────────────
# DEAM anota a 2 Hz empezando en el segundo 15 de cada fragmento.
HZ_ANOT = 2.0
INICIO_ANOT_S = 15.0
N_BINS = 60               # 30 s de anotación por fragmento

# ── mel ────────────────────────────────────────────────────────────────────
N_MELS = 128
