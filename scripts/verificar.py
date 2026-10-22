#!/usr/bin/env python3
"""
Comprueba que el entorno, la GPU y los datos están como el pipeline espera.

    python scripts/verificar.py

Devuelve 0 si todo lo imprescindible está en su sitio.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

OK, AVISO, MAL = "✓", "!", "✗"
fallos = 0
avisos = 0


def linea(estado: str, texto: str, detalle: str = "") -> None:
    global fallos, avisos
    if estado == MAL:
        fallos += 1
    elif estado == AVISO:
        avisos += 1
    print(f"  {estado} {texto}" + (f"  —  {detalle}" if detalle else ""))


def seccion(t: str) -> None:
    print(f"\n{t}")


def main() -> int:
    print("Verificación del entorno · visualización musical dinámica")

    # ── Python ──
    seccion("Python")
    v = sys.version_info
    if (v.major, v.minor) == (3, 12):
        linea(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        linea(AVISO, f"Python {v.major}.{v.minor}",
              "el proyecto se probó en 3.12; otras versiones pueden fallar")
    if sys.prefix == sys.base_prefix:
        linea(AVISO, "no parece un entorno virtual",
              "¿olvidaste 'source .venv/bin/activate'?")
    else:
        linea(OK, "entorno virtual activo")

    # ── librerías ──
    seccion("Librerías")
    for mod, imprescindible in [
        ("numpy", True), ("scipy", True), ("librosa", True), ("soundfile", True),
        ("pandas", True), ("sklearn", True), ("tqdm", True), ("matplotlib", False),
        ("tensorflow", False), ("torch", False),
    ]:
        try:
            m = importlib.import_module(mod)
            linea(OK, f"{mod:12}", getattr(m, "__version__", "?"))
        except ImportError:
            linea(MAL if imprescindible else AVISO, f"{mod:12}", "no instalado")

    # ── GPU ──
    seccion("GPU (opcional — el pipeline de datos no la necesita)")
    try:
        import torch
        if torch.cuda.is_available():
            d = torch.cuda.get_device_properties(0)
            linea(OK, "PyTorch ve la GPU",
                  f"{d.name}, {d.total_memory/1e9:.1f} GB, sm_{d.major}{d.minor}")
        else:
            linea(AVISO, "PyTorch no ve GPU", "entrenaría en CPU")
    except ImportError:
        linea(AVISO, "PyTorch no instalado")

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        g = tf.config.list_physical_devices("GPU")
        if g:
            linea(OK, "TensorFlow ve la GPU", f"{len(g)} dispositivo(s)")
        else:
            linea(AVISO, "TensorFlow no ve GPU",
                  "si tienes GPU, revisa que LD_LIBRARY_PATH apunte a "
                  "site-packages/nvidia/*/lib (lo hace .venv/bin/activate)")
    except ImportError:
        linea(AVISO, "TensorFlow no instalado")

    # ── datos ──
    seccion("Datos")
    try:
        import config as cfg
    except ImportError:
        linea(MAL, "no pude importar src/config.py")
        return 1

    linea(OK, "DEAM_DIR", str(cfg.DEAM_DIR))
    if not cfg.DEAM_DIR.exists():
        linea(AVISO, "el directorio no existe", "corre 'make datos'")
    else:
        n = len(list(cfg.AUDIO_DIR.glob("*.mp3"))) if cfg.AUDIO_DIR.exists() else 0
        linea(OK if n == 1802 else AVISO, f"{n} archivos mp3",
              "" if n == 1802 else "se esperaban 1802 — corre 'make datos'")
        for f in ("arousal.csv", "valence.csv"):
            p = cfg.ANOT_DIR / f
            linea(OK if p.exists() else AVISO, f"anotaciones {f}",
                  "" if p.exists() else "faltan")
        proc = cfg.SALIDA_DIR / "caracteristicas.npz"
        if proc.exists():
            import numpy as np
            d = np.load(proc, allow_pickle=True)
            linea(OK, "características extraídas",
                  f"X{d['X'].shape}  ·  {d['X'].shape[0]*d['X'].shape[1]:,} puntos etiquetados")
        else:
            linea(AVISO, "sin características extraídas", "corre 'make extraer'")

    # ── mapeo de color ──
    seccion("Mapeo emoción → color")
    try:
        import color
        casos = [("ira (V−,A+)", -0.8, 0.8, (255, 0, 0)),
                 ("alegría (V+,A+)", 0.8, 0.8, (255, 255, 0)),
                 ("tristeza (V−,A−)", -0.8, -0.8, (0, 0, 255))]
        for nom, val, act, esperado in casos:
            rgb = color.va_a_rgb(val, act)
            linea(OK if rgb == esperado else MAL, f"{nom:18}", f"RGB {rgb}")
    except Exception as e:
        linea(MAL, "color.py falló", str(e))

    # ── resumen ──
    print()
    if fallos:
        print(f"✗ {fallos} problema(s) que impiden correr el proyecto"
              + (f", {avisos} aviso(s)" if avisos else ""))
        return 1
    if avisos:
        print(f"✓ Lo imprescindible está listo — {avisos} aviso(s) arriba")
    else:
        print("✅ Todo en orden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
