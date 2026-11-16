"""
Recupera la imagen de OASIS cuya emoción anotada está más cerca de la predicha.

OASIS (Kurdi, Lozano & Banaji, 2017) valora 900 imágenes en **los mismos dos
ejes** que predice el modelo de audio. Por eso el enlace no necesita un puente
inventado: se busca el vecino más cercano en el plano (valencia, activación).

Es el método de Xing et al. (2019, IEEE Access), que enlazan imagen y música a
través del espacio emocional en vez de entrenar un generador transmodal.

    python src/imagenes.py     # comprueba el corpus y la recuperación
"""

from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# OASIS valora de 1 a 7, con 4 en el centro.
_CENTRO, _RADIO = 4.0, 3.0

CATEGORIAS = ("Object", "Scene", "Animal")
POR_DEFECTO = CATEGORIAS

# `Person` (346 de las 900) queda FUERA del corpus, no solo desmarcada en la
# interfaz. OASIS es un conjunto afectivo, no un banco de imágenes agradables:
# cubre la valencia negativa con desnudos, heridas y agresión, y casi todo eso
# son personas. Se filtra en `cargar()` para que no pueda colarse ni pasando la
# categoría a mano en la URL.
EXCLUIDAS = ("Person",)


@lru_cache(maxsize=1)
def cargar() -> list[dict]:
    """Lee el CSV normativo. Devuelve una entrada por imagen."""
    if not cfg.OASIS_CSV.exists():
        raise FileNotFoundError(
            f"No encuentro {cfg.OASIS_CSV}. Corre antes:  make imagenes")

    salida = []
    with open(cfg.OASIS_CSV, newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            if fila["Category"].strip() in EXCLUIDAS:
                continue
            tema = fila["Theme"].strip()
            archivo = f"{tema}.jpg"
            if not (cfg.OASIS_IMG / archivo).exists():
                continue
            salida.append({
                "tema": tema,
                "archivo": archivo,
                "categoria": fila["Category"].strip(),
                "valencia": (float(fila["Valence_mean"]) - _CENTRO) / _RADIO,
                "activacion": (float(fila["Arousal_mean"]) - _CENTRO) / _RADIO,
            })
    return salida


def _tabla(categorias: tuple[str, ...]) -> tuple[list[dict], np.ndarray]:
    sub = [e for e in cargar() if e["categoria"] in categorias]
    if not sub:
        raise ValueError(f"ninguna imagen en las categorías {categorias}")
    puntos = np.array([[e["valencia"], e["activacion"]] for e in sub])
    return sub, puntos


def buscar(valencia: float, activacion: float,
           categorias: tuple[str, ...] = POR_DEFECTO,
           evitar: set[str] | None = None) -> dict:
    """La imagen más cercana en el plano, sin repetir las de `evitar`."""
    sub, puntos = _tabla(tuple(categorias))
    d = np.hypot(puntos[:, 0] - valencia, puntos[:, 1] - activacion)
    if evitar:
        for i, e in enumerate(sub):
            if e["archivo"] in evitar:
                d[i] = np.inf
        if not np.isfinite(d).any():          # todas evitadas: se reinicia
            d = np.hypot(puntos[:, 0] - valencia, puntos[:, 1] - activacion)
    return sub[int(np.argmin(d))]


def secuencia(valencia, activacion,
              categorias: tuple[str, ...] = POR_DEFECTO,
              umbral: float = 0.22, minimo_bloques: int = 16,
              sin_repetir: int = 6) -> list[dict]:
    """
    Reparte la canción en tramos, uno por imagen.

    Cambiar de imagen en cada bloque de medio segundo daría un estroboscopio.
    La imagen se mantiene hasta que se cumplen **las dos** condiciones: que
    hayan pasado `minimo_bloques` y que el punto emocional se haya movido más
    de `umbral` respecto al que la eligió. Así los cambios caen donde la música
    de verdad cambia, no a intervalos fijos.

    Devuelve [{desde, hasta, archivo, tema, categoria, valencia, activacion}].
    """
    val = np.asarray(valencia, dtype=float)
    aro = np.asarray(activacion, dtype=float)
    n = len(val)
    cats = tuple(categorias)

    tramos: list[dict] = []
    recientes: list[str] = []
    i = 0
    while i < n:
        e = buscar(float(val[i]), float(aro[i]), cats, set(recientes))
        recientes.append(e["archivo"])
        del recientes[:-sin_repetir]

        j = i + 1
        while j < n:
            movido = np.hypot(val[j] - val[i], aro[j] - aro[i])
            if j - i >= minimo_bloques and movido > umbral:
                break
            j += 1

        tramos.append({"desde": i, "hasta": j, **e})
        i = j
    return tramos


def _verificar() -> int:
    try:
        corpus = cargar()
    except FileNotFoundError as err:
        print(f"✗ {err}", file=sys.stderr)
        return 1

    from collections import Counter
    print(f"OASIS: {len(corpus)} imágenes en {cfg.OASIS_IMG}\n")
    print(f"  {'categoría':<10}{'n':>5}{'valencia':>22}{'activación':>22}")
    for cat, n in Counter(e["categoria"] for e in corpus).most_common():
        sub = [e for e in corpus if e["categoria"] == cat]
        v = [e["valencia"] for e in sub]; a = [e["activacion"] for e in sub]
        print(f"  {cat:<10}{n:>5}{f'{min(v):+.2f} .. {max(v):+.2f}':>22}"
              f"{f'{min(a):+.2f} .. {max(a):+.2f}':>22}")

    print(f"\n  Excluidas del corpus: {', '.join(EXCLUIDAS)}")
    print(f"  Recuperación por cuadrante ({', '.join(POR_DEFECTO)})\n")
    print(f"  {'cuadrante':<20}{'imagen':<28}{'categoría':<10}{'V':>7}{'A':>7}")
    for nombre, v, a in [("tensión  (V− A+)", -0.7, 0.5),
                         ("alegría  (V+ A+)", 0.7, 0.5),
                         ("calma    (V+ A−)", 0.7, -0.5),
                         ("tristeza (V− A−)", -0.7, -0.5)]:
        e = buscar(v, a)
        print(f"  {nombre:<20}{e['tema'][:26]:<28}{e['categoria']:<10}"
              f"{e['valencia']:>+7.2f}{e['activacion']:>+7.2f}")

    # un recorrido sintético, para ver que los tramos no se atomizan
    t = np.linspace(0, 4 * np.pi, 540)
    tr = secuencia(0.5 * np.sin(t), 0.5 * np.cos(t))
    largos = [x["hasta"] - x["desde"] for x in tr]
    print(f"\n  Sobre un recorrido de 540 bloques (4.5 min): {len(tr)} tramos, "
          f"de {min(largos)} a {max(largos)} bloques "
          f"({min(largos)/2:.0f}–{max(largos)/2:.0f} s)")
    if min(largos) < 8:
        print("  ✗ hay tramos de menos de 4 s: parpadearía")
        return 1
    print("  ✅ sin tramos demasiado cortos")
    return 0


if __name__ == "__main__":
    raise SystemExit(_verificar())
