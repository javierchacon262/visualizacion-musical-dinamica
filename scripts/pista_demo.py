#!/usr/bin/env python3
"""
Arma las pistas de demostración empalmando fragmentos de DEAM.

    python scripts/pista_demo.py                      # todas
    python scripts/pista_demo.py --pista calma-alegria

Cada pista alterna fragmentos elegidos en los extremos de su cuadrante, así que
el color debería cambiar de forma visible al pasar de uno a otro.

Un aviso sobre `calma-alegria`: medido sobre el conjunto de test, el modelo
acierta el cuadrante de calma en 1 de 21 canciones (5 %) y lo confunde con
tristeza. Esa pista enseña el fallo, no el acierto; para una demostración que
luzca, `alegria-tristeza` usa los dos cuadrantes que el modelo sí maneja
(84 % y 71 %).

Se regenera en vez de versionarse porque son 6 MB de audio derivado: el
repositorio guarda la receta, no el resultado. El audio de DEAM tiene licencia
Creative Commons, así que la pista se puede mostrar y publicar sin problema.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as cfg

# Cada pista es una lista de (id en DEAM, cuadrante según la anotación humana).
# Los ids están elegidos en los extremos de su cuadrante.
PISTAS = {
    "cuadrantes": {
        "descripcion": "recorrido por los cuatro cuadrantes",
        "fragmentos": [
            (1334, "tensión   V− A+"), (375, "tristeza  V− A−"),
            (631, "calma     V+ A−"), (115, "alegría   V+ A+"),
            (488, "tristeza  V− A−"), (640, "alegría   V+ A+"),
        ],
    },
    "calma-alegria": {
        "descripcion": "alterna calma y alegría — enseña el fallo en calma",
        "fragmentos": [
            (631, "calma     V+ A−"), (115, "alegría   V+ A+"),
            (253, "calma     V+ A−"), (640, "alegría   V+ A+"),
            (45, "calma     V+ A−"), (960, "alegría   V+ A+"),
        ],
    },
    "alegria-tristeza": {
        "descripcion": "alterna alegría y tristeza — los dos cuadrantes que acierta",
        "fragmentos": [
            (115, "alegría   V+ A+"), (375, "tristeza  V− A−"),
            (640, "alegría   V+ A+"), (488, "tristeza  V− A−"),
            (960, "alegría   V+ A+"), (379, "tristeza  V− A−"),
        ],
    },
}


def armar(clave: str, destino: Path) -> int:
    """Empalma los fragmentos de una pista. Devuelve 0 si salió bien."""
    pista = PISTAS[clave]
    fragmentos = pista["fragmentos"]

    faltan = [i for i, _ in fragmentos if not (cfg.AUDIO_DIR / f"{i}.mp3").exists()]
    if faltan:
        print(f"ERROR: faltan los mp3 {faltan} en {cfg.AUDIO_DIR}.\n"
              f"       Corre antes:  make datos", file=sys.stderr)
        return 1

    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{clave} — {pista['descripcion']}")
    for i, q in fragmentos:
        print(f"  {i:>5}  {q}")

    # El demuxer concat de ffmpeg lee una lista de archivos; se escribe en un
    # temporal porque las rutas absolutas no caben cómodas en la orden.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for i, _ in fragmentos:
            f.write(f"file '{cfg.AUDIO_DIR / f'{i}.mp3'}'\n")
        lista = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", lista, "-ar", "44100", "-ac", "2", "-b:a", "192k",
             str(destino)],
            check=True)
    except FileNotFoundError:
        print("ERROR: no encuentro ffmpeg. Instálalo y vuelve a intentar.",
              file=sys.stderr)
        return 1
    finally:
        Path(lista).unlink(missing_ok=True)

    print(f"  → {destino}  ({destino.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parent.parent
    ap.add_argument("--pista", choices=[*PISTAS, "todas"], default="todas")
    ap.add_argument("--destino", type=Path, default=raiz / "demo")
    args = ap.parse_args()

    claves = list(PISTAS) if args.pista == "todas" else [args.pista]
    for clave in claves:
        codigo = armar(clave, args.destino / f"{clave}.mp3")
        if codigo:
            return codigo

    print(f"\nSiguiente paso:  make visor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
