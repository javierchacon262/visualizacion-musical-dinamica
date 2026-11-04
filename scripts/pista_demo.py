#!/usr/bin/env python3
"""
Arma la pista de demostración empalmando seis fragmentos de DEAM.

    python scripts/pista_demo.py

Los seis están elegidos en los extremos de los cuatro cuadrantes del plano
valencia–activación, así que el color tiene que cambiar de forma visible al
pasar de uno a otro. Sirve para comprobar el sistema de un vistazo.

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

# (id en DEAM, cuadrante según la anotación humana)
FRAGMENTOS = [
    (1334, "tensión   V− A+"),
    (375, "tristeza  V− A−"),
    (631, "calma     V+ A−"),
    (115, "alegría   V+ A+"),
    (488, "tristeza  V− A−"),
    (640, "alegría   V+ A+"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parent.parent
    ap.add_argument("--salida", type=Path,
                    default=raiz / "demo" / "recorrido-cuadrantes.mp3")
    args = ap.parse_args()

    faltan = [i for i, _ in FRAGMENTOS if not (cfg.AUDIO_DIR / f"{i}.mp3").exists()]
    if faltan:
        print(f"ERROR: faltan los mp3 {faltan} en {cfg.AUDIO_DIR}.\n"
              f"       Corre antes:  make datos", file=sys.stderr)
        return 1

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    print("Empalmando:")
    for i, q in FRAGMENTOS:
        print(f"  {i:>5}  {q}")

    # El demuxer concat de ffmpeg lee una lista de archivos; se escribe en un
    # temporal porque las rutas absolutas no caben cómodas en la línea de órdenes.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for i, _ in FRAGMENTOS:
            f.write(f"file '{cfg.AUDIO_DIR / f'{i}.mp3'}'\n")
        lista = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", lista, "-ar", "44100", "-ac", "2", "-b:a", "192k",
             str(args.salida)],
            check=True)
    except FileNotFoundError:
        print("ERROR: no encuentro ffmpeg. Instálalo y vuelve a intentar.",
              file=sys.stderr)
        return 1
    finally:
        Path(lista).unlink(missing_ok=True)

    mb = args.salida.stat().st_size / 1e6
    print(f"\n✓ {args.salida}  ({mb:.1f} MB)")
    print(f"\nSiguiente paso:  make demo ARCHIVO={args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
