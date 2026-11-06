#!/usr/bin/env python3
"""
Descarga OASIS: 900 imágenes con valoraciones de valencia y activación.

    python3 scripts/descargar_imagenes.py

Usa solo la biblioteca estándar, igual que `descargar_datos.py`, para poder
correr antes de que exista el entorno virtual.

OASIS (Kurdi, Lozano & Banaji, 2017) valora cada imagen en **los mismos dos
ejes** que predice el modelo de audio. Eso es lo que permite enlazar canción e
imagen sin inventar un puente: se busca la imagen cuya emoción anotada está más
cerca de la que el modelo predice para ese instante.

  Kurdi, B., Lozano, S., & Banaji, M. R. (2017). Introducing the Open Affective
  Standardized Image Set (OASIS). Behavior Research Methods, 49(2), 457–470.
  https://osf.io/6pnd7/

Los autores declaran uso libre y señalan que el conjunto no está sujeto a las
restricciones de copyright del IAPS, aunque no publican una licencia formal.

AVISO DE CONTENIDO: es un conjunto afectivo, no un banco de imágenes
agradables. Para cubrir la valencia negativa incluye heridas, agresión y
escenas angustiosas. El visor filtra por categoría y trae el filtro puesto.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

NODO = "6pnd7"
API = f"https://api.osf.io/v2/nodes/{NODO}/files/osfstorage/"
REINTENTOS = 3
T_CONEXION = 30
TAM_APROX = 92_100_000


def humano(n: float) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unidad}"
        n /= 1024
    return f"{n:,.1f} TB"


def _json(url: str) -> dict:
    pedido = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(pedido, timeout=T_CONEXION) as r:
        return json.load(r)


def enlace_del_zip() -> str | None:
    """
    Resuelve el enlace de descarga preguntando a la API de OSF.

    Se resuelve en vez de fijarlo en el código porque los identificadores de
    archivo de OSF cambian si los autores vuelven a subir algo; el del nodo no.
    """
    try:
        for f in _json(API)["data"]:
            if f["attributes"]["name"] != "Stimulus set":
                continue
            carpeta = f["relationships"]["files"]["links"]["related"]["href"]
            for g in _json(carpeta)["data"]:
                if g["attributes"]["name"] == "OASIS.zip":
                    return g["links"]["download"]
    except Exception as e:
        print(f"  ✗ no pude consultar OSF: {e}", file=sys.stderr)
    return None


def descargar(url: str, ruta: Path) -> bool:
    espera = 5
    for intento in range(1, REINTENTOS + 1):
        if _intentar(url, ruta):
            return True
        if intento < REINTENTOS:
            print(f"  ↻ reintento {intento + 1}/{REINTENTOS} en {espera}s…")
            time.sleep(espera)
            espera *= 2
    print(f"""
  ✗ No se pudo descargar OASIS.zip tras {REINTENTOS} intentos.
    Puedes bajarlo a mano desde https://osf.io/6pnd7/ (Stimulus set →
    OASIS.zip), dejarlo en {ruta.parent} y volver a correr este script:
    lo descomprimirá sin descargar nada.
""", file=sys.stderr)
    return False


def _intentar(url: str, ruta: Path) -> bool:
    ya = ruta.stat().st_size if ruta.exists() else 0
    pedido = urllib.request.Request(url)
    if ya:
        pedido.add_header("Range", f"bytes={ya}-")
        print(f"  ↻ reanudando desde {humano(ya)}")

    try:
        with urllib.request.urlopen(pedido, timeout=T_CONEXION) as r:
            total = int(r.headers.get("Content-Length", 0)) + ya
            modo = "ab" if ya and r.status == 206 else "wb"
            if modo == "wb":
                ya = 0           # el servidor ignoró el Range
            hecho = ya
            with open(ruta, modo) as f:
                while True:
                    trozo = r.read(1024 * 256)
                    if not trozo:
                        break
                    f.write(trozo)
                    hecho += len(trozo)
                    if total:
                        pct = 100 * hecho / total
                        barra = "█" * int(pct / 3) + "·" * (33 - int(pct / 3))
                        print(f"\r  OASIS.zip  [{barra}] {pct:5.1f}%  {humano(hecho)}",
                              end="", flush=True)
            print()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"\n  ✗ {getattr(e, 'reason', e)}", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print("\n  ⏸ Interrumpido. Vuelve a correr el comando para reanudar.")
        raise

    if ruta.stat().st_size < TAM_APROX * 0.5:
        print(f"  ✗ quedó demasiado pequeño ({humano(ruta.stat().st_size)}).",
              file=sys.stderr)
        return False
    return True


def descomprimir(zip_path: Path, destino: Path) -> bool:
    """
    Saca solo el CSV normativo y las imágenes.

    Se comprueba cada entrada contra el destino resuelto: un zip ajeno puede
    traer rutas con `..` que escribirían fuera de la carpeta.
    """
    raiz = destino.resolve()
    sacados = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                nombre = info.filename
                if nombre.startswith("__MACOSX/") or nombre.endswith("/"):
                    continue
                base = Path(nombre).name
                if base == "OASIS.csv":
                    final = raiz / "OASIS.csv"
                elif nombre.startswith("images/") and base.lower().endswith(
                        (".jpg", ".jpeg", ".png")):
                    final = raiz / "images" / base
                else:
                    continue

                if not str(final.resolve()).startswith(str(raiz)):
                    print(f"  ✗ {nombre} escribiría fuera de {destino}. Abortado.",
                          file=sys.stderr)
                    return False
                final.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as origen, open(final, "wb") as salida:
                    shutil.copyfileobj(origen, salida)
                sacados += 1
    except zipfile.BadZipFile:
        print(f"  ✗ {zip_path.name} está corrupto. Bórralo y vuelve a correr.",
              file=sys.stderr)
        return False
    print(f"  ✓ {sacados} archivos extraídos")
    return True


def verificar(destino: Path, silencioso: bool = False) -> bool:
    di = (lambda *a, **k: None) if silencioso else print
    csv = destino / "OASIS.csv"
    imgs = destino / "images"
    n = len(list(imgs.glob("*.jpg"))) + len(list(imgs.glob("*.png"))) if imgs.exists() else 0
    ok = csv.exists() and n >= 800
    di(f"  {'✓' if csv.exists() else '✗'} OASIS.csv")
    di(f"  {'✓' if n >= 800 else '✗'} {n} imágenes")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parent.parent
    # Mismo criterio que config.py, sin importarlo: este script corre antes de
    # que exista el entorno virtual.
    por_defecto = os.environ.get("OASIS_DIR") or (
        Path(os.environ.get("DEAM_DIR", raiz / "data")) / "oasis")
    ap.add_argument("--destino", type=Path, default=Path(por_defecto))
    ap.add_argument("--conservar-zip", action="store_true")
    args = ap.parse_args()

    destino: Path = args.destino
    destino.mkdir(parents=True, exist_ok=True)

    if verificar(destino, silencioso=True):
        print(f"OASIS ya está completo en {destino}.")
        verificar(destino)
        return 0

    print(f"Descargando OASIS a {destino}")
    print(f"Tamaño aproximado: {humano(TAM_APROX)}")
    print("Fuente: https://osf.io/6pnd7/  ·  Kurdi, Lozano & Banaji (2017)\n")

    libre = shutil.disk_usage(destino).free
    if libre < TAM_APROX * 2.5:
        print(f"⚠ Espacio libre: {humano(libre)}. Se recomiendan "
              f"{humano(TAM_APROX * 2.5)}.", file=sys.stderr)

    zip_path = destino / "OASIS.zip"
    if not zip_path.exists() or zip_path.stat().st_size < TAM_APROX * 0.5:
        url = enlace_del_zip()
        if not url:
            return 1
        if not descargar(url, zip_path):
            return 1
    else:
        print(f"  ✓ OASIS.zip ya estaba ({humano(zip_path.stat().st_size)})")

    if not descomprimir(zip_path, destino):
        return 1
    if not args.conservar_zip:
        zip_path.unlink(missing_ok=True)

    print()
    if not verificar(destino):
        return 1
    print(f"\n✅ OASIS listo en {destino}")
    print("\nSiguiente paso:  make visor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
