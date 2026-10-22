#!/usr/bin/env python3
"""
Descarga el corpus DEAM y lo deja listo para la extracción.

Usa solo la biblioteca estándar: se puede ejecutar con el Python del sistema,
antes de crear el entorno virtual.

    python3 scripts/descargar_datos.py              # a ./data
    python3 scripts/descargar_datos.py --destino X  # a otro sitio
    python3 scripts/descargar_datos.py --solo-anotaciones   # sin el audio (4.6 MB)

Las descargas se reanudan si se interrumpen: vuelve a correr el comando.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://cvml.unige.ch/databases/DEAM/"
REINTENTOS = 3
ESPERA = 5           # segundos entre reintentos, se duplica cada vez
T_CONEXION = 30      # el servidor de la UNIGE a veces tarda en responder

# (archivo, tamaño aproximado en bytes, imprescindible para entrenar)
ARCHIVOS = [
    ("DEAM_Annotations.zip", 4_800_000, True),
    ("metadata.zip", 345_000, True),
    ("DEAM_audio.zip", 1_350_000_000, True),
]

ESPERADO_MP3 = 1802


def humano(n: float) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unidad}"
        n /= 1024
    return f"{n:,.1f} TB"


def descargar(nombre: str, destino: Path, tam_aprox: int) -> bool:
    """Descarga con reanudación y reintentos. True si quedó completo."""
    espera = ESPERA
    for intento in range(1, REINTENTOS + 1):
        if _intentar(nombre, destino, tam_aprox):
            return True
        if intento < REINTENTOS:
            print(f"  ↻ reintento {intento + 1}/{REINTENTOS} en {espera}s…")
            time.sleep(espera)
            espera *= 2
    print(f"""
  ✗ No se pudo descargar {nombre} tras {REINTENTOS} intentos.

    El servidor que aloja DEAM (cvml.unige.ch) se cae con cierta frecuencia.
    Opciones:
      · Espera un rato y vuelve a correr este comando — las descargas
        parciales se reanudan solas.
      · Bájalo a mano desde https://cvml.unige.ch/databases/DEAM/
        y deja los .zip en {destino}; el script los descomprimirá.
""", file=sys.stderr)
    return False


def _intentar(nombre: str, destino: Path, tam_aprox: int) -> bool:
    """Un intento de descarga."""
    ruta = destino / nombre
    url = BASE + nombre

    ya = ruta.stat().st_size if ruta.exists() else 0
    if ya > 0:
        # ¿Ya está completo? Preguntamos el tamaño real al servidor.
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=T_CONEXION
            ) as r:
                total = int(r.headers.get("Content-Length", 0))
            if total and ya >= total:
                print(f"  ✓ {nombre} ya está completo ({humano(ya)})")
                return True
        except Exception:
            total = 0
        print(f"  ↻ reanudando {nombre} desde {humano(ya)}")

    pedido = urllib.request.Request(url)
    if ya:
        pedido.add_header("Range", f"bytes={ya}-")

    try:
        with urllib.request.urlopen(pedido, timeout=T_CONEXION) as r:
            total = int(r.headers.get("Content-Length", 0)) + ya
            modo = "ab" if ya and r.status == 206 else "wb"
            if modo == "wb":
                ya = 0  # el servidor ignoró el Range: empezamos de nuevo
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
                        print(
                            f"\r  {nombre}  [{barra}] {pct:5.1f}%  {humano(hecho)}",
                            end="", flush=True,
                        )
            print()
    except urllib.error.URLError as e:
        print(f"\n  ✗ {nombre}: {e.reason if hasattr(e, 'reason') else e}",
              file=sys.stderr)
        return False
    except TimeoutError:
        print(f"\n  ✗ {nombre}: el servidor no respondió", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print(f"\n  ⏸ Interrumpido. Vuelve a correr el comando para reanudar.")
        raise

    final = ruta.stat().st_size
    if final < tam_aprox * 0.5:
        print(f"  ✗ {nombre} quedó demasiado pequeño ({humano(final)}).", file=sys.stderr)
        return False
    return True


def descomprimir(nombre: str, destino: Path) -> bool:
    """Descomprime validando que ninguna entrada escape del destino."""
    ruta = destino / nombre
    raiz = destino.resolve()
    try:
        with zipfile.ZipFile(ruta) as z:
            for info in z.infolist():
                final = (raiz / info.filename).resolve()
                if not str(final).startswith(str(raiz)):
                    print(f"  ✗ {nombre} intenta escribir fuera de {destino} "
                          f"({info.filename}). Abortado.", file=sys.stderr)
                    return False
            z.extractall(destino)
    except zipfile.BadZipFile:
        print(f"  ✗ {nombre} está corrupto. Bórralo y vuelve a correr el script.",
              file=sys.stderr)
        return False
    print(f"  ✓ {nombre} descomprimido")
    return True


def verificar(destino: Path, con_audio: bool, silencioso: bool = False) -> bool:
    """Comprueba que la estructura quedó como la espera el pipeline."""
    di = (lambda *a, **k: None) if silencioso else print
    di("\nVerificando…")
    ok = True

    anot = destino / "annotations" / "annotations averaged per song" / "dynamic (per second annotations)"
    for f in ("arousal.csv", "valence.csv"):
        if (anot / f).exists():
            di(f"  ✓ {f}")
        else:
            di(f"  ✗ falta {anot / f}")
            ok = False

    if con_audio:
        audio = destino / "MEMD_audio"
        n = len(list(audio.glob("*.mp3"))) if audio.exists() else 0
        if n == ESPERADO_MP3:
            di(f"  ✓ {n} archivos mp3")
        else:
            di(f"  ✗ se esperaban {ESPERADO_MP3} mp3 y hay {n}")
            ok = False

    if ok:
        libre = shutil.disk_usage(destino).free
        di(f"\n✅ Datos listos en {destino}")
        di(f"   Espacio libre: {humano(libre)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parent.parent
    ap.add_argument("--destino", type=Path, default=raiz / "data")
    ap.add_argument("--solo-anotaciones", action="store_true",
                    help="omite el audio (1.3 GB); útil para inspeccionar etiquetas")
    ap.add_argument("--conservar-zips", action="store_true",
                    help="no borra los .zip tras descomprimir")
    args = ap.parse_args()

    destino: Path = args.destino
    destino.mkdir(parents=True, exist_ok=True)

    # ¿Ya está todo? Entonces no hay nada que bajar.
    if verificar(destino, con_audio=not args.solo_anotaciones, silencioso=True):
        print(f"Los datos ya están completos en {destino}.")
        verificar(destino, con_audio=not args.solo_anotaciones)
        print("\nSiguiente paso:  make extraer")
        return 0

    archivos = [a for a in ARCHIVOS
                if not (args.solo_anotaciones and a[0] == "DEAM_audio.zip")]
    total = sum(a[1] for a in archivos)

    print(f"Descargando DEAM a {destino}")
    print(f"Tamaño aproximado: {humano(total)}")
    print("Fuente: https://cvml.unige.ch/databases/DEAM/  ·  audio con licencia Creative Commons\n")

    libre = shutil.disk_usage(destino).free
    if libre < total * 2.2:
        print(f"⚠ Espacio libre: {humano(libre)}. Se recomiendan al menos "
              f"{humano(total * 2.2)} (el zip y su contenido conviven un momento).",
              file=sys.stderr)

    for nombre, tam, _ in archivos:
        if not descargar(nombre, destino, tam):
            return 1
        if not descomprimir(nombre, destino):
            return 1
        if not args.conservar_zips:
            (destino / nombre).unlink(missing_ok=True)

    if not verificar(destino, con_audio=not args.solo_anotaciones):
        return 1

    print("\nSiguiente paso:  make extraer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
