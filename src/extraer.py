"""
Corre la extracción sobre todo el corpus DEAM y guarda las matrices listas
para entrenar.

    python src/extraer.py              # todo el corpus
    python src/extraer.py --limite 50  # prueba rápida

Salidas en $DEAM_OUT (por defecto <DEAM_DIR>/procesado):

    caracteristicas.npz   X (N, 60, 78) · y_val, y_aro (N, 60) · ids · nombres
    mel/<id>.npy          mel-espectrograma float16 (128, 1500) por canción

El mel va en archivos sueltos a propósito: consolidado pesaría ~700 MB en RAM,
y los experimentos con Conv2D solo necesitan los lotes que están usando.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import features as ft

warnings.filterwarnings("ignore")


def cargar_etiquetas():
    """
    Carga valencia y activación como matrices numpy ya ordenadas por tiempo.

    El orden de columnas se resuelve UNA vez. Hacerlo por canción costaba
    4.4 millones de llamadas a regex y dejaba el script colgado antes de
    tocar una sola pista de audio.

    Devuelve (indice, matrices) donde indice mapea song_id → fila.
    """
    matrices = {}
    indice = None
    for dim in ("valence", "arousal"):
        d = pd.read_csv(cfg.ANOT_DIR / f"{dim}.csv")
        d.columns = [c.strip() for c in d.columns]
        cols = sorted(
            (c for c in d.columns if c.startswith("sample_")),
            key=lambda c: int(re.findall(r"\d+", c)[0]),
        )
        ids = d["song_id"].to_numpy(dtype=int)
        matrices[dim] = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        if indice is None:
            indice = {sid: i for i, sid in enumerate(ids)}
    return indice, matrices


def etiquetas_de(indice, matrices, song_id: int, n_bins: int):
    """
    Devuelve (valencia, activacion) de longitud n_bins, o None si la canción
    no tiene suficientes anotaciones válidas.
    """
    fila = indice.get(song_id)
    if fila is None:
        return None
    salida = []
    for dim in ("valence", "arousal"):
        v = matrices[dim][fila]
        v = v[~np.isnan(v)]
        if len(v) < n_bins:
            return None
        salida.append(v[:n_bins].astype(np.float32))
    return tuple(salida)


def _trabajo(args):
    ruta, song_id, n_bins = args
    try:
        feats, mel = ft.procesar_cancion(ruta, n_bins)
        return song_id, feats, mel, None
    except Exception as e:  # una canción corrupta no debe tumbar el lote
        return song_id, None, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="procesar solo N canciones")
    ap.add_argument("--procesos", type=int, default=None, help="núcleos (por defecto: todos menos 2)")
    args = ap.parse_args()

    import multiprocessing as mp
    from tqdm import tqdm

    if not cfg.AUDIO_DIR.exists():
        print(f"ERROR: no encuentro el audio en {cfg.AUDIO_DIR}", file=sys.stderr)
        print("Define DEAM_DIR si los datos están en otro sitio.", file=sys.stderr)
        return 1

    print("Cargando anotaciones…")
    indice, matrices = cargar_etiquetas()

    mp3 = sorted(
        cfg.AUDIO_DIR.glob("*.mp3"),
        key=lambda p: int(re.findall(r"\d+", p.stem)[0]),
    )
    tareas, y_val, y_aro, ids = [], [], [], []
    for p in mp3:
        sid = int(re.findall(r"\d+", p.stem)[0])
        et = etiquetas_de(indice, matrices, sid, cfg.N_BINS)
        if et is None:
            continue
        tareas.append((p, sid, cfg.N_BINS))
        y_val.append(et[0])
        y_aro.append(et[1])
        ids.append(sid)
        if args.limite and len(tareas) >= args.limite:
            break

    print(f"{len(mp3)} mp3 en disco · {len(tareas)} con ≥{cfg.N_BINS} anotaciones válidas")
    cfg.MEL_DIR.mkdir(parents=True, exist_ok=True)

    n_proc = args.procesos or max(1, mp.cpu_count() - 2)
    print(f"Extrayendo con {n_proc} procesos…")

    X = np.zeros((len(tareas), cfg.N_BINS, ft.N_FEATURES), dtype=np.float32)
    orden = {sid: i for i, sid in enumerate(ids)}
    fallos = []

    with mp.Pool(n_proc) as pool:
        for sid, feats, mel, err in tqdm(
            pool.imap_unordered(_trabajo, tareas, chunksize=4),
            total=len(tareas), unit="canción", ncols=78,
        ):
            if err:
                fallos.append((sid, err))
                continue
            X[orden[sid]] = feats
            np.save(cfg.MEL_DIR / f"{sid}.npy", mel)

    buenas = np.array([sid not in {f[0] for f in fallos} for sid in ids])
    X, ids_a = X[buenas], np.array(ids)[buenas]
    yv = np.array(y_val, dtype=np.float32)[buenas]
    ya = np.array(y_aro, dtype=np.float32)[buenas]

    cfg.SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    destino = cfg.SALIDA_DIR / "caracteristicas.npz"
    np.savez_compressed(
        destino, X=X, y_val=yv, y_aro=ya, ids=ids_a, nombres=np.array(ft.NOMBRES)
    )

    print(f"\nGuardado en {destino}")
    print(f"  X        {X.shape}  ({X.nbytes/1e6:.0f} MB en memoria)")
    print(f"  y_val    {yv.shape}   rango [{yv.min():.2f}, {yv.max():.2f}]")
    print(f"  y_aro    {ya.shape}   rango [{ya.min():.2f}, {ya.max():.2f}]")
    print(f"  mel      {len(list(cfg.MEL_DIR.glob('*.npy')))} archivos en {cfg.MEL_DIR}")
    if fallos:
        print(f"\n{len(fallos)} canciones fallaron:")
        for sid, err in fallos[:5]:
            print(f"  {sid}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
