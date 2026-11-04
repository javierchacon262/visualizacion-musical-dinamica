"""
Convierte una canción completa en un video de color sincronizado con el audio.

    python src/visualizar.py cancion.mp3
    python src/visualizar.py cancion.mp3 --modelo mlp --salida demo.mp4
    python src/visualizar.py cancion.mp3 --tira      # solo la franja PNG

Acepta cualquier formato que lea librosa (mp3, wav, m4a, flac, ogg).

El modelo se entrenó sobre ventanas de 60 bloques (30 s), así que la canción se
procesa por ventanas de ese tamaño y luego se concatenan las predicciones.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color as col
import config as cfg
import features as ft

warnings.filterwarnings("ignore")
MODELOS = cfg.SALIDA_DIR / "modelos"


def features_cancion_completa(ruta: Path):
    """
    Extrae las características de TODA la canción, desde el segundo 0.

    Devuelve (features (T, 78), rms (T,), duracion_segundos).
    `features.procesar_cancion` arranca en el segundo 15 porque ahí empiezan
    las anotaciones de DEAM; para una demo queremos la canción entera.
    """
    import librosa

    y, sr = librosa.load(str(ruta), sr=cfg.SR, mono=True)
    dur = len(y) / sr
    fino = ft._features_finas(y, sr)
    n_bloques = fino.shape[1] // cfg.BLOQUE
    if n_bloques < 1:
        raise ValueError(f"El audio dura {dur:.1f}s; hacen falta al menos "
                         f"{cfg.BLOQUE / cfg.FPS_ANALISIS:.1f}s.")
    X = ft._agregar_en_bloques(fino, n_bloques, offset=0)

    # el RMS es la primera característica base; su media está en la columna 0
    rms = X[:, 0].copy()
    return X, rms, dur


def predecir(X: np.ndarray, nombre: str, calibrar: bool = True) -> np.ndarray:
    """Devuelve (T, 2) con valencia y activación por bloque."""
    from tensorflow import keras

    ruta = MODELOS / f"{nombre}.keras"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encuentro {ruta}. Entrena primero:  python src/entrenar.py")
    modelo = keras.models.load_model(ruta)
    norm = np.load(MODELOS / "normalizacion.npz")
    Xn = (X - norm["mu"]) / norm["sd"]

    L = cfg.N_BINS
    T = len(Xn)
    # Se procesa por ventanas de L bloques; la última se rellena repitiendo
    # el último bloque en vez de con ceros, que el modelo leería como silencio.
    n_vent = int(np.ceil(T / L))
    relleno = n_vent * L - T
    if relleno:
        Xn = np.vstack([Xn, np.repeat(Xn[-1:], relleno, axis=0)])
    lotes = Xn.reshape(n_vent, L, Xn.shape[1])
    pred = modelo.predict(lotes, verbose=0).reshape(-1, 2)
    pred = pred[:T]

    if calibrar:
        ruta_cal = MODELOS / f"calibracion_{nombre}.npz"
        if ruta_cal.exists():
            c = np.load(ruta_cal)
            # Expande la dispersión alrededor de la media del corpus hasta
            # igualar la de las etiquetas. No cambia la correlación: corrige
            # la escala que el MSE comprime hacia el centro.
            pred = np.clip(c["media"] + (pred - c["media"]) * c["factor"], -1, 1)
        else:
            print(f"  ! sin {ruta_cal.name}: predicción sin calibrar",
                  file=sys.stderr)
    return pred


def suavizar(v: np.ndarray, ventana: int = 5) -> np.ndarray:
    """Media móvil — sin esto el color parpadea y resulta desagradable."""
    if ventana < 2 or len(v) < ventana:
        return v
    k = np.ones(ventana) / ventana
    pad = ventana // 2
    ext = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(ext, k, mode="valid")[:len(v)]


def construir_colores(pred: np.ndarray, rms: np.ndarray, ventana: int):
    val = suavizar(pred[:, 0], ventana)
    aro = suavizar(pred[:, 1], ventana)
    energia = col.normalizar_energia(rms)
    return col.va_a_rgb_lote(val, aro, energia), val, aro


def guardar_tira(rgb: np.ndarray, destino: Path, alto: int = 180) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tira = np.repeat(rgb[np.newaxis, :, :], alto, axis=0)
    fig, ax = plt.subplots(figsize=(len(rgb) / 40, alto / 80), dpi=140)
    ax.imshow(tira, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _disco_circumplejo(lado: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pinta el plano valencia–activación con el mismo mapeo que usa el proyecto.

    Devuelve (imagen RGB (lado, lado, 3), máscara dentro del círculo). Sirve de
    leyenda viva: el punto que se mueve encima explica el color sin palabras.
    """
    ej = np.linspace(-1, 1, lado)
    vx, ay = np.meshgrid(ej, -ej)            # -ej: activación positiva arriba
    r = np.hypot(vx, ay)
    dentro = r <= 1.0
    rgb = np.zeros((lado, lado, 3), dtype=np.uint8)
    plano = col.va_a_rgb_lote(vx[dentro], ay[dentro], np.ones(dentro.sum()))
    rgb[dentro] = plano
    return rgb, dentro


def render_video(rgb, val, aro, audio: Path, destino: Path, fps: int = 20,
                 ancho: int = 1280, alto: int = 720) -> None:
    """
    Compone el video cuadro a cuadro en numpy y lo canaliza a ffmpeg.

    Se hace así y no con matplotlib porque una canción de 4 minutos son más de
    5000 cuadros: armar cada uno como figura cuesta minutos, armarlo como
    arreglo cuesta segundos.
    """
    from PIL import Image, ImageDraw, ImageFont

    T = len(rgb)
    dur = T / cfg.HZ_ANOT
    n_frames = max(1, int(dur * fps))

    # ── geometría ──
    M = 40                                   # margen
    disco_lado = 300
    bloque_w = ancho - disco_lado - 3 * M
    bloque_h = 430
    y_bloque = M
    y_tira, tira_h = 540, 70
    disco, mascara = _disco_circumplejo(disco_lado)
    x_disco = ancho - M - disco_lado

    # la franja completa, estirada al ancho disponible
    tira_w = ancho - 2 * M
    idx = np.linspace(0, T - 1, tira_w).astype(int)
    tira = np.repeat(rgb[idx][np.newaxis, :, :], tira_h, axis=0)

    # Las rutas de fuentes cambian entre distribuciones: se prueban varias y
    # se cae a la de Pillow si no hay ninguna.
    def _fuente(candidatos, tam):
        for ruta in candidatos:
            try:
                return ImageFont.truetype(ruta, tam)
            except OSError:
                continue
        return ImageFont.load_default()

    fuente = _fuente([
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ], 20)
    fuente_lbl = _fuente([
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ], 15)

    tmp = destino.with_suffix(".sinaudio.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ancho}x{alto}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(tmp)],
        stdin=subprocess.PIPE)

    FONDO = np.array([15, 21, 25], dtype=np.uint8)
    try:
        for f in range(n_frames):
            i = min(int(f / fps * cfg.HZ_ANOT), T - 1)
            lienzo = np.broadcast_to(FONDO, (alto, ancho, 3)).copy()

            # color actual, a bloque grande
            lienzo[y_bloque:y_bloque + bloque_h, M:M + bloque_w] = rgb[i]

            # plano de Russell
            reg = lienzo[y_bloque:y_bloque + disco_lado,
                         x_disco:x_disco + disco_lado]
            reg[mascara] = disco[mascara]

            # franja temporal completa + cursor
            lienzo[y_tira:y_tira + tira_h, M:M + tira_w] = tira
            xc = M + int(i / max(T - 1, 1) * (tira_w - 1))
            lienzo[y_tira - 6:y_tira + tira_h + 6,
                   max(M, xc - 1):min(M + tira_w, xc + 2)] = 255

            img = Image.fromarray(lienzo)
            d = ImageDraw.Draw(img)

            # punto del estado actual sobre el plano
            cx = x_disco + disco_lado / 2 * (1 + val[i])
            cy = y_bloque + disco_lado / 2 * (1 - aro[i])
            d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8],
                      fill=(255, 255, 255), outline=(15, 21, 25), width=2)

            # Ejes y nombres de los cuadrantes: el plano se explica solo.
            cxd = x_disco + disco_lado / 2
            cyd = y_bloque + disco_lado / 2
            for t, ex, ey in [("activación +", cxd, y_bloque - 16),
                              ("valencia +", cxd, y_bloque + disco_lado + 18)]:
                d.text((ex, ey), t, font=fuente_lbl, fill=(130, 143, 158),
                       anchor="mm")
            q = 0.60 * disco_lado / 2
            for t, dx, dy in [("tensión", -q, -q), ("alegría", q, -q),
                              ("calma", q, q), ("tristeza", -q, q)]:
                d.text((cxd + dx, cyd + dy), t, font=fuente_lbl,
                       fill=(0, 0, 0), anchor="mm")

            d.text((M, y_tira + tira_h + 22),
                   f"t {f/fps:6.1f}s / {dur:.0f}s     "
                   f"valencia {val[i]:+.2f}     activación {aro[i]:+.2f}     "
                   f"RGB {tuple(int(x) for x in rgb[i])}",
                   font=fuente, fill=(148, 161, 175))

            ff.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
    finally:
        ff.stdin.close()
        ff.wait()

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(destino)],
        check=True)
    tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--modelo", default="conv1d", choices=["conv1d", "mlp"])
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--suavizado", type=int, default=5,
                    help="ventana de la media móvil en bloques (0.5 s cada uno)")
    ap.add_argument("--tira", action="store_true", help="solo la franja PNG")
    ap.add_argument("--sin-calibrar", action="store_true",
                    help="predicción cruda, sin igualar varianza")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: no existe {args.audio}", file=sys.stderr)
        return 1

    print(f"Analizando {args.audio.name}…")
    X, rms, dur = features_cancion_completa(args.audio)
    print(f"  {dur:.1f}s  →  {len(X)} bloques de 0.5 s")

    pred = predecir(X, args.modelo, calibrar=not args.sin_calibrar)
    rgb, val, aro = construir_colores(pred, rms, args.suavizado)
    print(f"  modelo {args.modelo}  ·  valencia [{val.min():+.2f}, {val.max():+.2f}]"
          f"  ·  activación [{aro.min():+.2f}, {aro.max():+.2f}]")

    salidas = Path("outputs"); salidas.mkdir(exist_ok=True)
    base = args.salida or (salidas / args.audio.stem)
    png = base.with_suffix(".png")
    guardar_tira(rgb, png)
    print(f"  franja  → {png}")

    if not args.tira:
        mp4 = base.with_suffix(".mp4")
        render_video(rgb, val, aro, args.audio, mp4)
        print(f"  video   → {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
