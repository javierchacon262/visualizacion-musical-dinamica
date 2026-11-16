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
import formas as fm
import imagenes as im

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


def _fuente(candidatos, tam):
    """Las rutas de fuentes cambian entre distribuciones: se prueban varias."""
    from PIL import ImageFont
    for ruta in candidatos:
        try:
            return ImageFont.truetype(ruta, tam)
        except OSError:
            continue
    return ImageFont.load_default()


MONO = ["/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
SANS = ["/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def _cubrir(src, w: int, h: int):
    """Redimensiona la imagen para cubrir w×h sin deformarla."""
    from PIL import Image
    e = max(w / src.width, h / src.height)
    src = src.resize((max(1, int(src.width * e)), max(1, int(src.height * e))),
                     Image.LANCZOS)
    x, y = (src.width - w) // 2, (src.height - h) // 2
    return src.crop((x, y, x + w, y + h))


def _tenir(lum, c, alfa: float):
    """
    Filtro de color: conserva la luminosidad de la foto y le impone el matiz y
    la saturación del color que predice el modelo.

    Es el equivalente al modo de fusión 'color' del lienzo que usa el visor.
    `lum` llega ya en escala de grises para no reconvertirla en cada cuadro.
    """
    from PIL import Image
    tinte = Image.new("RGB", lum.size, c).convert("HSV")
    h, sat, _ = tinte.split()
    tenida = Image.merge("HSV", (h, sat, lum)).convert("RGB")
    if alfa >= 1.0:
        return tenida
    return Image.blend(lum.convert("RGB"), tenida, alfa)


def _mascara_forma(f: dict, w: int, h: int, rot: float, theta):
    """Máscara en blanco y negro con la silueta de la superfórmula."""
    from PIL import Image, ImageDraw
    r = fm.radio(theta, f) * (min(w, h) * 0.46) * f["escala"]
    cx, cy = w / 2, h / 2
    pts = [(cx + np.cos(t + rot) * rr, cy + np.sin(t + rot) * rr)
           for t, rr in zip(theta, r)]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def render_video(rgb, val, aro, audio: Path, destino: Path, fps: int = 20,
                 ancho: int = 1600, alto: int = 900,
                 alfa_figura: float = fm.ALFA_FIGURA,
                 con_imagen: bool = True) -> None:
    """
    Compone el video cuadro a cuadro con PIL y lo canaliza a ffmpeg.

    Se hace así y no con matplotlib porque una canción de cuatro minutos son
    más de 5 000 cuadros: armar cada uno como figura cuesta minutos.

    El cuadro enseña las tres capas a la vez — la composición grande, y debajo
    las miniaturas de cada paso — para que se vea cómo se aplican y no solo el
    resultado.
    """
    from PIL import Image, ImageDraw

    T = len(rgb)
    dur = T / cfg.HZ_ANOT
    n_frames = max(1, int(dur * fps))
    theta = np.linspace(0, 2 * np.pi, 420, endpoint=False)

    # ── parámetros por bloque ──
    formas = fm.va_a_forma_lote(val, aro, _escala_desde_rgb(rgb))
    giros = np.array([f["giro"] for f in formas], dtype=float)
    giro_acum = np.concatenate([[0.0], np.cumsum(giros[:-1]) / cfg.HZ_ANOT])

    tramos = []
    if con_imagen:
        try:
            tramos = im.secuencia(val, aro)
        except Exception as e:
            print(f"  ! sin capa de imagen ({e})", file=sys.stderr)

    # ── geometría ──
    M = 40
    pw, ph = 1000, 516                       # composición principal
    dl = 430                                 # disco
    x_disco, y_disco = ancho - M - dl, M
    tw, th_ = 240, 132                       # miniaturas de cada capa
    y_min = M + ph + 46                      # fila de miniaturas
    y_tira, tira_h = alto - 130, 54
    tira_w = ancho - 2 * M

    if y_min + th_ > y_tira - 8:             # el cuadro no es de goma
        raise ValueError(f"las miniaturas invaden la franja "
                         f"({y_min + th_} > {y_tira - 8})")

    disco, mask_disco = _disco_circumplejo(dl)
    idx = np.linspace(0, T - 1, tira_w).astype(int)
    tira = np.repeat(rgb[idx][np.newaxis, :, :], tira_h, axis=0)

    fuente = _fuente(MONO, 20)
    f_rot = _fuente(SANS, 17)
    f_lbl = _fuente(SANS, 15)

    FONDO = (15, 21, 25)
    cache_img: dict[str, tuple] = {}         # archivo -> (cubierta, luminosidad)

    def capas(i: int, w: int, h: int):
        """Devuelve (fondo, figura) ya compuestas para el bloque i."""
        c = tuple(int(x) for x in rgb[i])
        tr = next((t for t in tramos if t["desde"] <= i < t["hasta"]),
                  tramos[-1] if tramos else None)
        if tr is None:
            base = Image.new("RGB", (w, h), c)
            return Image.new("RGB", (w, h), tuple(int(x * fm.ATENUACION_FONDO)
                                                  for x in c)), base, None
        clave = f"{tr['archivo']}|{w}x{h}"
        if clave not in cache_img:
            src = Image.open(cfg.OASIS_IMG / tr["archivo"]).convert("RGB")
            cub = _cubrir(src, w, h)
            cache_img[clave] = (cub, cub.convert("L"))
        cub, lum = cache_img[clave]
        figura = _tenir(lum, c, fm.ALFA_COLOR)
        fondo = Image.eval(figura, lambda v: int(v * fm.ATENUACION_FONDO))
        return fondo, figura, cub

    escritor = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ancho}x{alto}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(destino.with_suffix(".sinaudio.mp4"))],
        stdin=subprocess.PIPE)

    try:
        for n in range(n_frames):
            t_s = n / fps
            i = min(int(t_s * cfg.HZ_ANOT), T - 1)
            c = tuple(int(x) for x in rgb[i])
            f = formas[i]

            # rotación interpolada: función determinista del instante
            tb = t_s * cfg.HZ_ANOT
            i0 = min(int(tb), T - 1); i1 = min(i0 + 1, T - 1)
            gr = giro_acum[i0] + (giro_acum[i1] - giro_acum[i0]) * (tb - i0)
            rot = np.deg2rad(gr)

            lienzo = Image.new("RGB", (ancho, alto), FONDO)

            # ── composición principal: fondo + figura recortada ──
            fondo, figura, cruda = capas(i, pw, ph)
            mask = _mascara_forma(f, pw, ph, rot, theta)
            if alfa_figura < 1.0:
                mask = Image.eval(mask, lambda v: int(v * alfa_figura))
            principal = fondo.copy()
            principal.paste(figura, (0, 0), mask)
            lienzo.paste(principal, (M, M))

            # ── miniaturas: las tres capas por separado ──
            if cruda is not None:
                fondo_m, figura_m, cruda_m = capas(i, tw, th_)
                mask_m = _mascara_forma(f, tw, th_, rot, theta)
                comp_m = fondo_m.copy(); comp_m.paste(figura_m, (0, 0), mask_m)
                minis = [(cruda_m, "1 · imagen"), (figura_m, "2 · + color"),
                         (comp_m, "3 · + forma")]
            else:
                minis = []
            for k, (mini, _) in enumerate(minis):
                lienzo.paste(mini, (M + k * (tw + 26), y_min))

            # ── disco de Russell ──
            reg = np.array(lienzo)
            reg[y_disco:y_disco + dl, x_disco:x_disco + dl][mask_disco] = \
                disco[mask_disco]
            lienzo = Image.fromarray(reg)

            # ── franja temporal y cursor ──
            lienzo.paste(Image.fromarray(tira), (M, y_tira))
            d = ImageDraw.Draw(lienzo)
            xc = M + int(i / max(T - 1, 1) * (tira_w - 1))
            d.rectangle([xc - 1, y_tira - 6, xc + 1, y_tira + tira_h + 6],
                        fill=(255, 255, 255))

            # ── punto sobre el disco y rótulos ──
            cx = x_disco + dl / 2 * (1 + val[i])
            cy = y_disco + dl / 2 * (1 - aro[i])
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(255, 255, 255),
                      outline=FONDO, width=2)
            q = 0.62 * dl / 2
            for txt, dx, dy in [("tensión", -q, -q), ("alegría", q, -q),
                                ("calma", q, q), ("tristeza", -q, q)]:
                d.text((x_disco + dl / 2 + dx, y_disco + dl / 2 + dy), txt,
                       font=f_lbl, fill=(0, 0, 0), anchor="mm")
            for k, (_, nom) in enumerate(minis):
                d.text((M + k * (tw + 26), y_min - 24), nom, font=f_rot,
                       fill=(148, 161, 175))

            d.text((M, y_tira + tira_h + 20),
                   f"t {t_s:6.1f}s / {dur:.0f}s    "
                   f"V {val[i]:+.2f}  A {aro[i]:+.2f}    "
                   f"m{f['m']} n1 {f['n1']:.2f}    RGB {c}",
                   font=fuente, fill=(148, 161, 175))

            escritor.stdin.write(np.asarray(lienzo, dtype=np.uint8).tobytes())
    finally:
        escritor.stdin.close()
        escritor.wait()

    tmp = destino.with_suffix(".sinaudio.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(destino)],
        check=True)
    tmp.unlink(missing_ok=True)


def _escala_desde_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Recupera la energía normalizada a partir del brillo del color.

    `construir_colores` ya metió el RMS en el canal de valor; sacarlo de ahí
    evita volver a pasar el audio por librosa solo para la escala.
    """
    v = rgb.max(axis=1).astype(float) / 255.0
    return np.clip(v, 0.0, 1.0)


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
    ap.add_argument("--alfa-figura", type=float, default=fm.ALFA_FIGURA,
                    help=f"opacidad de la figura sobre la imagen "
                         f"(0–1, por defecto {fm.ALFA_FIGURA})")
    ap.add_argument("--sin-imagen", action="store_true",
                    help="solo color y forma, sin la capa de OASIS")
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
        render_video(rgb, val, aro, args.audio, mp4,
                     alfa_figura=args.alfa_figura,
                     con_imagen=not args.sin_imagen)
        print(f"  video   → {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
