"""
Extracción de características de audio alineadas con las anotaciones de DEAM.

Dos representaciones por canción, pensadas para la escalera de experimentos:

  1. Vector de características "a mano" por bloque de 0.5 s  →  MLP y Conv1D
  2. Mel-espectrograma a 50 Hz                               →  Conv2D y CRNN

El análisis espectral corre a 50 Hz y después se agrega en bloques de 25
fotogramas (0.5 s exactos) para casar con la rejilla de 2 Hz de las etiquetas.
De cada bloque se guardan media y desviación: la media describe el contenido
del medio segundo y la desviación cuánto varió dentro de él, que es
información rítmica que una sola media borraría.
"""

from __future__ import annotations

import numpy as np

import config as cfg

# Nombres de las características base, en el orden en que las apila
# `_features_finas`. El vector final tiene el doble: media y desviación.
NOMBRES_BASE = (
    ["rms", "centroide", "rolloff", "ancho_banda", "planitud", "cruces_cero", "fuerza_onset"]
    + [f"mfcc{i}" for i in range(1, 14)]
    + [f"chroma{i}" for i in range(1, 13)]
    + [f"contraste{i}" for i in range(1, 8)]
)
NOMBRES = [f"{n}_media" for n in NOMBRES_BASE] + [f"{n}_desv" for n in NOMBRES_BASE]
N_FEATURES = len(NOMBRES)


def _features_finas(y: np.ndarray, sr: int) -> np.ndarray:
    """Características a 50 Hz. Devuelve (D_base, T)."""
    import librosa

    kw = dict(hop_length=cfg.HOP, n_fft=cfg.N_FFT)
    S = np.abs(librosa.stft(y, **kw))

    filas = [
        librosa.feature.rms(S=S, hop_length=cfg.HOP),
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(y, frame_length=cfg.N_FFT, hop_length=cfg.HOP),
        librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr)[None, :],
        librosa.feature.mfcc(S=librosa.power_to_db(S**2), sr=sr, n_mfcc=13),
        librosa.feature.chroma_stft(S=S, sr=sr),
        librosa.feature.spectral_contrast(S=S, sr=sr),
    ]
    T = min(f.shape[-1] for f in filas)
    return np.vstack([f[..., :T] for f in filas])


def _agregar_en_bloques(fino: np.ndarray, n_bins: int, offset: int) -> np.ndarray:
    """
    Agrega (D, T) a 50 Hz en (n_bins, 2D) a 2 Hz, con media y desviación.

    `offset` es el fotograma fino donde empiezan las anotaciones.
    """
    salida = np.zeros((n_bins, fino.shape[0] * 2), dtype=np.float32)
    for i in range(n_bins):
        a = offset + i * cfg.BLOQUE
        b = a + cfg.BLOQUE
        if b > fino.shape[1]:
            # El fragmento se quedó corto: se repite el último bloque válido
            # en vez de rellenar con ceros, que el modelo leería como silencio.
            salida[i] = salida[i - 1] if i else 0.0
            continue
        bloque = fino[:, a:b]
        salida[i] = np.concatenate([bloque.mean(axis=1), bloque.std(axis=1)])
    return salida


def procesar_cancion(ruta, n_bins: int = cfg.N_BINS):
    """
    Extrae todo de una canción.

    Devuelve (features, mel) donde
        features : (n_bins, N_FEATURES)  float32, alineado con las etiquetas
        mel      : (N_MELS, n_bins*BLOQUE) float16 en dB, la región anotada
    """
    import librosa

    y, sr = librosa.load(str(ruta), sr=cfg.SR, mono=True)
    offset = int(round(cfg.INICIO_ANOT_S * cfg.FPS_ANALISIS))

    fino = _features_finas(y, sr)
    features = _agregar_en_bloques(fino, n_bins, offset)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg.N_MELS, hop_length=cfg.HOP, n_fft=cfg.N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fin = offset + n_bins * cfg.BLOQUE
    if mel_db.shape[1] < fin:
        faltan = fin - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, faltan)), mode="edge")
    mel_db = mel_db[:, offset:fin].astype(np.float16)

    return features, mel_db
