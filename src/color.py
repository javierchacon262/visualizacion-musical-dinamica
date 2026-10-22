"""
Mapeo emoción → color.

El modelo circumplejo de Russell (1980) ubica cualquier emoción en un plano de
dos ejes:

    valencia   (x)  desagradable  ←→  agradable
    activación (y)  calmado       ←→  excitado

La idea central de este módulo: **ese plano es un círculo, y el matiz (hue) del
espacio HSV también lo es**, así que mapear ángulo → matiz preserva la
estructura. No es una elección arbitraria de colores bonitos: dos emociones
cercanas en el plano de Russell quedan cercanas en el círculo cromático.

    ángulo en el plano V-A        emoción típica      matiz resultante
    ───────────────────────────────────────────────────────────────────
    135°  (V bajo,  A alto)       tensión, ira        rojo
     45°  (V alto,  A alto)       alegría, euforia    amarillo
    −45°  (V alto,  A bajo)       calma, serenidad    cian / verde
    −135° (V bajo,  A bajo)       tristeza, apatía    azul

Los tres canales HSV llevan tres señales distintas:

    matiz       H  ←  dirección en el plano V-A   (qué emoción)
    saturación  S  ←  distancia al origen         (qué tan intensa)
    valor       V  ←  energía RMS del audio        (qué tan fuerte suena)

Así un pasaje emocionalmente neutro sale gris, y un silencio sale negro,
independientemente de lo que prediga el modelo.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

# Anclas del mapeo ángulo → matiz, medidas como distancia angular desde el
# rojo (135°) girando en sentido horario. Son interpolación lineal por tramos
# en vez de una rampa única, para que los cuatro cuadrantes caigan exactamente
# en el color que la intuición espera: una rampa lineal mandaba la alegría a
# verde-amarillo en vez de amarillo.
_ANG_ROJO = 135.0
_SPAN_ANG = 270.0
_ANCLAS_ANG = [0.0, 90.0, 180.0, 270.0]   # ira, alegría, calma, tristeza
_ANCLAS_HUE = [0.0, 60.0, 150.0, 240.0]   # rojo, amarillo, verde-cian, azul


def va_a_hsv(
    valencia: float,
    activacion: float,
    energia: float = 1.0,
    sat_min: float = 0.15,
    val_min: float = 0.08,
) -> tuple[float, float, float]:
    """
    Convierte un punto (valencia, activación) más una energía en HSV.

    Parámetros
    ----------
    valencia, activacion : en [-1, 1]. (0, 0) es el centro emocional neutro.
    energia : en [0, 1], típicamente el RMS normalizado del segmento.
    sat_min : saturación mínima, para que el centro no salga gris puro.
    val_min : brillo mínimo, para que los silencios no sean negro absoluto.

    Devuelve (h, s, v) con cada componente en [0, 1].
    """
    # Dirección → matiz
    ang = math.degrees(math.atan2(activacion, valencia))
    # Normalizar el ángulo al rango (-180, 180] y luego medir desde el rojo
    delta = (_ANG_ROJO - ang) % 360.0
    if delta > _SPAN_ANG:
        # La zona muerta entre azul y rojo (el "atrás" del círculo emocional):
        # se reparte hacia el extremo más cercano en vez de dar un salto.
        delta = 0.0 if delta > (_SPAN_ANG + 360.0) / 2 else _SPAN_ANG
    h = float(np.interp(delta, _ANCLAS_ANG, _ANCLAS_HUE)) / 360.0

    # Intensidad emocional → saturación
    radio = min(1.0, math.hypot(valencia, activacion))
    s = sat_min + (1.0 - sat_min) * radio

    # Energía del audio → brillo
    v = val_min + (1.0 - val_min) * float(np.clip(energia, 0.0, 1.0))

    return h, s, v


def va_a_rgb(valencia: float, activacion: float, energia: float = 1.0) -> tuple[int, int, int]:
    """Igual que `va_a_hsv` pero devuelve RGB en 0–255, listo para pintar."""
    h, s, v = va_a_hsv(valencia, activacion, energia)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return round(r * 255), round(g * 255), round(b * 255)


def va_a_rgb_lote(
    valencia: np.ndarray, activacion: np.ndarray, energia: np.ndarray | None = None
) -> np.ndarray:
    """
    Versión vectorizada para una canción entera.

    Recibe arreglos de longitud T y devuelve (T, 3) en uint8 — la línea de
    tiempo de color que luego se renderiza como video.
    """
    valencia = np.asarray(valencia, dtype=float)
    activacion = np.asarray(activacion, dtype=float)
    if energia is None:
        energia = np.ones_like(valencia)
    energia = np.asarray(energia, dtype=float)

    salida = np.empty((len(valencia), 3), dtype=np.uint8)
    for i, (v, a, e) in enumerate(zip(valencia, activacion, energia)):
        salida[i] = va_a_rgb(v, a, e)
    return salida


def normalizar_energia(rms: np.ndarray, percentil: float = 95.0) -> np.ndarray:
    """
    Lleva el RMS a [0, 1] usando un percentil alto como techo.

    Se usa percentil y no el máximo porque un solo golpe de batería puede ser
    varias veces más fuerte que el resto y aplastaría toda la canción a oscuro.
    """
    rms = np.asarray(rms, dtype=float)
    techo = np.percentile(rms, percentil)
    if techo <= 0:
        return np.zeros_like(rms)
    return np.clip(rms / techo, 0.0, 1.0)


if __name__ == "__main__":
    print("Verificación del mapeo — los cuatro cuadrantes de Russell\n")
    casos = [
        ("tensión / ira      (V−, A+)", -0.8, 0.8),
        ("alegría / euforia  (V+, A+)", 0.8, 0.8),
        ("calma / serenidad  (V+, A−)", 0.8, -0.8),
        ("tristeza / apatía  (V−, A−)", -0.8, -0.8),
        ("neutro             (0,  0 )", 0.0, 0.0),
    ]
    for nombre, v, a in casos:
        h, s, val = va_a_hsv(v, a)
        rgb = va_a_rgb(v, a)
        print(f"  {nombre}  →  H={h*360:6.1f}°  S={s:.2f}  RGB={rgb}")
