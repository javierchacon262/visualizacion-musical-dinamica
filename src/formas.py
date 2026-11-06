"""
Mapeo del plano valencia–activación a la forma del polígono que se dibuja.

Mismo criterio que `color.py`: no es una elección de gusto, se apoya en una
regularidad perceptual ya medida, y vive aparte de la red para poder cambiarse
sin reentrenar nada.

La regularidad es la **angularidad**. Bar y Neta (2006) mostraron que un
contorno con vértices agudos se percibe como amenazante y uno curvo como
seguro, incluso en objetos sin significado. Eso da un eje que ya corre paralelo
a la valencia, así que se usa ése en vez de inventar una convención:

    valencia −1  →  triángulo agudo        valencia +1  →  casi un círculo

La activación, que es movimiento, va a lo que se mueve: la velocidad de giro y
la rugosidad del contorno.

    activación −1  →  contorno liso, giro lento
    activación +1  →  contorno dentado, giro rápido

El tamaño lo gobierna la **energía RMS del audio, no la red** — por la misma
razón que el brillo en `color.py`. Un silencio tiene que verse pequeño y quieto
aunque el modelo prediga lo que sea.

    python src/formas.py     # imprime la verificación de los cuatro cuadrantes
"""

from __future__ import annotations

import numpy as np

# Extremos de cada parámetro. Están aquí arriba y no repartidos por el código
# porque son la convención del proyecto: si alguien quiere otra estética, esto
# es lo único que toca.
LADOS_MIN, LADOS_MAX = 3, 24          # 24 ya se lee como círculo
RUGOSIDAD_MIN, RUGOSIDAD_MAX = 0.05, 0.45
GIRO_MIN, GIRO_MAX = 2.0, 60.0        # grados por segundo
ESCALA_MIN, ESCALA_MAX = 0.18, 1.0    # fracción del radio disponible

_EXP_LADOS = 2.0                      # curva la redondez hacia el lado agudo


def _interp(t: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * t


def va_a_forma(valencia: float, activacion: float,
               energia: float = 1.0) -> dict:
    """
    Devuelve los parámetros de la forma para un bloque.

    valencia, activacion en [−1, 1]; energia en [0, 1].

    Las claves son las que consume el dibujante (el navegador, en el visor):
    `lados`, `rugosidad`, `giro`, `escala`.
    """
    v = float(np.clip(valencia, -1.0, 1.0))
    a = float(np.clip(activacion, -1.0, 1.0))
    e = float(np.clip(energia, 0.0, 1.0))

    # La redondez crece con la valencia, pero elevada a una potencia > 1: así
    # la zona agudа ocupa más recorrido y el paso de 3 a 8 lados —que es el que
    # de verdad se nota— no se agota en el primer tramo.
    redondez = ((v + 1.0) / 2.0) ** _EXP_LADOS
    lados = int(round(_interp(redondez, LADOS_MIN, LADOS_MAX)))

    t_act = (a + 1.0) / 2.0
    return {
        "lados": lados,
        "rugosidad": round(_interp(t_act, RUGOSIDAD_MIN, RUGOSIDAD_MAX), 4),
        "giro": round(_interp(t_act, GIRO_MIN, GIRO_MAX), 2),
        "escala": round(_interp(e, ESCALA_MIN, ESCALA_MAX), 4),
    }


def va_a_forma_lote(valencia: np.ndarray, activacion: np.ndarray,
                    energia: np.ndarray | None = None) -> list[dict]:
    """Versión para una canción entera. Devuelve una lista de T diccionarios."""
    valencia = np.asarray(valencia, dtype=float)
    activacion = np.asarray(activacion, dtype=float)
    if energia is None:
        energia = np.ones_like(valencia)
    energia = np.asarray(energia, dtype=float)
    return [va_a_forma(v, a, e)
            for v, a, e in zip(valencia, activacion, energia)]


def _verificar() -> int:
    """Comprueba que los cuatro cuadrantes dan lo que la convención promete."""
    casos = [
        ("tensión  (V− A+)", -0.8, 0.8, "pocos lados, muy dentada, giro rápido"),
        ("alegría  (V+ A+)", 0.8, 0.8, "muchos lados, dentada, giro rápido"),
        ("calma    (V+ A−)", 0.8, -0.8, "muchos lados, lisa, giro lento"),
        ("tristeza (V− A−)", -0.8, -0.8, "pocos lados, lisa, giro lento"),
    ]
    print("Mapeo valencia–activación → forma\n")
    print(f"  {'cuadrante':<20}{'lados':>7}{'rugosidad':>11}{'giro':>8}   esperado")
    formas = {}
    for nombre, v, a, esperado in casos:
        f = va_a_forma(v, a)
        formas[nombre] = f
        print(f"  {nombre:<20}{f['lados']:>7}{f['rugosidad']:>11.2f}"
              f"{f['giro']:>8.1f}   {esperado}")

    fallos = []
    agudo = formas["tensión  (V− A+)"]["lados"]
    redondo = formas["alegría  (V+ A+)"]["lados"]
    if not agudo < redondo:
        fallos.append(f"valencia negativa debería dar menos lados ({agudo} vs {redondo})")
    activa = formas["alegría  (V+ A+)"]["rugosidad"]
    quieta = formas["calma    (V+ A−)"]["rugosidad"]
    if not quieta < activa:
        fallos.append(f"activación baja debería ser más lisa ({quieta} vs {activa})")
    if not formas["calma    (V+ A−)"]["giro"] < formas["alegría  (V+ A+)"]["giro"]:
        fallos.append("activación baja debería girar más lento")

    # el silencio tiene que verse pequeño pase lo que pase con la predicción
    silencio = va_a_forma(0.9, 0.9, energia=0.0)
    if silencio["escala"] > ESCALA_MIN + 1e-9:
        fallos.append(f"con energía 0 la escala debería ser {ESCALA_MIN}")
    print(f"\n  energía 0 con predicción extrema → escala {silencio['escala']}"
          f"  (mínima: la forma se encoge en los silencios)")

    print()
    for f in fallos:
        print(f"  ✗ {f}")
    if not fallos:
        print("  ✅ los cuatro cuadrantes se comportan como promete la convención")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_verificar())
