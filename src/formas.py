"""
Mapeo del plano valencia–activación a la forma que recorta la imagen.

Mismo criterio que `color.py`: se apoya en una regularidad perceptual ya medida
y vive fuera de la red, para poder cambiarse sin reentrenar nada.

La regularidad es la **angularidad**. Bar y Neta (2006) mostraron que un
contorno de vértices agudos se percibe como amenazante y uno curvo como seguro,
incluso en objetos sin significado — un eje que ya corre paralelo a la valencia.

LA FAMILIA DE FORMAS
--------------------
El contorno es una **superfórmula de Gielis (2003)**, en polares:

    r(θ) = [ |cos(mθ/4)|^n2 + |sin(mθ/4)|^n3 ] ^ (−1/n1)

Con cuatro números genera estrellas, flores, lentes, púas, gotas y círculos.
Sustituye al polígono regular de la primera versión, que daba siluetas casi
iguales entre sí.

Encima va una distorsión armónica, r ← r · (1 + espigas·cos(k·θ)), que es la
que produce las puntas largas.

POR QUÉ LOS CAMBIOS SON ABRUPTOS
--------------------------------
Dos decisiones deliberadas, porque con mapeos suaves no se notaba nada:

1. **Parámetros enteros.** `m` (lóbulos) y `armonico` (puntas) son enteros, así
   que no interpolan: saltan. Pasar de m=3 a m=4 cambia la silueta de golpe.

2. **Respuesta de ganancia alta.** `_ganancia` es una tanh con pendiente fuerte
   en el centro, de modo que un movimiento emocional pequeño alrededor del
   punto medio recorre buena parte del rango visual.

Y una tercera, a nivel de canción: `va_a_forma_lote` **estira el rango** de
valencia y activación de esa canción a [−1, 1] por percentiles. El modelo
predice comprimido —valencia rara vez sale de [−0.5, +0.6]— y sin estirar se
usaría un tercio del vocabulario. Es una decisión de presentación, no del
modelo, igual que la calibración de color: cámbiala con `estirar=False`.

El tamaño lo gobierna la **energía RMS del audio, no la red** — por la misma
razón que el brillo en `color.py`.

    python src/formas.py     # verificación de los cuatro cuadrantes
"""

from __future__ import annotations

import numpy as np

# Extremos de cada parámetro: la convención del proyecto en un solo sitio.
# M_MIN es 3 y no 2 a propósito: con dos lóbulos y n1 bajo la superfórmula
# degenera en un cuerpo redondo con una púa suelta, que no se lee como forma.
M_MIN, M_MAX = 3, 17              # lóbulos de la superfórmula (entero)
N1_MIN, N1_MAX = 0.16, 5.0        # bajo = púas; alto = redondo
N23_MIN, N23_MAX = 0.35, 2.4      # asimetría del contorno
ONDULACION_MAX = 0.44             # lóbulos visibles aunque n1 sea alto
ESPIGAS_MAX = 0.62                # amplitud de la distorsión armónica
ARM_MIN, ARM_MAX = 3, 15          # frecuencia de las puntas (entero)
GIRO_MIN, GIRO_MAX = 3.0, 95.0    # grados por segundo
ESCALA_MIN, ESCALA_MAX = 0.18, 1.0

_GANANCIA = 2.4                   # pendiente de la respuesta en el centro

# ── Composición: cuánto pesa cada capa sobre la de abajo ───────────────────
# Opacidad de la figura sobre la imagen de fondo. Con 1.0 la figura recorta la
# imagen de forma dura y fuera de ella solo queda el fondo atenuado; por debajo
# de 1.0 el fondo se transparenta a través de la figura, que es lo que permite
# enseñar las tres capas a la vez en lugar de solo el resultado.
ALFA_FIGURA = 0.92

# Cuánto tiñe el filtro de color a la imagen: 0 deja la foto cruda, 1 la lleva
# al color pleno del modelo. Conserva siempre la luminosidad de la foto.
ALFA_COLOR = 0.80

# Qué tan atenuada queda la imagen fuera de la figura.
ATENUACION_FONDO = 0.28


def _ganancia(x: float) -> float:
    """[−1,1] → [0,1] con pendiente fuerte en el centro."""
    return float(np.clip(0.5 + 0.5 * np.tanh(_GANANCIA * x), 0.0, 1.0))


def _log_interp(t: float, lo: float, hi: float) -> float:
    """Interpola en escala logarítmica: n1 se percibe así, no linealmente."""
    return float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * t))


def va_a_forma(valencia: float, activacion: float,
               energia: float = 1.0) -> dict:
    """
    Parámetros de la superfórmula para un bloque.

    valencia, activacion en [−1, 1]; energia en [0, 1].
    """
    v = float(np.clip(valencia, -1.0, 1.0))
    a = float(np.clip(activacion, -1.0, 1.0))
    e = float(np.clip(energia, 0.0, 1.0))

    tv, ta = _ganancia(v), _ganancia(a)

    # Valencia → qué tan punzante es el contorno. n1 bajo da púas, alto da un
    # cuerpo redondo: el eje de Bar y Neta, ahora sobre una familia mucho más
    # ancha que el polígono regular.
    n1 = _log_interp(tv, N1_MIN, N1_MAX)

    # Activación → cuántos lóbulos y cuántas puntas. Enteros, así que saltan.
    # La valencia negativa aporta un cuarto: una forma hostil se astilla en más
    # puntas y más finas, y es lo que separa la estrella de aguja (tensión) de
    # la flor de lóbulos anchos (alegría), que si no salían casi iguales.
    m = int(round(M_MIN + (M_MAX - M_MIN) * (0.75 * ta + 0.25 * (1.0 - tv))))
    armonico = int(round(ARM_MIN + (ARM_MAX - ARM_MIN) * ta))
    if armonico % 2 == 0:
        armonico += 1             # impar: evita puntas pegadas a los lóbulos

    # La asimetría cruza los dos ejes, para que cuadrantes distintos no den
    # siluetas simétricas entre sí.
    n2 = N23_MIN + (N23_MAX - N23_MIN) * ta
    n3 = N23_MIN + (N23_MAX - N23_MIN) * (1.0 - tv)

    # Pero la asimetría se apaga cuando n1 es pequeño. Ahí r = base^(−1/n1) es
    # brutalmente sensible: basta que `base` cruce 1 para que el radio pase de
    # desplomarse a dispararse, y con n2 ≠ n3 eso hace que lóbulos alternos
    # colapsen en pelos de un píxel. Por debajo de n1 ≈ 0.9 se igualan.
    k = float(np.clip((n1 - N1_MIN) / (0.9 - N1_MIN), 0.0, 1.0))
    medio = 0.5 * (n2 + n3)
    n2 = medio + (n2 - medio) * k
    n3 = medio + (n3 - medio) * k

    return {
        "m": m,
        "n1": round(n1, 4),
        "n2": round(n2, 4),
        "n3": round(n3, 4),
        # Las espigas las pide la activación, pero la valencia positiva las
        # apaga: si no, una canción alegre y movida salía tan punzante como una
        # tensa, y se perdía justo el eje de Bar y Neta que sostiene el mapeo.
        # La ondulación es lo que hace visible la activación en la mitad
        # positiva del plano: con n1 alto la superfórmula tiende al círculo y
        # los lóbulos se borran, de modo que alegría y calma salían iguales.
        # Los lóbulos se engordan con la valencia positiva, para que alegría
        # sea una flor carnosa y no una estrella más.
        "ondulacion": round(ONDULACION_MAX * ta * (0.45 + 0.55 * tv), 4),
        # Las espigas las pide la activación, pero la valencia positiva las
        # apaga: si no, una canción alegre y movida salía tan punzante como una
        # tensa, y se perdía justo el eje de Bar y Neta que sostiene el mapeo.
        "espigas": round(ESPIGAS_MAX * ta ** 1.2 * (1.0 - tv) ** 0.8, 4),
        "armonico": armonico,
        "giro": round(GIRO_MIN + (GIRO_MAX - GIRO_MIN) * ta, 2),
        "escala": round(ESCALA_MIN + (ESCALA_MAX - ESCALA_MIN) * e, 4),
    }


def _estirar(x: np.ndarray, pct: float = 6.0) -> np.ndarray:
    """
    Lleva el rango útil de esta canción a [−1, 1].

    Se usan percentiles y no el mínimo y el máximo porque un solo bloque
    extremo aplastaría todo lo demás contra el centro.
    """
    lo, hi = np.percentile(x, pct), np.percentile(x, 100 - pct)
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def va_a_forma_lote(valencia: np.ndarray, activacion: np.ndarray,
                    energia: np.ndarray | None = None,
                    estirar: bool = True) -> list[dict]:
    """Versión para una canción entera. Devuelve T diccionarios."""
    v = np.asarray(valencia, dtype=float)
    a = np.asarray(activacion, dtype=float)
    if energia is None:
        energia = np.ones_like(v)
    e = np.asarray(energia, dtype=float)

    if estirar and len(v) > 8:
        v, a = _estirar(v), _estirar(a)
    return [va_a_forma(vv, aa, ee) for vv, aa, ee in zip(v, a, e)]


def radio(theta: np.ndarray, f: dict) -> np.ndarray:
    """
    Evalúa el contorno. **El navegador replica esta función**; si cambia una,
    hay que cambiar la otra.

    Se manda el parámetro y no la silueta ya calculada porque son 540 bloques
    por canción: enviar 180 vértices de cada uno serían megabytes de JSON.
    """
    t = f["m"] * theta / 4.0
    base = (np.abs(np.cos(t)) ** f["n2"] + np.abs(np.sin(t)) ** f["n3"])
    r = np.power(np.maximum(base, 1e-9), -1.0 / f["n1"])
    r = r * (1.0 + f.get("ondulacion", 0.0) * np.cos(f["m"] * theta))
    r = r * (1.0 + f["espigas"] * np.cos(f["armonico"] * theta))
    pico = r.max()
    return r / pico if pico > 0 else r


def _verificar() -> int:
    casos = [
        ("tensión  (V− A+)", -0.8, 0.8), ("alegría  (V+ A+)", 0.8, 0.8),
        ("calma    (V+ A−)", 0.8, -0.8), ("tristeza (V− A−)", -0.8, -0.8),
    ]
    print("Mapeo valencia–activación → superfórmula\n")
    print(f"  {'cuadrante':<20}{'m':>4}{'n1':>8}{'ondul':>8}"
          f"{'espigas':>9}{'arm':>5}{'giro':>7}")
    fs = {}
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for nombre, v, a in casos:
        f = va_a_forma(v, a); fs[nombre] = f
        print(f"  {nombre:<20}{f['m']:>4}{f['n1']:>8.2f}{f['ondulacion']:>8.2f}"
              f"{f['espigas']:>9.2f}{f['armonico']:>5}{f['giro']:>7.1f}")

    print(f"\n  {'cuadrante':<20}{'r mín':>8}{'r medio':>9}"
          f"{'contraste':>11}{'cuerpo':>8}   silueta")
    fallos = []
    for nombre, _, _ in casos:
        r = radio(th, fs[nombre])
        contraste = float(r.max() / max(r.min(), 1e-6))
        desc = ("púas largas" if contraste > 4 else
                "lobulada" if contraste > 1.55 else "casi redonda")
        print(f"  {nombre:<20}{r.min():>8.3f}{r.mean():>9.3f}"
              f"{contraste:>11.1f}{100*(r > 0.45).mean():>8.0f}%   {desc}")
        if not np.isfinite(r).all():
            fallos.append(f"{nombre}: radio no finito")
        # Un contorno sano tiene cuerpo: si casi todas las muestras están
        # pegadas al centro, lo que queda son pelos, no una forma.
        cuerpo = float((r > 0.45).mean())
        if cuerpo < 0.10:
            fallos.append(f"{nombre}: silueta degenerada, solo el "
                          f"{100*cuerpo:.0f} % del contorno tiene cuerpo")

    # La valencia tiene que separar punzante de redondo
    c_neg = radio(th, fs["tensión  (V− A+)"])
    c_pos = radio(th, fs["alegría  (V+ A+)"])
    k = lambda r: r.max() / max(r.min(), 1e-6)
    if not k(c_neg) > k(c_pos):
        fallos.append(f"valencia negativa debería ser más punzante "
                      f"({k(c_neg):.1f} vs {k(c_pos):.1f})")

    # Y los cuatro cuadrantes tienen que dar siluetas distintas entre sí
    perfiles = {n: radio(th, f) for n, f in fs.items()}
    nombres = list(perfiles)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            d = float(np.abs(perfiles[nombres[i]] - perfiles[nombres[j]]).mean())
            if d < 0.04:
                fallos.append(f"{nombres[i]} y {nombres[j]} son casi idénticas "
                              f"(distancia {d:.3f})")

    print(f"\n  Rango medio, que es donde cae el dato real:\n")
    print(f"  {'(V, A)':<20}{'m':>4}{'n1':>8}{'ondul':>8}{'espigas':>9}"
          f"{'contraste':>11}")
    for v, a in [(-0.3, 0.3), (0.0, 0.0), (0.3, -0.3), (-0.15, -0.4), (0.4, 0.4)]:
        f = va_a_forma(v, a)
        r = radio(th, f)
        print(f"  {f'({v:+.2f}, {a:+.2f})':<20}{f['m']:>4}{f['n1']:>8.2f}"
              f"{f['ondulacion']:>8.2f}{f['espigas']:>9.2f}"
              f"{r.max()/max(r.min(),1e-6):>11.1f}")

    silencio = va_a_forma(0.9, 0.9, energia=0.0)
    if silencio["escala"] > ESCALA_MIN + 1e-9:
        fallos.append(f"con energía 0 la escala debería ser {ESCALA_MIN}")
    print(f"\n  energía 0 con predicción extrema → escala {silencio['escala']}")

    # El estirado por canción tiene que ensanchar el vocabulario
    rng = np.random.default_rng(0)
    comprimida = rng.uniform(-0.5, 0.6, 540)
    sin_e = {x["m"] for x in va_a_forma_lote(comprimida, comprimida,
                                             estirar=False)}
    con_e = {x["m"] for x in va_a_forma_lote(comprimida, comprimida)}
    print(f"  valores distintos de m sobre una predicción comprimida: "
          f"{len(sin_e)} sin estirar → {len(con_e)} estirando")
    if len(con_e) <= len(sin_e):
        fallos.append("el estirado no ensancha el vocabulario")

    print()
    for f in fallos:
        print(f"  ✗ {f}")
    if not fallos:
        print("  ✅ cuatro siluetas distintas, la valencia separa punzante de "
              "redondo")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_verificar())
