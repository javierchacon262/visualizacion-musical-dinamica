"""
Entrena y evalúa los modelos de regresión de valencia y activación.

    python src/entrenar.py              # MLP y Conv1D
    python src/entrenar.py --modelo mlp

La partición es **por canción**, no por bloque: si dos bloques de la misma
canción cayeran en train y test, el modelo podría memorizar la canción y las
métricas saldrían infladas.

La evaluación descompone la correlación en dos componentes, que es lo que de
verdad importa para este proyecto:

  · ENTRE canciones  — ¿distingue una canción intensa de una suave?  (fácil)
  · DENTRO de canción — ¿sigue el vaivén momento a momento?          (difícil)

Un modelo que solo acierte la primera produciría una visualización de color
casi constante, que es exactamente lo que no queremos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SEMILLA = 42
DESTINO = cfg.SALIDA_DIR / "modelos"


# ── métricas ───────────────────────────────────────────────────────────────
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluar(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_real, y_pred: (n_canciones, n_bloques)

    Devuelve RMSE y las tres correlaciones: agrupada, entre y dentro.
    """
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    agrupada = _pearson(y_real, y_pred)
    entre = _pearson(y_real.mean(axis=1), y_pred.mean(axis=1))

    dr = y_real - y_real.mean(axis=1, keepdims=True)
    dp = y_pred - y_pred.mean(axis=1, keepdims=True)
    rs = [_pearson(a, b) for a, b in zip(dr, dp)]
    rs = [r for r in rs if not np.isnan(r)]
    dentro = float(np.mean(rs)) if rs else float("nan")

    return {"rmse": rmse, "r_agrupada": agrupada, "r_entre": entre, "r_dentro": dentro}


def calibrar(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Igualación de varianza: cuánto hay que expandir las predicciones para que
    su dispersión coincida con la de las etiquetas.

    Una red entrenada con MSE regresa a la media — predice menos extremo de lo
    que es la realidad, porque equivocarse en el centro cuesta menos. Para la
    visualización eso significa colores apagados. El factor corrige la escala
    sin tocar la correlación, que es lo que mide el acierto.

    Se calcula sobre el conjunto de test, no sobre train.
    """
    out = {}
    for j, dim in enumerate(("valencia", "activacion")):
        P, R = y_pred[..., j], y_real[..., j]
        dP = (P - P.mean(axis=1, keepdims=True)).std()
        dR = (R - R.mean(axis=1, keepdims=True)).std()
        out[dim] = {
            "media_pred": float(P.mean()),
            "factor_total": float(R.std() / P.std()),
            "factor_dentro": float(dR / dP),
            "sd_pred": float(P.std()),
            "sd_real": float(R.std()),
            "sd_dentro_pred": float(dP),
            "sd_dentro_real": float(dR),
        }
    return out


# ── datos ──────────────────────────────────────────────────────────────────
def cargar():
    d = np.load(cfg.SALIDA_DIR / "caracteristicas.npz", allow_pickle=True)
    X, yv, ya = d["X"], d["y_val"], d["y_aro"]
    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(len(X))
    n = len(X)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    part = {
        "train": orden[:i_tr],
        "val": orden[i_tr:i_va],
        "test": orden[i_va:],
    }
    # normalización con estadísticas de train únicamente
    plano = X[part["train"]].reshape(-1, X.shape[2])
    mu, sd = plano.mean(axis=0), plano.std(axis=0) + 1e-8
    Xn = (X - mu) / sd
    Y = np.stack([yv, ya], axis=-1)          # (n, 60, 2)
    return Xn, Y, part, (mu, sd)


# ── modelos ────────────────────────────────────────────────────────────────
def construir(nombre: str, n_bloques: int, n_feat: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    if nombre == "mlp":
        # Trata cada bloque de forma independiente: no ve contexto temporal.
        return keras.Sequential([
            layers.Input(shape=(n_bloques, n_feat)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(2),
        ], name="mlp")

    if nombre == "conv1d":
        # Ve una ventana temporal: patrones locales que se repiten en el tiempo.
        return keras.Sequential([
            layers.Input(shape=(n_bloques, n_feat)),
            layers.Conv1D(128, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(128, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv1D(64, 3, padding="same", activation="relu"),
            layers.Conv1D(2, 1, padding="same"),
        ], name="conv1d")

    raise ValueError(f"modelo desconocido: {nombre}")


def entrenar_uno(nombre, X, Y, part, epochs, verbose=1):
    from tensorflow import keras

    m = construir(nombre, X.shape[1], X.shape[2])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    hist = m.fit(
        X[part["train"]], Y[part["train"]],
        validation_data=(X[part["val"]], Y[part["val"]]),
        epochs=epochs, batch_size=32, verbose=verbose,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=12, restore_best_weights=True, monitor="val_loss")],
    )
    pred = m.predict(X[part["test"]], verbose=0)
    real = Y[part["test"]]
    res = {
        "valencia": evaluar(real[..., 0], pred[..., 0]),
        "activacion": evaluar(real[..., 1], pred[..., 1]),
        "params": int(m.count_params()),
        "epocas_corridas": len(hist.history["loss"]),
        "calibracion": calibrar(real, pred),
    }
    return m, res


# ── informe ────────────────────────────────────────────────────────────────
# Techo de la componente DENTRO de canción: alfa de Cronbach por canción, la
# medida que usan los autores de DEAM (Tabla 2 del artículo: 0.28–0.31 en
# activación y 0.20–0.28 en valencia para 2013–2014). Reproducido sobre el
# corpus completo da 0.32 y 0.25.
#
# NO usar aquí el 0.60 / 0.43 que sale de aplicar Spearman-Brown a la
# correlación AGRUPADA entre dos anotadores: ese número incluye el acuerdo en
# distinguir unas canciones de otras, y compararlo con r_dentro infla el
# resultado al doble.
TECHO = {"valencia": 0.25, "activacion": 0.32}


def imprimir(nombre: str, r: dict) -> None:
    print(f"\n── {nombre.upper()}  ·  {r['params']:,} parámetros  ·  "
          f"{r['epocas_corridas']} épocas ──")
    print(f"  {'dimensión':<12}{'RMSE':>8}{'r agrup.':>10}{'r entre':>9}"
          f"{'r dentro':>10}{'techo':>8}{'% techo':>9}")
    for dim in ("activacion", "valencia"):
        m = r[dim]; techo = TECHO[dim]
        print(f"  {dim:<12}{m['rmse']:>8.4f}{m['r_agrupada']:>10.3f}"
              f"{m['r_entre']:>9.3f}{m['r_dentro']:>10.3f}{techo:>8.2f}"
              f"{100*m['r_dentro']/techo:>8.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="todos", choices=["mlp", "conv1d", "todos"])
    ap.add_argument("--epocas", type=int, default=120)
    ap.add_argument("--silencioso", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf
    tf.random.set_seed(SEMILLA)
    np.random.seed(SEMILLA)

    ruta = cfg.SALIDA_DIR / "caracteristicas.npz"
    if not ruta.exists():
        print(f"ERROR: no encuentro {ruta}. Corre antes:  make extraer", file=sys.stderr)
        return 1

    X, Y, part, (mu, sd) = cargar()
    print(f"Corpus: {len(X)} canciones × {X.shape[1]} bloques × {X.shape[2]} características")
    print(f"Partición POR CANCIÓN — train {len(part['train'])} · "
          f"val {len(part['val'])} · test {len(part['test'])}")

    DESTINO.mkdir(parents=True, exist_ok=True)
    np.savez(DESTINO / "normalizacion.npz", mu=mu, sd=sd)

    nombres = ["mlp", "conv1d"] if args.modelo == "todos" else [args.modelo]
    todos = {}
    for n in nombres:
        m, r = entrenar_uno(n, X, Y, part, args.epocas,
                            verbose=0 if args.silencioso else 1)
        imprimir(n, r)
        m.save(DESTINO / f"{n}.keras")
        c = r["calibracion"]
        np.savez(DESTINO / f"calibracion_{n}.npz",
                 media=np.array([c["valencia"]["media_pred"],
                                 c["activacion"]["media_pred"]]),
                 factor=np.array([c["valencia"]["factor_total"],
                                  c["activacion"]["factor_total"]]))
        todos[n] = r

    (DESTINO / "resultados.json").write_text(json.dumps(todos, indent=2))
    print(f"\nModelos y resultados en {DESTINO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
