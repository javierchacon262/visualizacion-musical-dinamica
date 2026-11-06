"""
Visor web: reproduce una canción y pinta el color predicho, sincronizado.

    python src/servidor.py              # http://127.0.0.1:8765
    python src/servidor.py --puerto 9000 --carpeta demo

A diferencia del video de `visualizar.py`, aquí el color no está grabado: el
navegador recibe la predicción por bloque y pinta el que toca según el reloj
del audio. Eso permite saltar a cualquier punto de la canción, que es lo que
hace falta para enseñar un pasaje concreto sin esperar a que llegue.

Usa solo la biblioteca estándar para el servidor — el proyecto no gana una
dependencia web por una herramienta de demostración.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import color as col
import config as cfg
import visualizar as vis

EXTENSIONES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
TIPOS = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
         ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg"}

_cache: dict[str, dict] = {}
_cerrojo = threading.Lock()
_carpeta: Path = Path("demo")
_modelo: str = "conv1d"


def _analizar(nombre: str) -> dict:
    """Analiza una canción (o devuelve el análisis ya hecho)."""
    with _cerrojo:
        if nombre in _cache:
            return _cache[nombre]

    ruta = (_carpeta / nombre).resolve()
    X, rms, dur = vis.features_cancion_completa(ruta)
    pred = vis.predecir(X, _modelo)
    rgb, val, aro = vis.construir_colores(pred, rms, 5)

    datos = {
        "nombre": nombre,
        "duracion": round(dur, 2),
        "hz": cfg.HZ_ANOT,
        "modelo": _modelo,
        "valencia": [round(float(v), 4) for v in val],
        "activacion": [round(float(a), 4) for a in aro],
        "rgb": [[int(c) for c in fila] for fila in rgb],
    }
    with _cerrojo:
        _cache[nombre] = datos
    return datos


def _canciones() -> list[dict]:
    if not _carpeta.exists():
        return []
    return [
        {"nombre": p.name, "mb": round(p.stat().st_size / 1e6, 1),
         "analizada": p.name in _cache}
        for p in sorted(_carpeta.iterdir())
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    ]


def _disco_png() -> bytes:
    """El plano de Russell, pintado con el mismo mapeo que el resto."""
    from PIL import Image

    lado = 420
    disco, mascara = vis._disco_circumplejo(lado)
    rgba = np.zeros((lado, lado, 4), dtype=np.uint8)
    rgba[..., :3] = disco
    rgba[..., 3] = np.where(mascara, 255, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


class Manejador(BaseHTTPRequestHandler):
    server_version = "VisualizacionMusical/1.0"

    def log_message(self, formato, *args):        # menos ruido en consola
        if "/api/analizar" in (args[0] if args else ""):
            sys.stderr.write(f"  analizando… {args[0]}\n")

    # ── utilidades de respuesta ──
    def _enviar(self, cuerpo: bytes, tipo: str, codigo: int = 200,
                extra: dict | None = None) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, obj, codigo: int = 200) -> None:
        self._enviar(json.dumps(obj).encode(), "application/json; charset=utf-8",
                     codigo)

    def _audio(self, nombre: str) -> None:
        """Sirve el audio con soporte de Range, que es lo que permite saltar."""
        ruta = (_carpeta / nombre).resolve()
        # El nombre viene de la URL: hay que comprobar que no se escape de la
        # carpeta antes de abrir nada.
        if _carpeta.resolve() not in ruta.parents or not ruta.is_file():
            self._json({"error": "no encontrado"}, 404)
            return

        tam = ruta.stat().st_size
        tipo = TIPOS.get(ruta.suffix.lower(), "application/octet-stream")
        rango = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)", rango)

        if not m:
            with open(ruta, "rb") as f:
                self._enviar(f.read(), tipo, 200, {"Accept-Ranges": "bytes"})
            return

        ini = int(m.group(1)) if m.group(1) else 0
        fin = int(m.group(2)) if m.group(2) else tam - 1
        fin = min(fin, tam - 1)
        if ini > fin:
            self._json({"error": "rango inválido"}, 416)
            return

        with open(ruta, "rb") as f:
            f.seek(ini)
            trozo = f.read(fin - ini + 1)
        self._enviar(trozo, tipo, 206, {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {ini}-{fin}/{tam}",
        })

    # ── rutas ──
    def do_GET(self):                              # noqa: N802
        u = urlparse(self.path)
        ruta = u.path

        if ruta == "/":
            self._enviar(PAGINA.encode(), "text/html; charset=utf-8")
        elif ruta == "/disco.png":
            self._enviar(_disco_png(), "image/png",
                         extra={"Cache-Control": "max-age=86400"})
        elif ruta == "/api/canciones":
            self._json(_canciones())
        elif ruta == "/api/analizar":
            nombre = unquote((parse_qs(u.query).get("f") or [""])[0])
            if not nombre or "/" in nombre or "\\" in nombre:
                self._json({"error": "nombre inválido"}, 400)
                return
            if not (_carpeta / nombre).is_file():
                self._json({"error": f"no existe {nombre}"}, 404)
                return
            try:
                self._json(_analizar(nombre))
            except Exception as e:                 # el navegador debe verlo
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        elif ruta.startswith("/audio/"):
            self._audio(unquote(ruta[len("/audio/"):]))
        else:
            self._json({"error": "no encontrado"}, 404)


PAGINA = """<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visualización musical dinámica</title>
<style>
:root{--fondo:#0F1519;--panel:#151D23;--borde:#27333B;--tinta:#E8EDF2;
      --suave:#94A1AF;--acento:#3987E5}
*{box-sizing:border-box}
body{margin:0;background:var(--fondo);color:var(--tinta);
     font-family:'IBM Plex Sans',system-ui,sans-serif;padding:24px;
     display:flex;flex-direction:column;gap:20px;min-height:100vh}
h1{font-size:22px;font-weight:600;margin:0;letter-spacing:-.3px}
h1 small{color:var(--suave);font-weight:400;font-size:15px;margin-left:10px}
.barra{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button{background:var(--panel);color:var(--tinta);border:1px solid var(--borde);
    border-radius:9px;padding:10px 14px;font-size:14px;font-family:inherit;cursor:pointer}
button:hover:not(:disabled){border-color:var(--acento)}
button:disabled{opacity:.45;cursor:default}
.escenario{display:flex;gap:20px;align-items:stretch;flex-wrap:wrap}
#color{flex:1 1 440px;min-height:320px;border-radius:14px;background:#000;
       transition:background-color .12s linear}
.lateral{width:260px;display:flex;flex-direction:column;gap:14px;flex-shrink:0}
.disco{position:relative;width:260px;height:260px}
.disco img{width:100%;height:100%;display:block}
#punto{position:absolute;width:17px;height:17px;border-radius:50%;background:#fff;
       border:2px solid var(--fondo);transform:translate(-50%,-50%);
       left:50%;top:50%;transition:left .12s linear,top .12s linear}
.lbl{position:absolute;font-size:12px;color:#000;transform:translate(-50%,-50%);
     pointer-events:none}
.eje{text-align:center;font-size:12px;color:var(--suave)}
#franja{width:100%;height:66px;border-radius:9px;display:block;cursor:pointer}
.cifras{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:14px;
        color:var(--suave);display:flex;gap:22px;flex-wrap:wrap}
.cifras b{color:var(--tinta);font-weight:600}
audio{width:100%;height:38px}
#aviso{color:var(--suave);font-size:14px;min-height:20px}
#aviso.mal{color:#E8793F}
</style></head><body>

<h1>Visualización musical dinámica <small>Conv1D · predicción cada 0,5 s</small></h1>

<div class="barra">
  <select id="sel"></select>
  <button id="cargar">Analizar</button>
  <button id="recargar" title="Volver a leer la carpeta demo/">↻ carpeta</button>
  <span id="aviso"></span>
</div>

<div class="escenario">
  <div id="color"></div>
  <div class="lateral">
    <div class="disco">
      <img src="/disco.png" alt="Plano valencia-activación">
      <div id="punto"></div>
      <span class="lbl" style="left:28%;top:24%">tensión</span>
      <span class="lbl" style="left:74%;top:24%">alegría</span>
      <span class="lbl" style="left:76%;top:76%">calma</span>
      <span class="lbl" style="left:26%;top:76%">tristeza</span>
    </div>
    <div class="eje">← valencia →&nbsp;&nbsp;·&nbsp;&nbsp;↑ activación</div>
  </div>
</div>

<canvas id="franja" height="66"></canvas>
<audio id="audio" controls preload="auto"></audio>
<div class="cifras">
  <span>t <b id="t">0,0</b> s</span>
  <span>valencia <b id="v">—</b></span>
  <span>activación <b id="a">—</b></span>
  <span>RGB <b id="rgb">—</b></span>
</div>

<script>
const $ = id => document.getElementById(id);
let D = null;                        // análisis de la canción actual

const avisar = (t, mal=false) => { $('aviso').textContent = t;
                                   $('aviso').classList.toggle('mal', mal); };

async function listar(){
  const r = await fetch('/api/canciones');
  const cs = await r.json();
  $('sel').innerHTML = cs.length
    ? cs.map(c => `<option value="${c.nombre}">${c.nombre} · ${c.mb} MB</option>`).join('')
    : '<option value="">(no hay audio en demo/)</option>';
  $('cargar').disabled = !cs.length;
  if(!cs.length) avisar('Deja un mp3, wav, m4a o flac en demo/ y pulsa ↻ carpeta.');
}

function pintarFranja(){
  const c = $('franja'), g = c.getContext('2d');
  c.width = c.clientWidth;
  const n = D.rgb.length, w = c.width / n;
  for(let i=0;i<n;i++){
    const [r,gg,b] = D.rgb[i];
    g.fillStyle = `rgb(${r},${gg},${b})`;
    g.fillRect(i*w, 0, Math.ceil(w)+1, c.height);
  }
}

function marcar(){                   // cursor sobre la franja
  if(!D) return;
  const c = $('franja'), g = c.getContext('2d');
  pintarFranja();
  const x = ($('audio').currentTime / D.duracion) * c.width;
  g.fillStyle = '#fff'; g.fillRect(x-1, 0, 2, c.height);
}

async function cargar(){
  const f = $('sel').value; if(!f) return;
  $('cargar').disabled = true; D = null;
  avisar('Analizando… la primera vez tarda unos segundos.');
  try{
    const r = await fetch('/api/analizar?f=' + encodeURIComponent(f));
    const j = await r.json();
    if(j.error) throw new Error(j.error);
    D = j;
    $('audio').src = '/audio/' + encodeURIComponent(f);
    pintarFranja();
    avisar(`${D.rgb.length} bloques · ${D.duracion.toFixed(1)} s · modelo ${D.modelo}`);
  }catch(e){ avisar('Error: ' + e.message, true); }
  finally{ $('cargar').disabled = false; }
}

function tic(){
  if(D){
    const i = Math.min(Math.floor($('audio').currentTime * D.hz), D.rgb.length-1);
    if(i >= 0){
      const [r,g,b] = D.rgb[i];
      $('color').style.backgroundColor = `rgb(${r},${g},${b})`;
      $('punto').style.left = (50 + D.valencia[i]*50) + '%';
      $('punto').style.top  = (50 - D.activacion[i]*50) + '%';
      $('t').textContent   = $('audio').currentTime.toFixed(1).replace('.', ',');
      $('v').textContent   = D.valencia[i].toFixed(2).replace('.', ',');
      $('a').textContent   = D.activacion[i].toFixed(2).replace('.', ',');
      $('rgb').textContent = `${r}, ${g}, ${b}`;
      marcar();
    }
  }
  requestAnimationFrame(tic);
}

$('cargar').onclick   = cargar;
$('recargar').onclick = listar;
$('franja').onclick = e => {         // clic en la franja = saltar ahí
  if(!D) return;
  const c = e.target.getBoundingClientRect();
  $('audio').currentTime = ((e.clientX - c.left) / c.width) * D.duracion;
};
window.addEventListener('resize', () => D && marcar());
listar(); tic();
</script></body></html>
"""


def main() -> int:
    global _carpeta, _modelo
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parent.parent
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--carpeta", type=Path, default=raiz / "demo",
                    help="de dónde salen las canciones (por defecto demo/)")
    ap.add_argument("--modelo", default="conv1d", choices=["conv1d", "mlp"])
    args = ap.parse_args()

    _carpeta = args.carpeta.resolve()
    _modelo = args.modelo
    if not (cfg.SALIDA_DIR / "modelos" / f"{_modelo}.keras").exists():
        print(f"ERROR: no encuentro el modelo {_modelo}. Corre antes:  make entrenar",
              file=sys.stderr)
        return 1
    _carpeta.mkdir(parents=True, exist_ok=True)

    n = len(_canciones())
    # Solo 127.0.0.1: es una herramienta de demostración, no se expone a la red.
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador)
    print(f"\n  Visor listo en   http://127.0.0.1:{args.puerto}")
    print(f"  Carpeta          {_carpeta}  ({n} archivo{'s' if n != 1 else ''})")
    print(f"  Modelo           {_modelo}")
    print("\n  Ctrl+C para parar.\n")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n  Parado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
