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
import formas as fm
import imagenes as im
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

    energia = col.normalizar_energia(rms)
    formas = fm.va_a_forma_lote(val, aro, energia)

    # El giro se integra aquí y no en el navegador: así la rotación es una
    # función determinista del instante, y saltar a mitad de canción deja la
    # forma donde le toca en vez de donde la hubiera dejado la animación.
    giros = np.array([f["giro"] for f in formas], dtype=float)
    acum = np.concatenate([[0.0], np.cumsum(giros[:-1]) / cfg.HZ_ANOT])

    datos = {
        "nombre": nombre,
        "duracion": round(dur, 2),
        "hz": cfg.HZ_ANOT,
        "modelo": _modelo,
        "valencia": [round(float(v), 4) for v in val],
        "activacion": [round(float(a), 4) for a in aro],
        "rgb": [[int(c) for c in fila] for fila in rgb],
        "formas": formas,
        "giro_acumulado": [round(float(g), 2) for g in acum],
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
    protocol_version = "HTTP/1.1"                 # el audio se sirve mejor así

    def log_message(self, formato, *args):
        # Una línea por petición: sin esto no hay forma de ver si el navegador
        # llegó siquiera, que es justo lo que hace falta cuando algo no suena.
        sys.stderr.write("  " + (formato % args) + "\n")

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

    def _imagen(self, nombre: str) -> None:
        """Sirve una imagen de OASIS, comprobando que no se salga de su carpeta."""
        raiz = cfg.OASIS_IMG.resolve()
        ruta = (raiz / nombre).resolve()
        if raiz not in ruta.parents or not ruta.is_file():
            self._json({"error": "no encontrada"}, 404)
            return
        tipo = "image/png" if ruta.suffix.lower() == ".png" else "image/jpeg"
        with open(ruta, "rb") as f:
            self._enviar(f.read(), tipo, extra={"Cache-Control": "max-age=86400"})

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
        elif ruta == "/api/categorias":
            try:
                from collections import Counter
                c = Counter(e["categoria"] for e in im.cargar())
                self._json({"categorias": [
                    {"nombre": k, "n": c[k], "defecto": k in im.POR_DEFECTO}
                    for k in im.CATEGORIAS if c.get(k)]})
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        elif ruta == "/api/imagenes":
            q = parse_qs(u.query)
            nombre = unquote((q.get("f") or [""])[0])
            cats = tuple(c for c in (q.get("cats") or [""])[0].split(",")
                         if c in im.CATEGORIAS) or im.POR_DEFECTO
            if not nombre or "/" in nombre or "\\" in nombre:
                self._json({"error": "nombre inválido"}, 400)
                return
            try:
                d = _analizar(nombre)
                self._json({"categorias": list(cats),
                            "tramos": im.secuencia(d["valencia"],
                                                   d["activacion"], cats)})
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        elif ruta.startswith("/oasis/"):
            self._imagen(unquote(ruta[len("/oasis/"):]))
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
#escena{flex:1 1 440px;min-height:340px;border-radius:14px;background:#000;
        display:block}
label.conm{display:flex;align-items:center;gap:7px;font-size:14px;color:var(--suave);
           cursor:pointer;user-select:none}
label.conm input{accent-color:var(--acento);width:16px;height:16px;cursor:pointer}
.cats{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
      padding-left:10px;border-left:1px solid var(--borde)}
.cats .av{font-size:12px;color:#E8793F}
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
  <label class="conm"><input type="checkbox" id="conFormas" checked> formas</label>
  <label class="conm"><input type="checkbox" id="conImagen" checked> imagen</label>
  <span id="cats" class="cats"></span>
  <span id="aviso"></span>
</div>

<div class="escenario">
  <canvas id="escena"></canvas>
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
  <span>forma <b id="forma">—</b></span>
  <span>imagen <b id="img">—</b></span>
</div>

<script>
const $ = id => document.getElementById(id);
let D = null, tira = null, animando = false;
let T = null, ultimoTramo = null;   // tramos de imagen

const avisar = (t, mal=false) => { $('aviso').textContent = t;
                                   $('aviso').classList.toggle('mal', mal); };

async function listar(auto){
  try{
    const cs = await (await fetch('/api/canciones')).json();
    $('sel').innerHTML = cs.length
      ? cs.map(c => `<option value="${c.nombre}">${c.nombre} · ${c.mb} MB</option>`).join('')
      : '<option value="">(no hay audio en demo/)</option>';
    $('cargar').disabled = !cs.length;
    if(!cs.length){ avisar('Deja un mp3, wav, m4a o flac en demo/ y pulsa ↻ carpeta.'); return; }
    if(auto) cargar();
  }catch(e){ avisar('No pude hablar con el servidor: ' + e.message, true); }
}

async function listarCategorias(){
  try{
    const j = await (await fetch('/api/categorias')).json();
    if(j.error){ $('cats').textContent = 'sin imágenes (' + j.error + ')'; return; }
    $('cats').innerHTML = j.categorias.map(c =>
      `<label class="conm"><input type="checkbox" class="cat" value="${c.nombre}"`
      + `${c.defecto ? ' checked' : ''}> ${c.nombre} <span style="opacity:.55">`
      + `${c.n}</span></label>`).join('')
      + '<span class="av" id="avisoCat"></span>';
    document.querySelectorAll('.cat').forEach(el => el.onchange = () => {
      avisarCategorias(); cargarImagenes();
    });
    avisarCategorias();
  }catch(e){ $('cats').textContent = 'sin imágenes'; }
}

function catsElegidas(){
  return [...document.querySelectorAll('.cat:checked')].map(e => e.value);
}

function avisarCategorias(){
  // Las imágenes de personas no están en el corpus (ver imagenes.EXCLUIDAS),
  // así que aquí solo queda avisar si no queda ninguna categoría activa.
  const a = $('avisoCat');
  if(a) a.textContent = catsElegidas().length ? '' : '⚠ sin categorías: no hay imagen';
}

async function cargarImagenes(){
  T = null; cacheImg.clear();
  if(!D) return;
  const cats = catsElegidas();
  if(!cats.length) return;
  try{
    const j = await (await fetch('/api/imagenes?f='
      + encodeURIComponent(D.nombre) + '&cats=' + cats.join(','))).json();
    if(j.error) throw new Error(j.error);
    T = j.tramos;
    // Se precargan para que el cambio de tramo no deje un hueco.
    T.forEach(t => imagenDe(t.desde));
    pintar();
  }catch(e){ avisar('Error con las imágenes: ' + e.message, true); }
}

function construirTira(){
  // La franja se dibuja UNA vez en un lienzo de 1 px de alto; cada cuadro solo
  // la estira. Repintarla entera 60 veces por segundo dejaba la página pegada.
  const n = D.rgb.length;
  tira = document.createElement('canvas');
  tira.width = n; tira.height = 1;
  const g = tira.getContext('2d'), img = g.createImageData(n, 1);
  for(let i=0;i<n;i++){
    const [r,gg,b] = D.rgb[i];
    img.data[i*4]=r; img.data[i*4+1]=gg; img.data[i*4+2]=b; img.data[i*4+3]=255;
  }
  g.putImageData(img, 0, 0);
}

function dibujarFranja(){
  if(!tira) return;
  const c = $('franja'), g = c.getContext('2d');
  if(c.width !== c.clientWidth) c.width = c.clientWidth;
  g.imageSmoothingEnabled = false;
  g.drawImage(tira, 0, 0, c.width, c.height);
  const x = (($('audio').currentTime || 0) / D.duracion) * c.width;
  g.fillStyle = '#fff'; g.fillRect(x - 1, 0, 2, c.height);
}

const cacheImg = new Map();            // archivo -> HTMLImageElement

function imagenDe(i){
  if(!T || !$('conImagen').checked) return null;
  const tr = T.find(t => i >= t.desde && i < t.hasta) || T[T.length - 1];
  if(!tr) return null;
  let el = cacheImg.get(tr.archivo);
  if(!el){
    el = new Image();
    el.src = '/oasis/' + encodeURIComponent(tr.archivo);
    cacheImg.set(tr.archivo, el);
  }
  ultimoTramo = tr;
  return el.complete && el.naturalWidth ? el : null;
}

function dibujarCover(g, el, w, h){
  // La imagen cubre el lienzo sin deformarse; sobra por el lado largo.
  const e = Math.max(w / el.naturalWidth, h / el.naturalHeight);
  const aw = el.naturalWidth * e, ah = el.naturalHeight * e;
  g.drawImage(el, (w - aw)/2, (h - ah)/2, aw, ah);
}

const NV = 420;                        // vértices por contorno

// Réplica de formas.radio(). Si cambia una, hay que cambiar la otra: se manda
// el parámetro y no la silueta porque son 540 bloques por canción, y enviar
// 420 vértices de cada uno serían megabytes de JSON.
function perfil(f){
  const rs = new Float64Array(NV);
  let pico = 0;
  for(let k=0;k<NV;k++){
    const th = k * 2*Math.PI / NV, t = f.m * th / 4;
    const base = Math.pow(Math.abs(Math.cos(t)), f.n2)
               + Math.pow(Math.abs(Math.sin(t)), f.n3);
    let r = Math.pow(Math.max(base, 1e-9), -1/f.n1);
    r *= (1 + (f.ondulacion || 0) * Math.cos(f.m * th));
    r *= (1 + f.espigas * Math.cos(f.armonico * th));
    rs[k] = r; if(r > pico) pico = r;
  }
  if(pico > 0) for(let k=0;k<NV;k++) rs[k] /= pico;
  return rs;
}

// El perfil solo depende de los parámetros, y hay 540 bloques pero pocos
// juegos distintos: se cachean por clave para no recalcular en cada cuadro.
const cachePerfil = new Map();
function perfilDe(f){
  const k = `${f.m}|${f.n1}|${f.n2}|${f.n3}|${f.ondulacion}|${f.espigas}|${f.armonico}`;
  let r = cachePerfil.get(k);
  if(!r){ r = perfil(f); cachePerfil.set(k, r); }
  return r;
}

function trazarForma(g, f, rot, cx, cy, R){
  const rs = perfilDe(f);
  g.beginPath();
  for(let k=0;k<NV;k++){
    const th = rot + k * 2*Math.PI / NV, rr = R * rs[k];
    const x = cx + Math.cos(th)*rr, y = cy + Math.sin(th)*rr;
    k ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.closePath();
}

function dibujarEscena(i){
  const c = $('escena'), g = c.getContext('2d');
  const w = c.clientWidth, h = c.clientHeight;
  if(c.width !== w || c.height !== h){ c.width = w; c.height = h; }
  const [r,gg,b] = D.rgb[i];
  const color = `rgb(${r},${gg},${b})`;
  const el = imagenDe(i);

  // El fondo lleva el color atenuado: así se ve el color aunque la forma sea
  // pequeña en un pasaje flojo.
  g.fillStyle = `rgb(${r*0.22|0},${gg*0.22|0},${b*0.22|0})`;
  g.fillRect(0, 0, w, h);

  const conFormas = $('conFormas').checked;

  // Las tres capas, de abajo arriba: imagen, filtro de color, máscara de
  // forma. La máscara va primero en el código porque recortar es un `clip`
  // que condiciona todo lo que se pinte después.
  const pintarCapas = () => {
    if(el){
      dibujarCover(g, el, w, h);
      // 'color' toma matiz y saturación de lo que se pinta y la luminosidad
      // de lo que hay debajo: eso es exactamente un filtro de color.
      g.globalCompositeOperation = 'color';
      g.globalAlpha = 0.8;
      g.fillStyle = color; g.fillRect(0, 0, w, h);
      g.globalCompositeOperation = 'source-over';
      g.globalAlpha = 1;
    }else{
      g.fillStyle = color; g.fillRect(0, 0, w, h);
    }
  };

  if(!conFormas){
    g.save(); pintarCapas(); g.restore();
    return;
  }

  const f = D.formas[i];
  // Giro interpolado entre bloques, a partir de la integral que manda el
  // servidor: el mismo instante da siempre la misma rotación.
  const t = $('audio').currentTime * D.hz;
  const i0 = Math.max(0, Math.min(Math.floor(t), D.giro_acumulado.length - 1));
  const i1 = Math.min(i0 + 1, D.giro_acumulado.length - 1);
  const gr = D.giro_acumulado[i0]
           + (D.giro_acumulado[i1] - D.giro_acumulado[i0]) * (t - i0);
  const rot = gr * Math.PI / 180;

  const R = Math.min(w, h) * 0.40 * f.escala;
  const cx = w/2, cy = h/2;

  // El halo se pinta antes del recorte, porque un `clip` lo cortaría.
  g.save();
  g.shadowColor = color; g.shadowBlur = 40 * f.escala;
  trazarForma(g, f, rot, cx, cy, R);
  g.fillStyle = color; g.fill();
  g.restore();

  g.save();
  trazarForma(g, f, rot, cx, cy, R);
  g.clip();
  pintarCapas();
  g.restore();
}

function pintar(){
  if(!D) return;
  const n = D.rgb.length;
  const i = Math.max(0, Math.min(Math.floor($('audio').currentTime * D.hz), n - 1));
  const [r,g,b] = D.rgb[i];
  dibujarEscena(i);
  $('punto').style.left = (50 + D.valencia[i]*50) + '%';
  $('punto').style.top  = (50 - D.activacion[i]*50) + '%';
  $('t').textContent   = $('audio').currentTime.toFixed(1).replace('.', ',');
  $('v').textContent   = D.valencia[i].toFixed(2).replace('.', ',');
  $('a').textContent   = D.activacion[i].toFixed(2).replace('.', ',');
  $('rgb').textContent = `${r}, ${g}, ${b}`;
  const f = D.formas[i];
  $('forma').textContent = `m${f.m} · n1 ${f.n1.toFixed(1).replace('.',',')}`
                         + ` · púas ${f.espigas.toFixed(2).replace('.',',')}`
                         + ` · ${Math.round(f.giro)}°/s`;
  $('img').textContent = ultimoTramo
    ? `${ultimoTramo.tema} (${ultimoTramo.categoria})` : '—';
  dibujarFranja();
}

function bucle(){ pintar(); if(animando) requestAnimationFrame(bucle); }

async function cargar(){
  const f = $('sel').value; if(!f) return;
  $('cargar').disabled = true; D = null; tira = null; animando = false;
  avisar('Analizando «' + f + '»… la primera vez tarda unos segundos.');
  try{
    const j = await (await fetch('/api/analizar?f=' + encodeURIComponent(f))).json();
    if(j.error) throw new Error(j.error);
    D = j;
    construirTira();
    $('audio').src = '/audio/' + encodeURIComponent(f);
    avisar(`${D.rgb.length} bloques · ${D.duracion.toFixed(1)} s · modelo ${D.modelo} — dale al play`);
    cargarImagenes();
  }catch(e){ avisar('Error al analizar: ' + e.message, true); }
  finally{ $('cargar').disabled = false; }
}

const A = $('audio');
A.addEventListener('playing', () => { animando = true; bucle(); });
A.addEventListener('pause',  () => { animando = false; pintar(); });
A.addEventListener('ended',  () => { animando = false; pintar(); });
A.addEventListener('seeked', pintar);
A.addEventListener('loadeddata', pintar);
A.addEventListener('error', () => {
  const c = A.error ? A.error.code : '?';
  avisar(`El navegador no pudo reproducir el audio (código ${c}). `
       + 'Mira la consola del servidor.', true);
});

$('cargar').onclick   = () => cargar();
$('recargar').onclick = () => listar(false);
$('franja').onclick = e => {
  if(!D) return;
  const c = e.target.getBoundingClientRect();
  A.currentTime = ((e.clientX - c.left) / c.width) * D.duracion;
  pintar();
};
$('conFormas').onchange = () => pintar();
$('conImagen').onchange = () => pintar();
window.addEventListener('resize', () => { if(D) pintar(); });
listarCategorias();
listar(true);           // analiza la primera canción al abrir
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
