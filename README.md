# Visualización musical dinámica

Traduce una canción en una línea de tiempo de color: a medida que avanza el
audio, la pantalla muestra un color que evoluciona con la **emoción** y la
**energía** de la música.

El modelo no predice un color. Predice dos números por cada medio segundo —
**valencia** (qué tan agradable suena) y **activación** (qué tan calmado o
excitado) — y un mapeo explícito los convierte en color. Separar esos dos
problemas es deliberado: uno es de aprendizaje, el otro de diseño.

> Proyecto final de Deep Learning · Universidad Industrial de Santander · 2026

---

## Empezar

```bash
git clone https://github.com/javierchacon262/visualizacion-musical-dinamica.git
cd visualizacion-musical-dinamica

./setup.sh          # crea el entorno (Python 3.12 vía uv) e instala todo
make datos          # descarga el corpus DEAM (~1.3 GB, reanudable)
make extraer        # procesa 1802 canciones (~3 min con 22 núcleos)
make entrenar       # entrena y evalúa el MLP y el Conv1D
```

Y para ver el resultado sobre audio:

```bash
make pista                                   # arma las tres pistas de prueba
make visor                                   # visor web en vivo
make demo ARCHIVO=demo/cuadrantes.mp3        # franja PNG + video MP4
```

Y para comprobar que todo quedó bien:

```bash
source .venv/bin/activate
make verificar
```

Si solo quieres mirar las etiquetas sin bajar 1.3 GB de audio:
`make datos-min` (4.6 MB).

**El servidor de DEAM se cae con cierta frecuencia.** Si `make datos` falla,
no es tu red: espera un rato y vuelve a correrlo — las descargas parciales se
reanudan solas. Como alternativa, baja los `.zip` a mano desde
<https://cvml.unige.ch/databases/DEAM/>, déjalos en `data/` y corre `make datos`
otra vez: los descomprimirá sin volver a descargarlos. Y si ya tienes el corpus
en otro sitio, apunta `export DEAM_DIR=/ruta/a/deam` y te ahorras la descarga.

### Por qué `uv` y no el `venv` del sistema

El proyecto necesita **Python 3.12**. Muchas distribuciones ya traen 3.13 o
3.14, versiones para las que TensorFlow y numba todavía no publican binarios.
`uv` descarga el intérprete correcto sin tocar el Python del sistema. Si no lo
tienes instalado, `setup.sh` lo instala.

### GPU

Opcional: el pipeline de datos no la necesita. Si la tienes, `setup.sh`
instala PyTorch con CUDA 12.8 (compatible con Blackwell / RTX 50) y
TensorFlow con CUDA.

Hay un detalle que cuesta encontrar: **TensorFlow no localiza por sí solo las
librerías CUDA que instala pip**. Reporta `Cannot dlopen some GPU libraries` y
cae a CPU en silencio. `setup.sh` parcha `.venv/bin/activate` para que
`LD_LIBRARY_PATH` apunte a `site-packages/nvidia/*/lib`. Con eso ambos
frameworks corren a la misma velocidad.

Para instalar solo versiones de CPU: `./setup.sh --cpu`.

**Aviso para GPUs Blackwell (RTX 50):** TensorFlow 2.21 todavía no trae kernels
precompilados para `compute capability 12.0`, así que los compila desde PTX la
primera vez — puede tardar bastante en el primer entrenamiento. Funciona, pero
si quieres resultados rápidos desde el principio, usa PyTorch para entrenar.

---

## El mapeo emoción → color

El plano valencia–activación del **modelo circumplejo de Russell (1980)** es un
círculo, y el matiz del espacio HSV también. Mapear ángulo → matiz preserva la
estructura: emociones vecinas dan colores vecinos. No es una paleta elegida a
gusto.

| Canal HSV | Lleva | Significa |
|---|---|---|
| Matiz | dirección en el plano V–A | **qué** emoción |
| Saturación | distancia al origen | qué tan **intensa** |
| Valor | energía RMS del audio | qué tan **fuerte** suena |

Anclado en los cuatro cuadrantes:

| Cuadrante | Emoción | Color |
|---|---|---|
| V−, A+ | tensión, ira | rojo `(255,0,0)` |
| V+, A+ | alegría, euforia | amarillo `(255,255,0)` |
| V+, A− | calma, serenidad | verde-cian `(0,255,128)` |
| V−, A− | tristeza, apatía | azul `(0,0,255)` |

El detalle que lo hace robusto: **el brillo lo gobierna el RMS del audio, no la
red**. Un silencio sale oscuro aunque el modelo prediga cualquier cosa — evita
el fallo típico de estos visualizadores, que es mostrar colores intensos
durante los silencios.

```bash
python src/color.py     # imprime la verificación de los cuatro cuadrantes
```

---

## Dos hallazgos sobre los datos

Estos condicionan cómo hay que evaluar, y conviene leerlos antes de entrenar.

### 1. Las anotaciones por segundo son muy ruidosas

DEAM trae **10 anotadores por canción**, lo que permite medir cuánto coinciden
entre sí:

| | r entre dos anotadores | Fiabilidad del promedio de 10 |
|---|---|---|
| Activación | 0.132 | **0.60** |
| Valencia | 0.071 | **0.43** |

El 24 % de los anotadores dejó la curva plana. Dos personas anotando la misma
canción coinciden en r ≈ 0.1: la emoción momento a momento es intrínsecamente
subjetiva.

**Ese 0.60 / 0.43 es el techo.** Un modelo que alcance r = 0.50 en activación
está cerca del límite de lo predecible, no fallando.

### 2. No confundir correlación agrupada con seguimiento temporal

Una sola característica (`mfcc1_media`, esencialmente log-energía) da r = 0.619
con la activación si se agrupan los 108 120 puntos. Al descomponerla:

| | Entre canciones | Dentro de canción | Agrupada |
|---|---|---|---|
| Activación | 0.668 | **0.150** | 0.619 |
| Valencia | 0.387 | **0.081** | 0.357 |

La agrupada está dominada por distinguir canciones intensas de suaves, que es
lo fácil. **Seguir el vaivén dentro de una canción es r = 0.15**, y ese es el
número comparable con el techo humano.

**Consecuencia:** hay que reportar las dos componentes por separado. Un modelo
que solo aprenda «esta canción es intensa» daría una visualización de color
casi constante — exactamente lo que el proyecto no quiere.

---

## Pipeline

```
mp3 → librosa 22 050 Hz mono
    ├── análisis espectral a 50 Hz (hop 441 = 20 ms)
    │   └── agregado en bloques de 25 fotogramas = 0.5 s exactos
    │       → 78 características (39 base × media y desviación)
    └── mel-espectrograma 128 bandas a 50 Hz
        → (128, 1500) float16 por canción
                    ▼
        MODELO → (valencia, activación) por bloque
                    ▼
        suavizado temporal → color → video / demo
```

El análisis corre a 50 Hz y *después* se agrega, en vez de calcularlo directo a
2 Hz: con un salto de 0.5 s, una FFT de 2048 muestras solo miraría 93 ms de
cada medio segundo y desperdiciaría el 80 % del audio. De cada bloque se
guardan media **y desviación** — la desviación captura variación rítmica
interna que una media sola borraría.

La rejilla encaja exacta con las anotaciones de DEAM, que vienen a 2 Hz
empezando en el segundo 15 de cada fragmento. No hay que interpolar nada.

### Lo que produce `make extraer`

```
1802 canciones · 60 bloques · 78 características   →  108 120 puntos etiquetados
sin NaN ni Inf · etiquetas en [−0.77, 0.83]
1802 mel-espectrogramas (128, 1500) float16
```

---

## Resultados

Dos modelos sobre las mismas características, para aislar una sola variable:
si el modelo ve contexto temporal o no. El MLP trata cada bloque de medio
segundo por separado; el Conv1D convoluciona a lo largo del tiempo.

| correlación · test | MLP | Conv1D |
|---|---:|---:|
| activación · entre canciones | 0.791 | 0.790 |
| activación · **dentro de canción** | 0.079 | **0.134** |
| valencia · entre canciones | 0.675 | 0.649 |
| valencia · **dentro de canción** | 0.035 | **0.056** |
| parámetros | 18 498 | 157 890 |

La fila «entre» no se mueve: para distinguir una canción intensa de una suave
basta el nivel medio de energía, y el contexto temporal no aporta nada. La fila
«dentro» casi se duplica. Es la asimetría que uno espera si la hipótesis es
correcta — **el contexto temporal solo puede ayudar en la parte que depende del
tiempo**.

### La etiqueta se mueve menos que el modelo

| desviación dentro de canción | activación |
|---|---:|
| etiqueta de DEAM | 0.035 |
| predicción del Conv1D | 0.082 |

Promediar diez anotadores cancela casi todo el vaivén dentro de un fragmento de
30 s y deja una curva casi plana. El modelo ya oscila más que su propio
objetivo, así que el problema no es solo seguir mal la curva: la curva de
referencia apenas se mueve.

### Calibración

Una red entrenada con MSE regresa a la media — equivocarse en el centro cuesta
menos — y comprime la dispersión en un factor 1.4. En pantalla eso da colores
apagados. `entrenar.py` mide el factor sobre el conjunto de test y guarda
`calibracion_<modelo>.npz`; `visualizar.py` lo aplica. **No cambia la
correlación**, que es lo que mide el acierto: solo la escala. Para ver la salida
cruda, `--sin-calibrar`.

### Acierta la activación, falla la valencia

Acierto de cuadrante sobre las 271 canciones de test, usando la media por
canción:

| cuadrante real | n | acierto | se confunde con |
|---|---:|---:|---|
| alegría (V+ A+) | 155 | **84 %** | — |
| tristeza (V− A−) | 62 | **71 %** | — |
| tensión (V− A+) | 33 | **39 %** | alegría ×15 |
| **calma (V+ A−)** | 21 | **5 %** | **tristeza ×14** |
| total | 271 | 69 % | |

Los dos cuadrantes que salen bien son los que se separan por activación.
Los dos que fallan —tensión y calma— son justo los que exigen acertar la
**valencia** a activación dada. No es falta de datos: train tiene 161 canciones
de calma (13 %). Es que con MSE y una activación media de +0.135 en el corpus,
apostar por activación positiva sale barato, y las canciones calmadas extremas
se arrastran al otro lado del cero.

Sobre ocho canciones elegidas en el extremo de cada cuadrante, el modelo acierta
**8 de 8** en alegría y **0 de 8** en calma.

### Sobre audio real

`make pista` arma tres pistas de demostración, cada una empalmando seis
fragmentos de DEAM elegidos en el extremo de su cuadrante:

| pista | qué enseña |
|---|---|
| `alegria-tristeza` | los dos cuadrantes que el modelo maneja — amarillo ↔ azul |
| `cuadrantes` | los cuatro: **cinco de seis** correctos |
| `calma-alegria` | el fallo sistemático en calma |

## Imágenes

La tercera capa. `src/imagenes.py` recupera de **OASIS** la imagen cuya emoción
anotada está más cerca de la que el modelo predice en ese instante.

OASIS (Kurdi, Lozano & Banaji, 2017) valora 900 imágenes en **los mismos dos
ejes** que predice el modelo, así que el enlace no necesita un puente inventado:
es una búsqueda de vecino más cercano en el plano (valencia, activación). Es el
método de **Xing et al. (2019, IEEE Access)**, que enlazan imagen y música por
similitud emocional en vez de entrenar un generador transmodal.

```bash
make imagenes          # descarga OASIS (92 MB)
python src/imagenes.py # comprueba el corpus y la recuperación
```

Lo que recupera, por cuadrante:

| cuadrante | imagen | categoría |
|---|---|---|
| tensión (V− A+) | Explosion 5 | Scene |
| alegría (V+ A+) | Fireworks 2 | Scene |
| calma (V+ A−) | Flowers 6 | Object |
| tristeza (V− A−) | Garbage dump 1 | Scene |

**La imagen no cambia cada medio segundo**, que sería un estroboscopio. Se
mantiene hasta que se cumplen las dos condiciones: que hayan pasado 8 segundos
y que el punto emocional se haya movido más de 0.22. Así los cambios caen donde
la música cambia, no a intervalos fijos. Sobre una pista de 4.5 minutos salen
unos 20 tramos.

### Aviso de contenido

OASIS es un conjunto afectivo, no un banco de imágenes agradables: para cubrir
la valencia negativa incluye heridas, agresión y escenas angustiosas, y casi
todo eso cae en la categoría `Person`. **El visor filtra por categoría y trae
`Person` desactivada**, con un aviso si se reactiva.

| categoría | n | por defecto |
|---|---:|---|
| Object | 200 | ✅ |
| Scene | 220 | ✅ |
| Animal | 134 | ✅ |
| Person | 346 | ❌ |

## Formas

`src/formas.py` mapea el mismo par (valencia, activación) a la forma de un
polígono, con el mismo criterio que el color: una regularidad perceptual ya
medida, fuera de la red, cambiable sin reentrenar.

La regularidad es la **angularidad**. Bar y Neta (2006) mostraron que un
contorno de vértices agudos se percibe como amenazante y uno curvo como seguro,
incluso en objetos sin significado — un eje que ya corre paralelo a la valencia.

| parámetro | lo gobierna | de | a |
|---|---|---|---|
| lados | valencia | 3 (agudo) | 24 (círculo) |
| rugosidad | activación | 0.05 | 0.45 |
| giro | activación | 2 °/s | 60 °/s |
| escala | **energía RMS** | 0.18 | 1.0 |

El tamaño lo gobierna el RMS y no la red, por la misma razón que el brillo: un
silencio tiene que verse pequeño aunque el modelo prediga lo que sea.

```bash
python src/formas.py    # verificación de los cuatro cuadrantes
```

Es un mapeo paramétrico, no un modelo generativo. Una GAN condicionada a
(valencia, activación) iría **después** de esta etapa, no en lugar del Conv1D:
el cuello de botella de la predicción es el ruido de etiqueta, no la capacidad.

---

## Estructura

```
.
├── setup.sh                  entorno virtual + dependencias + parche CUDA
├── Makefile                  make datos / extraer / verificar
├── scripts/
│   ├── descargar_datos.py    descarga DEAM (stdlib: corre sin el venv)
│   ├── descargar_imagenes.py descarga OASIS (stdlib)
│   ├── pista_demo.py         arma las pistas de prueba
│   └── verificar.py          comprueba entorno, GPU y datos
├── src/
│   ├── config.py             rutas y constantes
│   ├── color.py              Russell → HSV, verificado en 4 cuadrantes
│   ├── formas.py             Russell → polígono, verificado en 4 cuadrantes
│   ├── imagenes.py           recuperación en OASIS por cercanía emocional
│   ├── features.py           extracción alineada con las anotaciones
│   ├── extraer.py            corre el corpus en paralelo
│   ├── entrenar.py           MLP y Conv1D, evaluación descompuesta
│   ├── visualizar.py         canción → franja PNG y video MP4
│   └── servidor.py           visor web (biblioteca estándar)
├── demo/                     tu audio (ignorado por git salvo el README)
└── data/                     (ignorado por git) corpus y características
```

---

## Estado

| Etapa | Estado |
|---|---|
| Mapeo emoción → color | ✅ verificado |
| Descarga y verificación del corpus | ✅ |
| Extracción de características | ✅ 108 120 puntos |
| MLP (línea base) | ✅ r dentro 0.079 |
| Conv1D sobre secuencias | ✅ r dentro 0.134 |
| Conv2D sobre mel-espectrograma | pendiente |
| CRNN | pendiente |
| Autoencoder (no supervisado) | pendiente |
| Renderizado de video | ✅ |
| Visor web en vivo | ✅ |
| Mapeo a formas (paramétrico) | ✅ |
| Capa de imagen (OASIS) | ✅ |

Métricas: RMSE y correlación de Pearson, **separadas en componente entre
canciones y dentro de canción**, reportadas contra el techo humano.

---

## Datos y licencia

El corpus es **DEAM** (MediaEval Database for Emotional Analysis in Music):
1802 fragmentos con anotaciones continuas de valencia y activación a 2 Hz.
Su audio tiene **licencia Creative Commons**, así que los resultados se pueden
mostrar y publicar sin problema de derechos.

> Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017).
> *Developing a benchmark for emotional analysis of music.* PLoS ONE 12(3).
> <https://cvml.unige.ch/databases/DEAM/>

El código de este repositorio está bajo licencia MIT (ver `LICENSE`). La
licencia del corpus la fija su fuente original.
