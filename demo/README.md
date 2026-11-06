# Carpeta de demostración

Aquí van los archivos de audio con los que quieras ver el sistema funcionando.
**El audio de esta carpeta no se versiona** (`.gitignore`), así que puedes dejar
tu propia música sin arrastrarla al repositorio.

```bash
make demo ARCHIVO=demo/mi-cancion.mp3
```

Produce dos cosas en `outputs/`:

| Archivo | Qué es |
|---|---|
| `mi-cancion.png` | la franja de color, la canción entera de un vistazo |
| `mi-cancion.mp4` | el video sincronizado con el audio |

Acepta mp3, wav, m4a, flac y ogg. Cuatro minutos de audio se renderizan en
menos de dos.

## Las pistas de prueba

```bash
python scripts/pista_demo.py                      # las tres
python scripts/pista_demo.py --pista calma-alegria
```

Cada una empalma seis fragmentos de DEAM elegidos en el extremo de su cuadrante.

| pista | qué enseña |
|---|---|
| `alegria-tristeza` | **para enseñar que funciona** — amarillo ↔ azul, los dos cuadrantes que el modelo acierta (84 % y 71 %) |
| `cuadrantes` | los cuatro cuadrantes; cinco de seis correctos |
| `calma-alegria` | **enseña el fallo**: el modelo acierta calma en 1 de 21 canciones de test, y la pinta como tristeza |

Si vas a mostrar el sistema funcionando, usa `alegria-tristeza`. La de
`calma-alegria` está para la conversación sobre limitaciones.

Se regenera en vez de versionarse: el repositorio guarda la receta, no los 6 MB
de audio derivado. El audio de DEAM tiene licencia Creative Commons, así que
esa pista se puede mostrar y publicar sin problema.

## Música con derechos de autor

Para una demostración en clase no hay inconveniente en usar tus propios
archivos: reproducir música en un aula es uso educativo y el audio nunca sale
de tu máquina.

Publicar el video resultante es otra cosa — llevaría la grabación completa
dentro. Para cualquier versión que vaya a subirse a algún sitio, usa la pista
de DEAM.
