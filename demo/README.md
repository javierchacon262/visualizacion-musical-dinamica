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

## La pista de prueba

```bash
python scripts/pista_demo.py
```

Empalma seis fragmentos de DEAM elegidos en los extremos de los cuatro
cuadrantes del plano valencia–activación. El color cambia de forma visible al
pasar de uno a otro, así que sirve para comprobar de un vistazo que todo
funciona. **Cinco de los seis** caen en el cuadrante correcto; el sexto, una
pieza anotada como *calma*, se predice como tensión.

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
