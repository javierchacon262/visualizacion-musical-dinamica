.PHONY: ayuda setup datos datos-min verificar extraer entrenar pista demo visor limpiar

VENV := .venv/bin
PY   := $(VENV)/python

ayuda:
	@echo "Visualización musical dinámica — comandos disponibles"
	@echo
	@echo "  make setup       crea el entorno virtual e instala todo"
	@echo "  make datos       descarga el corpus DEAM (~1.3 GB)"
	@echo "  make datos-min   solo anotaciones (4.6 MB, sin audio)"
	@echo "  make verificar   comprueba entorno, GPU y datos"
	@echo "  make extraer     procesa el corpus y guarda las características"
	@echo "  make entrenar    entrena y evalúa el MLP y el Conv1D"
	@echo "  make pista       arma la pista de prueba con fragmentos de DEAM"
	@echo "  make demo ARCHIVO=cancion.mp3    franja y video de una canción"
	@echo "  make visor       visor web en http://127.0.0.1:8765"
	@echo "  make limpiar     borra lo generado (conserva el audio descargado)"
	@echo
	@echo "Primera vez:  ./setup.sh && make datos && make extraer && make entrenar"

setup:
	@./setup.sh

datos:
	@python3 scripts/descargar_datos.py

datos-min:
	@python3 scripts/descargar_datos.py --solo-anotaciones

verificar: | $(PY)
	@$(PY) scripts/verificar.py

extraer: | $(PY)
	@$(PY) src/extraer.py

entrenar: | $(PY)
	@$(PY) src/entrenar.py

pista: | $(PY)
	@$(PY) scripts/pista_demo.py

visor: | $(PY)
	@$(PY) src/servidor.py

demo: | $(PY)
	@test -n "$(ARCHIVO)" || { \
		echo "Uso:  make demo ARCHIVO=ruta/a/cancion.mp3" >&2; exit 1; }
	@$(PY) src/visualizar.py "$(ARCHIVO)"

$(PY):
	@echo "No encuentro $(PY) — corre primero:  ./setup.sh" >&2
	@exit 1

limpiar:
	@rm -rf data/procesado
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Listo. El audio en data/ se conservó."
