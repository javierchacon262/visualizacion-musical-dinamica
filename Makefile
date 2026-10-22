.PHONY: ayuda setup datos datos-min verificar extraer limpiar

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
	@echo "  make limpiar     borra lo generado (conserva el audio descargado)"
	@echo
	@echo "Primera vez:  ./setup.sh && make datos && make extraer"

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

$(PY):
	@echo "No encuentro $(PY) — corre primero:  ./setup.sh" >&2
	@exit 1

limpiar:
	@rm -rf data/procesado
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Listo. El audio en data/ se conservó."
