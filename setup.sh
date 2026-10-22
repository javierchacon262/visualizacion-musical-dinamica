#!/usr/bin/env bash
# Crea el entorno virtual y deja el proyecto listo para correr.
#
#   ./setup.sh           # instala todo, con soporte de GPU si la hay
#   ./setup.sh --cpu     # fuerza versiones de CPU (más liviano, ~1 GB)
#
# Por qué uv y no venv del sistema: este proyecto necesita Python 3.12, y
# muchas distribuciones ya traen 3.13 o 3.14, para las que TensorFlow y numba
# todavía no publican binarios. uv descarga el intérprete correcto sin tocar
# el Python del sistema.

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ"
PYVER=3.12
SOLO_CPU=0
[[ "${1:-}" == "--cpu" ]] && SOLO_CPU=1

azul()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
verde() { printf '\033[1;32m%s\033[0m\n' "$*"; }
rojo()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }
gris()  { printf '\033[0;90m%s\033[0m\n' "$*"; }

azul "▸ 1/4 · Comprobando uv"
if ! command -v uv >/dev/null 2>&1; then
  gris "  uv no está instalado. Instalándolo…"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    rojo "  Necesito curl o wget para instalar uv."
    rojo "  Instálalo a mano: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    rojo "  uv se instaló pero no está en el PATH."
    rojo "  Abre una terminal nueva y vuelve a correr ./setup.sh"
    exit 1
  }
fi
gris "  $(uv --version)"

azul "▸ 2/4 · Creando entorno con Python $PYVER"
uv venv --python "$PYVER" >/dev/null
gris "  .venv creado"

azul "▸ 3/4 · Instalando dependencias"
gris "  núcleo (numpy, librosa, scikit-learn…)"
uv pip install -q numpy scipy librosa soundfile matplotlib pandas scikit-learn tqdm

if [[ $SOLO_CPU -eq 1 ]]; then
  gris "  PyTorch y TensorFlow (CPU)"
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q tensorflow
else
  gris "  PyTorch (CUDA 12.8 — compatible con Blackwell/RTX 50)"
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
  gris "  TensorFlow con CUDA"
  uv pip install -q 'tensorflow[and-cuda]'
fi

azul "▸ 4/4 · Parche de CUDA para TensorFlow"
# TensorFlow no encuentra por sí solo las librerías CUDA que pip deja en
# site-packages/nvidia/*/lib: reporta "Cannot dlopen some GPU libraries" y
# cae a CPU en silencio. Lo resolvemos desde el activate del entorno.
if ! grep -q "_NVLIBS" .venv/bin/activate 2>/dev/null; then
  cat >> .venv/bin/activate <<'PARCHE'

# --- CUDA: TensorFlow no localiza solo las libs que instala pip ---
_NVLIBS=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH:-}"
unset _NVLIBS
PARCHE
  gris "  añadido a .venv/bin/activate"
else
  gris "  ya estaba aplicado"
fi

echo
verde "✅ Entorno listo."
echo
echo "   source .venv/bin/activate"
echo "   make verificar      # comprueba GPU y librerías"
echo "   make datos          # descarga DEAM (~1.3 GB)"
echo "   make extraer        # procesa el corpus (~3 min con 22 núcleos)"
