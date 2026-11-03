#!/usr/bin/env bash
set -euo pipefail

SUPERMIMIC_ROOT="${SUPERMIMIC_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAACSIM_ROOT_PATH="${ISAACSIM_ROOT_PATH:-/isaac-sim}"
GPUFREE_PYTHON="${GPUFREE_PYTHON:-/opt/isaaclab-venv/bin/python}"

test -x "${GPUFREE_PYTHON}"
test -x "${ISAACSIM_ROOT_PATH}/python.sh"
test -f "${ISAACSIM_ROOT_PATH}/setup_python_env.sh"
test -d "${SUPERMIMIC_ROOT}/src/supermimic_rl"

# Non-interactive SSH does not load the image's .bashrc. The bundled Python
# must resolve its own libpython before any system copy with the same SONAME.
export LD_LIBRARY_PATH="${ISAACSIM_ROOT_PATH}/kit/python/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${SUPERMIMIC_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ISAACSIM_ROOT_PATH
export ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
export PYTHONEXE="${GPUFREE_PYTHON}"
export OMNI_KIT_ACCEPT_EULA=yes
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/gpufree-data/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

cd -- "${SUPERMIMIC_ROOT}"
# Same environment as run_gpufree.sh; the PPO teacher has its own entry script.
# The installed official launcher sources setup_python_env.sh and configures
# CARB_APP_PATH, EXP_PATH, and libcarb preload for this standalone distribution.
# SUPERMIMIC_SCRIPT runs a repository script with the same environment instead of the CLI.
if [[ -n "${SUPERMIMIC_SCRIPT:-}" ]]; then
  exec "${ISAACSIM_ROOT_PATH}/python.sh" --no-ros-env -u "${SUPERMIMIC_SCRIPT}" "$@"
fi
exec "${ISAACSIM_ROOT_PATH}/python.sh" --no-ros-env -u "${SUPERMIMIC_ROOT}/scripts/train_ppo.py" "$@"
