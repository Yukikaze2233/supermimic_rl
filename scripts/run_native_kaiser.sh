#!/usr/bin/env bash
set -euo pipefail

SUPERMIMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${SUPERMIMIC_ROOT}/src"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/kaiser/software/isaaclab-3.0.0/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMNI_KIT_ACCEPT_EULA=yes
export XDG_CACHE_HOME=/home/kaiser/software/isaaclab-3.0.0/runtime/cache
export XDG_CONFIG_HOME=/home/kaiser/software/isaaclab-3.0.0/runtime/config
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd -- "${SUPERMIMIC_ROOT}"
exec /home/kaiser/software/isaaclab-3.0.0/IsaacLab/.venv/bin/python -u -m supermimic_rl "$@"
