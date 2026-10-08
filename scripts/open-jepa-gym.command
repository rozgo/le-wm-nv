#!/bin/sh
set -eu
gym_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
viewer_python=${JEPA_GYM_PYTHON:-"$HOME/.cache/lewm-mujoco-viewer/bin/mjpython"}
if [ ! -x "$viewer_python" ]; then
  echo "Set JEPA_GYM_PYTHON to a Python launcher with MuJoCo installed (mjpython on macOS)." >&2
  exit 1
fi
cd "$gym_root"
export PYTHONPATH="$gym_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$viewer_python" -m mujoco_compare.viewer "$@"
