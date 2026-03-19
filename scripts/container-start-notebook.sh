#!/usr/bin/env bash
set -euo pipefail

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_BIND_HOST="${JUPYTER_BIND_HOST:-127.0.0.1}"

mkdir -p /workspace/logs
if [[ -f /workspace/logs/jupyter.pid ]] && kill -0 "$(cat /workspace/logs/jupyter.pid)" 2>/dev/null; then
  echo "JupyterLab already running"
  exit 0
fi

pkill -f "jupyter-lab.*${JUPYTER_PORT}" || true
source /etc/profile.d/strix-halo-env.sh
cd /workspace
nohup jupyter lab \
  --ip=0.0.0.0 \
  --port="${JUPYTER_PORT}" \
  --allow-root \
  --ServerApp.allow_origin="http://${JUPYTER_BIND_HOST}:${JUPYTER_PORT}" \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.allow_remote_access=False \
  --ServerApp.root_dir=/workspace \
  --no-browser \
  > /workspace/logs/jupyter.log 2>&1 &
echo $! > /workspace/logs/jupyter.pid
sleep 5
tail -n 20 /workspace/logs/jupyter.log
