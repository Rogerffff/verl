#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

LISTEN_PORT=${LISTEN_PORT:-8090}
BASE_PORT=${BASE_PORT:-8081}
MAX_BACKENDS=${MAX_BACKENDS:-4}
PID_ROOT=${PID_ROOT:-"$(default_pid_root)"}
STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
TEMPLATE_PATH=${TEMPLATE_PATH:-"$SCRIPT_DIR/nginx_sandbox_lb.conf.template"}
OUTPUT_PATH=${OUTPUT_PATH:-"$SCRIPT_DIR/generated/nginx_sandbox_lb.conf"}
ACCESS_LOG=${ACCESS_LOG:-"$STATE_ROOT/nginx/sandbox_lb.access.log"}
ERROR_LOG=${ERROR_LOG:-"$STATE_ROOT/nginx/sandbox_lb.error.log"}
PID_FILE=${PID_FILE:-"$STATE_ROOT/nginx/nginx.pid"}
BACKEND_PORTS_CSV=${BACKEND_PORTS_CSV:-}
NGINX_TEST=${NGINX_TEST:-false}

ensure_dir "$(dirname "$OUTPUT_PATH")"
ensure_dir "$(dirname "$ACCESS_LOG")"
ensure_dir "$(dirname "$ERROR_LOG")"
ensure_dir "$(dirname "$PID_FILE")"

declare -a backend_ports=()
if [[ -n "$BACKEND_PORTS_CSV" ]]; then
    IFS=',' read -r -a backend_ports <<<"$BACKEND_PORTS_CSV"
else
    while IFS= read -r port; do
        [[ -n "$port" ]] && backend_ports+=("$port")
    done < <(active_backend_ports "$PID_ROOT" "$BASE_PORT" "$MAX_BACKENDS")
fi

if (( ${#backend_ports[@]} == 0 )); then
    echo "No backend ports available for Nginx upstream rendering." >&2
    exit 1
fi

server_lines=()
for port in "${backend_ports[@]}"; do
    server_lines+=("        server 127.0.0.1:${port};")
done
printf -v upstream_servers '%s\n' "${server_lines[@]}"

python3 - "$TEMPLATE_PATH" "$OUTPUT_PATH" "$LISTEN_PORT" "$ACCESS_LOG" "$ERROR_LOG" "$PID_FILE" "$upstream_servers" <<'PY'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
listen_port = sys.argv[3]
access_log = sys.argv[4]
error_log = sys.argv[5]
pid_file = sys.argv[6]
upstream_servers = sys.argv[7]

text = template_path.read_text(encoding="utf-8")
text = text.replace("__LISTEN_PORT__", listen_port)
text = text.replace("__ACCESS_LOG__", access_log)
text = text.replace("__ERROR_LOG__", error_log)
text = text.replace("__PID_FILE__", pid_file)
text = text.replace("__UPSTREAM_SERVERS__", upstream_servers.rstrip("\n"))
output_path.write_text(text, encoding="utf-8")
PY

echo "Rendered Nginx config to $OUTPUT_PATH"
echo "Upstream backends: ${backend_ports[*]}"

if [[ "$NGINX_TEST" == "true" ]]; then
    nginx -t -c "$OUTPUT_PATH"
fi
