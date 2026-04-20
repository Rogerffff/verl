#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/sandbox_backend_common.sh"

STATE_ROOT=${STATE_ROOT:-"$(default_state_root)"}
CONFIG_PATH=${CONFIG_PATH:-"$SCRIPT_DIR/generated/nginx_sandbox_lb.conf"}
PID_FILE=${PID_FILE:-"$STATE_ROOT/nginx/nginx.pid"}
NGINX_BIN=${NGINX_BIN:-nginx}
POST_APPLY_SETTLE_SECONDS=${POST_APPLY_SETTLE_SECONDS:-2}

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Nginx config not found at $CONFIG_PATH" >&2
    exit 1
fi

if ! command -v "$NGINX_BIN" >/dev/null 2>&1; then
    echo "nginx binary not found: $NGINX_BIN" >&2
    exit 1
fi

"$NGINX_BIN" -t -c "$CONFIG_PATH"

nginx_pid=""
if [[ -s "$PID_FILE" ]]; then
    nginx_pid="$(<"$PID_FILE")"
fi

if [[ -n "$nginx_pid" ]] && kill -0 "$nginx_pid" >/dev/null 2>&1; then
    "$NGINX_BIN" -s reload -c "$CONFIG_PATH"
    echo "Reloaded Nginx using $CONFIG_PATH"
else
    rm -f "$PID_FILE"
    "$NGINX_BIN" -c "$CONFIG_PATH"
    echo "Started Nginx using $CONFIG_PATH"
fi

if [[ "$POST_APPLY_SETTLE_SECONDS" != "0" ]]; then
    sleep "$POST_APPLY_SETTLE_SECONDS"
    echo "Waited ${POST_APPLY_SETTLE_SECONDS}s for Nginx workers to settle"
fi
