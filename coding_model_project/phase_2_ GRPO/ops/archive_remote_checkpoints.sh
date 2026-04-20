#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-vastai2}
BASE_DIR=${BASE_DIR:-/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume10_seed0}
RUN_NAME=${RUN_NAME:-grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0}
RUN_DIR=${RUN_DIR:-/workspace/verl/checkpoints/rlvr_coding_model/$RUN_NAME}
LOCAL_ROOT=${LOCAL_ROOT:-/Users/roger/Desktop/coding_RL_project/verl/checkpoint_archives/grpo_a1_formal_observe_lb24_multi2_resume100_to200_seed0}
ARCHIVE_LOG=${ARCHIVE_LOG:-$LOCAL_ROOT/archive_remote_checkpoints.log}
ARCHIVE_STEPS=${ARCHIVE_STEPS:-100,120,140,160,180,200}
REMOTE_DELETE_STEP100_FREE_BELOW_GB=${REMOTE_DELETE_STEP100_FREE_BELOW_GB:-180}

mkdir -p "$LOCAL_ROOT"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$ARCHIVE_LOG"
}

remote_step_dir() {
    local step="$1"
    if [[ "$step" == "100" ]]; then
        printf '%s/global_step_%s' "$BASE_DIR" "$step"
    else
        printf '%s/global_step_%s' "$RUN_DIR" "$step"
    fi
}

remote_step_stable() {
    local step="$1"
    local dir
    dir="$(remote_step_dir "$step")"
    local snapshot1 snapshot2
    snapshot1="$(ssh -o ClearAllForwardings=yes "$REMOTE_HOST" "if [ -d '$dir' ]; then size=\$(du -sb '$dir' | awk '{print \$1}'); files=\$(find '$dir' -type f | wc -l); latest=\$(cat '$RUN_DIR/latest_checkpointed_iteration.txt' 2>/dev/null || echo none); echo \$size \$files \$latest; else echo missing; fi")"
    if [[ "$snapshot1" == "missing" ]]; then
        return 1
    fi
    sleep 30
    snapshot2="$(ssh -o ClearAllForwardings=yes "$REMOTE_HOST" "if [ -d '$dir' ]; then size=\$(du -sb '$dir' | awk '{print \$1}'); files=\$(find '$dir' -type f | wc -l); latest=\$(cat '$RUN_DIR/latest_checkpointed_iteration.txt' 2>/dev/null || echo none); echo \$size \$files \$latest; else echo missing; fi")"
    [[ "$snapshot1" == "$snapshot2" ]]
}

archive_step() {
    local step="$1"
    local remote_dir local_dir tmp_parent
    remote_dir="$(remote_step_dir "$step")"
    local_dir="$LOCAL_ROOT/global_step_$step"
    tmp_parent="$LOCAL_ROOT/.partial"

    if [[ -f "$local_dir/.archived" ]]; then
        return 0
    fi

    if ! remote_step_stable "$step"; then
        return 1
    fi

    log "archiving step $step from $remote_dir"
    mkdir -p "$tmp_parent"
    rm -rf "$tmp_parent/global_step_$step"
    ssh -o ClearAllForwardings=yes "$REMOTE_HOST" \
        "cd '$(dirname "$remote_dir")' && tar cf - '$(basename "$remote_dir")'" \
        | tar xf - -C "$tmp_parent"
    rm -rf "$local_dir"
    mv "$tmp_parent/global_step_$step" "$local_dir"
    touch "$local_dir/.archived"
    du -sh "$local_dir" | tee -a "$ARCHIVE_LOG"
    log "archive complete for step $step"
    return 0
}

remote_free_gb() {
    ssh -o ClearAllForwardings=yes "$REMOTE_HOST" "python3 - <<'PY'
import shutil
print(shutil.disk_usage('/workspace').free / (1024**3))
PY"
}

maybe_delete_remote_step100() {
    local local_dir="$LOCAL_ROOT/global_step_100"
    if [[ ! -f "$local_dir/.archived" ]]; then
        return 0
    fi
    local free_gb
    free_gb="$(remote_free_gb | tail -n 1)"
    if awk "BEGIN {exit !($free_gb < $REMOTE_DELETE_STEP100_FREE_BELOW_GB)}"; then
        local remote_dir="$BASE_DIR/global_step_100"
        if ssh -o ClearAllForwardings=yes "$REMOTE_HOST" "[ -d '$remote_dir' ]"; then
            log "remote free ${free_gb} GiB is below ${REMOTE_DELETE_STEP100_FREE_BELOW_GB}; deleting remote step100 after verified local archive"
            ssh -o ClearAllForwardings=yes "$REMOTE_HOST" "rm -rf '$remote_dir' && df -h /workspace"
        fi
    fi
}

main() {
    log "archive watcher started for steps: $ARCHIVE_STEPS"
    IFS=',' read -r -a steps <<< "$ARCHIVE_STEPS"
    while true; do
        local all_done=1
        for step in "${steps[@]}"; do
            if [[ ! -f "$LOCAL_ROOT/global_step_${step}/.archived" ]]; then
                all_done=0
                archive_step "$step" || true
            fi
        done
        maybe_delete_remote_step100 || true
        if [[ "$all_done" == "1" ]]; then
            log "all requested checkpoints archived"
            break
        fi
        sleep 60
    done
}

main "$@"
