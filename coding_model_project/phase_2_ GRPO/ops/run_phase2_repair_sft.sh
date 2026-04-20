#!/bin/bash

set -euo pipefail

NPROC_PER_NODE=${1:-${NPROC_PER_NODE:-4}}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PHASE2_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_DIR=$(cd "${PHASE2_DIR}/.." && pwd)
VERL_DIR=$(cd "${PROJECT_DIR}/.." && pwd)

SFT_REPAIR_DIR="${PHASE2_DIR}/sft_repair_data"
REPAIR_DIR="${SFT_REPAIR_DIR}/v1"
ANCHOR_DIR="${SFT_REPAIR_DIR}/anchor_v1"
FINAL_DIR="${SFT_REPAIR_DIR}/final_v1"

REPAIR_TRAIN_FILE=${REPAIR_TRAIN_FILE:-"${REPAIR_DIR}/repair_sft_train_v1.parquet"}
REPAIR_VAL_FILE=${REPAIR_VAL_FILE:-"${REPAIR_DIR}/repair_sft_val_v1.parquet"}
ANCHOR_TRAIN_FILE=${ANCHOR_TRAIN_FILE:-"${ANCHOR_DIR}/anchor_train_v1.parquet"}
ANCHOR_VAL_FILE=${ANCHOR_VAL_FILE:-}

USE_ANCHOR=${USE_ANCHOR:-auto}
VAL_MODE=${VAL_MODE:-repair_only}
DISABLE_INTERNAL_VAL=${DISABLE_INTERNAL_VAL:-false}
REPAIR_ANCHOR_RATIO=${REPAIR_ANCHOR_RATIO:-1:1}
REBUILD_FINAL_V1=${REBUILD_FINAL_V1:-false}

MIX_TRAIN_FILE=${MIX_TRAIN_FILE:-"${FINAL_DIR}/phase2_sft_train_v1.parquet"}
MIX_VAL_FILE=${MIX_VAL_FILE:-"${FINAL_DIR}/phase2_sft_val_v1.parquet"}
MIX_MANIFEST_FILE=${MIX_MANIFEST_FILE:-"${FINAL_DIR}/phase2_mix_manifest_v1.json"}

STEP200_CKPT_DIR=${STEP200_CKPT_DIR:-}
MODEL_PATH=${MODEL_PATH:-}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase2_repair_sft_step200_r1a1_v1}
SAVE_DIR=${SAVE_DIR:-"${VERL_DIR}/checkpoints/rlvr_coding_model/${EXPERIMENT_NAME}"}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
DATA_MAX_LENGTH=${DATA_MAX_LENGTH:-4096}
TRUNCATION=${TRUNCATION:-error}
LR=${LR:-5e-7}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
SAVE_FREQ=${SAVE_FREQ:-4}
TEST_FREQ=${TEST_FREQ:-2}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-4}
SAVE_HF_MODEL=${SAVE_HF_MODEL:-true}
USE_WANDB=${USE_WANDB:-false}
SEED=${SEED:-42}
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-false}
FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-false}
FSDP_OFFLOAD_PARAMS=${FSDP_OFFLOAD_PARAMS:-false}

mkdir -p "${FINAL_DIR}" "${SAVE_DIR}"

has_hf_weights() {
    local model_dir="$1"
    [[ -f "${model_dir}/config.json" ]] || return 1
    find "${model_dir}" -maxdepth 1 -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) \
        -print -quit | grep -q .
}

checkpoint_ready() {
    local ckpt_root="$1"
    [[ -f "${ckpt_root}/data.pt" ]] || return 1
    [[ -f "${ckpt_root}/actor/fsdp_config.json" ]] || return 1
    [[ -f "${ckpt_root}/actor/huggingface/config.json" ]] || return 1
    local rank
    for rank in 0 1 2 3; do
        [[ -f "${ckpt_root}/actor/model_world_size_4_rank_${rank}.pt" ]] || return 1
    done
}

resolve_step200_ckpt_dir() {
    if [[ -n "${STEP200_CKPT_DIR}" ]]; then
        if ! checkpoint_ready "${STEP200_CKPT_DIR}"; then
            echo "ERROR: STEP200_CKPT_DIR is set but incomplete: ${STEP200_CKPT_DIR}" >&2
            exit 1
        fi
        echo "${STEP200_CKPT_DIR}"
        return 0
    fi

    local candidate
    local -a candidates=(
        "${VERL_DIR}/checkpoint_archives/grpo_a1_formal_observe_lb24_multi2_resume100_to200_seed0/global_step_200"
        "${VERL_DIR}/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0_resume120_keep3_to200/global_step_200"
        "${VERL_DIR}/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0/global_step_200"
        "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0_resume120_keep3_to200/global_step_200"
        "/workspace/verl/checkpoints/rlvr_coding_model/grpo_a1_formal_observe_lb24_multi2_resume100_to160_seed0/global_step_200"
    )

    for candidate in "${candidates[@]}"; do
        if checkpoint_ready "${candidate}"; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "ERROR: Could not resolve a valid step200 checkpoint. Set STEP200_CKPT_DIR or MODEL_PATH explicitly." >&2
    exit 1
}

resolve_model_path() {
    if [[ -n "${MODEL_PATH}" ]]; then
        if ! has_hf_weights "${MODEL_PATH}"; then
            echo "ERROR: MODEL_PATH does not look like a complete HF model dir: ${MODEL_PATH}" >&2
            exit 1
        fi
        echo "${MODEL_PATH}"
        return 0
    fi

    if [[ -n "${MERGED_MODEL_DIR}" ]] && has_hf_weights "${MERGED_MODEL_DIR}"; then
        echo "${MERGED_MODEL_DIR}"
        return 0
    fi

    local resolved_ckpt_dir
    resolved_ckpt_dir="$(resolve_step200_ckpt_dir)"

    local merged_model_dir
    merged_model_dir="${MERGED_MODEL_DIR:-"${resolved_ckpt_dir}/actor_merged_hf_phase2_sft"}"

    if ! has_hf_weights "${merged_model_dir}"; then
        echo "Merging RL actor checkpoint into HF model dir: ${merged_model_dir}" >&2
        cd "${VERL_DIR}"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${resolved_ckpt_dir}/actor" \
            --target_dir "${merged_model_dir}" \
            1>&2
    fi
    echo "${merged_model_dir}"
}

prepare_mix() {
    if [[ "${REBUILD_FINAL_V1}" == "false" ]]; then
        if [[ ! -f "${MIX_TRAIN_FILE}" ]]; then
            echo "ERROR: missing final_v1 train parquet: ${MIX_TRAIN_FILE}" >&2
            echo "Set REBUILD_FINAL_V1=true to rebuild from repair/anchor assets, or point MIX_TRAIN_FILE to an existing parquet." >&2
            exit 1
        fi
        if [[ "${DISABLE_INTERNAL_VAL}" != "true" && ! -f "${MIX_VAL_FILE}" ]]; then
            echo "ERROR: missing final_v1 val parquet: ${MIX_VAL_FILE}" >&2
            echo "Set REBUILD_FINAL_V1=true to rebuild from repair/anchor assets, or point MIX_VAL_FILE to an existing parquet." >&2
            exit 1
        fi
        if [[ ! -f "${MIX_MANIFEST_FILE}" ]]; then
            echo "ERROR: missing final_v1 mix manifest: ${MIX_MANIFEST_FILE}" >&2
            echo "Set REBUILD_FINAL_V1=true to rebuild from repair/anchor assets, or point MIX_MANIFEST_FILE to an existing manifest." >&2
            exit 1
        fi
        return 0
    fi

    if [[ ! -f "${REPAIR_TRAIN_FILE}" ]]; then
        echo "ERROR: missing repair train parquet: ${REPAIR_TRAIN_FILE}" >&2
        exit 1
    fi
    if [[ ! -f "${REPAIR_VAL_FILE}" ]]; then
        echo "ERROR: missing repair val parquet: ${REPAIR_VAL_FILE}" >&2
        exit 1
    fi

    local use_anchor_resolved=false
    local ratio_repair ratio_anchor
    IFS=':' read -r ratio_repair ratio_anchor <<< "${REPAIR_ANCHOR_RATIO}"
    if [[ -z "${ratio_repair}" || -z "${ratio_anchor}" ]]; then
        echo "ERROR: REPAIR_ANCHOR_RATIO must look like 2:1" >&2
        exit 1
    fi

    case "${USE_ANCHOR}" in
        true)
            if [[ ! -f "${ANCHOR_TRAIN_FILE}" ]]; then
                echo "ERROR: USE_ANCHOR=true but anchor parquet is missing: ${ANCHOR_TRAIN_FILE}" >&2
                exit 1
            fi
            use_anchor_resolved=true
            ;;
        false)
            use_anchor_resolved=false
            ;;
        auto)
            if [[ -f "${ANCHOR_TRAIN_FILE}" ]]; then
                use_anchor_resolved=true
            fi
            ;;
        *)
            echo "ERROR: USE_ANCHOR must be one of: auto, true, false" >&2
            exit 1
            ;;
    esac

    if [[ "${use_anchor_resolved}" != "true" ]]; then
        echo "ERROR: REBUILD_FINAL_V1=true currently expects anchor parquet to be available. Set USE_ANCHOR=true/auto with a valid ANCHOR_TRAIN_FILE, or reuse existing final_v1." >&2
        exit 1
    fi

    python3 "${SFT_REPAIR_DIR}/scripts/build_phase2_mixed_sft.py" \
        --repair-train "${REPAIR_TRAIN_FILE}" \
        --repair-val "${REPAIR_VAL_FILE}" \
        --anchor-train "${ANCHOR_TRAIN_FILE}" \
        --repair_weight "${ratio_repair}" \
        --anchor_weight "${ratio_anchor}" \
        --seed "${SEED}"
}

read_mix_stats() {
    python3 - "$MIX_MANIFEST_FILE" "$MIX_TRAIN_FILE" "$TRAIN_BATCH_SIZE" "$TOTAL_TRAINING_STEPS" "$NPROC_PER_NODE" <<'PY'
import json, math, sys
manifest_path, train_parquet_path, train_batch_size, total_training_steps, world_size = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
)
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

def read_parquet_rows(path):
    try:
        import pyarrow.parquet as pq
        return int(pq.read_table(path).num_rows)
    except Exception:
        pass
    try:
        import pandas as pd
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None

def get_train_rows_total(m):
    if "train_rows_total" in m:
        return int(m["train_rows_total"])
    if "train_effective_count" in m:
        return int(m["train_effective_count"])
    if "row_count" in m:
        return int(m["row_count"])
    return int(m["final_train_count"])

def get_train_rows_repair(m):
    if "train_rows_repair" in m:
        return int(m["train_rows_repair"])
    if "repair_unique_count" in m:
        return int(m["repair_unique_count"])
    if "row_count" in m:
        return int(m["row_count"])
    return int(m["repair_train_count"])

def get_train_rows_anchor(m):
    if "train_rows_anchor" in m:
        return int(m["train_rows_anchor"])
    if "anchor_effective_count" in m:
        return int(m["anchor_effective_count"])
    if "row_count" in m:
        return 0
    return int(m["anchor_selected_count"])

def get_val_rows_total(m):
    if "val_rows_total" in m:
        return int(m["val_rows_total"])
    if "row_count" in m:
        return 0
    return int(m["final_val_count"])

if train_batch_size % world_size != 0:
    raise SystemExit(
        f"Global train_batch_size={train_batch_size} must be divisible by world_size={world_size}."
    )

per_rank_batch = train_batch_size // world_size
train_rows_manifest = get_train_rows_total(manifest)
train_rows_parquet = read_parquet_rows(train_parquet_path)
train_rows = train_rows_parquet if train_rows_parquet is not None else train_rows_manifest

if train_rows % world_size != 0:
    sampler_num_samples = math.ceil((train_rows - world_size) / world_size)
else:
    sampler_num_samples = math.ceil(train_rows / world_size)
sampler_num_samples = max(1, sampler_num_samples)
steps_per_epoch = sampler_num_samples // per_rank_batch
if steps_per_epoch <= 0:
    raise SystemExit(
        f"Mixed train rows ({train_rows}) are too small for global train batch size ({train_batch_size}) "
        f"with world_size={world_size} when trainer/drop_last semantics are applied."
    )
total_epochs = max(1, math.ceil(total_training_steps / steps_per_epoch))
print(json.dumps({
    "train_rows_total": train_rows,
    "train_rows_manifest": train_rows_manifest,
    "train_rows_parquet": train_rows_parquet,
    "train_rows_repair": get_train_rows_repair(manifest),
    "train_rows_anchor": get_train_rows_anchor(manifest),
    "val_rows_total": get_val_rows_total(manifest),
    "per_rank_batch_size": per_rank_batch,
    "sampler_num_samples_per_rank": sampler_num_samples,
    "steps_per_epoch": steps_per_epoch,
    "total_epochs": total_epochs,
}))
PY
}

main() {
    local final_model_path
    final_model_path="$(resolve_model_path)"

    prepare_mix

    local mix_stats_json
    mix_stats_json="$(read_mix_stats)"

    local train_rows_total train_rows_manifest train_rows_parquet train_rows_repair train_rows_anchor val_rows_total per_rank_batch_size sampler_num_samples_per_rank steps_per_epoch total_epochs
    train_rows_total="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_rows_total"])' "${mix_stats_json}")"
    train_rows_manifest="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_rows_manifest"])' "${mix_stats_json}")"
    train_rows_parquet="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_rows_parquet"])' "${mix_stats_json}")"
    train_rows_repair="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_rows_repair"])' "${mix_stats_json}")"
    train_rows_anchor="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_rows_anchor"])' "${mix_stats_json}")"
    val_rows_total="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["val_rows_total"])' "${mix_stats_json}")"
    per_rank_batch_size="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["per_rank_batch_size"])' "${mix_stats_json}")"
    sampler_num_samples_per_rank="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["sampler_num_samples_per_rank"])' "${mix_stats_json}")"
    steps_per_epoch="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["steps_per_epoch"])' "${mix_stats_json}")"
    total_epochs="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["total_epochs"])' "${mix_stats_json}")"

    local logger_override save_contents_override val_files_override effective_test_freq
    if [[ "${USE_WANDB}" == "true" ]]; then
        logger_override='trainer.logger=["console","wandb"]'
    else
        logger_override='trainer.logger=["console"]'
    fi

    if [[ "${SAVE_HF_MODEL}" == "true" ]]; then
        save_contents_override='trainer.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'
    else
        save_contents_override='trainer.checkpoint.save_contents=["model","optimizer","extra"]'
    fi

    if [[ "${DISABLE_INTERNAL_VAL}" == "true" ]]; then
        val_files_override='data.val_files=[]'
        effective_test_freq=0
    else
        val_files_override="data.val_files=${MIX_VAL_FILE}"
        effective_test_freq="${TEST_FREQ}"
    fi

    echo "============================================"
    echo "Phase 2 Repair-SFT"
    echo "  GPUs: ${NPROC_PER_NODE}"
    echo "  Init model: ${final_model_path}"
    echo "  final_v1 manifest: ${MIX_MANIFEST_FILE}"
    echo "  Train rows (manifest): ${train_rows_manifest}"
    echo "  Train rows (parquet): ${train_rows_parquet}"
    echo "  Repair train rows: ${train_rows_repair}"
    echo "  Anchor train rows: ${train_rows_anchor}"
    echo "  Mixed train rows: ${train_rows_total}"
    echo "  Val rows: ${val_rows_total}"
    echo "  Internal val enabled: $([[ \"${DISABLE_INTERNAL_VAL}\" == \"true\" ]] && echo no || echo yes)"
    echo "  Sequence parallel size: ${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    echo "  Remove padding: ${USE_REMOVE_PADDING}"
    echo "  CPU offload: ${FSDP_CPU_OFFLOAD}"
    echo "  Offload params: ${FSDP_OFFLOAD_PARAMS}"
    echo "  Per-rank batch size: ${per_rank_batch_size}"
    echo "  Sampler rows/rank: ${sampler_num_samples_per_rank}"
    echo "  Steps/epoch: ${steps_per_epoch}"
    echo "  Total epochs: ${total_epochs}"
    echo "  Total training steps: ${TOTAL_TRAINING_STEPS}"
    echo "  Output dir: ${SAVE_DIR}"
    echo "============================================"

    cd "${VERL_DIR}"

    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
        -m verl.trainer.fsdp_sft_trainer \
        "data.train_files=${MIX_TRAIN_FILE}" \
        "${val_files_override}" \
        data.multiturn.enable=true \
        data.multiturn.messages_key=messages \
        "data.max_length=${DATA_MAX_LENGTH}" \
        "data.truncation=${TRUNCATION}" \
        "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
        "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}" \
        data.balance_dp_token=True \
        "model.partial_pretrain=${final_model_path}" \
        model.trust_remote_code=true \
        model.enable_gradient_checkpointing=true \
        model.strategy=fsdp2 \
        model.fsdp_config.model_dtype=bf16 \
        "model.fsdp_config.cpu_offload=${FSDP_CPU_OFFLOAD}" \
        "model.fsdp_config.offload_params=${FSDP_OFFLOAD_PARAMS}" \
        "optim.lr=${LR}" \
        'optim.betas=[0.9,0.95]' \
        optim.weight_decay=0.01 \
        "optim.lr_warmup_steps_ratio=${WARMUP_RATIO}" \
        optim.clip_grad=1.0 \
        optim.lr_scheduler=cosine \
        "ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
        "use_remove_padding=${USE_REMOVE_PADDING}" \
        "trainer.default_local_dir=${SAVE_DIR}" \
        trainer.project_name=rlvr_coding_model \
        "trainer.experiment_name=${EXPERIMENT_NAME}" \
        "trainer.total_epochs=${total_epochs}" \
        "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
        "trainer.save_freq=${SAVE_FREQ}" \
        "trainer.test_freq=${effective_test_freq}" \
        "trainer.n_gpus_per_node=${NPROC_PER_NODE}" \
        "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}" \
        "trainer.seed=${SEED}" \
        trainer.resume_mode=disable \
        "${logger_override}" \
        "${save_contents_override}"
}

main "$@"
