from __future__ import annotations

import base64
import json
import pickle
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


BUCKETS = ("U_unseen", "A_retention", "B_near_miss", "C_hard_partial", "D_dead_hard")
BUCKET_RANK = {
    "U_unseen": 0,
    "D_dead_hard": 1,
    "C_hard_partial": 2,
    "B_near_miss": 3,
    "A_retention": 4,
}
FALLBACK_ORDER = ("B_near_miss", "A_retention", "U_unseen", "C_hard_partial", "D_dead_hard")
DEFAULT_PHASE_QUOTAS = (
    {"U_unseen": 8, "A_retention": 3, "B_near_miss": 3, "C_hard_partial": 1, "D_dead_hard": 1},
    {"U_unseen": 3, "A_retention": 4, "B_near_miss": 6, "C_hard_partial": 2, "D_dead_hard": 1},
    {"U_unseen": 2, "A_retention": 5, "B_near_miss": 6, "C_hard_partial": 2, "D_dead_hard": 1},
)


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _encode_rng_state(rng_state: object) -> str:
    return base64.b64encode(pickle.dumps(rng_state)).decode("ascii")


def _decode_rng_state(payload: str) -> object:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


def _normalize_phase_quota(raw: Any, default_quota: dict[str, int], batch_size: int) -> dict[str, int]:
    if raw is None:
        quota = dict(default_quota)
    else:
        quota_payload = json.loads(raw) if isinstance(raw, str) else raw
        quota = {}
        for bucket in BUCKETS:
            if bucket not in quota_payload:
                raise ValueError(f"Curriculum phase quota missing bucket {bucket}")
            quota[bucket] = int(quota_payload[bucket])
    quota_sum = sum(quota.values())
    if quota_sum != batch_size:
        raise ValueError(f"Curriculum phase quota sum {quota_sum} does not match batch size {batch_size}")
    return quota


class Step580CurriculumSampler(AbstractCurriculumSampler):
    """Dynamic curriculum sampler for the step580 pilot."""

    def __init__(self, data_source, data_config):
        self.data_source = data_source
        self.data_config = data_config
        curriculum_cfg = data_config.get("curriculum", {})
        self.state_dir = Path(str(curriculum_cfg.get("state_dir") or Path.cwd() / "curriculum_state"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        resume_state_path = curriculum_cfg.get("resume_state_path")
        self.resume_state_path = Path(str(resume_state_path)) if resume_state_path else None

        self.start_global_step = int(curriculum_cfg.get("start_global_step", 580))
        self.phase_boundaries = list(curriculum_cfg.get("phase_boundaries", [20, 60, 80]))
        self.snapshot_steps = set(curriculum_cfg.get("snapshot_steps", [600, 620, 640, 660]))
        self.ema_alpha = float(curriculum_cfg.get("ema_alpha", 0.4))
        self.min_visits_for_online = int(curriculum_cfg.get("min_visits_for_online", 2))
        self.recent_exclusion_window = int(curriculum_cfg.get("recent_exclusion_window", 2))
        self.reset_state_on_dataset_mismatch = bool(curriculum_cfg.get("reset_state_on_dataset_mismatch", False))
        self.batch_size = int(data_config.get("gen_batch_size", data_config.get("train_batch_size", 16)))
        self.phase_quotas = [
            _normalize_phase_quota(curriculum_cfg.get("phase0_quotas"), DEFAULT_PHASE_QUOTAS[0], self.batch_size),
            _normalize_phase_quota(curriculum_cfg.get("phase1_quotas"), DEFAULT_PHASE_QUOTAS[1], self.batch_size),
            _normalize_phase_quota(curriculum_cfg.get("phase2_quotas"), DEFAULT_PHASE_QUOTAS[2], self.batch_size),
        ]
        self.phase_u_revisit_quotas = [
            int(curriculum_cfg.get("phase0_u_revisit_quota", 0)),
            int(curriculum_cfg.get("phase1_u_revisit_quota", 0)),
            int(curriculum_cfg.get("phase2_u_revisit_quota", 0)),
        ]
        for phase_idx, revisit_quota in enumerate(self.phase_u_revisit_quotas):
            unseen_quota = self.phase_quotas[phase_idx]["U_unseen"]
            if revisit_quota < 0 or revisit_quota > unseen_quota:
                raise ValueError(
                    "Curriculum U_revisit quota must be between 0 and the U_unseen quota "
                    f"(phase={phase_idx}, revisit={revisit_quota}, unseen={unseen_quota})"
                )
        self.prefer_low_visits = bool(curriculum_cfg.get("prefer_low_visits", False))
        self.snapshot_loaded = False
        self._loaded_sampler_state_echo = None

        seed = data_config.get("seed")
        self.rng = random.Random(seed if seed is not None else 0)

        self.local_update_step = 0
        self.recent_index_queue: deque[list[int]] = deque(maxlen=self.recent_exclusion_window)
        self.sampled_counts: Counter[str] = Counter()
        self.promotion_counts: Counter[str] = Counter()
        self.demotion_counts: Counter[str] = Counter()
        self.invalid_group_skip_count = 0
        self.u_first_touch_sampled_count = 0
        self.u_revisit_sampled_count = 0
        self.u_to_bucket_counts: Counter[str] = Counter()

        if not hasattr(data_source, "curriculum_initial_bucket_by_index"):
            raise TypeError("Step580CurriculumSampler requires Step580CurriculumDataset-compatible data_source")

        self.index_states: list[dict[str, Any]] = []
        self.problem_id_to_indices: dict[str, list[int]] = defaultdict(list)
        self.dataset_problem_to_index: dict[tuple[str, str], int] = {}
        for key, seed_bucket in zip(data_source.curriculum_key_by_index, data_source.curriculum_initial_bucket_by_index, strict=True):
            index = len(self.index_states)
            self.index_states.append(self._make_default_state(key, seed_bucket))
            self.problem_id_to_indices[key[1]].append(index)
            self.dataset_problem_to_index[key] = index
        self.bucket_counts: Counter[str] = Counter(state["bucket"] for state in self.index_states)

    def _make_default_state(self, key: tuple[str, str], seed_bucket: str) -> dict[str, Any]:
        bucket = seed_bucket if seed_bucket in BUCKETS else "U_unseen"
        return {
            "dataset": key[0],
            "problem_id": key[1],
            "bucket": bucket,
            "seed_bucket": bucket,
            "visits": 0,
            "ema_group_pass_ratio": 0.0,
            "ema_group_accept_rate": 0.0,
            "ema_runtime_error_rate": 0.0,
            "ema_timeout_rate": 0.0,
            "dominant_error_type": "",
            "last_seen_update_step": -1,
        }

    def _reset_index_states_from_current_seed_buckets(self) -> None:
        self.index_states = [
            self._make_default_state(key, seed_bucket)
            for key, seed_bucket in zip(
                self.data_source.curriculum_key_by_index,
                self.data_source.curriculum_initial_bucket_by_index,
                strict=True,
            )
        ]
        self.bucket_counts = Counter(state["bucket"] for state in self.index_states)
        self.sampled_counts = Counter()
        self.promotion_counts = Counter()
        self.demotion_counts = Counter()
        self.invalid_group_skip_count = 0
        self.u_first_touch_sampled_count = 0
        self.u_revisit_sampled_count = 0
        self.u_to_bucket_counts = Counter()
        self.recent_index_queue = deque(maxlen=self.recent_exclusion_window)

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, Any]:
        return {
            # Checkpoints are saved before sampler.update(batch), so the matching
            # post-label curriculum snapshot is always one label ahead of the
            # currently completed local update count.
            "snapshot_step_echo": self.start_global_step + self.local_update_step + 1,
            "sampler_type": self.__class__.__name__,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loaded_sampler_state_echo = state_dict.get("snapshot_step_echo")

    def __iter__(self):
        self._ensure_resume_snapshot_loaded()
        yielded = 0
        total = len(self.data_source)
        while yielded < total:
            batch_indices = self._sample_batch_indices()
            for index in batch_indices:
                yield index
                yielded += 1
                if yielded >= total:
                    break

    def _ensure_resume_snapshot_loaded(self) -> None:
        if self.resume_state_path is None or self.snapshot_loaded:
            return
        self.load_snapshot(self.resume_state_path)
        self.snapshot_loaded = True
        print(
            "CURRICULUM_SNAPSHOT_LOADED "
            f"step={self.start_global_step + self.local_update_step} local_update_step={self.local_update_step}"
        )

    def _phase_config(self) -> tuple[dict[str, int], int]:
        if self.local_update_step < self.phase_boundaries[0]:
            return dict(self.phase_quotas[0]), self.phase_u_revisit_quotas[0]
        if self.local_update_step < self.phase_boundaries[1]:
            return dict(self.phase_quotas[1]), self.phase_u_revisit_quotas[1]
        return dict(self.phase_quotas[2]), self.phase_u_revisit_quotas[2]

    def _recent_exclusion_set(self) -> set[int]:
        recent: set[int] = set()
        for batch_indices in self.recent_index_queue:
            recent.update(batch_indices)
        return recent

    def _candidate_indices(
        self,
        bucket_names: list[str],
        *,
        chosen: set[int],
        respect_recent: bool,
        predicate: Any = None,
    ) -> list[int]:
        recent = self._recent_exclusion_set() if respect_recent else set()
        out: list[int] = []
        for index, state in enumerate(self.index_states):
            if state["bucket"] not in bucket_names:
                continue
            if index in chosen or index in recent:
                continue
            if predicate is not None and not predicate(index, state):
                continue
            out.append(index)
        return out

    def _order_candidates(self, candidates: list[int]) -> list[int]:
        if not self.prefer_low_visits:
            shuffled = list(candidates)
            self.rng.shuffle(shuffled)
            return shuffled
        ranked = sorted(candidates, key=lambda idx: (int(self.index_states[idx]["visits"]), self.rng.random()))
        return ranked

    def _pick_from_buckets(
        self,
        bucket_names: list[str],
        need: int,
        *,
        chosen: set[int],
        respect_recent: bool,
        predicate: Any = None,
    ) -> list[int]:
        if need <= 0:
            return []
        candidates = self._candidate_indices(
            bucket_names,
            chosen=chosen,
            respect_recent=respect_recent,
            predicate=predicate,
        )
        if not candidates:
            return []
        ordered = self._order_candidates(candidates)
        take = min(need, len(ordered))
        return ordered[:take]

    def _sample_for_quota(self, primary_bucket: str, quota: int, chosen: set[int]) -> list[int]:
        picked: list[int] = []
        fallback_buckets = [bucket for bucket in FALLBACK_ORDER if bucket != primary_bucket]
        for respect_recent in (True, False):
            if len(picked) < quota:
                primary = self._pick_from_buckets(
                    [primary_bucket],
                    quota - len(picked),
                    chosen=chosen | set(picked),
                    respect_recent=respect_recent,
                )
                picked.extend(primary)
            if len(picked) < quota:
                fallback = self._pick_from_buckets(
                    fallback_buckets,
                    quota - len(picked),
                    chosen=chosen | set(picked),
                    respect_recent=respect_recent,
                )
                picked.extend(fallback)
        return picked

    def _sample_unseen_quota(self, quota: int, chosen: set[int], *, revisit_quota: int) -> list[int]:
        picked: list[int] = []
        fallback_buckets = [bucket for bucket in FALLBACK_ORDER if bucket != "U_unseen"]
        revisit_quota = min(quota, revisit_quota)
        for respect_recent in (True, False):
            if len(picked) < revisit_quota:
                revisit = self._pick_from_buckets(
                    ["U_unseen"],
                    revisit_quota - len(picked),
                    chosen=chosen | set(picked),
                    respect_recent=respect_recent,
                    predicate=lambda _idx, state: int(state["visits"]) == 1,
                )
                picked.extend(revisit)
            if len(picked) < quota:
                first_touch = self._pick_from_buckets(
                    ["U_unseen"],
                    quota - len(picked),
                    chosen=chosen | set(picked),
                    respect_recent=respect_recent,
                    predicate=lambda _idx, state: int(state["visits"]) == 0,
                )
                picked.extend(first_touch)
            if len(picked) < quota:
                any_unseen = self._pick_from_buckets(
                    ["U_unseen"],
                    quota - len(picked),
                    chosen=chosen | set(picked),
                    respect_recent=respect_recent,
                )
                picked.extend(any_unseen)
            if len(picked) < quota:
                fallback = self._pick_from_buckets(
                    fallback_buckets,
                    quota - len(picked),
                    chosen=chosen | set(picked),
                    respect_recent=respect_recent,
                )
                picked.extend(fallback)
        return picked

    def _record_pre_sample_observability(self, indices: list[int]) -> None:
        for index in indices:
            state = self.index_states[index]
            if state["bucket"] != "U_unseen":
                continue
            visits = int(state["visits"])
            if visits <= 0:
                self.u_first_touch_sampled_count += 1
            elif visits == 1:
                self.u_revisit_sampled_count += 1

    def _sample_batch_indices(self) -> list[int]:
        quotas, unseen_revisit_quota = self._phase_config()
        chosen: list[int] = []
        chosen_set: set[int] = set()
        unseen_batch_slice = self._sample_unseen_quota(
            quotas["U_unseen"],
            chosen_set,
            revisit_quota=unseen_revisit_quota,
        )
        chosen.extend(unseen_batch_slice)
        chosen_set.update(unseen_batch_slice)
        self._record_pre_sample_observability(unseen_batch_slice)

        for bucket in ("A_retention", "B_near_miss", "C_hard_partial", "D_dead_hard"):
            batch_slice = self._sample_for_quota(bucket, quotas[bucket], chosen_set)
            chosen.extend(batch_slice)
            chosen_set.update(batch_slice)
            self._record_pre_sample_observability(batch_slice)

        if len(chosen) != self.batch_size:
            raise RuntimeError(
                f"Curriculum sampler could not fill a full batch: got {len(chosen)} expected {self.batch_size}"
            )

        self.recent_index_queue.append(list(chosen))
        for index in chosen:
            self.sampled_counts[self.index_states[index]["bucket"]] += 1
        return chosen

    def _bucket_for_state(self, state: dict[str, Any]) -> str:
        visits = int(state["visits"])
        pass_ratio = float(state["ema_group_pass_ratio"])
        accept_rate = float(state["ema_group_accept_rate"])
        dominant_error = str(state["dominant_error_type"])
        if visits < self.min_visits_for_online:
            return state["seed_bucket"]
        if accept_rate >= 0.25 or pass_ratio >= 0.85:
            return "A_retention"
        if 0.35 <= pass_ratio < 0.85 and dominant_error not in {"runtime_error", "timeout"}:
            return "B_near_miss"
        if pass_ratio < 0.05:
            return "D_dead_hard"
        return "C_hard_partial"

    def _update_ema(self, old: float, new: float, *, visits: int) -> float:
        if visits <= 1:
            return new
        alpha = self.ema_alpha
        return alpha * new + (1.0 - alpha) * old

    def _most_common_error(self, error_types: list[str]) -> str:
        if not error_types:
            return ""
        counts = Counter(error_types)
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _resolve_indices(self, non_tensor: dict[str, Any], expected_len: int) -> list[int]:
        if "index" in non_tensor:
            return [int(value) for value in _to_list(non_tensor["index"])]

        if "problem_id" not in non_tensor:
            raise KeyError("Curriculum sampler requires either non_tensor['index'] or non_tensor['problem_id']")

        problem_ids = [str(value) for value in _to_list(non_tensor["problem_id"])]
        if len(problem_ids) != expected_len:
            raise RuntimeError(
                f"Curriculum sampler got {len(problem_ids)} problem_ids for {expected_len} uid entries"
            )

        datasets = None
        if "dataset" in non_tensor:
            datasets = [str(value) for value in _to_list(non_tensor["dataset"])]
            if len(datasets) != expected_len:
                raise RuntimeError(
                    f"Curriculum sampler got {len(datasets)} dataset values for {expected_len} uid entries"
                )

        indices: list[int] = []
        for offset, problem_id in enumerate(problem_ids):
            if datasets is not None:
                key = (datasets[offset], problem_id)
                if key not in self.dataset_problem_to_index:
                    raise KeyError(f"Curriculum sampler could not resolve dataset/problem_id key {key}")
                indices.append(self.dataset_problem_to_index[key])
                continue

            candidate_indices = self.problem_id_to_indices.get(problem_id, [])
            if len(candidate_indices) != 1:
                raise KeyError(
                    "Curriculum sampler could not uniquely resolve problem_id "
                    f"{problem_id!r}; candidates={candidate_indices}"
                )
            indices.append(candidate_indices[0])
        return indices

    def update(self, batch: DataProto) -> None:
        non_tensor = batch.non_tensor_batch
        uids = _to_list(non_tensor["uid"])
        indices = self._resolve_indices(non_tensor, len(uids))
        pass_ratios = [float(value) for value in _to_list(non_tensor["pass_ratio_all"])]
        accepted = [bool(value) for value in _to_list(non_tensor["accepted"])]
        error_types = [str(value) for value in _to_list(non_tensor["error_type"])]
        invalid_flags = [bool(value) for value in _to_list(non_tensor.get("invalid_for_rl", [False] * len(uids)))]

        groups: dict[str, list[int]] = defaultdict(list)
        for offset, uid in enumerate(uids):
            groups[str(uid)].append(offset)

        for offsets in groups.values():
            unique_indices = {indices[offset] for offset in offsets}
            if len(unique_indices) != 1:
                raise RuntimeError(f"Curriculum sampler expected one dataset index per uid group, got {unique_indices}")
            index = next(iter(unique_indices))
            valid_offsets = [offset for offset in offsets if not invalid_flags[offset]]
            if not valid_offsets:
                self.invalid_group_skip_count += 1
                continue

            group_pass_ratio = sum(pass_ratios[offset] for offset in valid_offsets) / len(valid_offsets)
            group_accept_rate = sum(1.0 for offset in valid_offsets if accepted[offset]) / len(valid_offsets)
            group_runtime_error_rate = (
                sum(1.0 for offset in valid_offsets if error_types[offset] == "runtime_error") / len(valid_offsets)
            )
            group_timeout_rate = (
                sum(1.0 for offset in valid_offsets if error_types[offset] == "timeout") / len(valid_offsets)
            )
            dominant_error_type = self._most_common_error([error_types[offset] for offset in valid_offsets])

            state = self.index_states[index]
            old_bucket = state["bucket"]
            state["visits"] += 1
            state["ema_group_pass_ratio"] = self._update_ema(
                float(state["ema_group_pass_ratio"]),
                group_pass_ratio,
                visits=state["visits"],
            )
            state["ema_group_accept_rate"] = self._update_ema(
                float(state["ema_group_accept_rate"]),
                group_accept_rate,
                visits=state["visits"],
            )
            state["ema_runtime_error_rate"] = self._update_ema(
                float(state["ema_runtime_error_rate"]),
                group_runtime_error_rate,
                visits=state["visits"],
            )
            state["ema_timeout_rate"] = self._update_ema(
                float(state["ema_timeout_rate"]),
                group_timeout_rate,
                visits=state["visits"],
            )
            state["dominant_error_type"] = dominant_error_type
            state["last_seen_update_step"] = self.local_update_step

            new_bucket = self._bucket_for_state(state)
            if new_bucket != old_bucket:
                self.bucket_counts[old_bucket] -= 1
                self.bucket_counts[new_bucket] += 1
                state["bucket"] = new_bucket
                transition = f"{old_bucket}->{new_bucket}"
                if old_bucket == "U_unseen" and new_bucket in {"A_retention", "B_near_miss", "C_hard_partial", "D_dead_hard"}:
                    self.u_to_bucket_counts[new_bucket] += 1
                if BUCKET_RANK[new_bucket] > BUCKET_RANK[old_bucket]:
                    self.promotion_counts[transition] += 1
                else:
                    self.demotion_counts[transition] += 1

        self.local_update_step += 1
        completed_label_step = self.start_global_step + self.local_update_step
        if completed_label_step in self.snapshot_steps:
            self.save_snapshot(completed_label_step)

    def _snapshot_payload(self, step: int) -> dict[str, Any]:
        return {
            "global_step": step,
            "local_update_step": self.local_update_step,
            "bucket_counts": {bucket: self.bucket_counts.get(bucket, 0) for bucket in BUCKETS},
            "sampled_counts": dict(self.sampled_counts),
            "promotion_counts": dict(self.promotion_counts),
            "demotion_counts": dict(self.demotion_counts),
            "invalid_group_skip_count": self.invalid_group_skip_count,
            "unseen_coverage": 1.0 - (self.bucket_counts.get("U_unseen", 0) / max(len(self.index_states), 1)),
            "u_first_touch_sampled_count": self.u_first_touch_sampled_count,
            "u_revisit_sampled_count": self.u_revisit_sampled_count,
            "u_to_A": self.u_to_bucket_counts.get("A_retention", 0),
            "u_to_B": self.u_to_bucket_counts.get("B_near_miss", 0),
            "u_to_C": self.u_to_bucket_counts.get("C_hard_partial", 0),
            "u_to_D": self.u_to_bucket_counts.get("D_dead_hard", 0),
            "u_visit1_backlog": sum(
                1
                for state in self.index_states
                if state["bucket"] == "U_unseen" and int(state["visits"]) == 1
            ),
            "recent_index_queue": list(self.recent_index_queue),
            "rng_state": _encode_rng_state(self.rng.getstate()),
            "index_keys": list(self.data_source.curriculum_key_by_index),
            "index_states": self.index_states,
        }

    def save_snapshot(self, step: int) -> Path:
        path = self.state_dir / f"curriculum_state_step_{step}.json"
        tmp_path = path.with_suffix(".json.tmp")
        payload = self._snapshot_payload(step)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return path

    def load_snapshot(self, path: str | Path) -> None:
        snapshot_path = Path(path)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Curriculum snapshot not found: {snapshot_path}")
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        expected_local_update_step = int(payload["global_step"]) - self.start_global_step
        if int(payload["local_update_step"]) != expected_local_update_step:
            raise RuntimeError(
                "Curriculum snapshot local_update_step does not match global_step "
                f"({payload['local_update_step']} != {expected_local_update_step})"
            )
        resume_checkpoint_step = self.data_config.get("curriculum", {}).get("resume_checkpoint_step")
        if resume_checkpoint_step is not None and int(payload["global_step"]) != int(resume_checkpoint_step):
            raise RuntimeError(
                "Curriculum snapshot global_step does not match resume checkpoint step "
                f"({payload['global_step']} != {resume_checkpoint_step})"
            )
        if self._loaded_sampler_state_echo is not None and self._loaded_sampler_state_echo != payload["global_step"]:
            raise RuntimeError(
                "Sampler dataloader echo does not match curriculum snapshot "
                f"({self._loaded_sampler_state_echo} != {payload['global_step']})"
            )

        self.local_update_step = int(payload["local_update_step"])
        self.bucket_counts = Counter(payload["bucket_counts"])
        self.sampled_counts = Counter(payload.get("sampled_counts", {}))
        self.promotion_counts = Counter(payload.get("promotion_counts", {}))
        self.demotion_counts = Counter(payload.get("demotion_counts", {}))
        self.invalid_group_skip_count = int(payload.get("invalid_group_skip_count", 0))
        self.u_first_touch_sampled_count = int(payload.get("u_first_touch_sampled_count", 0))
        self.u_revisit_sampled_count = int(payload.get("u_revisit_sampled_count", 0))
        self.u_to_bucket_counts = Counter(
            {
                "A_retention": int(payload.get("u_to_A", 0)),
                "B_near_miss": int(payload.get("u_to_B", 0)),
                "C_hard_partial": int(payload.get("u_to_C", 0)),
                "D_dead_hard": int(payload.get("u_to_D", 0)),
            }
        )
        self.recent_index_queue = deque(
            [list(batch_indices) for batch_indices in payload.get("recent_index_queue", [])],
            maxlen=self.recent_exclusion_window,
        )
        self.rng.setstate(_decode_rng_state(payload["rng_state"]))

        current_keys = list(self.data_source.curriculum_key_by_index)
        payload_index_states = list(payload["index_states"])
        payload_index_keys = payload.get("index_keys")

        if len(payload_index_states) == len(current_keys):
            self.index_states = payload_index_states
            self._overlay_current_seed_buckets()
            return

        if payload_index_keys and len(payload_index_keys) == len(payload_index_states):
            payload_state_by_key = {
                (str(key[0]), str(key[1])): dict(state)
                for key, state in zip(payload_index_keys, payload_index_states, strict=True)
            }
            remapped_states: list[dict[str, Any]] = []
            missing_keys: list[tuple[str, str]] = []
            for key, seed_bucket in zip(
                self.data_source.curriculum_key_by_index,
                self.data_source.curriculum_initial_bucket_by_index,
                strict=True,
            ):
                state = payload_state_by_key.get(key)
                if state is None:
                    missing_keys.append(key)
                    remapped_states.append(self._make_default_state(key, seed_bucket))
                else:
                    state["dataset"] = key[0]
                    state["problem_id"] = key[1]
                    remapped_states.append(state)
            self.index_states = remapped_states
            self._overlay_current_seed_buckets()
            if missing_keys:
                print(
                    "CURRICULUM_SNAPSHOT_REMAP_MISSING_KEYS "
                    f"count={len(missing_keys)} example={missing_keys[:3]}"
                )
            return

        if self.reset_state_on_dataset_mismatch:
            print(
                "CURRICULUM_SNAPSHOT_RESET_ON_DATASET_MISMATCH "
                f"old_len={len(payload_index_states)} new_len={len(current_keys)} "
                f"step={payload['global_step']}"
            )
            self._reset_index_states_from_current_seed_buckets()
            return

        raise RuntimeError(
            "Curriculum snapshot index_states length does not match current dataset length "
            f"({len(payload_index_states)} != {len(current_keys)})"
        )

    def _overlay_current_seed_buckets(self) -> None:
        if len(self.index_states) != len(self.data_source.curriculum_initial_bucket_by_index):
            raise RuntimeError(
                "Curriculum snapshot index_states length does not match current dataset length "
                f"({len(self.index_states)} != {len(self.data_source.curriculum_initial_bucket_by_index)})"
            )

        for idx, current_seed_bucket in enumerate(self.data_source.curriculum_initial_bucket_by_index):
            state = self.index_states[idx]
            state["seed_bucket"] = current_seed_bucket
            if int(state["visits"]) < self.min_visits_for_online:
                state["bucket"] = current_seed_bucket

        self.bucket_counts = Counter(state["bucket"] for state in self.index_states)
