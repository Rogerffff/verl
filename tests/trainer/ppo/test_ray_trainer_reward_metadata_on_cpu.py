import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.reward import merge_generation_metadata_into_extra_info


def test_merge_generation_metadata_into_extra_info_copies_per_sample_dicts():
    shared_extra_info = {"problem_id": "prob-1"}
    batch = DataProto(
        batch=TensorDict({"responses": torch.zeros((2, 1), dtype=torch.long)}, batch_size=[2]),
        non_tensor_batch={
            "extra_info": np.array([shared_extra_info, shared_extra_info], dtype=object),
            "finish_reason": np.array(["length", "stop"], dtype=object),
            "truncated_by_max_tokens": np.array([True, False], dtype=object),
        },
        meta_info={},
    )

    merge_generation_metadata_into_extra_info(batch)

    merged = batch.non_tensor_batch["extra_info"]
    assert merged[0] is not merged[1]
    assert merged[0]["problem_id"] == "prob-1"
    assert merged[0]["finish_reason"] == "length"
    assert merged[0]["truncated_by_max_tokens"] is True
    assert merged[1]["finish_reason"] == "stop"
    assert merged[1]["truncated_by_max_tokens"] is False
    assert "finish_reason" not in shared_extra_info
    assert "truncated_by_max_tokens" not in shared_extra_info


def test_merge_generation_metadata_into_extra_info_handles_missing_or_non_dict_values():
    batch = DataProto(
        batch=TensorDict({"responses": torch.zeros((2, 1), dtype=torch.long)}, batch_size=[2]),
        non_tensor_batch={
            "extra_info": np.array([None, "not-a-dict"], dtype=object),
            "finish_reason": np.array([None, "length"], dtype=object),
        },
        meta_info={},
    )

    merge_generation_metadata_into_extra_info(batch)

    merged = batch.non_tensor_batch["extra_info"]
    assert merged[0] == {"finish_reason": None, "truncated_by_max_tokens": False}
    assert merged[1] == {"finish_reason": "length", "truncated_by_max_tokens": True}
