import hashlib
import json
import os

from failure_prob.data.utils import Rollout


def _rollout_signature_record(rollout: Rollout) -> dict:
    return {
        "task_suite_name": str(rollout.task_suite_name),
        "task_id": int(rollout.task_id),
        "episode_idx": int(rollout.episode_idx),
        "episode_success": int(rollout.episode_success),
        "mp4_path": str(rollout.mp4_path),
        "task_min_step": None if rollout.task_min_step is None else int(rollout.task_min_step),
        "seq_len": int(rollout.hidden_states.shape[0]),
        "exec_horizon": None if rollout.exec_horizon is None else int(rollout.exec_horizon),
    }


def build_split_signature(
    rollouts_by_split_name: dict[str, list[Rollout]],
) -> dict:
    splits = {
        split_name: [_rollout_signature_record(rollout) for rollout in rollouts]
        for split_name, rollouts in rollouts_by_split_name.items()
    }
    hash_payload = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    split_md5 = hashlib.md5(hash_payload.encode("utf-8")).hexdigest()
    split_counts = {
        split_name: len(rollouts)
        for split_name, rollouts in rollouts_by_split_name.items()
    }
    return {
        "split_md5": split_md5,
        "split_counts": split_counts,
    }


def save_split_signature(
    split_path: str,
    rollouts_by_split_name: dict[str, list[Rollout]],
) -> dict:
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    signature = build_split_signature(rollouts_by_split_name)
    with open(split_path, "w") as f:
        json.dump(signature, f, indent=2)
    return signature


def load_split_signature(split_path: str) -> dict:
    with open(split_path, "r") as f:
        signature = json.load(f)

    if "split_md5" not in signature:
        raise ValueError(f"Invalid split signature format: {split_path}")
    return signature


def validate_split_signature(
    rollouts_by_split_name: dict[str, list[Rollout]],
    signature: dict,
) -> str:
    current_signature = build_split_signature(rollouts_by_split_name)
    expected = signature["split_md5"]
    actual = current_signature["split_md5"]
    if actual != expected:
        raise ValueError(
            f"Split MD5 mismatch: expected {expected}, got {actual}"
        )
    return actual
