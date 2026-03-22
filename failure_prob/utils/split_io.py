import hashlib
import json
import os

from omegaconf import OmegaConf

from failure_prob.conf import Config
from failure_prob.data.utils import Rollout

_DATA_HASH_EXCLUDED_DATASET_KEYS = {
    "use_cache",
    "refresh_cache",
    "cache_path",
    "cache_dir",
    "load_to_cuda",
}


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


def _normalize_for_hash(value):
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    return value


def _dataset_hash_payload(cfg: Config) -> dict:
    dataset_cfg = OmegaConf.to_container(
        cfg.dataset,
        resolve=False,
        throw_on_missing=False,
    )
    dataset_cfg = {
        key: value
        for key, value in dataset_cfg.items()
        if key not in _DATA_HASH_EXCLUDED_DATASET_KEYS
    }
    payload = {
        "dataset": _normalize_for_hash(dataset_cfg),
        "train": {
            "log_precomputed": bool(cfg.train.log_precomputed),
            "log_precomputed_only": bool(cfg.train.log_precomputed_only),
        },
    }
    return payload


def build_split_signature(
    cfg: Config,
    rollouts_by_split_name: dict[str, list[Rollout]],
) -> dict:
    splits = {
        split_name: [_rollout_signature_record(rollout) for rollout in rollouts]
        for split_name, rollouts in rollouts_by_split_name.items()
    }
    hash_payload = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    split_md5 = hashlib.md5(hash_payload.encode("utf-8")).hexdigest()
    data_payload = _dataset_hash_payload(cfg)
    data_hash_payload = json.dumps(data_payload, sort_keys=True, separators=(",", ":"))
    data_md5 = hashlib.md5(data_hash_payload.encode("utf-8")).hexdigest()
    signature_payload = json.dumps(
        {
            "split_md5": split_md5,
            "data_md5": data_md5,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    signature_md5 = hashlib.md5(signature_payload.encode("utf-8")).hexdigest()
    split_counts = {
        split_name: len(rollouts)
        for split_name, rollouts in rollouts_by_split_name.items()
    }
    split_tasks = {}
    for split_name, rollouts in rollouts_by_split_name.items():
        task_items = sorted({
            (
                int(rollout.task_id),
                str(rollout.task_suite_name),
                str(rollout.task_description),
            )
            for rollout in rollouts
        })
        split_tasks[split_name] = [
            {
                "task_id": task_id,
                "task_suite_name": task_suite_name,
                "task_description": task_description,
            }
            for task_id, task_suite_name, task_description in task_items
        ]
    return {
        "split_md5": split_md5,
        "data_md5": data_md5,
        "signature_md5": signature_md5,
        "split_counts": split_counts,
        "split_tasks": split_tasks,
        "splits": splits,
        "data_payload": data_payload,
    }


def save_split_signature(
    split_path: str,
    cfg: Config,
    rollouts_by_split_name: dict[str, list[Rollout]],
) -> dict:
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    signature = build_split_signature(cfg, rollouts_by_split_name)
    with open(split_path, "w") as f:
        json.dump(signature, f, indent=2)
    return signature


def load_split_signature(split_path: str) -> dict:
    with open(split_path, "r") as f:
        signature = json.load(f)

    if "split_md5" not in signature:
        raise ValueError(f"Invalid split signature format: {split_path}")
    return signature


def restore_rollouts_by_split_signature(
    all_rollouts: list[Rollout],
    signature: dict,
) -> dict[str, list[Rollout]] | None:
    splits = signature.get("splits")
    if not splits:
        return None

    rollout_buckets = {}
    for rollout in all_rollouts:
        record = _rollout_signature_record(rollout)
        key = json.dumps(record, sort_keys=True, separators=(",", ":"))
        rollout_buckets.setdefault(key, []).append(rollout)

    restored = {}
    for split_name, records in splits.items():
        restored_rollouts = []
        for record in records:
            key = json.dumps(record, sort_keys=True, separators=(",", ":"))
            bucket = rollout_buckets.get(key)
            if not bucket:
                raise ValueError(
                    f"Saved split rollout not found in loaded data for split '{split_name}': {record}"
                )
            restored_rollouts.append(bucket.pop())
        restored[split_name] = restored_rollouts
    return restored


def validate_split_signature(
    cfg: Config,
    rollouts_by_split_name: dict[str, list[Rollout]],
    signature: dict,
) -> str:
    current_signature = build_split_signature(cfg, rollouts_by_split_name)
    expected = signature.get("signature_md5", signature["split_md5"])
    actual = current_signature.get("signature_md5", current_signature["split_md5"])
    if actual != expected:
        raise ValueError(
            f"Split signature mismatch: expected {expected}, got {actual}"
        )
    return actual
