import importlib
import json
import os
import pickle
from hashlib import md5
from pathlib import Path

from failure_prob.conf import Config
from failure_prob.data.utils import Rollout
from omegaconf import OmegaConf

_CACHE_EXCLUDED_DATASET_KEYS = {
    "use_cache",
    "refresh_cache",
    "cache_path",
    "cache_dir",
    "load_to_cuda",
    "normalize_hidden_states",
    "pred_horizon",
    "exec_horizon",
    "dim_features",
    "dim_action",
}

def _import_data_module(data_type: str):
    # Dynamically import a module named after the data_type
    # Assumes each module implements load_rollouts(cfg) and split_rollouts(cfg, all_rollouts)
    try:
        return importlib.import_module(f'.{data_type}', package=__name__)
    except ImportError:
        raise ValueError(f"No module named '{data_type}' found, or it doesn't implement the required functions.")


def _build_cache_meta(cfg: Config) -> tuple[dict, str]:
    dataset_cfg = OmegaConf.to_container(
        cfg.dataset,
        resolve=False,
        throw_on_missing=False,
    )
    dataset_cfg = {
        key: value
        for key, value in dataset_cfg.items()
        if key not in _CACHE_EXCLUDED_DATASET_KEYS
    }
    meta = {
        "cache_version": 1,
        "dataset": dataset_cfg,
        "train": {
            "log_precomputed": bool(cfg.train.log_precomputed),
            "log_precomputed_only": bool(cfg.train.log_precomputed_only),
        },
    }
    fingerprint = md5(
        json.dumps(meta, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return meta, fingerprint


def _resolve_cache_path(cfg: Config, fingerprint: str) -> str:
    if cfg.dataset.cache_path:
        cache_path = str(cfg.dataset.cache_path).format(
            dataset=cfg.dataset.name,
            subset=cfg.dataset.subset_name,
            cache_key=fingerprint,
        )
        return os.path.abspath(os.path.expanduser(cache_path))

    cache_dir = Path(os.path.expanduser(str(cfg.dataset.cache_dir)))
    subset_name = str(getattr(cfg.dataset, "subset_name", "default"))
    return str((cache_dir / str(cfg.dataset.name) / subset_name / f"{fingerprint}.pkl").resolve())


def _get_runtime_dataset_fields(cfg: Config) -> dict:
    return {
        "dim_features": getattr(cfg.dataset, "dim_features", None),
        "dim_action": getattr(cfg.dataset, "dim_action", None),
        "pred_horizon": getattr(cfg.dataset, "pred_horizon", None),
        "exec_horizon": getattr(cfg.dataset, "exec_horizon", None),
    }


def _restore_runtime_dataset_fields(cfg: Config, runtime_fields: dict | None) -> None:
    if not runtime_fields:
        return
    for key, value in runtime_fields.items():
        if value is None:
            continue
        setattr(cfg.dataset, key, value)


def _load_rollouts_from_cache(cache_path: str) -> tuple[list[Rollout], dict | None, dict | None]:
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "rollouts" in payload:
        return payload["rollouts"], payload.get("meta"), payload.get("runtime")
    return payload, None, None


def _save_rollouts_to_cache(
    cache_path: str,
    rollouts: list[Rollout],
    meta: dict,
    runtime_fields: dict,
) -> None:
    cache_payload = {
        "meta": meta,
        "runtime": runtime_fields,
        "rollouts": rollouts,
    }
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_path = f"{cache_path}.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "cache_path": cache_path,
                "meta": meta,
                "runtime": runtime_fields,
                "num_rollouts": len(rollouts),
            },
            f,
            indent=2,
        )


def load_rollouts(cfg: Config) -> list[Rollout]:
    module = _import_data_module(cfg.dataset.name)
    if not getattr(cfg.dataset, "use_cache", False):
        return module.load_rollouts(cfg)

    cache_meta, fingerprint = _build_cache_meta(cfg)
    cache_path = _resolve_cache_path(cfg, fingerprint)
    refresh_cache = bool(getattr(cfg.dataset, "refresh_cache", False))

    if os.path.isfile(cache_path) and not refresh_cache:
        rollouts, cached_meta, runtime_fields = _load_rollouts_from_cache(cache_path)
        if cached_meta == cache_meta:
            _restore_runtime_dataset_fields(cfg, runtime_fields)
            print(f"Loaded rollout cache from {cache_path}")
            return rollouts
        print(f"Ignoring stale rollout cache at {cache_path}; cache metadata mismatch.")

    rollouts = module.load_rollouts(cfg)
    runtime_fields = _get_runtime_dataset_fields(cfg)
    _save_rollouts_to_cache(cache_path, rollouts, cache_meta, runtime_fields)
    print(f"Saved rollout cache to {cache_path}")
    return rollouts


def split_rollouts(cfg: Config, all_rollouts) -> dict[str, list[Rollout]]:
    # Dynamically load the appropriate module and call its split_rollouts
    module = _import_data_module(cfg.dataset.name)
    return module.split_rollouts(cfg, all_rollouts)
