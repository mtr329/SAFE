import json
import os
import sys
from numbers import Number
from pathlib import Path
import warnings

import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch.utils.data import DataLoader

from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import RolloutDataset, normalize_rollouts_hidden_states
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.utils.constants import MANUAL_METRICS, EVAL_TIME_QUANTILES
from failure_prob.utils.random import seed_everything
from failure_prob.utils.routines import (
    eval_metrics_and_log,
    eval_model_and_log,
    eval_save_timing_plots,
)
from failure_prob.utils.split_io import load_split_signature, validate_split_signature
from failure_prob.utils.timer import Timer
from failure_prob.utils.video import eval_save_videos, eval_save_videos_functional_cp
from failure_prob.utils.routines import (
    model_forward_dataloader
)

from failure_prob.conf import Config, process_cfg

from failure_prob.mrefine.ori_metrics import (
    get_ori_metrics,
    summar_ori_metrics,
)


def parse_seeds(seed_cfg: str | int) -> list[int]:
    if isinstance(seed_cfg, int):
        return [seed_cfg]
    if seed_cfg.isnumeric() and "-" not in seed_cfg:
        return [int(seed_cfg)]

    assert all(s.isdigit() for s in seed_cfg.split("-")), (
        "All seeds must be integers separated by '-'"
    )
    return [int(s) for s in seed_cfg.split("-")]


def resolve_ckpt_path(ckpt_path_cfg: str | None, seed: int) -> str:
    if ckpt_path_cfg is None:
        raise ValueError(
            "train.eval_ckpt_path must be set for evaluation. "
            "It may be a file, a directory, or a template such as "
            "'/path/to/model_seed{seed}.ckpt'."
        )

    ckpt_path = os.path.expanduser(str(ckpt_path_cfg)).format(seed=seed)
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, f"model_seed{seed}.ckpt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def resolve_split_path(
    split_path_cfg: str | None,
    ckpt_path: str | None,
    seed: int,
) -> str | None:
    if split_path_cfg is not None:
        split_path = os.path.expanduser(str(split_path_cfg)).format(seed=seed)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Saved split not found: {split_path}")
        return split_path

    if ckpt_path is None:
        return None

    ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
    candidate = os.path.join(ckpt_dir, f"split_seed{seed}.json")
    if os.path.exists(candidate):
        return candidate
    return None


def load_model_checkpoint(model: BaseModel, ckpt_path: str) -> tuple[BaseModel, int | None]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    loaded_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model, loaded_epoch


def collect_scalar_logs(logs: dict) -> dict[str, float]:
    scalar_logs = {}
    for key, value in logs.items():
        if isinstance(value, Number) and not isinstance(value, bool):
            scalar_logs[key] = float(value)
    return scalar_logs


def mean_scalar_logs(logs_by_seed: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    if not logs_by_seed:
        return {}

    common_keys = set(logs_by_seed[0][1].keys())
    for _, seed_logs in logs_by_seed[1:]:
        common_keys &= set(seed_logs.keys())

    mean_logs = {}
    for key in sorted(common_keys):
        values = [seed_logs[key] for _, seed_logs in logs_by_seed]
        mean_logs[key] = sum(values) / len(values)
    return mean_logs


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def resolve_eval_output_dir(ckpt_path: str | None) -> str:
    if ckpt_path is None:
        return os.getcwd()
    return ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)


def collect_eval_dirs(target_root: Path) -> list[Path]:
    if not target_root.exists():
        raise FileNotFoundError(f"Directory not found: {target_root}")
    if target_root.is_file():
        raise ValueError(f"Expected a directory, got file: {target_root}")
    if (target_root / "config.yaml").is_file():
        return [target_root.resolve()]

    config_dirs = sorted({p.parent.resolve() for p in target_root.rglob("config.yaml")})
    return config_dirs


def run_batch_eval(target_root: Path) -> None:
    target_dirs = collect_eval_dirs(target_root)
    if not target_dirs:
        raise FileNotFoundError(f"No evaluation directories found under: {target_root}")

    for log_dir in target_dirs:
        print("Running:", log_dir)
        cfg = OmegaConf.load(log_dir / "config.yaml")
        cfg.train.eval_ckpt_path = str(log_dir)
        cfg.dataset.data_path_prefix = ""
        evaluate_cfg(cfg)


def resolve_default_debug_run_dir(repo_root: Path) -> Path:
    for root_name in ("log_ckpt", "logs"):
        root_dir = repo_root / root_name
        if not root_dir.exists():
            continue

        config_dirs = sorted({p.parent.resolve() for p in root_dir.rglob("config.yaml")})
        if config_dirs:
            return config_dirs[0]

    return repo_root / "log_ckpt"


def evaluate_cfg(cfg: Config) -> None:
    cfg = process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))

    method_name = cfg.model.name
    if "distance" in cfg.model:
        method_name += f"_{cfg.model.distance}"

    if (
        cfg.train.eval_save_logs
        or cfg.train.eval_save_video
        or cfg.train.eval_save_video_functional
        or cfg.train.eval_save_timing_plots
    ):
        os.makedirs(cfg.train.logs_save_path, exist_ok=True)
        with open(os.path.join(cfg.train.logs_save_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # Match train.py exactly: use a fixed seed while loading raw rollouts.
    seed_everything(0)
    with Timer("Loading rollouts"):
        all_rollouts = load_rollouts(cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        if cfg.dataset.load_to_cuda:
            all_rollouts = [r.to("cuda") for r in all_rollouts]

    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")

    if cfg.dataset.normalize_hidden_states:
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)

    seeds = parse_seeds(cfg.train.seed)
    scalar_logs_by_seed = []
    my_logs = {}
    my_logs_save_dir = None
    ori_logs = {}
    for seed in seeds:
        print(f"Evaluating seed {seed}")
        cfg.train.seed = seed
        # Match train.py exactly: reseed immediately before split_rollouts.
        seed_everything(seed)

        ckpt_path = None
        if not cfg.train.log_precomputed_only:
            ckpt_path = resolve_ckpt_path(cfg.train.eval_ckpt_path, seed)

        split_path = resolve_split_path(cfg.train.eval_split_path, ckpt_path, seed)
        rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
        if split_path is not None:
            print("Loading split signature from", os.path.abspath(split_path))
            split_signature = load_split_signature(split_path)
            split_md5 = validate_split_signature(
                rollouts_by_split_name,
                split_signature,
            )
            print(f"Validated split md5: {split_md5}")

        dataset_by_split_name = {
            split: RolloutDataset(cfg, rollouts)
            for split, rollouts in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            split: DataLoader(
                dataset,
                batch_size=cfg.model.batch_size,
                shuffle="train" in split,
                num_workers=0,
            )
            for split, dataset in dataset_by_split_name.items()
        }

        to_be_logged = {"epoch": 0}
        
        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            pass
            # my eval
            

        if not cfg.train.log_precomputed_only:
            if my_logs_save_dir is None:
                my_logs_save_dir = resolve_eval_output_dir(ckpt_path)
            input_dim = rollouts_by_split_name["train"][0].hidden_states.shape[-1]
            model: BaseModel = get_model(cfg, input_dim)
            print("Loading checkpoint from", os.path.abspath(ckpt_path))
            model, loaded_epoch = load_model_checkpoint(model, ckpt_path)
            if loaded_epoch is not None:
                print(f"Loaded checkpoint epoch: {loaded_epoch}")
            model.to("cuda")
            if cfg.model.name == "embed" and not getattr(model, "trained", True):
                print("Rebuilding embed state from the training split for evaluation")
                model.train_epoch(None, dataloader_by_split_name["train"], force_retrain=True)
            model.eval()
            
            #### Forward the model and compute the scores ####
            scores_by_split_name = {}
            for split, dataloader in dataloader_by_split_name.items():
                # Re-create a dataset to disable shuffling
                dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
                with torch.no_grad():
                    scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
                scores = scores.detach().cpu().numpy()
                seq_lengths = valid_masks.sum(dim=-1).cpu().numpy() # (B,)
                scores_by_split_name[split] = [scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]
            
            # ori metrics
            my_logs.setdefault("ori", {})
            get_ori_metrics(
                scores_by_split_name, 
                rollouts_by_split_name,
                method_name,
                my_logs["ori"],
            )


    if my_logs_save_dir is None:
        my_logs_save_dir = resolve_eval_output_dir(cfg.train.eval_ckpt_path)
    my_logs_save_dir = os.path.join(my_logs_save_dir, "eval")
    os.makedirs(my_logs_save_dir, exist_ok=True)
    my_logs_save_path = os.path.join(my_logs_save_dir, "my_logs.json")
    with open(my_logs_save_path, "w") as f:
        json.dump(to_jsonable(my_logs), f, indent=2)
    print("Saved my_logs to", os.path.abspath(my_logs_save_path))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    evaluate_cfg(cfg)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    debug_run_dir = resolve_default_debug_run_dir(repo_root)
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--config-path",
            str(debug_run_dir),
            "--config-name",
            "config",
            f"train.eval_ckpt_path={debug_run_dir}",
            "dataset.data_path_prefix=",
        ])
    main()
