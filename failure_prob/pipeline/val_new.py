import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.conf import Config, process_cfg
from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import RolloutDataset, normalize_rollouts_hidden_states
from failure_prob.eval import (
    collect_eval_dirs,
    load_model_checkpoint,
    parse_seeds,
    resolve_ckpt_path,
    resolve_split_path,
    to_jsonable,
)
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.mrefine.new_eval import _get_delay_calib_res, _get_new_static_metrics
from failure_prob.mrefine.ori_eval import get_ori_metrics
from failure_prob.mrefine.utils import get_func_conformal_bands
from failure_prob.utils.constants import MANUAL_METRICS
from failure_prob.utils.metrics import get_metrics_curve
from failure_prob.utils.random import seed_everything
from failure_prob.utils.routines import model_forward_dataloader
from failure_prob.utils.split_io import (
    load_split_signature,
    restore_rollouts_by_split_signature,
    validate_split_signature,
)
from failure_prob.utils.timer import Timer

VAL_SPLITS = ("train", "val_seen", "val_unseen")
ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]


def _resolve_default_logs_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    for dirname in ("log_ckpt_new", "log_ckpt", "logs"):
        candidate = repo_root / dirname
        if candidate.is_dir():
            return str(candidate)
    return str(repo_root / "log_ckpt_new")


def _clear_redundant_data_path_prefix(cfg) -> None:
    data_path = cfg.dataset.data_path
    data_path_unseen = cfg.dataset.data_path_unseen

    def _is_abs_path(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return os.path.isabs(value)
        try:
            return all(isinstance(v, str) and os.path.isabs(v) for v in value)
        except TypeError:
            return False

    if _is_abs_path(data_path) or _is_abs_path(data_path_unseen):
        cfg.dataset.data_path_prefix = None


def _method_name(cfg: Config) -> str:
    method_name = cfg.model.name
    if "distance" in cfg.model:
        method_name += f"_{cfg.model.distance}"
    return method_name


def _val_dir(log_dir: str) -> str:
    return os.path.join(os.path.abspath(log_dir), "val")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _metric_mean(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.nan
    return float(array.mean())


def _require_val_splits(rollouts_by_split_name: dict[str, list]) -> dict[str, list]:
    missing = [split for split in VAL_SPLITS if split not in rollouts_by_split_name]
    if missing:
        raise KeyError(f"Missing required splits: {missing}")
    return {split: rollouts_by_split_name[split] for split in VAL_SPLITS}


def _cfg_for_split_validation(cfg: Config, split_signature: dict) -> Config:
    saved_dataset_cfg = split_signature.get("data_payload", {}).get("dataset", {})
    if not saved_dataset_cfg:
        return cfg

    cfg_for_validation = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in saved_dataset_cfg.items():
        cfg_for_validation.dataset[key] = value
    return cfg_for_validation



def _load_or_rebuild_training_split(
    cfg: Config,
    all_rollouts: list,
    split_path: str | None,
) -> dict[str, list]:
    if split_path is None:
        raise ValueError("Saved split_seed*.json is required to guarantee an exact train/val/test split match.")
    split_signature = load_split_signature(split_path)
    if split_signature is not None:
        restored_rollouts_by_split_name = restore_rollouts_by_split_signature(all_rollouts, split_signature)
        if restored_rollouts_by_split_name is not None:
            cfg_for_validation = _cfg_for_split_validation(cfg, split_signature)
            validate_split_signature(cfg_for_validation, restored_rollouts_by_split_name, split_signature)
            return restored_rollouts_by_split_name

    full_rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
    if split_signature is not None:
        cfg_for_validation = _cfg_for_split_validation(cfg, split_signature)
        validate_split_signature(cfg_for_validation, full_rollouts_by_split_name, split_signature)
    return full_rollouts_by_split_name


def _build_model_scores(
    cfg: Config,
    rollouts_by_split_name: dict[str, list],
    ckpt_path: str,
) -> dict[str, list[np.ndarray]]:
    datasets = {
        split: RolloutDataset(cfg, rollouts)
        for split, rollouts in rollouts_by_split_name.items()
    }
    input_dim = rollouts_by_split_name["train"][0].hidden_states.shape[-1]
    model: BaseModel = get_model(cfg, input_dim)
    model, _ = load_model_checkpoint(model, ckpt_path)
    model.to("cuda")
    model.eval()

    scores_by_split_name = {}
    for split, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()
        scores_by_split_name[split] = [scores[i, : int(seq_lengths[i])] for i in range(len(seq_lengths))]
    return scores_by_split_name


def _get_new_metrics_for_validation(
    scores_by_split_name: dict[str, list[np.ndarray]],
    rollouts_by_split_name: dict[str, list],
    method_name: str,
    res_dict: dict,
) -> None:
    _get_new_static_metrics(scores_by_split_name, rollouts_by_split_name, method_name, res_dict)

    cal_rollouts = list(rollouts_by_split_name["val_seen"])
    cal_scores = [np.asarray(scores, dtype=float) for scores in scores_by_split_name["val_seen"]]
    val_rollouts = list(rollouts_by_split_name["val_unseen"])
    val_scores = [np.asarray(scores, dtype=float) for scores in scores_by_split_name["val_unseen"]]

    max_length = max(len(scores) for scores in cal_scores + val_scores)
    cal_scores = [np.pad(scores, (0, max_length - len(scores)), mode="edge") for scores in cal_scores]
    val_scores = [np.pad(scores, (0, max_length - len(scores)), mode="edge") for scores in val_scores]

    cp_bands_by_alpha = get_func_conformal_bands(cal_rollouts, cal_scores, ALPHAS)
    _get_delay_calib_res(val_rollouts, val_scores, cp_bands_by_alpha, ALPHAS, method_name, res_dict)


def _get_handcrafted_scores(
    cfg: Config,
    rollouts_by_split_name: dict[str, list],
) -> dict[str, dict[str, list[np.ndarray]]]:
    metric_keys = MANUAL_METRICS[cfg.dataset.name]
    if metric_keys is None:
        metric_keys = rollouts_by_split_name["train"][0].logs.columns

    scores_by_metric = {}
    for metric_key in metric_keys:
        if metric_key not in rollouts_by_split_name["train"][0].logs.columns:
            continue
        metric_name = metric_key.split("/")[-1]
        scores_by_metric[metric_name] = {
            split: get_metrics_curve(rollouts, metric_key)
            for split, rollouts in rollouts_by_split_name.items()
        }
    return scores_by_metric


def _save_ori_outputs(save_dir: str, ori_logs: dict) -> None:
    metrics_rows = []
    roc_rows = []
    pr_rows = []

    static_logs_by_method = ori_logs.get("static", {})
    for method_name, static_logs in static_logs_by_method.items():
        for split in VAL_SPLITS:
            task_dict = static_logs.get(split, {}).get("all", {})
            metrics_rows.append({
                "method": method_name,
                "split": split,
                "roc_auc_early": _metric_mean(task_dict.get("roc_auc", [])),
                "prc_auc_early": _metric_mean(task_dict.get("prc_auc", [])),
            })

            for curve_index, (fpr, tpr) in enumerate(zip(task_dict.get("fpr", []), task_dict.get("tpr", []))):
                for x, y in zip(np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)):
                    roc_rows.append({
                        "method": method_name,
                        "split": split,
                        "curve_index": curve_index,
                        "fpr": float(x),
                        "tpr": float(y),
                    })
            for curve_index, (rec, pre) in enumerate(zip(task_dict.get("rec", []), task_dict.get("pre", []))):
                for x, y in zip(np.asarray(rec, dtype=float), np.asarray(pre, dtype=float)):
                    pr_rows.append({
                        "method": method_name,
                        "split": split,
                        "curve_index": curve_index,
                        "recall": float(x),
                        "precision": float(y),
                    })

            safe_method = _safe_name(method_name)
            safe_split = _safe_name(split)

            roc_fig = Figure(figsize=(6, 5))
            roc_ax = roc_fig.subplots()
            for fpr, tpr in zip(task_dict.get("fpr", []), task_dict.get("tpr", [])):
                roc_ax.plot(np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float), alpha=0.35, linewidth=1.5)
            roc_ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#666666")
            roc_ax.set_xlabel("FPR")
            roc_ax.set_ylabel("TPR")
            roc_ax.set_xlim(0.0, 1.0)
            roc_ax.set_ylim(0.0, 1.0)
            roc_ax.set_title(f"{method_name} {split} ROC (early)")
            roc_ax.grid(True, alpha=0.3)
            roc_fig.tight_layout()
            roc_fig.savefig(os.path.join(save_dir, f"ori_{safe_method}_{safe_split}_roc.png"), dpi=300)

            pr_fig = Figure(figsize=(6, 5))
            pr_ax = pr_fig.subplots()
            for rec, pre in zip(task_dict.get("rec", []), task_dict.get("pre", [])):
                pr_ax.plot(np.asarray(rec, dtype=float), np.asarray(pre, dtype=float), alpha=0.35, linewidth=1.5)
            pr_ax.set_xlabel("Recall")
            pr_ax.set_ylabel("Precision")
            pr_ax.set_xlim(0.0, 1.0)
            pr_ax.set_ylim(0.0, 1.0)
            pr_ax.set_title(f"{method_name} {split} PR (early)")
            pr_ax.grid(True, alpha=0.3)
            pr_fig.tight_layout()
            pr_fig.savefig(os.path.join(save_dir, f"ori_{safe_method}_{safe_split}_pr.png"), dpi=300)

    pd.DataFrame(metrics_rows).to_csv(os.path.join(save_dir, "ori_metrics_early.csv"), index=False)
    pd.DataFrame(roc_rows).to_csv(os.path.join(save_dir, "ori_roc_curve_points.csv"), index=False)
    pd.DataFrame(pr_rows).to_csv(os.path.join(save_dir, "ori_pr_curve_points.csv"), index=False)


def _mark_pareto_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    marked = [dict(row) for row in rows]
    for row in marked:
        row["is_pareto"] = False
        row["pareto_rank"] = np.nan

    best_bal_acc = -np.inf
    pareto_rank = 0
    order = sorted(range(len(marked)), key=lambda i: (marked[i]["avg_det_time"], -marked[i]["bal_acc"], marked[i]["alpha"]))
    for idx in order:
        if marked[idx]["bal_acc"] > best_bal_acc:
            marked[idx]["is_pareto"] = True
            marked[idx]["pareto_rank"] = pareto_rank
            pareto_rank += 1
            best_bal_acc = marked[idx]["bal_acc"]
    return marked


def _save_new_outputs(save_dir: str, new_logs: dict) -> None:
    raw_rows = []
    for method_name, calib_logs in new_logs.get("calib", {}).items():
        for mode, delta_dict in calib_logs.items():
            for delta, alpha_dict in delta_dict.items():
                rows = []
                for alpha, metrics in alpha_dict.items():
                    row = {
                        "method": method_name,
                        "mode": mode,
                        "delta": float(delta),
                        "alpha": float(alpha),
                        "avg_det_time": _metric_mean(metrics.get("avg_det_time", [])),
                        "bal_acc": _metric_mean(metrics.get("bal_acc", [])),
                        "acc": _metric_mean(metrics.get("acc", [])),
                        "f1": _metric_mean(metrics.get("f1", [])),
                        "fpr": _metric_mean(metrics.get("fpr", [])),
                        "fnr": _metric_mean(metrics.get("fnr", [])),
                        "tpr": _metric_mean(metrics.get("tpr", [])),
                        "tnr": _metric_mean(metrics.get("tnr", [])),
                    }
                    if "ttd_auc" in metrics:
                        row["ttd_auc"] = _metric_mean(metrics.get("ttd_auc", []))
                    rows.append(row)
                raw_rows.extend(_mark_pareto_rows(rows))

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(os.path.join(save_dir, "new_t_at_balacc_raw.csv"), index=False)
    raw_df[raw_df["is_pareto"]].to_csv(os.path.join(save_dir, "new_t_at_balacc_pareto.csv"), index=False)


def evaluate_run(log_dir: str, logs_dir: str) -> dict:
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    _clear_redundant_data_path_prefix(cfg)
    cfg.train.eval_ckpt_path = log_dir
    cfg = process_cfg(cfg)

    seed_everything(0)
    with Timer("Loading rollouts"):
        all_rollouts = load_rollouts(cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        if cfg.dataset.load_to_cuda:
            all_rollouts = [rollout.to("cuda") for rollout in all_rollouts]

    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")

    if cfg.dataset.normalize_hidden_states:
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)

    ori_logs = {}
    new_logs = {}
    method_name = _method_name(cfg)
    is_handcrafted = bool(cfg.train.log_precomputed or cfg.train.log_precomputed_only)

    for seed in parse_seeds(cfg.train.seed):
        cfg.train.seed = seed
        seed_everything(seed)

        ckpt_path = None if cfg.train.log_precomputed_only else resolve_ckpt_path(cfg.train.eval_ckpt_path, seed)
        split_path = resolve_split_path(cfg.train.eval_split_path, ckpt_path, seed)
        if split_path is None:
            expected_split_path = None
            if ckpt_path is not None:
                ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
                expected_split_path = os.path.join(ckpt_dir, f"split_seed{seed}.json")
            raise ValueError(
                "Saved split_seed*.json is required to guarantee an exact train/val/test split match. "
                f"run_dir={os.path.abspath(log_dir)} seed={seed} expected_split_path={expected_split_path}"
            )
        full_rollouts_by_split_name = _load_or_rebuild_training_split(cfg, all_rollouts, split_path)
        rollouts_by_split_name = _require_val_splits(full_rollouts_by_split_name)

        if is_handcrafted:
            for metric_name, scores_by_split_name in _get_handcrafted_scores(cfg, rollouts_by_split_name).items():
                get_ori_metrics(scores_by_split_name, rollouts_by_split_name, metric_name, ori_logs)
                _get_new_metrics_for_validation(scores_by_split_name, rollouts_by_split_name, metric_name, new_logs)
        else:
            if ckpt_path is None:
                raise ValueError("Missing checkpoint path for model validation")
            scores_by_split_name = _build_model_scores(cfg, rollouts_by_split_name, ckpt_path)
            get_ori_metrics(scores_by_split_name, rollouts_by_split_name, method_name, ori_logs)
            _get_new_metrics_for_validation(scores_by_split_name, rollouts_by_split_name, method_name, new_logs)

    save_dir = _val_dir(log_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "ori_logs.json"), "w") as f:
        json.dump(to_jsonable(ori_logs), f, indent=2)
    with open(os.path.join(save_dir, "new_logs.json"), "w") as f:
        json.dump(to_jsonable(new_logs), f, indent=2)

    _save_ori_outputs(save_dir, ori_logs)
    _save_new_outputs(save_dir, new_logs)

    return {
        "run_name": os.path.relpath(log_dir, os.path.abspath(logs_dir)),
        "run_dir": os.path.abspath(log_dir),
        "method": method_name,
        "val_dir": save_dir,
        "ori_log_path": os.path.join(save_dir, "ori_logs.json"),
        "new_log_path": os.path.join(save_dir, "new_logs.json"),
        "ori_metrics_path": os.path.join(save_dir, "ori_metrics_early.csv"),
        "new_raw_path": os.path.join(save_dir, "new_t_at_balacc_raw.csv"),
        "new_pareto_path": os.path.join(save_dir, "new_t_at_balacc_pareto.csv"),
    }


def run_batch_validation(logs_dir: str, save_dir: str) -> dict:
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    failures = []
    for log_dir in collect_eval_dirs(Path(logs_dir)):
        try:
            rows.append(evaluate_run(str(log_dir), logs_dir))
        except Exception as exc:
            failure = {
                "run_name": os.path.relpath(str(log_dir), logs_dir),
                "run_dir": os.path.abspath(str(log_dir)),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"[val_new] Failed evaluating {failure['run_dir']}: {failure['error']}")

    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "val_summary.csv"), index=False)
    if failures:
        pd.DataFrame([{k: v for k, v in failure.items() if k != "traceback"} for failure in failures]).to_csv(
            os.path.join(save_dir, "val_failures.csv"), index=False
        )
        with open(os.path.join(save_dir, "val_failures.json"), "w") as f:
            json.dump(failures, f, indent=2)

    payload = {
        "logs_dir": logs_dir,
        "save_dir": save_dir,
        "num_runs": len(rows),
        "num_failures": len(failures),
    }
    with open(os.path.join(save_dir, "val_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train/validation-only metrics for trained checkpoints and save ori/new validation logs.",
    )
    parser.add_argument(
        "--logs-dir",
        default=_resolve_default_logs_dir(),
        help="Root directory containing trained run folders with config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save batch validation summaries. Defaults to <logs-dir>/pipeline_val_new.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.abspath(args.logs_dir), "pipeline_val_new")
    result = run_batch_validation(args.logs_dir, save_dir)
    print("Saved validation summary to", os.path.abspath(os.path.join(save_dir, "val_summary.json")))
    print(f"runs={result['num_runs']}")


if __name__ == "__main__":
    main()
