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
from failure_prob.mrefine.delay_summary import (
    PENALIZED_DET_TIME,
    _compute_max_bal_acc_from_pareto,
    _compute_pareto_balacc_coverage,
    _compute_penalized_mean_t_at_balacc,
    _get_pareto_curve_from_alpha_dict,
)
from failure_prob.mrefine.new_eval import _get_delay_calib_res, _get_new_static_metrics
from failure_prob.mrefine.new_summary import _split_new_logs_by_method
from failure_prob.mrefine.ori_eval import get_ori_metrics
from failure_prob.mrefine.ori_summary import _split_ori_logs_by_method
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

VAL_SPLIT = VAL_SPLITS[-1]
PARETO_INTEGRAL_EARLY_BAL_ACC_MIN = 0.6
PARETO_INTEGRAL_EARLY_BAL_ACC_MAX = 0.8
PARETO_INTEGRAL_LAST_BAL_ACC_MIN = 0.7
PARETO_INTEGRAL_LAST_BAL_ACC_MAX = 0.9


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
    if cfg.model.name == "embed" and not getattr(model, "trained", True):
        print("Rebuilding embed state from the training split for validation")
        train_dataloader = DataLoader(datasets["train"], batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
        model.train_epoch(None, train_dataloader, force_retrain=True)
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


def _extract_seed_from_run_name(run_name: str) -> int | None:
    for part in Path(run_name).parts:
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def _weight_key_from_run_name(run_name: str) -> str:
    parts = list(Path(run_name).parts)
    if parts and parts[0].startswith("seed") and parts[0][4:].isdigit():
        parts = parts[1:]
    return str(Path(*parts)) if parts else run_name


def _to_jsonable_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
        })
    return records


def _write_dataframe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _save_payload(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _summarize_ori_metric(ori_logs: dict, split_name: str) -> tuple[float, float]:
    task_dict = ori_logs.get("static", {}).get(split_name, {}).get("all", {})
    return _metric_mean(task_dict.get("roc_auc", [])), _metric_mean(task_dict.get("prc_auc", []))


def _mean_alpha_dict(alpha_dict: dict) -> dict:
    mean_alpha_dict = {}
    for alpha, metrics in alpha_dict.items():
        mean_alpha_dict[alpha] = {
            key: (value if key == "detect_method" else _metric_mean(value))
            for key, value in metrics.items()
        }
    return mean_alpha_dict


def _seed_runs_json(group: pd.DataFrame) -> str:
    mapping = {}
    for row in group.sort_values(by=["seed", "run_name"]).itertuples(index=False):
        if pd.isna(row.seed):
            key = "seed?"
        else:
            key = f"seed{int(row.seed)}"
        mapping[key] = row.run_name
    return json.dumps(mapping, sort_keys=True)


def collect_val_candidates_from_rows(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        ori_path = row.get("ori_log_path")
        new_path = row.get("new_log_path")
        if not ori_path or not new_path or not os.path.isfile(ori_path) or not os.path.isfile(new_path):
            continue

        with open(ori_path, "r") as f:
            ori_logs = json.load(f)
        with open(new_path, "r") as f:
            new_logs = json.load(f)

        method_name = str(row["method"])
        run_name = str(row["run_name"])
        candidates.append({
            "method": method_name,
            "seed": _extract_seed_from_run_name(run_name),
            "weight_key": _weight_key_from_run_name(run_name),
            "run_name": run_name,
            "run_dir": str(row["run_dir"]),
            "val_dir": str(row["val_dir"]),
            "val_ori": _split_ori_logs_by_method(ori_logs, method_name).get(method_name, {"static": {}, "calib": {}}),
            "val_new": _split_new_logs_by_method(new_logs, method_name).get(method_name, {"static": {}, "calib": {}}),
        })

    return sorted(candidates, key=lambda row: (row["method"], row["weight_key"], row["seed"], row["run_name"]))


def summarize_mean_val_ori_metric(
    candidates: list[dict],
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for candidate in candidates:
        roc_auc, prc_auc = _summarize_ori_metric(candidate["val_ori"], VAL_SPLIT)
        rows.append({
            "method": candidate["method"],
            "seed": candidate["seed"],
            "weight_key": candidate["weight_key"],
            "run_name": candidate["run_name"],
            "run_dir": candidate["run_dir"],
            "val_split": VAL_SPLIT,
            "val_roc_auc_early": roc_auc,
            "val_prc_auc_early": prc_auc,
        })

    by_seed_df = pd.DataFrame(rows)
    metric_cols = [
        "method", "weight_key", "num_seeds", "seed_runs",
        "mean_val_roc_auc_early", "std_val_roc_auc_early",
        "mean_val_prc_auc_early", "std_val_prc_auc_early",
    ]
    if by_seed_df.empty:
        empty_seed = pd.DataFrame(columns=[
            "method", "seed", "weight_key", "run_name", "run_dir", "val_split", "val_roc_auc_early", "val_prc_auc_early",
        ])
        empty_weight = pd.DataFrame(columns=metric_cols)
        return empty_seed, empty_weight, empty_weight.copy()

    agg_rows = []
    for (method, weight_key), group in by_seed_df.groupby(["method", "weight_key"], dropna=False):
        agg_rows.append({
            "method": method,
            "weight_key": weight_key,
            "num_seeds": int(group["seed"].dropna().nunique()),
            "seed_runs": _seed_runs_json(group),
            "mean_val_roc_auc_early": float(group["val_roc_auc_early"].mean()),
            "std_val_roc_auc_early": float(group["val_roc_auc_early"].std(ddof=0)),
            "mean_val_prc_auc_early": float(group["val_prc_auc_early"].mean()),
            "std_val_prc_auc_early": float(group["val_prc_auc_early"].std(ddof=0)),
        })

    by_weight_df = pd.DataFrame(agg_rows)
    primary_col = f"mean_val_{primary_metric}_auc_early"
    secondary_col = "mean_val_prc_auc_early" if primary_metric == "roc" else "mean_val_roc_auc_early"
    by_weight_df["missing_score"] = ~np.isfinite(by_weight_df[primary_col].to_numpy(dtype=float))
    by_weight_df = by_weight_df.sort_values(
        by=["method", "missing_score", "num_seeds", primary_col, secondary_col, "weight_key"],
        ascending=[True, True, False, False, False, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    best_df = by_weight_df.drop_duplicates(subset=["method"], keep="first").drop(columns=["missing_score"]).reset_index(drop=True)
    by_weight_df = by_weight_df.drop(columns=["missing_score"])
    by_seed_df = by_seed_df.sort_values(by=["method", "weight_key", "seed", "run_name"]).reset_index(drop=True)
    return by_seed_df, by_weight_df, best_df


def _range_for_mode(
    mode: str,
    early_bal_acc_min: float,
    early_bal_acc_max: float,
    last_bal_acc_min: float,
    last_bal_acc_max: float,
) -> tuple[float, float]:
    if str(mode) == "early":
        return float(early_bal_acc_min), float(early_bal_acc_max)
    if str(mode) == "last":
        return float(last_bal_acc_min), float(last_bal_acc_max)
    raise ValueError(f"Unsupported pareto mode: {mode}")


def _build_fixed_balacc_range(
    intervals: list[dict],
    method: str,
    mode: str,
    bal_acc_min: float,
    bal_acc_max: float,
) -> dict:
    coverage_fraction = 0.0
    if intervals and bal_acc_max > bal_acc_min:
        mins = np.asarray([interval["bal_acc_min"] for interval in intervals], dtype=float)
        maxs = np.asarray([interval["bal_acc_max"] for interval in intervals], dtype=float)
        coverage_fraction = float(np.mean(np.logical_and(mins <= bal_acc_min, maxs >= bal_acc_max)))

    return {
        "method": method,
        "mode": mode,
        "curve_fraction_target": 1.0,
        "curve_fraction_achieved": coverage_fraction,
        "num_curves": len(intervals),
        "bal_acc_min": float(bal_acc_min),
        "bal_acc_max": float(bal_acc_max),
        "bal_acc_width": max(0.0, float(bal_acc_max) - float(bal_acc_min)),
    }


def build_val_pareto_ranges(
    candidates: list[dict],
    early_bal_acc_min: float,
    early_bal_acc_max: float,
    last_bal_acc_min: float,
    last_bal_acc_max: float,
) -> tuple[dict[tuple[str, str], dict], pd.DataFrame]:
    intervals_by_group: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        calib_logs = candidate["val_new"].get("calib", {})
        for mode, delta_dict in calib_logs.items():
            group_key = (candidate["method"], mode)
            for alpha_dict in delta_dict.values():
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(_mean_alpha_dict(alpha_dict))
                if pareto_det_times.size == 0 or pareto_bal_accs.size == 0:
                    continue
                intervals_by_group.setdefault(group_key, []).append({
                    "bal_acc_min": float(np.min(pareto_bal_accs)),
                    "bal_acc_max": float(np.max(pareto_bal_accs)),
                })

    rows = []
    all_group_keys = sorted({
        (candidate["method"], mode)
        for candidate in candidates
        for mode in candidate["val_new"].get("calib", {}).keys()
    })
    range_by_group = {}
    for method, mode in all_group_keys:
        group_bal_acc_min, group_bal_acc_max = _range_for_mode(
            mode,
            early_bal_acc_min,
            early_bal_acc_max,
            last_bal_acc_min,
            last_bal_acc_max,
        )
        stats = _build_fixed_balacc_range(
            intervals_by_group.get((method, mode), []),
            method=method,
            mode=mode,
            bal_acc_min=group_bal_acc_min,
            bal_acc_max=group_bal_acc_max,
        )
        range_by_group[(method, mode)] = stats
        rows.append(stats)

    range_df = pd.DataFrame(rows)
    if range_df.empty:
        range_df = pd.DataFrame(columns=[
            "method", "mode", "curve_fraction_target", "curve_fraction_achieved", "num_curves",
            "bal_acc_min", "bal_acc_max", "bal_acc_width",
        ])
    else:
        range_df = range_df.sort_values(by=["method", "mode"]).reset_index(drop=True)
    return range_by_group, range_df


def score_val_pareto_candidates(
    candidates: list[dict],
    range_by_group: dict[tuple[str, str], dict],
    penalty_det_time: float,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        calib_logs = candidate["val_new"].get("calib", {})
        for mode, delta_dict in calib_logs.items():
            stats = range_by_group.get((candidate["method"], mode))
            if stats is None:
                continue
            bal_acc_min = stats["bal_acc_min"]
            bal_acc_max = stats["bal_acc_max"]
            for delta_key, alpha_dict in delta_dict.items():
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(_mean_alpha_dict(alpha_dict))
                score = _compute_penalized_mean_t_at_balacc(
                    pareto_det_times,
                    pareto_bal_accs,
                    bal_acc_min,
                    bal_acc_max,
                    penalty_det_time=penalty_det_time,
                )
                integral = (
                    score * (bal_acc_max - bal_acc_min)
                    if np.isfinite(score) and np.isfinite(bal_acc_min) and np.isfinite(bal_acc_max)
                    else np.nan
                )
                rows.append({
                    "method": candidate["method"],
                    "seed": candidate["seed"],
                    "weight_key": candidate["weight_key"],
                    "mode": mode,
                    "delta": float(delta_key),
                    "run_name": candidate["run_name"],
                    "run_dir": candidate["run_dir"],
                    "val_bal_acc_min": bal_acc_min,
                    "val_bal_acc_max": bal_acc_max,
                    "val_bal_acc_width": stats["bal_acc_width"],
                    "val_curve_fraction_achieved": stats["curve_fraction_achieved"],
                    "val_penalty_det_time": float(penalty_det_time),
                    "val_pareto_points": int(len(pareto_det_times)),
                    "val_pareto_coverage": _compute_pareto_balacc_coverage(
                        pareto_bal_accs,
                        bal_acc_min,
                        bal_acc_max,
                    ),
                    "val_pareto_max_bal_acc": _compute_max_bal_acc_from_pareto(pareto_bal_accs),
                    "val_penalized_mean_t_at_balacc": score,
                    "val_integral_t_at_balacc": integral,
                })

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return pd.DataFrame(columns=[
            "method", "seed", "weight_key", "mode", "delta", "run_name", "run_dir",
            "val_bal_acc_min", "val_bal_acc_max", "val_bal_acc_width", "val_curve_fraction_achieved",
            "val_penalty_det_time", "val_pareto_points", "val_pareto_coverage", "val_pareto_max_bal_acc",
            "val_penalized_mean_t_at_balacc", "val_integral_t_at_balacc",
        ])
    return score_df.sort_values(by=["method", "mode", "weight_key", "seed", "delta", "run_name"]).reset_index(drop=True)


def summarize_mean_val_pareto(
    candidates: list[dict],
    early_bal_acc_min: float = PARETO_INTEGRAL_EARLY_BAL_ACC_MIN,
    early_bal_acc_max: float = PARETO_INTEGRAL_EARLY_BAL_ACC_MAX,
    last_bal_acc_min: float = PARETO_INTEGRAL_LAST_BAL_ACC_MIN,
    last_bal_acc_max: float = PARETO_INTEGRAL_LAST_BAL_ACC_MAX,
    penalty_det_time: float = PENALIZED_DET_TIME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    range_by_group, range_df = build_val_pareto_ranges(
        candidates,
        early_bal_acc_min=early_bal_acc_min,
        early_bal_acc_max=early_bal_acc_max,
        last_bal_acc_min=last_bal_acc_min,
        last_bal_acc_max=last_bal_acc_max,
    )
    by_seed_df = score_val_pareto_candidates(candidates, range_by_group=range_by_group, penalty_det_time=penalty_det_time)
    agg_cols = [
        "method", "mode", "weight_key", "delta", "num_seeds", "seed_runs",
        "mean_val_bal_acc_min", "mean_val_bal_acc_max", "mean_val_bal_acc_width",
        "mean_val_curve_fraction_achieved", "mean_val_penalty_det_time",
        "mean_val_pareto_points", "mean_val_pareto_coverage", "mean_val_pareto_max_bal_acc",
        "mean_val_penalized_mean_t_at_balacc", "std_val_penalized_mean_t_at_balacc",
        "mean_val_integral_t_at_balacc", "std_val_integral_t_at_balacc",
    ]
    if by_seed_df.empty:
        empty = pd.DataFrame(columns=agg_cols)
        return range_df, by_seed_df, empty, empty.copy()

    agg_rows = []
    for (method, mode, weight_key, delta), group in by_seed_df.groupby(["method", "mode", "weight_key", "delta"], dropna=False):
        agg_rows.append({
            "method": method,
            "mode": mode,
            "weight_key": weight_key,
            "delta": float(delta),
            "num_seeds": int(group["seed"].dropna().nunique()),
            "seed_runs": _seed_runs_json(group),
            "mean_val_bal_acc_min": float(group["val_bal_acc_min"].mean()),
            "mean_val_bal_acc_max": float(group["val_bal_acc_max"].mean()),
            "mean_val_bal_acc_width": float(group["val_bal_acc_width"].mean()),
            "mean_val_curve_fraction_achieved": float(group["val_curve_fraction_achieved"].mean()),
            "mean_val_penalty_det_time": float(group["val_penalty_det_time"].mean()),
            "mean_val_pareto_points": float(group["val_pareto_points"].mean()),
            "mean_val_pareto_coverage": float(group["val_pareto_coverage"].mean()),
            "mean_val_pareto_max_bal_acc": float(group["val_pareto_max_bal_acc"].mean()),
            "mean_val_penalized_mean_t_at_balacc": float(group["val_penalized_mean_t_at_balacc"].mean()),
            "std_val_penalized_mean_t_at_balacc": float(group["val_penalized_mean_t_at_balacc"].std(ddof=0)),
            "mean_val_integral_t_at_balacc": float(group["val_integral_t_at_balacc"].mean()),
            "std_val_integral_t_at_balacc": float(group["val_integral_t_at_balacc"].std(ddof=0)),
        })

    by_weight_df = pd.DataFrame(agg_rows)
    by_weight_df["missing_score"] = ~np.isfinite(by_weight_df["mean_val_integral_t_at_balacc"].to_numpy(dtype=float))
    by_weight_df = by_weight_df.sort_values(
        by=[
            "method", "mode", "missing_score", "num_seeds",
            "mean_val_integral_t_at_balacc", "mean_val_penalized_mean_t_at_balacc", "delta", "weight_key",
        ],
        ascending=[True, True, True, False, True, True, True, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    best_df = by_weight_df.drop_duplicates(subset=["method", "mode"], keep="first").drop(columns=["missing_score"]).reset_index(drop=True)
    by_weight_df = by_weight_df.drop(columns=["missing_score"])
    return range_df, by_seed_df, by_weight_df, best_df


def write_val_selection_summaries(
    rows: list[dict],
    save_dir: str,
    early_bal_acc_min: float = PARETO_INTEGRAL_EARLY_BAL_ACC_MIN,
    early_bal_acc_max: float = PARETO_INTEGRAL_EARLY_BAL_ACC_MAX,
    last_bal_acc_min: float = PARETO_INTEGRAL_LAST_BAL_ACC_MIN,
    last_bal_acc_max: float = PARETO_INTEGRAL_LAST_BAL_ACC_MAX,
    penalty_det_time: float = PENALIZED_DET_TIME,
) -> dict:
    candidates = collect_val_candidates_from_rows(rows)
    if not candidates:
        return {}

    roc_seed_df, roc_weight_df, roc_best_df = summarize_mean_val_ori_metric(candidates, primary_metric="roc")
    prc_seed_df, prc_weight_df, prc_best_df = summarize_mean_val_ori_metric(candidates, primary_metric="prc")
    range_df, pareto_seed_df, pareto_weight_df, pareto_best_df = summarize_mean_val_pareto(
        candidates,
        early_bal_acc_min=early_bal_acc_min,
        early_bal_acc_max=early_bal_acc_max,
        last_bal_acc_min=last_bal_acc_min,
        last_bal_acc_max=last_bal_acc_max,
        penalty_det_time=penalty_det_time,
    )

    roc_dir = os.path.join(save_dir, "roc_auc")
    prc_dir = os.path.join(save_dir, "prc_auc")
    pareto_dir = os.path.join(save_dir, "pareto")

    _write_dataframe(roc_seed_df, os.path.join(roc_dir, "candidates_by_seed.csv"))
    _write_dataframe(roc_weight_df, os.path.join(roc_dir, "weights_mean_across_seeds.csv"))
    _write_dataframe(roc_best_df, os.path.join(roc_dir, "best_weights.csv"))
    _save_payload(os.path.join(roc_dir, "summary.json"), {
        "strategy": "roc_auc",
        "num_candidates": int(len(roc_seed_df)),
        "num_weight_groups": int(len(roc_weight_df)),
        "best_weights": _to_jsonable_records(roc_best_df),
    })

    _write_dataframe(prc_seed_df, os.path.join(prc_dir, "candidates_by_seed.csv"))
    _write_dataframe(prc_weight_df, os.path.join(prc_dir, "weights_mean_across_seeds.csv"))
    _write_dataframe(prc_best_df, os.path.join(prc_dir, "best_weights.csv"))
    _save_payload(os.path.join(prc_dir, "summary.json"), {
        "strategy": "prc_auc",
        "num_candidates": int(len(prc_seed_df)),
        "num_weight_groups": int(len(prc_weight_df)),
        "best_weights": _to_jsonable_records(prc_best_df),
    })

    _write_dataframe(range_df, os.path.join(pareto_dir, "bal_acc_ranges_by_method_mode.csv"))
    _write_dataframe(pareto_seed_df, os.path.join(pareto_dir, "candidates_by_seed.csv"))
    _write_dataframe(pareto_weight_df, os.path.join(pareto_dir, "weights_mean_across_seeds.csv"))
    _write_dataframe(pareto_best_df, os.path.join(pareto_dir, "best_weights.csv"))
    _save_payload(os.path.join(pareto_dir, "summary.json"), {
        "strategy": "pareto_integral_t_at_balacc",
        "integral_early_bal_acc_min": float(early_bal_acc_min),
        "integral_early_bal_acc_max": float(early_bal_acc_max),
        "integral_last_bal_acc_min": float(last_bal_acc_min),
        "integral_last_bal_acc_max": float(last_bal_acc_max),
        "num_candidates": int(len(pareto_seed_df)),
        "num_weight_groups": int(len(pareto_weight_df)),
        "bal_acc_ranges": _to_jsonable_records(range_df),
        "best_weights": _to_jsonable_records(pareto_best_df),
    })

    return {
        "roc_auc_summary": os.path.join(roc_dir, "summary.json"),
        "roc_auc_best_weights": os.path.join(roc_dir, "best_weights.csv"),
        "prc_auc_summary": os.path.join(prc_dir, "summary.json"),
        "prc_auc_best_weights": os.path.join(prc_dir, "best_weights.csv"),
        "pareto_summary": os.path.join(pareto_dir, "summary.json"),
        "pareto_best_weights": os.path.join(pareto_dir, "best_weights.csv"),
    }


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

    selection_payload = write_val_selection_summaries(rows, save_dir)
    payload = {
        "logs_dir": logs_dir,
        "save_dir": save_dir,
        "num_runs": len(rows),
        "num_failures": len(failures),
        **selection_payload,
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
