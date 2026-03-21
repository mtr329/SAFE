import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.mrefine.delay_summary import (
    PENALIZED_DET_TIME,
    _compute_max_bal_acc_from_pareto,
    _compute_pareto_balacc_coverage,
    _compute_penalized_mean_t_at_balacc,
    _get_pareto_curve_from_alpha_dict,
    _select_auto_common_balacc_ranges,
)
from failure_prob.mrefine.new_summary import (
    _collect_new_log_paths,
    _get_run_meta_from_config,
    _split_new_logs_by_method,
    _summarize_new_calib,
)

IGNORED_DATASET_FIELDS = {
    "data_path",
    "data_path_prefix",
    "data_path_unseen",
    "use_cache",
    "refresh_cache",
    "cache_path",
    "cache_dir",
    "load_to_cuda",
    "normalize_hidden_states",
    "pred_horizon",
    "exec_horizon",
    "unseen_task_ratio",
    "seen_train_ratio",
    "dim_features",
    "dim_action",
    "failure_time_label_path",
}

IGNORED_TRAIN_FIELDS = {
    "seed",
    "wandb_project",
    "wandb_dir",
    "wandb_group_name",
    "exp_name",
    "debug",
    "vis_every",
    "roc_every",
    "eval_save_video",
    "eval_save_video_functional",
    "eval_save_video_multiproc",
    "eval_save_timing_plots",
    "eval_save_logs",
    "eval_save_ckpt",
    "eval_ckpt_path",
    "eval_split_path",
    "logs_save_root",
    "logs_save_path",
}


def _resolve_default_logs_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    for dirname in ("log_ckpt", "logs"):
        candidate = repo_root / dirname
        if candidate.is_dir():
            return str(candidate)
    return str(repo_root / "log_ckpt")


def _to_plain_dict(cfg_node) -> dict:
    return OmegaConf.to_container(cfg_node, resolve=False, throw_on_missing=False)


def _filter_cfg_dict(cfg_dict: dict, ignored_keys: set[str]) -> dict:
    return {k: v for k, v in cfg_dict.items() if k not in ignored_keys}


def _extract_seed(cfg, run_name: str) -> int | None:
    if cfg is not None:
        seed_value = getattr(cfg.train, "seed", None)
        if isinstance(seed_value, int):
            return int(seed_value)
        if isinstance(seed_value, str) and seed_value.isdigit():
            return int(seed_value)

    match = re.search(r"(?:^|/)seed(\d+)(?:/|$)", run_name)
    if match:
        return int(match.group(1))
    return None


def _seed_key(seed: int | None) -> str:
    if seed is None:
        return "unknown"
    return f"seed{seed}"


def build_candidate_signature(cfg, method_name: str) -> tuple[str, dict]:
    signature_payload = {
        "method_name": method_name,
        "dataset": _filter_cfg_dict(_to_plain_dict(cfg.dataset), IGNORED_DATASET_FIELDS),
        "model": _to_plain_dict(cfg.model),
        "train": _filter_cfg_dict(_to_plain_dict(cfg.train), IGNORED_TRAIN_FIELDS),
    }
    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(encoded.encode("utf-8")).hexdigest(), signature_payload


def collect_new_candidates(logs_dir: str) -> list[dict]:
    logs_dir = os.path.abspath(logs_dir)
    candidates = []
    for log_path in _collect_new_log_paths(logs_dir):
        with open(log_path, "r") as f:
            raw_logs = json.load(f)

        if os.path.basename(log_path) == "new_logs.json":
            new_logs = raw_logs
        else:
            if "new" not in raw_logs:
                continue
            new_logs = raw_logs["new"]

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        method_logs_by_name = _split_new_logs_by_method(new_logs, run_meta["method_name"])

        cfg = None
        cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(cfg_path):
            cfg = OmegaConf.load(cfg_path)
        seed = _extract_seed(cfg, run_name)

        for method_name, method_logs in sorted(method_logs_by_name.items()):
            calib_summary = _summarize_new_calib(method_logs)
            if not calib_summary:
                continue

            if cfg is not None:
                config_signature, signature_payload = build_candidate_signature(cfg, method_name)
                exp_suffix = getattr(cfg.train, "exp_suffix", None)
            else:
                config_signature = hashlib.md5(
                    f"{method_name}:{run_name}".encode("utf-8")
                ).hexdigest()
                signature_payload = {"method_name": method_name, "run_name": run_name}
                exp_suffix = None

            candidate_id = f"{method_name}::{run_name}"
            candidates.append({
                "candidate_id": candidate_id,
                "method_name": method_name,
                "run_name": run_name,
                "run_dir": os.path.abspath(run_dir),
                "log_path": os.path.abspath(log_path),
                "seed": seed,
                "exp_suffix": exp_suffix,
                "config_signature": config_signature,
                "signature_payload": signature_payload,
                "new_summary": {
                    "calib": calib_summary,
                },
            })

    return candidates


def compute_common_ranges(candidates: list[dict], min_curve_fraction: float) -> dict[str, dict]:
    range_stats_by_seed = {}
    for seed in sorted({candidate["seed"] for candidate in candidates}, key=lambda x: (x is None, x)):
        seed_candidates = [candidate for candidate in candidates if candidate["seed"] == seed]
        pseudo_summary = {
            candidate["candidate_id"]: candidate["new_summary"]
            for candidate in seed_candidates
        }
        range_stats_by_seed[_seed_key(seed)] = _select_auto_common_balacc_ranges(
            pseudo_summary,
            min_curve_fraction=min_curve_fraction,
        )
    return range_stats_by_seed


def summarize_candidate_scores(
    candidates: list[dict],
    range_stats: dict[str, dict],
    penalty_det_time: float = PENALIZED_DET_TIME,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        calib_summary = candidate["new_summary"]["calib"]
        seed_key = _seed_key(candidate["seed"])
        seed_range_stats = range_stats.get(seed_key, {})
        for mode in ("early", "last"):
            stats = seed_range_stats.get(mode, {})
            range_min = stats.get("bal_acc_min", np.nan)
            range_max = stats.get("bal_acc_max", np.nan)
            for delta, alpha_dict in sorted(calib_summary.get(mode, {}).items(), key=lambda x: float(x[0])):
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(alpha_dict)
                score = _compute_penalized_mean_t_at_balacc(
                    pareto_det_times,
                    pareto_bal_accs,
                    range_min,
                    range_max,
                    penalty_det_time=penalty_det_time,
                )
                coverage = _compute_pareto_balacc_coverage(
                    pareto_bal_accs,
                    range_min,
                    range_max,
                )
                max_bal_acc = _compute_max_bal_acc_from_pareto(pareto_bal_accs)
                rows.append({
                    "candidate_id": candidate["candidate_id"],
                    "method": candidate["method_name"],
                    "seed": candidate["seed"],
                    "mode": mode,
                    "delta": float(delta),
                    "run_name": candidate["run_name"],
                    "run_dir": candidate["run_dir"],
                    "log_path": candidate["log_path"],
                    "exp_suffix": candidate["exp_suffix"],
                    "config_signature": candidate["config_signature"],
                    "bal_acc_min": range_min,
                    "bal_acc_max": range_max,
                    "curve_fraction_target": stats.get("curve_fraction_target", np.nan),
                    "curve_fraction_achieved": stats.get("curve_fraction_achieved", np.nan),
                    "num_curves": stats.get("num_curves", np.nan),
                    "pareto_points": int(len(pareto_det_times)),
                    "pareto_coverage": coverage,
                    "pareto_max_bal_acc": max_bal_acc,
                    "penalty_det_time": float(penalty_det_time),
                    "penalized_mean_t_at_balacc": score,
                    "integral_t_at_balacc": (
                        score * (range_max - range_min)
                        if np.isfinite(score) and np.isfinite(range_min) and np.isfinite(range_max)
                        else np.nan
                    ),
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "method",
                "seed",
                "mode",
                "delta",
                "run_name",
                "run_dir",
                "log_path",
                "exp_suffix",
                "config_signature",
                "bal_acc_min",
                "bal_acc_max",
                "curve_fraction_target",
                "curve_fraction_achieved",
                "num_curves",
                "pareto_points",
                "pareto_coverage",
                "pareto_max_bal_acc",
                "penalty_det_time",
                "penalized_mean_t_at_balacc",
                "integral_t_at_balacc",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["method", "mode", "penalized_mean_t_at_balacc", "delta", "run_name"],
        na_position="last",
    ).reset_index(drop=True)


def select_best_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return score_df.copy()

    sortable = score_df.copy()
    sortable["score_for_sort"] = sortable["penalized_mean_t_at_balacc"].fillna(np.inf)
    sortable["coverage_for_sort"] = sortable["pareto_coverage"].fillna(-np.inf)
    sortable["max_bal_acc_for_sort"] = sortable["pareto_max_bal_acc"].fillna(-np.inf)
    sortable = sortable.sort_values(
        by=[
            "method",
            "seed",
            "mode",
            "score_for_sort",
            "coverage_for_sort",
            "max_bal_acc_for_sort",
            "delta",
            "run_name",
        ],
        ascending=[True, True, True, True, False, False, True, True],
        na_position="last",
    )
    selected = sortable.groupby(["method", "seed", "mode"], as_index=False).head(1).copy()
    selected = selected.drop(
        columns=["score_for_sort", "coverage_for_sort", "max_bal_acc_for_sort"]
    )
    return selected.reset_index(drop=True)


def summarize_selected_candidates(selection_df: pd.DataFrame) -> pd.DataFrame:
    if selection_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "mode",
                "num_seeds",
                "mean_penalized_mean_t_at_balacc",
                "std_penalized_mean_t_at_balacc",
                "mean_integral_t_at_balacc",
                "std_integral_t_at_balacc",
            ]
        )

    rows = []
    for (method, mode), group in selection_df.groupby(["method", "mode"], dropna=False):
        rows.append({
            "method": method,
            "mode": mode,
            "num_seeds": int(len(group)),
            "mean_penalized_mean_t_at_balacc": float(group["penalized_mean_t_at_balacc"].mean()),
            "std_penalized_mean_t_at_balacc": float(group["penalized_mean_t_at_balacc"].std(ddof=0)),
            "mean_integral_t_at_balacc": float(group["integral_t_at_balacc"].mean()),
            "std_integral_t_at_balacc": float(group["integral_t_at_balacc"].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(by=["mode", "mean_penalized_mean_t_at_balacc", "method"]).reset_index(drop=True)


def range_stats_to_df(range_stats: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for seed_key, seed_stats in range_stats.items():
        for mode in ("early", "last"):
            if mode not in seed_stats:
                continue
            row = dict(seed_stats[mode])
            row["seed_key"] = seed_key
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "seed_key",
                "mode",
                "curve_fraction_target",
                "curve_fraction_achieved",
                "num_curves",
                "bal_acc_min",
                "bal_acc_max",
                "bal_acc_width",
            ]
        )
    return pd.DataFrame(rows).sort_values(by=["seed_key", "mode"]).reset_index(drop=True)


def _selection_to_json_records(selection_df: pd.DataFrame) -> list[dict]:
    records = []
    for row in selection_df.to_dict(orient="records"):
        records.append({
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in row.items()
        })
    return records


def run_validation_pipeline(
    logs_dir: str,
    save_dir: str,
    min_curve_fraction: float,
    penalty_det_time: float,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)

    candidates = collect_new_candidates(logs_dir)
    range_stats = compute_common_ranges(candidates, min_curve_fraction=min_curve_fraction)
    score_df = summarize_candidate_scores(
        candidates,
        range_stats,
        penalty_det_time=penalty_det_time,
    )
    selection_df = select_best_candidates(score_df)
    selection_summary_df = summarize_selected_candidates(selection_df)
    range_df = range_stats_to_df(range_stats)

    score_df.to_csv(os.path.join(save_dir, "val_all_candidates.csv"), index=False)
    selection_df.to_csv(os.path.join(save_dir, "val_best_by_method_seed.csv"), index=False)
    selection_summary_df.to_csv(os.path.join(save_dir, "val_best_aggregate.csv"), index=False)
    range_df.to_csv(os.path.join(save_dir, "val_common_balacc_range.csv"), index=False)

    payload = {
        "logs_dir": os.path.abspath(logs_dir),
        "save_dir": os.path.abspath(save_dir),
        "penalty_det_time": float(penalty_det_time),
        "range_stats": range_stats,
        "selected": _selection_to_json_records(selection_df),
    }
    with open(os.path.join(save_dir, "val_selection.json"), "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best hyper-parameter setting for each method using Pareto T@BalAcc on validation logs.",
    )
    parser.add_argument(
        "--logs-dir",
        default=_resolve_default_logs_dir(),
        help="Root directory that contains run folders with eval/new_logs.json.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save validation summaries. Defaults to <logs-dir>/pipeline_val_new.",
    )
    parser.add_argument(
        "--min-curve-fraction",
        type=float,
        default=0.8,
        help="Minimum fraction of Pareto curves that must cover the chosen common BalAcc interval.",
    )
    parser.add_argument(
        "--penalty-det-time",
        type=float,
        default=PENALIZED_DET_TIME,
        help="Penalty detection time used when a Pareto curve cannot reach a target BalAcc.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.abspath(args.logs_dir), "pipeline_val_new")
    result = run_validation_pipeline(
        logs_dir=args.logs_dir,
        save_dir=save_dir,
        min_curve_fraction=args.min_curve_fraction,
        penalty_det_time=args.penalty_det_time,
    )

    print("Saved validation selection to", os.path.abspath(os.path.join(save_dir, "val_selection.json")))
    for record in result["selected"]:
        print(
            f"[{record['mode']}] {record['method']}: "
            f"run={record['run_name']} delta={record['delta']:.3f} "
            f"score={record['penalized_mean_t_at_balacc']:.6f}"
        )


if __name__ == "__main__":
    main()
