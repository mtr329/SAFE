import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.eval import collect_eval_dirs, evaluate_cfg
from failure_prob.mrefine.delay_summary import (
    PENALIZED_DET_TIME,
    _compute_max_bal_acc_from_pareto,
    _compute_pareto_balacc_coverage,
    _compute_penalized_mean_t_at_balacc,
    _get_pareto_curve_from_alpha_dict,
)
from failure_prob.mrefine.new_summary import _split_new_logs_by_method
from failure_prob.mrefine.ori_summary import _split_ori_logs_by_method
from failure_prob.pipeline.val_new import (
    VAL_SPLITS,
    _method_name,
    _metric_mean,
    _resolve_default_logs_dir,
    write_val_selection_summaries,
)
from failure_prob.utils.split_io import _dataset_hash_payload, load_split_signature

VAL_SPLIT = VAL_SPLITS[-1]
PARETO_INTEGRAL_BAL_ACC_MIN = 0.7
PARETO_INTEGRAL_BAL_ACC_MAX = 1.0


def _extract_seed(cfg, run_name: str) -> int | None:
    train_seed = None if cfg is None else getattr(cfg.train, "seed", None)
    if isinstance(train_seed, int):
        return int(train_seed)
    if isinstance(train_seed, str) and train_seed.isdigit() and "-" not in train_seed:
        return int(train_seed)

    for part in Path(run_name).parts:
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def _seed_sort_value(seed) -> tuple[int, int]:
    if seed is None or (isinstance(seed, float) and np.isnan(seed)):
        return (1, -1)
    return (0, int(seed))


def _weight_key_from_run_name(run_name: str) -> str:
    parts = list(Path(run_name).parts)
    if parts and parts[0].startswith("seed") and parts[0][4:].isdigit():
        parts = parts[1:]
    return str(Path(*parts)) if parts else run_name



def _resolve_val_summary_dir(logs_dir: str) -> str:
    return os.path.join(os.path.abspath(logs_dir), "pipeline_val_new")



def _ensure_val_best_weight_outputs(logs_dir: str) -> dict[str, str]:
    val_dir = _resolve_val_summary_dir(logs_dir)
    val_summary_path = os.path.join(val_dir, "val_summary.csv")
    if not os.path.isfile(val_summary_path):
        raise FileNotFoundError(
            f"Missing validation summary: {val_summary_path}. Run failure_prob/pipeline/val_new.py first."
        )

    expected = {
        "roc_auc": os.path.join(val_dir, "roc_auc", "best_weights.csv"),
        "prc_auc": os.path.join(val_dir, "prc_auc", "best_weights.csv"),
        "pareto": os.path.join(val_dir, "pareto", "best_weights.csv"),
    }
    if all(os.path.isfile(path) for path in expected.values()):
        return expected

    rows = pd.read_csv(val_summary_path).to_dict(orient="records")
    write_val_selection_summaries(rows, val_dir)
    missing = [path for path in expected.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing val best-weight outputs after rebuild: {missing}")
    return expected



def _load_best_weights_df(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing best-weight file: {path}")
    return pd.read_csv(path)



def _select_ori_best_weights(val_df: pd.DataFrame, best_weights_df: pd.DataFrame) -> pd.DataFrame:
    if val_df.empty or best_weights_df.empty:
        return pd.DataFrame(columns=list(val_df.columns) + [col for col in best_weights_df.columns if col not in val_df.columns])

    merge_cols = ["method", "weight_key"]
    selected_df = val_df.merge(best_weights_df, on=merge_cols, how="inner")
    if selected_df.empty:
        return selected_df
    return selected_df.sort_values(by=["method", "seed", "run_name"]).reset_index(drop=True)



def _select_pareto_best_weights(val_pareto_df: pd.DataFrame, best_weights_df: pd.DataFrame) -> pd.DataFrame:
    if val_pareto_df.empty or best_weights_df.empty:
        return pd.DataFrame(columns=list(val_pareto_df.columns) + [col for col in best_weights_df.columns if col not in val_pareto_df.columns])

    merge_cols = ["method", "mode", "weight_key", "delta"]
    selected_df = val_pareto_df.merge(best_weights_df, on=merge_cols, how="inner")
    if selected_df.empty:
        return selected_df
    return selected_df.sort_values(by=["method", "mode", "seed", "run_name"]).reset_index(drop=True)



def _to_jsonable_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
        })
    return records


def _find_delta_entry(delta_dict: dict, target_delta: float) -> tuple[str | None, dict | None]:
    for delta_key, alpha_dict in delta_dict.items():
        if np.isclose(float(delta_key), float(target_delta)):
            return delta_key, alpha_dict
    return None, None


def _mean_alpha_dict(alpha_dict: dict) -> dict:
    mean_alpha_dict = {}
    for alpha, metrics in alpha_dict.items():
        mean_alpha_dict[alpha] = {
            key: (value if key == "detect_method" else _metric_mean(value))
            for key, value in metrics.items()
        }
    return mean_alpha_dict


def _summarize_ori_metric(ori_logs: dict, split_name: str) -> tuple[float, float]:
    task_dict = ori_logs.get("static", {}).get(split_name, {}).get("all", {})
    return _metric_mean(task_dict.get("roc_auc", [])), _metric_mean(task_dict.get("prc_auc", []))


def collect_val_candidates(logs_dir: str) -> list[dict]:
    logs_dir = os.path.abspath(logs_dir)
    candidates = []
    run_dirs = [run_dir.resolve() for run_dir in collect_eval_dirs(Path(logs_dir))]
    pbar = tqdm(run_dirs, desc="Loading val logs", unit="run")
    for run_dir in pbar:
        val_dir = run_dir / "val"
        ori_path = val_dir / "ori_logs.json"
        new_path = val_dir / "new_logs.json"
        if not ori_path.is_file() or not new_path.is_file():
            continue

        cfg = None
        cfg_path = run_dir / "config.yaml"
        if cfg_path.is_file():
            cfg = OmegaConf.load(cfg_path)

        fallback_method = _method_name(cfg) if cfg is not None else run_dir.name
        run_name = os.path.relpath(str(run_dir), logs_dir)
        seed = _extract_seed(cfg, run_name)
        pbar.set_postfix_str(f"seed={seed} method={fallback_method} run={run_name}")

        with open(ori_path, "r") as f:
            ori_logs = json.load(f)
        with open(new_path, "r") as f:
            new_logs = json.load(f)

        ori_by_method = _split_ori_logs_by_method(ori_logs, fallback_method)
        new_by_method = _split_new_logs_by_method(new_logs, fallback_method)
        method_names = sorted(set(ori_by_method) | set(new_by_method))
        for method_name in method_names:
            candidates.append({
                "method": method_name,
                "seed": seed,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "val_dir": str(val_dir),
                "val_ori": ori_by_method.get(method_name, {"static": {}, "calib": {}}),
                "val_new": new_by_method.get(method_name, {"static": {}, "calib": {}}),
            })

    return sorted(
        candidates,
        key=lambda row: (row["method"], _seed_sort_value(row["seed"]), row["run_name"]),
    )


def summarize_val_roc_auc(candidates: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for candidate in candidates:
        roc_auc, prc_auc = _summarize_ori_metric(candidate["val_ori"], VAL_SPLIT)
        rows.append({
            "method": candidate["method"],
            "seed": candidate["seed"],
            "weight_key": _weight_key_from_run_name(candidate["run_name"]),
            "run_name": candidate["run_name"],
            "run_dir": candidate["run_dir"],
            "val_split": VAL_SPLIT,
            "val_roc_auc_early": roc_auc,
            "val_prc_auc_early": prc_auc,
        })

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        empty = pd.DataFrame(columns=[
            "method", "seed", "weight_key", "run_name", "run_dir", "val_split", "val_roc_auc_early", "val_prc_auc_early",
        ])
        return empty, empty

    score_df["missing_score"] = ~np.isfinite(score_df["val_roc_auc_early"].to_numpy(dtype=float))
    score_df = score_df.sort_values(
        by=["method", "seed", "missing_score", "val_roc_auc_early", "val_prc_auc_early", "run_name"],
        ascending=[True, True, True, False, False, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    selected_df = score_df.drop_duplicates(subset=["method", "seed"], keep="first").reset_index(drop=True)
    score_df = score_df.drop(columns=["missing_score"])
    selected_df = selected_df.drop(columns=["missing_score"])
    return score_df, selected_df


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
    bal_acc_min: float,
    bal_acc_max: float,
) -> tuple[dict[tuple[str, str], dict], pd.DataFrame]:
    intervals_by_group: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        calib_logs = candidate["val_new"].get("calib", {})
        for mode, delta_dict in calib_logs.items():
            group_key = (candidate["method"], mode)
            for delta_key, alpha_dict in delta_dict.items():
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(_mean_alpha_dict(alpha_dict))
                if pareto_det_times.size == 0 or pareto_bal_accs.size == 0:
                    continue
                intervals_by_group.setdefault(group_key, []).append({
                    "delta": float(delta_key),
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
        stats = _build_fixed_balacc_range(
            intervals_by_group.get((method, mode), []),
            method=method,
            mode=mode,
            bal_acc_min=bal_acc_min,
            bal_acc_max=bal_acc_max,
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


def score_pareto_candidates(
    candidates: list[dict],
    range_by_group: dict[tuple[str, str], dict],
    penalty_det_time: float,
    source_key: str,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        calib_logs = candidate[source_key].get("calib", {})
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
                    "weight_key": _weight_key_from_run_name(candidate["run_name"]),
                    "mode": mode,
                    "delta": float(delta_key),
                    "run_name": candidate["run_name"],
                    "run_dir": candidate["run_dir"],
                    f"{prefix}_bal_acc_min": bal_acc_min,
                    f"{prefix}_bal_acc_max": bal_acc_max,
                    f"{prefix}_bal_acc_width": stats["bal_acc_width"],
                    f"{prefix}_curve_fraction_achieved": stats["curve_fraction_achieved"],
                    f"{prefix}_penalty_det_time": float(penalty_det_time),
                    f"{prefix}_pareto_points": int(len(pareto_det_times)),
                    f"{prefix}_pareto_coverage": _compute_pareto_balacc_coverage(
                        pareto_bal_accs,
                        bal_acc_min,
                        bal_acc_max,
                    ),
                    f"{prefix}_pareto_max_bal_acc": _compute_max_bal_acc_from_pareto(pareto_bal_accs),
                    f"{prefix}_penalized_mean_t_at_balacc": score,
                    f"{prefix}_integral_t_at_balacc": integral,
                })

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return pd.DataFrame(columns=[
            "method", "seed", "weight_key", "mode", "delta", "run_name", "run_dir",
            f"{prefix}_bal_acc_min", f"{prefix}_bal_acc_max", f"{prefix}_bal_acc_width",
            f"{prefix}_curve_fraction_achieved", f"{prefix}_penalty_det_time", f"{prefix}_pareto_points",
            f"{prefix}_pareto_coverage", f"{prefix}_pareto_max_bal_acc",
            f"{prefix}_penalized_mean_t_at_balacc", f"{prefix}_integral_t_at_balacc",
        ])

    sort_col = f"{prefix}_penalized_mean_t_at_balacc"
    score_df["missing_score"] = ~np.isfinite(score_df[sort_col].to_numpy(dtype=float))
    score_df = score_df.sort_values(
        by=["method", "seed", "mode", "missing_score", sort_col, f"{prefix}_integral_t_at_balacc", "delta", "run_name"],
        ascending=[True, True, True, True, True, True, True, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    score_df = score_df.drop(columns=["missing_score"])
    return score_df


def select_best_val_pareto(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return score_df.copy()
    return score_df.drop_duplicates(subset=["method", "seed", "mode"], keep="first").reset_index(drop=True)


def select_weight_locked_val_pareto(selected_df: pd.DataFrame, val_pareto_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty or val_pareto_df.empty:
        return pd.DataFrame(columns=val_pareto_df.columns)

    locked_df = val_pareto_df.merge(
        selected_df[["method", "seed", "weight_key", "run_name", "run_dir"]].drop_duplicates(),
        on=["method", "seed", "weight_key", "run_name", "run_dir"],
        how="inner",
    )
    return select_best_val_pareto(locked_df)


def _resolve_saved_split_path(cfg, run_dir: str) -> str:
    seed = _extract_seed(cfg, run_dir)
    if seed is None:
        raise FileNotFoundError(f"Could not determine seed for run: {run_dir}")

    split_path = os.path.join(run_dir, f"split_seed{seed}.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Missing saved split for run: {split_path}")
    return split_path


def _validate_dataset_match(cfg, split_path: str, run_dir: str) -> None:
    signature = load_split_signature(split_path)
    expected = signature.get("data_payload")
    actual = _dataset_hash_payload(cfg)
    if actual != expected:
        raise ValueError(
            "Dataset config mismatch with saved split signature for "
            f"{run_dir}: expected={expected} actual={actual}"
        )


def ensure_test_eval(run_dir: str, force_eval: bool = False) -> str:
    eval_dir = os.path.join(run_dir, "eval")
    ori_path = os.path.join(eval_dir, "ori_logs.json")
    new_path = os.path.join(eval_dir, "new_logs.json")

    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.yaml for selected run: {run_dir}")

    cfg = OmegaConf.load(cfg_path)
    split_path = _resolve_saved_split_path(cfg, run_dir)
    _validate_dataset_match(cfg, split_path, run_dir)

    if not force_eval and os.path.isfile(ori_path) and os.path.isfile(new_path):
        return "cached"

    cfg.train.eval_ckpt_path = run_dir
    cfg.train.eval_split_path = split_path
    evaluate_cfg(cfg)

    if not os.path.isfile(ori_path) or not os.path.isfile(new_path):
        raise FileNotFoundError(f"Expected eval outputs were not written under {eval_dir}")
    return "ran"


def load_test_logs(run_dir: str, method: str) -> dict:
    eval_dir = os.path.join(run_dir, "eval")
    ori_path = os.path.join(eval_dir, "ori_logs.json")
    new_path = os.path.join(eval_dir, "new_logs.json")
    with open(ori_path, "r") as f:
        ori_logs = json.load(f)
    with open(new_path, "r") as f:
        new_logs = json.load(f)

    ori_by_method = _split_ori_logs_by_method(ori_logs, method)
    new_by_method = _split_new_logs_by_method(new_logs, method)
    if method not in ori_by_method:
        raise KeyError(f"Method {method} missing from {ori_path}")
    if method not in new_by_method:
        raise KeyError(f"Method {method} missing from {new_path}")
    return {
        "ori": ori_by_method[method],
        "new": new_by_method[method],
        "eval_dir": eval_dir,
        "ori_path": ori_path,
        "new_path": new_path,
    }


def _filter_available_selected(selected_df: pd.DataFrame, test_logs_by_key: dict) -> pd.DataFrame:
    if selected_df.empty:
        return selected_df.copy()
    mask = [
        (row.run_dir, row.method) in test_logs_by_key
        for row in selected_df.itertuples(index=False)
    ]
    return selected_df.loc[mask].reset_index(drop=True)


def summarize_selected_test_ori(selected_df: pd.DataFrame, test_logs_by_key: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    include_mode = "mode" in selected_df.columns
    for row in selected_df.itertuples(index=False):
        test_logs = test_logs_by_key[(row.run_dir, row.method)]
        ori_logs = test_logs["ori"]
        for split_name in VAL_SPLITS:
            roc_auc, prc_auc = _summarize_ori_metric(ori_logs, split_name)
            payload = {
                "method": row.method,
                "seed": row.seed,
                "run_name": row.run_name,
                "run_dir": row.run_dir,
                "split": split_name,
                "test_roc_auc_early": roc_auc,
                "test_prc_auc_early": prc_auc,
            }
            if include_mode:
                payload["mode"] = row.mode
            rows.append(payload)

    ori_df = pd.DataFrame(rows)
    if ori_df.empty:
        cols = ["method", "seed", "run_name", "run_dir", "split", "test_roc_auc_early", "test_prc_auc_early"]
        if include_mode:
            cols.insert(2, "mode")
        empty = pd.DataFrame(columns=cols)
        return empty, pd.DataFrame(columns=[
            "method", *( ["mode"] if include_mode else [] ), "split", "num_runs",
            "mean_test_roc_auc_early", "std_test_roc_auc_early", "mean_test_prc_auc_early", "std_test_prc_auc_early",
        ])

    group_cols = ["method"]
    if include_mode:
        group_cols.append("mode")
    group_cols.append("split")

    agg_rows = []
    for group_key, group in ori_df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        payload = {col: value for col, value in zip(group_cols, group_key)}
        payload.update({
            "num_runs": int(len(group)),
            "mean_test_roc_auc_early": float(group["test_roc_auc_early"].mean()),
            "std_test_roc_auc_early": float(group["test_roc_auc_early"].std(ddof=0)),
            "mean_test_prc_auc_early": float(group["test_prc_auc_early"].mean()),
            "std_test_prc_auc_early": float(group["test_prc_auc_early"].std(ddof=0)),
        })
        agg_rows.append(payload)
    agg_df = pd.DataFrame(agg_rows).sort_values(by=group_cols).reset_index(drop=True)
    ori_df = ori_df.sort_values(by=[col for col in ["method", "mode" if include_mode else None, "seed", "split", "run_name"] if col]).reset_index(drop=True)
    return ori_df, agg_df


def _merge_selected_ori_logs(selected_df: pd.DataFrame, test_logs_by_key: dict) -> dict:
    merged = {"static": {}, "calib": {}}
    for row in selected_df.itertuples(index=False):
        ori_logs = test_logs_by_key[(row.run_dir, row.method)]["ori"]
        for split_name, split_dict in ori_logs.get("static", {}).items():
            task_dict = split_dict.get("all", {})
            dst = merged["static"].setdefault(row.method, {}).setdefault(split_name, {}).setdefault("all", {})
            for key in ("roc_auc", "prc_auc", "fpr", "tpr", "rec", "pre"):
                dst.setdefault(key, [])
                dst[key].extend(task_dict.get(key, []))
    return merged


def _remove_test_curve_outputs(save_dir: str) -> None:
    if not os.path.isdir(save_dir):
        return
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if not os.path.isfile(path):
            continue
        if name in {"ori_roc_curve_points.csv", "ori_pr_curve_points.csv"}:
            os.remove(path)
            continue
        if name == "avg_det_time_vs_bal_acc_points.csv":
            os.remove(path)
            continue
        if not name.startswith("ori_"):
            if name.startswith("avg_det_time_vs_bal_acc_") and name.endswith(".png"):
                os.remove(path)
            continue
        if name.endswith("_roc.png") or name.endswith("_pr.png"):
            os.remove(path)


def _merge_mean_alpha_dicts(alpha_dicts: list[dict]) -> dict:
    merged = {}
    for alpha_dict in alpha_dicts:
        for alpha, metrics in _mean_alpha_dict(alpha_dict).items():
            alpha_key = float(alpha)
            dst = merged.setdefault(alpha_key, {})
            for metric_name, value in metrics.items():
                if metric_name == "detect_method":
                    dst[metric_name] = value
                else:
                    dst.setdefault(metric_name, []).append(float(value))

    output = {}
    for alpha_key in sorted(merged):
        metrics = merged[alpha_key]
        output[alpha_key] = {
            metric_name: (value if metric_name == "detect_method" else float(np.mean(value)))
            for metric_name, value in metrics.items()
        }
    return output


def _safe_file_stem(value) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _mean_ori_curve_points(selected_df: pd.DataFrame, test_logs_by_key: dict, eval_time: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_to_avg_det_times: dict[float, list[float]] = {}
    alpha_to_bal_accs: dict[float, list[float]] = {}

    for row in selected_df.itertuples(index=False):
        test_logs = test_logs_by_key.get((row.run_dir, row.method))
        if test_logs is None:
            continue
        calib_logs = test_logs["ori"].get("calib", {}).get(eval_time, {})
        if not calib_logs:
            continue
        for alpha_str, alpha_dict in calib_logs.items():
            if "avg_det_time" not in alpha_dict or "bal_acc" not in alpha_dict:
                continue
            alpha = float(alpha_str)
            alpha_to_avg_det_times.setdefault(alpha, []).append(
                float(np.asarray(alpha_dict["avg_det_time"], dtype=float).mean())
            )
            alpha_to_bal_accs.setdefault(alpha, []).append(
                float(np.asarray(alpha_dict["bal_acc"], dtype=float).mean())
            )

    rows = []
    for alpha in sorted(alpha_to_avg_det_times):
        if alpha not in alpha_to_bal_accs:
            continue
        rows.append((
            float(alpha),
            float(np.mean(alpha_to_avg_det_times[alpha])),
            float(np.mean(alpha_to_bal_accs[alpha])),
        ))

    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    alpha_values = np.asarray([row[0] for row in rows], dtype=float)
    avg_det_times = np.asarray([row[1] for row in rows], dtype=float)
    bal_accs = np.asarray([row[2] for row in rows], dtype=float)
    return alpha_values, avg_det_times, bal_accs


def save_test_selection_curves(
    save_dir: str,
    roc_selected_df: pd.DataFrame,
    prc_selected_df: pd.DataFrame,
    pareto_selected_df: pd.DataFrame,
    test_logs_by_key: dict,
) -> dict[str, str]:
    plot_dir = os.path.join(save_dir, "selection_curves")
    os.makedirs(plot_dir, exist_ok=True)
    _remove_test_curve_outputs(plot_dir)

    curve_rows = []
    outputs: dict[str, str] = {"selection_curves_dir": plot_dir}
    methods = sorted({
        *roc_selected_df.get("method", pd.Series(dtype=object)).dropna().tolist(),
        *prc_selected_df.get("method", pd.Series(dtype=object)).dropna().tolist(),
        *pareto_selected_df.get("method", pd.Series(dtype=object)).dropna().tolist(),
    }, key=str)

    strategy_inputs = [
        ("roc_auc", "Selected By Val ROC", "#1f77b4", "-", "o", 0.55, roc_selected_df),
        ("prc_auc", "Selected By Val PRC", "#ff7f0e", "--", "s", 0.55, prc_selected_df),
        ("pareto", "Selected By Val Integral", "#2ca02c", "-.", "^", 0.55, pareto_selected_df),
    ]

    for eval_time in ("early", "last"):
        for method in methods:
            fig = Figure(figsize=(7, 5))
            ax = fig.subplots()
            plotted = False

            for strategy_name, label, color, linestyle, marker, alpha, selected_df in strategy_inputs:
                if selected_df.empty or "method" not in selected_df.columns:
                    continue
                use_df = selected_df.loc[selected_df["method"] == method].copy()
                if use_df.empty:
                    continue
                if strategy_name == "pareto":
                    if "mode" not in use_df.columns:
                        continue
                    use_df = use_df.loc[use_df["mode"] == eval_time].copy()
                    if use_df.empty:
                        continue

                alpha_values, avg_det_times, bal_accs = _mean_ori_curve_points(use_df, test_logs_by_key, eval_time)
                if avg_det_times.size == 0 or bal_accs.size == 0:
                    continue

                plotted = True
                ax.plot(
                    avg_det_times,
                    bal_accs,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.0,
                    markersize=5,
                    label=label,
                    color=color,
                    alpha=alpha,
                    markeredgewidth=0.8,
                    markeredgecolor=color,
                    markerfacecolor="white",
                )
                for alpha, avg_det_time, bal_acc in zip(alpha_values, avg_det_times, bal_accs):
                    curve_rows.append({
                        "method": method,
                        "mode": eval_time,
                        "selection_strategy": strategy_name,
                        "alpha": float(alpha),
                        "avg_det_time": float(avg_det_time),
                        "bal_acc": float(bal_acc),
                    })

            if not plotted:
                continue

            ax.set_xlabel("Average Detection Time")
            ax.set_ylabel("Balanced Accuracy")
            ax.set_title(f"{method}: Test Average Detection Time vs Balanced Accuracy ({eval_time})")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
            fig.tight_layout()

            filename = f"{_safe_file_stem(method)}_avg_det_time_vs_bal_acc_{eval_time}.png"
            path_out = os.path.join(plot_dir, filename)
            fig.savefig(path_out, dpi=300)
            outputs[f"{_safe_file_stem(method)}_{eval_time}_selection_curve"] = path_out

    if curve_rows:
        points_path = os.path.join(plot_dir, "selection_curve_points.csv")
        _write_dataframe(
            pd.DataFrame(curve_rows).sort_values(by=["method", "mode", "selection_strategy", "alpha"]).reset_index(drop=True),
            points_path,
        )
        outputs["selection_curve_points"] = points_path

    return outputs


_SELECTION_STRATEGY_SPECS = [
    ("roc_auc", "Selected By Val ROC", "#1f77b4"),
    ("prc_auc", "Selected By Val PRC", "#ff7f0e"),
    ("pareto", "Selected By Val Integral", "#2ca02c"),
]


def _selection_barplot_rows(
    roc_best_test_df: pd.DataFrame,
    prc_best_test_df: pd.DataFrame,
    pareto_best_test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add_rows(df: pd.DataFrame, strategy: str, metric_name: str, value_col: str, mode: str | None = None) -> None:
        if df.empty or value_col not in df.columns:
            return
        use_df = df
        if mode is not None:
            if "mode" not in use_df.columns:
                return
            use_df = use_df.loc[use_df["mode"] == mode].copy()
        if use_df.empty:
            return
        for method, group in use_df.groupby("method", dropna=False):
            values = pd.to_numeric(group[value_col], errors="coerce")
            rows.append({
                "method": method,
                "selection_strategy": strategy,
                "metric": metric_name,
                "mode": mode,
                "num_runs": int(values.notna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            })

    add_rows(roc_best_test_df, "roc_auc", "test_roc_auc", "test_roc_auc_early")
    add_rows(prc_best_test_df, "prc_auc", "test_roc_auc", "test_roc_auc_early")
    add_rows(pareto_best_test_df, "pareto", "test_roc_auc", "test_roc_auc_early", mode="early")

    add_rows(roc_best_test_df, "roc_auc", "test_prc_auc", "test_prc_auc_early")
    add_rows(prc_best_test_df, "prc_auc", "test_prc_auc", "test_prc_auc_early")
    add_rows(pareto_best_test_df, "pareto", "test_prc_auc", "test_prc_auc_early", mode="early")

    add_rows(roc_best_test_df, "roc_auc", "test_integral_t_at_balacc_early", "test_integral_t_at_balacc_early")
    add_rows(prc_best_test_df, "prc_auc", "test_integral_t_at_balacc_early", "test_integral_t_at_balacc_early")
    add_rows(pareto_best_test_df, "pareto", "test_integral_t_at_balacc_early", "test_integral_t_at_balacc", mode="early")

    add_rows(roc_best_test_df, "roc_auc", "test_integral_t_at_balacc_last", "test_integral_t_at_balacc_last")
    add_rows(prc_best_test_df, "prc_auc", "test_integral_t_at_balacc_last", "test_integral_t_at_balacc_last")
    add_rows(pareto_best_test_df, "pareto", "test_integral_t_at_balacc_last", "test_integral_t_at_balacc", mode="last")

    if not rows:
        return pd.DataFrame(columns=["method", "selection_strategy", "metric", "mode", "num_runs", "mean", "std"])
    return pd.DataFrame(rows).sort_values(by=["metric", "method", "selection_strategy"]).reset_index(drop=True)


def _save_selection_metric_barplot(plot_df: pd.DataFrame, save_path: str, title: str, ylabel: str, lower_is_better: bool = False) -> None:
    strategy_specs = list(_SELECTION_STRATEGY_SPECS)
    methods = sorted(plot_df["method"].dropna().unique().tolist())
    if not methods:
        return

    width = 0.24
    x = np.arange(len(methods), dtype=float)
    fig_width = max(9.0, 1.0 * len(methods) + 2.0)
    fig = Figure(figsize=(fig_width, 5.5))
    ax = fig.subplots()

    for idx, (strategy, label, color) in enumerate(strategy_specs):
        offsets = x + (idx - 1) * width
        heights = []
        errors = []
        positions = []
        for method, xpos in zip(methods, offsets):
            row = plot_df.loc[
                (plot_df["method"] == method) & (plot_df["selection_strategy"] == strategy)
            ]
            if row.empty:
                continue
            mean = float(row["mean"].iloc[0])
            if not np.isfinite(mean):
                continue
            std = float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else 0.0
            positions.append(xpos)
            heights.append(mean)
            errors.append(std if np.isfinite(std) else 0.0)
        if positions:
            ax.bar(positions, heights, width=width, label=label, color=color, yerr=errors, capsize=3)

    title_suffix = " (lower is better)" if lower_is_better else ""
    ax.set_title(f"{title}{title_suffix}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def save_test_selection_barplots(
    save_dir: str,
    roc_best_test_df: pd.DataFrame,
    prc_best_test_df: pd.DataFrame,
    pareto_best_test_df: pd.DataFrame,
) -> dict[str, str]:
    plot_dir = os.path.join(save_dir, "selection_barplots")
    os.makedirs(plot_dir, exist_ok=True)

    summary_df = _selection_barplot_rows(roc_best_test_df, prc_best_test_df, pareto_best_test_df)
    summary_path = os.path.join(plot_dir, "selection_barplot_summary.csv")
    _write_dataframe(summary_df, summary_path)

    outputs = {"selection_barplot_summary": summary_path}
    metric_specs = [
        ("test_roc_auc", "Test ROC-AUC By Val Selection", "Test ROC-AUC", False, "test_roc_auc_by_val_selection.png"),
        ("test_prc_auc", "Test PRC-AUC By Val Selection", "Test PRC-AUC", False, "test_prc_auc_by_val_selection.png"),
        ("test_integral_t_at_balacc_early", "Test Integral t@bal_acc By Val Selection (early)", "Test Integral t@bal_acc", True, "test_integral_t_at_balacc_early_by_val_selection.png"),
        ("test_integral_t_at_balacc_last", "Test Integral t@bal_acc By Val Selection (last)", "Test Integral t@bal_acc", True, "test_integral_t_at_balacc_last_by_val_selection.png"),
    ]
    for metric, title, ylabel, lower_is_better, filename in metric_specs:
        metric_df = summary_df.loc[summary_df["metric"] == metric].copy()
        if metric_df.empty:
            continue
        path = os.path.join(plot_dir, filename)
        _save_selection_metric_barplot(metric_df, path, title, ylabel, lower_is_better=lower_is_better)
        outputs[metric] = path
    return outputs


def summarize_selected_test_pareto(
    selected_df: pd.DataFrame,
    test_logs_by_key: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in selected_df.itertuples(index=False):
        test_logs = test_logs_by_key[(row.run_dir, row.method)]
        calib_logs = test_logs["new"].get("calib", {}).get(row.mode, {})
        delta_key, alpha_dict = _find_delta_entry(calib_logs, float(row.delta))
        if alpha_dict is None:
            rows.append({
                "method": row.method,
                "seed": row.seed,
                "mode": row.mode,
                "delta": float(row.delta),
                "run_name": row.run_name,
                "run_dir": row.run_dir,
                "test_bal_acc_min": row.val_bal_acc_min,
                "test_bal_acc_max": row.val_bal_acc_max,
                "test_pareto_points": 0,
                "test_pareto_coverage": 0.0,
                "test_pareto_max_bal_acc": np.nan,
                "test_penalized_mean_t_at_balacc": np.nan,
                "test_integral_t_at_balacc": np.nan,
                "missing_delta": True,
            })
            continue

        pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(_mean_alpha_dict(alpha_dict))
        score = _compute_penalized_mean_t_at_balacc(
            pareto_det_times,
            pareto_bal_accs,
            row.val_bal_acc_min,
            row.val_bal_acc_max,
            penalty_det_time=row.val_penalty_det_time,
        )
        rows.append({
            "method": row.method,
            "seed": row.seed,
            "mode": row.mode,
            "delta": float(delta_key),
            "run_name": row.run_name,
            "run_dir": row.run_dir,
            "test_bal_acc_min": row.val_bal_acc_min,
            "test_bal_acc_max": row.val_bal_acc_max,
            "test_pareto_points": int(len(pareto_det_times)),
            "test_pareto_coverage": _compute_pareto_balacc_coverage(
                pareto_bal_accs,
                row.val_bal_acc_min,
                row.val_bal_acc_max,
            ),
            "test_pareto_max_bal_acc": _compute_max_bal_acc_from_pareto(pareto_bal_accs),
            "test_penalized_mean_t_at_balacc": score,
            "test_integral_t_at_balacc": (
                score * (row.val_bal_acc_max - row.val_bal_acc_min)
                if np.isfinite(score) and np.isfinite(row.val_bal_acc_min) and np.isfinite(row.val_bal_acc_max)
                else np.nan
            ),
            "missing_delta": False,
        })

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        empty = pd.DataFrame(columns=[
            "method", "seed", "mode", "delta", "run_name", "run_dir",
            "test_bal_acc_min", "test_bal_acc_max", "test_pareto_points", "test_pareto_coverage",
            "test_pareto_max_bal_acc", "test_penalized_mean_t_at_balacc", "test_integral_t_at_balacc", "missing_delta",
        ])
        return empty, pd.DataFrame(columns=[
            "method", "mode", "num_runs", "mean_test_penalized_mean_t_at_balacc", "std_test_penalized_mean_t_at_balacc",
            "mean_test_integral_t_at_balacc", "std_test_integral_t_at_balacc",
        ])

    agg_rows = []
    for (method, mode), group in score_df.groupby(["method", "mode"], dropna=False):
        agg_rows.append({
            "method": method,
            "mode": mode,
            "num_runs": int(len(group)),
            "mean_test_penalized_mean_t_at_balacc": float(group["test_penalized_mean_t_at_balacc"].mean()),
            "std_test_penalized_mean_t_at_balacc": float(group["test_penalized_mean_t_at_balacc"].std(ddof=0)),
            "mean_test_integral_t_at_balacc": float(group["test_integral_t_at_balacc"].mean()),
            "std_test_integral_t_at_balacc": float(group["test_integral_t_at_balacc"].std(ddof=0)),
        })
    agg_df = pd.DataFrame(agg_rows).sort_values(by=["mode", "method"]).reset_index(drop=True)
    score_df = score_df.sort_values(by=["method", "mode", "seed", "run_name"]).reset_index(drop=True)
    return score_df, agg_df


def _extract_test_split_metrics(ori_df: pd.DataFrame) -> pd.DataFrame:
    if ori_df.empty:
        cols = [col for col in ori_df.columns if col != "split"]
        return pd.DataFrame(columns=["test_split", *cols])

    test_split_df = ori_df.loc[ori_df["split"] == VAL_SPLIT].copy()
    test_split_df = test_split_df.rename(columns={"split": "test_split"})
    sort_cols = [col for col in ["method", "mode", "seed", "run_name"] if col in test_split_df.columns]
    if sort_cols:
        test_split_df = test_split_df.sort_values(by=sort_cols).reset_index(drop=True)
    return test_split_df


def _flatten_test_pareto_metrics_by_mode(test_pareto_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["method", "seed", "run_name", "run_dir"]
    if test_pareto_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (method, seed, run_name, run_dir), group in test_pareto_df.groupby(
        ["method", "seed", "run_name", "run_dir"],
        dropna=False,
    ):
        row = {
            "method": method,
            "seed": seed,
            "run_name": run_name,
            "run_dir": run_dir,
        }
        for mode, mode_group in group.groupby("mode", dropna=False):
            safe_mode = str(mode)
            if mode_group.empty:
                continue
            first = mode_group.iloc[0]
            row[f"test_delta_{safe_mode}"] = first.get("delta", np.nan)
            row[f"test_bal_acc_min_{safe_mode}"] = first.get("test_bal_acc_min", np.nan)
            row[f"test_bal_acc_max_{safe_mode}"] = first.get("test_bal_acc_max", np.nan)
            row[f"test_pareto_points_{safe_mode}"] = first.get("test_pareto_points", np.nan)
            row[f"test_pareto_coverage_{safe_mode}"] = first.get("test_pareto_coverage", np.nan)
            row[f"test_pareto_max_bal_acc_{safe_mode}"] = first.get("test_pareto_max_bal_acc", np.nan)
            row[f"test_penalized_mean_t_at_balacc_{safe_mode}"] = first.get("test_penalized_mean_t_at_balacc", np.nan)
            row[f"test_integral_t_at_balacc_{safe_mode}"] = first.get("test_integral_t_at_balacc", np.nan)
            row[f"test_missing_delta_{safe_mode}"] = first.get("missing_delta", np.nan)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by=["method", "seed", "run_name"]).reset_index(drop=True)


def build_roc_best_test_metrics_by_seed(
    selected_df: pd.DataFrame,
    test_ori_df: pd.DataFrame,
    test_pareto_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if selected_df.empty:
        base_columns = [
            "method",
            "seed",
            "run_name",
            "run_dir",
            "val_split",
            "val_roc_auc_early",
            "val_prc_auc_early",
            "test_split",
            "test_roc_auc_early",
            "test_prc_auc_early",
        ]
        extra_columns = [] if test_pareto_df is None else [
            "test_delta_early",
            "test_bal_acc_min_early",
            "test_bal_acc_max_early",
            "test_pareto_points_early",
            "test_pareto_coverage_early",
            "test_pareto_max_bal_acc_early",
            "test_penalized_mean_t_at_balacc_early",
            "test_integral_t_at_balacc_early",
            "test_missing_delta_early",
            "test_delta_last",
            "test_bal_acc_min_last",
            "test_bal_acc_max_last",
            "test_pareto_points_last",
            "test_pareto_coverage_last",
            "test_pareto_max_bal_acc_last",
            "test_penalized_mean_t_at_balacc_last",
            "test_integral_t_at_balacc_last",
            "test_missing_delta_last",
        ]
        return pd.DataFrame(columns=[*base_columns, *extra_columns])

    test_split_df = _extract_test_split_metrics(test_ori_df)
    metric_cols = ["method", "seed", "run_name", "run_dir", "test_split", "test_roc_auc_early", "test_prc_auc_early"]
    if not test_split_df.empty:
        test_split_df = test_split_df[metric_cols]

    summary_df = selected_df.merge(
        test_split_df,
        on=["method", "seed", "run_name", "run_dir"],
        how="left",
    )
    if test_pareto_df is not None:
        flat_pareto_df = _flatten_test_pareto_metrics_by_mode(test_pareto_df)
        summary_df = summary_df.merge(
            flat_pareto_df,
            on=["method", "seed", "run_name", "run_dir"],
            how="left",
        )
    return summary_df.sort_values(by=["method", "seed", "run_name"]).reset_index(drop=True)


def build_pareto_best_test_metrics_by_seed(
    selected_df: pd.DataFrame,
    test_ori_df: pd.DataFrame,
    test_pareto_df: pd.DataFrame,
) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame(columns=[
            "method",
            "seed",
            "mode",
            "delta",
            "run_name",
            "run_dir",
            "val_bal_acc_min",
            "val_bal_acc_max",
            "val_bal_acc_width",
            "val_curve_fraction_achieved",
            "val_penalty_det_time",
            "val_pareto_points",
            "val_pareto_coverage",
            "val_pareto_max_bal_acc",
            "val_penalized_mean_t_at_balacc",
            "val_integral_t_at_balacc",
            "test_split",
            "test_roc_auc_early",
            "test_prc_auc_early",
            "test_bal_acc_min",
            "test_bal_acc_max",
            "test_pareto_points",
            "test_pareto_coverage",
            "test_pareto_max_bal_acc",
            "test_penalized_mean_t_at_balacc",
            "test_integral_t_at_balacc",
            "missing_delta",
        ])

    test_split_df = _extract_test_split_metrics(test_ori_df)
    if not test_split_df.empty:
        test_split_df = test_split_df[
            [
                "method",
                "seed",
                "mode",
                "run_name",
                "run_dir",
                "test_split",
                "test_roc_auc_early",
                "test_prc_auc_early",
            ]
        ]

    summary_df = selected_df.merge(
        test_split_df,
        on=["method", "seed", "mode", "run_name", "run_dir"],
        how="left",
    )
    summary_df = summary_df.merge(
        test_pareto_df,
        on=["method", "seed", "mode", "delta", "run_name", "run_dir"],
        how="left",
    )
    return summary_df.sort_values(by=["method", "mode", "seed", "run_name"]).reset_index(drop=True)


def build_selected_eval_manifest(
    eval_df: pd.DataFrame,
    roc_selected_df: pd.DataFrame,
    prc_selected_df: pd.DataFrame,
    pareto_selected_df: pd.DataFrame,
) -> pd.DataFrame:
    records = {}

    for row in eval_df.itertuples(index=False):
        key = (row.run_dir, row.method)
        records[key] = {
            "method": row.method,
            "seed": _extract_seed(None, row.run_name),
            "run_name": row.run_name,
            "run_dir": row.run_dir,
            "eval_status": row.eval_status,
            "eval_dir": row.eval_dir,
            "ori_path": row.ori_path,
            "new_path": row.new_path,
            "selected_for_roc_auc": False,
            "selected_for_prc_auc": False,
            "selected_for_pareto": False,
            "selected_pareto_modes": "",
            "selected_pareto_deltas": "",
        }

    for row in roc_selected_df.itertuples(index=False):
        key = (row.run_dir, row.method)
        record = records.get(key)
        if record is None:
            continue
        record["seed"] = row.seed
        record["selected_for_roc_auc"] = True

    for row in prc_selected_df.itertuples(index=False):
        key = (row.run_dir, row.method)
        record = records.get(key)
        if record is None:
            continue
        record["seed"] = row.seed
        record["selected_for_prc_auc"] = True

    for row in pareto_selected_df.itertuples(index=False):
        key = (row.run_dir, row.method)
        record = records.get(key)
        if record is None:
            continue
        record["seed"] = row.seed
        record["selected_for_pareto"] = True
        modes = [value for value in record["selected_pareto_modes"].split(",") if value]
        deltas = [value for value in record["selected_pareto_deltas"].split(",") if value]
        mode_value = str(row.mode)
        delta_value = str(float(row.delta))
        if mode_value not in modes:
            modes.append(mode_value)
        if delta_value not in deltas:
            deltas.append(delta_value)
        record["selected_pareto_modes"] = ",".join(sorted(modes))
        record["selected_pareto_deltas"] = ",".join(sorted(deltas, key=float))

    manifest_df = pd.DataFrame(records.values())
    if manifest_df.empty:
        return pd.DataFrame(columns=[
            "method",
            "seed",
            "run_name",
            "run_dir",
            "eval_status",
            "eval_dir",
            "ori_path",
            "new_path",
            "selected_for_roc_auc",
            "selected_for_prc_auc",
            "selected_for_pareto",
            "selected_pareto_modes",
            "selected_pareto_deltas",
        ])
    return manifest_df.sort_values(by=["method", "seed", "run_name"]).reset_index(drop=True)


def _write_dataframe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _save_strategy_payload(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_test_pipeline(
    logs_dir: str,
    save_dir: str,
    force_eval: bool = False,
    bal_acc_min: float = PARETO_INTEGRAL_BAL_ACC_MIN,
    bal_acc_max: float = PARETO_INTEGRAL_BAL_ACC_MAX,
    penalty_det_time: float = PENALIZED_DET_TIME,
) -> dict:
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    candidates = collect_val_candidates(logs_dir)
    if not candidates:
        raise FileNotFoundError(f"No run directories with val/ori_logs.json and val/new_logs.json found under {logs_dir}")

    best_weight_paths = _ensure_val_best_weight_outputs(logs_dir)
    val_ori_df, _ = summarize_val_roc_auc(candidates)
    roc_best_weights_df = _load_best_weights_df(best_weight_paths["roc_auc"])
    prc_best_weights_df = _load_best_weights_df(best_weight_paths["prc_auc"])
    roc_selected_df = _select_ori_best_weights(val_ori_df, roc_best_weights_df)
    prc_selected_df = _select_ori_best_weights(val_ori_df, prc_best_weights_df)

    range_by_group, range_df = build_val_pareto_ranges(
        candidates,
        bal_acc_min=bal_acc_min,
        bal_acc_max=bal_acc_max,
    )
    val_pareto_df = score_pareto_candidates(
        candidates,
        range_by_group=range_by_group,
        penalty_det_time=penalty_det_time,
        source_key="val_new",
        prefix="val",
    )
    pareto_best_weights_df = _load_best_weights_df(best_weight_paths["pareto"])
    pareto_selected_df = _select_pareto_best_weights(val_pareto_df, pareto_best_weights_df)

    selected_pairs = {}
    for row in roc_selected_df.itertuples(index=False):
        selected_pairs[(row.run_dir, row.method)] = {"run_dir": row.run_dir, "method": row.method, "run_name": row.run_name}
    for row in prc_selected_df.itertuples(index=False):
        selected_pairs[(row.run_dir, row.method)] = {"run_dir": row.run_dir, "method": row.method, "run_name": row.run_name}
    for row in pareto_selected_df.itertuples(index=False):
        selected_pairs[(row.run_dir, row.method)] = {"run_dir": row.run_dir, "method": row.method, "run_name": row.run_name}

    test_logs_by_key = {}
    eval_rows = []
    eval_failures = []
    selected_items = sorted(selected_pairs.items(), key=lambda kv: (kv[1]["method"], kv[1]["run_name"]))
    pbar = tqdm(selected_items, desc="Loading test eval", unit="run")
    for key, item in pbar:
        run_name = item["run_name"]
        seed = next((part[4:] for part in Path(run_name).parts if part.startswith("seed") and part[4:].isdigit()), "?")
        pbar.set_postfix_str(f"seed={seed} method={item['method']} run={run_name}")
        try:
            status = ensure_test_eval(item["run_dir"], force_eval=force_eval)
            test_logs_by_key[key] = load_test_logs(item["run_dir"], item["method"])
            eval_rows.append({
                "method": item["method"],
                "run_name": item["run_name"],
                "run_dir": item["run_dir"],
                "eval_status": status,
                "eval_dir": test_logs_by_key[key]["eval_dir"],
                "ori_path": test_logs_by_key[key]["ori_path"],
                "new_path": test_logs_by_key[key]["new_path"],
            })
        except Exception as exc:
            eval_failures.append({
                "method": item["method"],
                "run_name": item["run_name"],
                "run_dir": item["run_dir"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        eval_df = pd.DataFrame(columns=["method", "run_name", "run_dir", "eval_status", "eval_dir", "ori_path", "new_path"])
    else:
        eval_df = eval_df.sort_values(by=["method", "run_name"]).reset_index(drop=True)
    eval_manifest_df = build_selected_eval_manifest(eval_df, roc_selected_df, prc_selected_df, pareto_selected_df)
    _write_dataframe(eval_manifest_df, os.path.join(save_dir, "selected_test_eval_runs.csv"))
    if eval_failures:
        _write_dataframe(
            pd.DataFrame([{k: v for k, v in row.items() if k != "traceback"} for row in eval_failures]),
            os.path.join(save_dir, "test_eval_failures.csv"),
        )
        _save_strategy_payload(os.path.join(save_dir, "test_eval_failures.json"), eval_failures)

    roc_dir = os.path.join(save_dir, "roc_auc")
    os.makedirs(roc_dir, exist_ok=True)
    _write_dataframe(val_ori_df, os.path.join(roc_dir, "val_candidates_by_seed.csv"))
    _write_dataframe(roc_best_weights_df, os.path.join(roc_dir, "val_best_weights.csv"))
    _write_dataframe(roc_selected_df, os.path.join(roc_dir, "val_selected_by_seed.csv"))
    available_roc_selected_df = _filter_available_selected(roc_selected_df, test_logs_by_key)
    roc_test_ori_df, roc_test_ori_agg_df = summarize_selected_test_ori(available_roc_selected_df, test_logs_by_key)
    roc_pareto_selected_df = select_weight_locked_val_pareto(available_roc_selected_df, val_pareto_df)
    roc_test_pareto_df, _ = summarize_selected_test_pareto(roc_pareto_selected_df, test_logs_by_key)
    roc_best_test_df = build_roc_best_test_metrics_by_seed(available_roc_selected_df, roc_test_ori_df, roc_test_pareto_df)
    _write_dataframe(roc_test_ori_df, os.path.join(roc_dir, "test_ori_metrics_by_seed.csv"))
    _write_dataframe(roc_test_ori_agg_df, os.path.join(roc_dir, "test_ori_metrics_aggregate.csv"))
    _write_dataframe(roc_best_test_df, os.path.join(roc_dir, "best_test_metrics_by_seed.csv"))
    _remove_test_curve_outputs(roc_dir)
    _save_strategy_payload(os.path.join(roc_dir, "summary.json"), {
        "strategy": "roc_auc",
        "selection_source": "val_best_weights_mean_across_seeds",
        "num_val_candidates": int(len(val_ori_df)),
        "num_best_weights": int(len(roc_best_weights_df)),
        "num_selected": int(len(roc_selected_df)),
        "num_selected_with_test_logs": int(len(available_roc_selected_df)),
        "val_best_weights": _to_jsonable_records(roc_best_weights_df),
        "best_test_metrics_by_seed": _to_jsonable_records(roc_best_test_df),
        "results": _to_jsonable_records(roc_test_ori_df),
        "aggregate": _to_jsonable_records(roc_test_ori_agg_df),
    })

    prc_dir = os.path.join(save_dir, "prc_auc")
    os.makedirs(prc_dir, exist_ok=True)
    _write_dataframe(val_ori_df, os.path.join(prc_dir, "val_candidates_by_seed.csv"))
    _write_dataframe(prc_best_weights_df, os.path.join(prc_dir, "val_best_weights.csv"))
    _write_dataframe(prc_selected_df, os.path.join(prc_dir, "val_selected_by_seed.csv"))
    available_prc_selected_df = _filter_available_selected(prc_selected_df, test_logs_by_key)
    prc_test_ori_df, prc_test_ori_agg_df = summarize_selected_test_ori(available_prc_selected_df, test_logs_by_key)
    prc_pareto_selected_df = select_weight_locked_val_pareto(available_prc_selected_df, val_pareto_df)
    prc_test_pareto_df, _ = summarize_selected_test_pareto(prc_pareto_selected_df, test_logs_by_key)
    prc_best_test_df = build_roc_best_test_metrics_by_seed(available_prc_selected_df, prc_test_ori_df, prc_test_pareto_df)
    _write_dataframe(prc_test_ori_df, os.path.join(prc_dir, "test_ori_metrics_by_seed.csv"))
    _write_dataframe(prc_test_ori_agg_df, os.path.join(prc_dir, "test_ori_metrics_aggregate.csv"))
    _write_dataframe(prc_best_test_df, os.path.join(prc_dir, "best_test_metrics_by_seed.csv"))
    _remove_test_curve_outputs(prc_dir)
    _save_strategy_payload(os.path.join(prc_dir, "summary.json"), {
        "strategy": "prc_auc",
        "selection_source": "val_best_weights_mean_across_seeds",
        "num_val_candidates": int(len(val_ori_df)),
        "num_best_weights": int(len(prc_best_weights_df)),
        "num_selected": int(len(prc_selected_df)),
        "num_selected_with_test_logs": int(len(available_prc_selected_df)),
        "val_best_weights": _to_jsonable_records(prc_best_weights_df),
        "best_test_metrics_by_seed": _to_jsonable_records(prc_best_test_df),
        "results": _to_jsonable_records(prc_test_ori_df),
        "aggregate": _to_jsonable_records(prc_test_ori_agg_df),
    })

    pareto_dir = os.path.join(save_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    _write_dataframe(range_df, os.path.join(pareto_dir, "val_bal_acc_ranges_by_method_mode.csv"))
    _write_dataframe(val_pareto_df, os.path.join(pareto_dir, "val_pareto_candidates.csv"))
    _write_dataframe(pareto_best_weights_df, os.path.join(pareto_dir, "val_best_weights.csv"))
    _write_dataframe(pareto_selected_df, os.path.join(pareto_dir, "val_selected_by_seed.csv"))
    available_pareto_selected_df = _filter_available_selected(pareto_selected_df, test_logs_by_key)
    pareto_test_ori_df, pareto_test_ori_agg_df = summarize_selected_test_ori(available_pareto_selected_df, test_logs_by_key)
    pareto_test_df, pareto_test_agg_df = summarize_selected_test_pareto(available_pareto_selected_df, test_logs_by_key)
    pareto_best_test_df = build_pareto_best_test_metrics_by_seed(
        available_pareto_selected_df,
        pareto_test_ori_df,
        pareto_test_df,
    )
    _write_dataframe(pareto_test_ori_df, os.path.join(pareto_dir, "test_ori_metrics_by_seed.csv"))
    _write_dataframe(pareto_test_ori_agg_df, os.path.join(pareto_dir, "test_ori_metrics_aggregate.csv"))
    _write_dataframe(pareto_test_df, os.path.join(pareto_dir, "test_pareto_by_seed.csv"))
    _write_dataframe(pareto_test_agg_df, os.path.join(pareto_dir, "test_pareto_aggregate.csv"))
    _write_dataframe(pareto_best_test_df, os.path.join(pareto_dir, "best_test_metrics_by_seed.csv"))
    _remove_test_curve_outputs(pareto_dir)
    selection_curve_paths = save_test_selection_curves(
        save_dir,
        available_roc_selected_df,
        available_prc_selected_df,
        available_pareto_selected_df,
        test_logs_by_key,
    )
    selection_plot_paths = save_test_selection_barplots(save_dir, roc_best_test_df, prc_best_test_df, pareto_best_test_df)
    _save_strategy_payload(os.path.join(pareto_dir, "summary.json"), {
        "strategy": "pareto",
        "selection_source": "val_best_weights_mean_across_seeds",
        "num_val_candidates": int(len(val_pareto_df)),
        "num_best_weights": int(len(pareto_best_weights_df)),
        "num_selected": int(len(pareto_selected_df)),
        "num_selected_with_test_logs": int(len(available_pareto_selected_df)),
        "ranges": _to_jsonable_records(range_df),
        "val_best_weights": _to_jsonable_records(pareto_best_weights_df),
        "best_test_metrics_by_seed": _to_jsonable_records(pareto_best_test_df),
        "ori_results": _to_jsonable_records(pareto_test_ori_df),
        "ori_aggregate": _to_jsonable_records(pareto_test_ori_agg_df),
        "pareto_results": _to_jsonable_records(pareto_test_df),
        "pareto_aggregate": _to_jsonable_records(pareto_test_agg_df),
    })

    payload = {
        "logs_dir": logs_dir,
        "save_dir": save_dir,
        "force_eval": bool(force_eval),
        "integral_bal_acc_min": float(bal_acc_min),
        "integral_bal_acc_max": float(bal_acc_max),
        "penalty_det_time": float(penalty_det_time),
        "num_candidates": int(len(candidates)),
        "num_selected_runs": int(len(selected_pairs)),
        "num_eval_success": int(len(eval_df)),
        "num_eval_failures": int(len(eval_failures)),
        "selected_test_eval_runs": os.path.join(save_dir, "selected_test_eval_runs.csv"),
        "roc_auc_summary": os.path.join(roc_dir, "summary.json"),
        "roc_auc_best_test_metrics_by_seed": os.path.join(roc_dir, "best_test_metrics_by_seed.csv"),
        "prc_auc_summary": os.path.join(prc_dir, "summary.json"),
        "prc_auc_best_test_metrics_by_seed": os.path.join(prc_dir, "best_test_metrics_by_seed.csv"),
        "pareto_summary": os.path.join(pareto_dir, "summary.json"),
        "pareto_best_test_metrics_by_seed": os.path.join(pareto_dir, "best_test_metrics_by_seed.csv"),
        **selection_curve_paths,
        **selection_plot_paths,
    }
    _save_strategy_payload(os.path.join(save_dir, "test_summary.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select val-best checkpoints/weights, run test eval on the selected runs, and summarize ROC-AUC, PRC-AUC, and Pareto t@bal_acc metrics."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        default=_resolve_default_logs_dir(),
        help="Root directory containing trained run folders with config.yaml and val outputs.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save test summaries. Defaults to <logs-dir>/pipeline_test_new.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run test evaluation even if eval/ori_logs.json and eval/new_logs.json already exist.",
    )
    parser.add_argument(
        "--pareto-bal-acc-min",
        type=float,
        default=PARETO_INTEGRAL_BAL_ACC_MIN,
        help="Lower bound of the fixed bal_acc range used for Pareto integration.",
    )
    parser.add_argument(
        "--pareto-bal-acc-max",
        type=float,
        default=PARETO_INTEGRAL_BAL_ACC_MAX,
        help="Upper bound of the fixed bal_acc range used for Pareto integration.",
    )
    parser.add_argument(
        "--penalty-det-time",
        type=float,
        default=PENALIZED_DET_TIME,
        help="Detection-time penalty used when a Pareto curve does not cover the selected bal_acc range.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.abspath(args.logs_dir), "pipeline_test_new")
    result = run_test_pipeline(
        logs_dir=args.logs_dir,
        save_dir=save_dir,
        force_eval=args.force_eval,
        bal_acc_min=args.pareto_bal_acc_min,
        bal_acc_max=args.pareto_bal_acc_max,
        penalty_det_time=args.penalty_det_time,
    )

    print("Saved test summary to", os.path.abspath(os.path.join(save_dir, "test_summary.json")))
    print(f"candidates={result['num_candidates']}")
    print(f"selected_runs={result['num_selected_runs']}")
    print(f"eval_success={result['num_eval_success']}")
    print(f"eval_failures={result['num_eval_failures']}")


if __name__ == "__main__":
    main()
