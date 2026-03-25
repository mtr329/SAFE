import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.mrefine.const import DELAY_DELTAS
from failure_prob.mrefine.delay_summary import (
    PENALIZED_DET_TIME,
    _compute_max_bal_acc_from_pareto,
    _compute_penalized_mean_t_at_balacc,
    _get_pareto_curve_from_alpha_dict,
    _split_delay_logs_by_method,
)
from failure_prob.pipeline.test_new import (
    VAL_SPLIT,
    _ensure_val_best_weight_outputs,
    _load_best_weights_df,
    _select_ori_best_weights,
    collect_val_candidates,
    ensure_test_eval,
    summarize_val_roc_auc,
)
from failure_prob.pipeline.val_new import _resolve_default_logs_dir

DEFAULT_DELAYS = tuple(float(delta) for delta in DELAY_DELTAS if 0.1 <= float(delta) <= 0.6)
DELAY_PARETO_INTEGRAL_EARLY_BAL_ACC_MIN = 0.6
DELAY_PARETO_INTEGRAL_EARLY_BAL_ACC_MAX = 0.7
DELAY_PARETO_INTEGRAL_LAST_BAL_ACC_MIN = 0.6
DELAY_PARETO_INTEGRAL_LAST_BAL_ACC_MAX = 0.75


def _parse_methods(values: list[str] | None) -> list[str]:
    if not values:
        return []
    methods = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                methods.append(part)
    return sorted(set(methods))


def _metric_mean(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.nan
    return float(array.mean())


def _find_float_key(data: dict, target: float) -> tuple[str | None, dict | None]:
    for key, value in data.items():
        if np.isclose(float(key), float(target)):
            return key, value
    return None, None


def _mean_alpha_dict(alpha_dict: dict) -> dict:
    mean_alpha_dict = {}
    for alpha, metrics in alpha_dict.items():
        mean_alpha_dict[alpha] = {
            key: (value if key == "detect_method" else _metric_mean(value))
            for key, value in metrics.items()
        }
    return mean_alpha_dict


def _range_for_mode(mode: str) -> tuple[float, float]:
    if str(mode) == "early":
        return (
            float(DELAY_PARETO_INTEGRAL_EARLY_BAL_ACC_MIN),
            float(DELAY_PARETO_INTEGRAL_EARLY_BAL_ACC_MAX),
        )
    if str(mode) == "last":
        return (
            float(DELAY_PARETO_INTEGRAL_LAST_BAL_ACC_MIN),
            float(DELAY_PARETO_INTEGRAL_LAST_BAL_ACC_MAX),
        )
    raise ValueError(f"Unsupported delay mode: {mode}")


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


def _build_eval_manifest(eval_rows: list[dict], selected_df: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for row in eval_rows:
        key = (row["run_dir"], row["method"])
        records[key] = {
            "method": row["method"],
            "seed": row["seed"],
            "run_name": row["run_name"],
            "run_dir": row["run_dir"],
            "eval_status": row["eval_status"],
            "eval_dir": row["eval_dir"],
            "delay_log_path": row["delay_log_path"],
            "selected_for_roc_auc": False,
        }

    for row in selected_df.itertuples(index=False):
        key = (row.run_dir, row.method)
        if key not in records:
            continue
        records[key]["selected_for_roc_auc"] = True

    if not records:
        return pd.DataFrame(columns=[
            "method",
            "seed",
            "run_name",
            "run_dir",
            "eval_status",
            "eval_dir",
            "delay_log_path",
            "selected_for_roc_auc",
        ])

    return pd.DataFrame(records.values()).sort_values(
        by=["method", "seed", "run_name"],
    ).reset_index(drop=True)


def ensure_test_delay_eval(run_dir: str, force_eval: bool = False) -> tuple[str, str]:
    delay_path = os.path.join(run_dir, "eval", "delay_logs.json")
    rerun = bool(force_eval or not os.path.isfile(delay_path))
    status = ensure_test_eval(run_dir, force_eval=rerun)
    if not os.path.isfile(delay_path):
        raise FileNotFoundError(f"Expected delay logs were not written under {os.path.dirname(delay_path)}")
    return ("ran" if rerun else status), delay_path


def load_test_delay_logs(run_dir: str, method: str) -> dict:
    delay_path = os.path.join(run_dir, "eval", "delay_logs.json")
    with open(delay_path, "r") as f:
        delay_logs = json.load(f)

    delay_by_method = _split_delay_logs_by_method(delay_logs, method)
    if method not in delay_by_method:
        raise KeyError(f"Method {method} missing from {delay_path}")

    return {
        "delay": delay_by_method[method],
        "eval_dir": os.path.dirname(delay_path),
        "delay_path": delay_path,
    }


def summarize_selected_test_delay_static(
    selected_df: pd.DataFrame,
    test_logs_by_key: dict,
    split_name: str,
    delays: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in selected_df.itertuples(index=False):
        delay_logs = test_logs_by_key[(row.run_dir, row.method)]["delay"]
        split_logs = delay_logs.get("static", {}).get(split_name, {}).get("all", {})
        for delta in delays:
            delta_key, delta_dict = _find_float_key(split_logs, delta)
            rows.append({
                "method": row.method,
                "seed": row.seed,
                "weight_key": row.weight_key,
                "run_name": row.run_name,
                "run_dir": row.run_dir,
                "split": split_name,
                "delta": float(delta),
                "test_roc_auc": np.nan if delta_dict is None else _metric_mean(delta_dict.get("roc_auc", [])),
                "test_prc_auc": np.nan if delta_dict is None else _metric_mean(delta_dict.get("prc_auc", [])),
                "missing_delta": delta_dict is None,
                "matched_delta_key": delta_key,
            })

    by_seed_df = pd.DataFrame(rows)
    if by_seed_df.empty:
        empty = pd.DataFrame(columns=[
            "method",
            "seed",
            "weight_key",
            "run_name",
            "run_dir",
            "split",
            "delta",
            "test_roc_auc",
            "test_prc_auc",
            "missing_delta",
            "matched_delta_key",
        ])
        return empty, pd.DataFrame(columns=[
            "method",
            "split",
            "delta",
            "num_runs",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_prc_auc",
            "std_test_prc_auc",
        ])

    agg_rows = []
    for (method, split, delta), group in by_seed_df.groupby(["method", "split", "delta"], dropna=False):
        agg_rows.append({
            "method": method,
            "split": split,
            "delta": float(delta),
            "num_runs": int(len(group)),
            "mean_test_roc_auc": float(group["test_roc_auc"].mean()),
            "std_test_roc_auc": float(group["test_roc_auc"].std(ddof=0)),
            "mean_test_prc_auc": float(group["test_prc_auc"].mean()),
            "std_test_prc_auc": float(group["test_prc_auc"].std(ddof=0)),
        })

    agg_df = pd.DataFrame(agg_rows).sort_values(
        by=["method", "split", "delta"],
    ).reset_index(drop=True)
    by_seed_df = by_seed_df.sort_values(
        by=["method", "seed", "delta", "run_name"],
    ).reset_index(drop=True)
    return by_seed_df, agg_df


def summarize_selected_test_delay_calib(
    selected_df: pd.DataFrame,
    test_logs_by_key: dict,
    delays: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in selected_df.itertuples(index=False):
        delay_logs = test_logs_by_key[(row.run_dir, row.method)]["delay"]
        calib_logs = delay_logs.get("calib", {})
        for mode, delta_logs in calib_logs.items():
            for delta in delays:
                _, alpha_dict = _find_float_key(delta_logs, delta)
                if alpha_dict is None:
                    rows.append({
                        "method": row.method,
                        "seed": row.seed,
                        "weight_key": row.weight_key,
                        "run_name": row.run_name,
                        "run_dir": row.run_dir,
                        "mode": mode,
                        "delta": float(delta),
                        "alpha": np.nan,
                        "avg_det_time": np.nan,
                        "bal_acc": np.nan,
                        "acc": np.nan,
                        "f1": np.nan,
                        "fpr": np.nan,
                        "fnr": np.nan,
                        "tpr": np.nan,
                        "tnr": np.nan,
                        "missing_delta": True,
                    })
                    continue

                for alpha, metrics in alpha_dict.items():
                    rows.append({
                        "method": row.method,
                        "seed": row.seed,
                        "weight_key": row.weight_key,
                        "run_name": row.run_name,
                        "run_dir": row.run_dir,
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
                        "missing_delta": False,
                    })

    by_seed_df = pd.DataFrame(rows)
    if by_seed_df.empty:
        empty = pd.DataFrame(columns=[
            "method",
            "seed",
            "weight_key",
            "run_name",
            "run_dir",
            "mode",
            "delta",
            "alpha",
            "avg_det_time",
            "bal_acc",
            "acc",
            "f1",
            "fpr",
            "fnr",
            "tpr",
            "tnr",
            "missing_delta",
        ])
        return empty, pd.DataFrame(columns=[
            "method",
            "mode",
            "delta",
            "alpha",
            "num_runs",
            "mean_avg_det_time",
            "std_avg_det_time",
            "mean_bal_acc",
            "std_bal_acc",
            "mean_acc",
            "std_acc",
            "mean_f1",
            "std_f1",
            "mean_fpr",
            "std_fpr",
            "mean_fnr",
            "std_fnr",
            "mean_tpr",
            "std_tpr",
            "mean_tnr",
            "std_tnr",
        ])

    value_columns = ["avg_det_time", "bal_acc", "acc", "f1", "fpr", "fnr", "tpr", "tnr"]
    agg_rows = []
    valid_df = by_seed_df.loc[~by_seed_df["missing_delta"]].copy()
    for (method, mode, delta, alpha), group in valid_df.groupby(["method", "mode", "delta", "alpha"], dropna=False):
        payload = {
            "method": method,
            "mode": mode,
            "delta": float(delta),
            "alpha": float(alpha),
            "num_runs": int(len(group)),
        }
        for col in value_columns:
            payload[f"mean_{col}"] = float(group[col].mean())
            payload[f"std_{col}"] = float(group[col].std(ddof=0))
        agg_rows.append(payload)

    agg_df = pd.DataFrame(agg_rows)
    if agg_df.empty:
        agg_df = pd.DataFrame(columns=[
            "method",
            "mode",
            "delta",
            "alpha",
            "num_runs",
            "mean_avg_det_time",
            "std_avg_det_time",
            "mean_bal_acc",
            "std_bal_acc",
            "mean_acc",
            "std_acc",
            "mean_f1",
            "std_f1",
            "mean_fpr",
            "std_fpr",
            "mean_fnr",
            "std_fnr",
            "mean_tpr",
            "std_tpr",
            "mean_tnr",
            "std_tnr",
        ])
    else:
        agg_df = agg_df.sort_values(by=["method", "mode", "delta", "alpha"]).reset_index(drop=True)

    by_seed_df = by_seed_df.sort_values(
        by=["method", "mode", "seed", "delta", "alpha", "run_name"],
        na_position="last",
    ).reset_index(drop=True)
    return by_seed_df, agg_df


def summarize_selected_test_delay_integral(
    selected_df: pd.DataFrame,
    test_logs_by_key: dict,
    delays: tuple[float, ...],
    penalty_det_time: float = PENALIZED_DET_TIME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in selected_df.itertuples(index=False):
        delay_logs = test_logs_by_key[(row.run_dir, row.method)]["delay"]
        calib_logs = delay_logs.get("calib", {})
        for mode, delta_logs in calib_logs.items():
            bal_acc_min, bal_acc_max = _range_for_mode(mode)
            for delta in delays:
                _, alpha_dict = _find_float_key(delta_logs, delta)
                if alpha_dict is None:
                    rows.append({
                        "method": row.method,
                        "seed": row.seed,
                        "weight_key": row.weight_key,
                        "run_name": row.run_name,
                        "run_dir": row.run_dir,
                        "mode": mode,
                        "delta": float(delta),
                        "bal_acc_min": bal_acc_min,
                        "bal_acc_max": bal_acc_max,
                        "penalty_det_time": float(penalty_det_time),
                        "num_pareto_points": 0,
                        "test_penalized_mean_t_at_balacc": np.nan,
                        "test_integral_t_at_balacc": np.nan,
                        "missing_delta": True,
                    })
                    continue

                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(_mean_alpha_dict(alpha_dict))
                score = _compute_penalized_mean_t_at_balacc(
                    pareto_det_times,
                    pareto_bal_accs,
                    bal_acc_min,
                    bal_acc_max,
                    penalty_det_time=penalty_det_time,
                )
                rows.append({
                    "method": row.method,
                    "seed": row.seed,
                    "weight_key": row.weight_key,
                    "run_name": row.run_name,
                    "run_dir": row.run_dir,
                    "mode": mode,
                    "delta": float(delta),
                    "bal_acc_min": bal_acc_min,
                    "bal_acc_max": bal_acc_max,
                    "penalty_det_time": float(penalty_det_time),
                    "num_pareto_points": int(len(pareto_det_times)),
                    "test_penalized_mean_t_at_balacc": score,
                    "test_integral_t_at_balacc": (
                        score * (bal_acc_max - bal_acc_min)
                        if np.isfinite(score) and np.isfinite(bal_acc_min) and np.isfinite(bal_acc_max)
                        else np.nan
                    ),
                    "missing_delta": False,
                })

    by_seed_df = pd.DataFrame(rows)
    if by_seed_df.empty:
        empty = pd.DataFrame(columns=[
            "method",
            "seed",
            "weight_key",
            "run_name",
            "run_dir",
            "mode",
            "delta",
            "bal_acc_min",
            "bal_acc_max",
            "penalty_det_time",
            "num_pareto_points",
            "test_penalized_mean_t_at_balacc",
            "test_integral_t_at_balacc",
            "missing_delta",
        ])
        return empty, pd.DataFrame(columns=[
            "method",
            "mode",
            "delta",
            "num_runs",
            "mean_test_penalized_mean_t_at_balacc",
            "std_test_penalized_mean_t_at_balacc",
            "mean_test_integral_t_at_balacc",
            "std_test_integral_t_at_balacc",
        ])

    agg_rows = []
    valid_df = by_seed_df.loc[~by_seed_df["missing_delta"]].copy()
    for (method, mode, delta), group in valid_df.groupby(["method", "mode", "delta"], dropna=False):
        agg_rows.append({
            "method": method,
            "mode": mode,
            "delta": float(delta),
            "num_runs": int(len(group)),
            "mean_test_penalized_mean_t_at_balacc": float(group["test_penalized_mean_t_at_balacc"].mean()),
            "std_test_penalized_mean_t_at_balacc": float(group["test_penalized_mean_t_at_balacc"].std(ddof=0)),
            "mean_test_integral_t_at_balacc": float(group["test_integral_t_at_balacc"].mean()),
            "std_test_integral_t_at_balacc": float(group["test_integral_t_at_balacc"].std(ddof=0)),
        })

    agg_df = pd.DataFrame(agg_rows)
    if agg_df.empty:
        agg_df = pd.DataFrame(columns=[
            "method",
            "mode",
            "delta",
            "num_runs",
            "mean_test_penalized_mean_t_at_balacc",
            "std_test_penalized_mean_t_at_balacc",
            "mean_test_integral_t_at_balacc",
            "std_test_integral_t_at_balacc",
        ])
    else:
        agg_df = agg_df.sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)

    by_seed_df = by_seed_df.sort_values(
        by=["method", "mode", "seed", "delta", "run_name"],
    ).reset_index(drop=True)
    return by_seed_df, agg_df


def summarize_selected_test_delay_pareto_max_bal_acc(
    selected_df: pd.DataFrame,
    test_logs_by_key: dict,
    delays: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in selected_df.itertuples(index=False):
        delay_logs = test_logs_by_key[(row.run_dir, row.method)]["delay"]
        calib_logs = delay_logs.get("calib", {})
        for mode, delta_logs in calib_logs.items():
            for delta in delays:
                _, alpha_dict = _find_float_key(delta_logs, delta)
                if alpha_dict is None:
                    rows.append({
                        "method": row.method,
                        "seed": row.seed,
                        "weight_key": row.weight_key,
                        "run_name": row.run_name,
                        "run_dir": row.run_dir,
                        "mode": mode,
                        "delta": float(delta),
                        "num_pareto_points": 0,
                        "test_pareto_max_bal_acc": np.nan,
                        "missing_delta": True,
                    })
                    continue

                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(_mean_alpha_dict(alpha_dict))
                rows.append({
                    "method": row.method,
                    "seed": row.seed,
                    "weight_key": row.weight_key,
                    "run_name": row.run_name,
                    "run_dir": row.run_dir,
                    "mode": mode,
                    "delta": float(delta),
                    "num_pareto_points": int(len(pareto_det_times)),
                    "test_pareto_max_bal_acc": _compute_max_bal_acc_from_pareto(pareto_bal_accs),
                    "missing_delta": False,
                })

    by_seed_df = pd.DataFrame(rows)
    if by_seed_df.empty:
        empty = pd.DataFrame(columns=[
            "method",
            "seed",
            "weight_key",
            "run_name",
            "run_dir",
            "mode",
            "delta",
            "num_pareto_points",
            "test_pareto_max_bal_acc",
            "missing_delta",
        ])
        return empty, pd.DataFrame(columns=[
            "method",
            "mode",
            "delta",
            "num_runs",
            "mean_test_pareto_max_bal_acc",
            "std_test_pareto_max_bal_acc",
        ])

    agg_rows = []
    valid_df = by_seed_df.loc[~by_seed_df["missing_delta"]].copy()
    for (method, mode, delta), group in valid_df.groupby(["method", "mode", "delta"], dropna=False):
        agg_rows.append({
            "method": method,
            "mode": mode,
            "delta": float(delta),
            "num_runs": int(len(group)),
            "mean_test_pareto_max_bal_acc": float(group["test_pareto_max_bal_acc"].mean()),
            "std_test_pareto_max_bal_acc": float(group["test_pareto_max_bal_acc"].std(ddof=0)),
        })

    agg_df = pd.DataFrame(agg_rows)
    if agg_df.empty:
        agg_df = pd.DataFrame(columns=[
            "method",
            "mode",
            "delta",
            "num_runs",
            "mean_test_pareto_max_bal_acc",
            "std_test_pareto_max_bal_acc",
        ])
    else:
        agg_df = agg_df.sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)

    by_seed_df = by_seed_df.sort_values(
        by=["method", "mode", "seed", "delta", "run_name"],
    ).reset_index(drop=True)
    return by_seed_df, agg_df


def _safe_file_stem(value: str) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _save_delay_metric_plot(
    df: pd.DataFrame,
    y_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    save_path: str,
    mode: str | None = None,
) -> str | None:
    if df.empty:
        return None

    plot_df = df.copy()
    if mode is not None:
        if "mode" not in plot_df.columns:
            return None
        plot_df = plot_df.loc[plot_df["mode"].astype(str) == str(mode)].copy()
    if plot_df.empty:
        return None

    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    plotted = False
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, method in enumerate(sorted(plot_df["method"].dropna().astype(str).unique().tolist())):
        method_df = plot_df.loc[plot_df["method"].astype(str) == method].copy()
        method_df = method_df.sort_values(by="delta")
        x = method_df["delta"].to_numpy(dtype=float)
        y = method_df[y_col].to_numpy(dtype=float)
        if x.size == 0 or y.size == 0 or not np.isfinite(y).any():
            continue
        color = colors[idx % len(colors)]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=5, label=method, color=color)
        if std_col in method_df.columns:
            std = method_df[std_col].to_numpy(dtype=float)
            if std.size == y.size and np.isfinite(std).any():
                lower = y - np.nan_to_num(std, nan=0.0)
                upper = y + np.nan_to_num(std, nan=0.0)
                ax.fill_between(x, lower, upper, color=color, alpha=0.15)
        plotted = True

    if not plotted:
        return None

    ax.set_title(title)
    ax.set_xlabel("Delay")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(np.min(plot_df["delta"])), float(np.max(plot_df["delta"])))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return save_path


def save_delay_pareto_front_plots(
    save_dir: str,
    delay_calib_agg_df: pd.DataFrame,
) -> dict[str, str]:
    plot_dir = os.path.join(save_dir, "pareto_front_by_delay")
    os.makedirs(plot_dir, exist_ok=True)
    outputs: dict[str, str] = {}

    if delay_calib_agg_df.empty:
        return outputs

    methods = sorted(delay_calib_agg_df["method"].dropna().astype(str).unique().tolist())
    for method in methods:
        method_df = delay_calib_agg_df.loc[delay_calib_agg_df["method"].astype(str) == method].copy()
        if method_df.empty:
            continue

        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        has_curve = False
        for ax, mode in zip(axes, ("early", "last")):
            mode_df = method_df.loc[method_df["mode"].astype(str) == mode].copy()
            if mode_df.empty:
                ax.set_title(mode)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
                continue

            delta_values = sorted(mode_df["delta"].dropna().astype(float).unique().tolist())
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            plotted_mode = False

            for idx, delta in enumerate(delta_values):
                delta_df = mode_df.loc[np.isclose(mode_df["delta"].astype(float), float(delta))].copy()
                if delta_df.empty:
                    continue
                delta_df = delta_df.sort_values(
                    by=["mean_avg_det_time", "mean_bal_acc", "alpha"],
                    ascending=[True, False, True],
                ).reset_index(drop=True)

                x = delta_df["mean_avg_det_time"].to_numpy(dtype=float)
                y = delta_df["mean_bal_acc"].to_numpy(dtype=float)
                if x.size == 0 or y.size == 0:
                    continue

                pareto_x = []
                pareto_y = []
                best_bal_acc = -np.inf
                for det_time, bal_acc in zip(x, y):
                    if not np.isfinite(det_time) or not np.isfinite(bal_acc):
                        continue
                    if bal_acc > best_bal_acc:
                        pareto_x.append(float(det_time))
                        pareto_y.append(float(bal_acc))
                        best_bal_acc = float(bal_acc)

                if not pareto_x:
                    continue

                color = colors[idx % len(colors)]
                ax.plot(
                    pareto_x,
                    pareto_y,
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    color=color,
                    alpha=0.95,
                    label=f"delay={delta:.1f}",
                )
                plotted_mode = True
                has_curve = True

            ax.set_title(mode)
            ax.set_xlabel("Average Detection Time")
            ax.set_ylabel("Balanced Accuracy")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if plotted_mode:
                ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

        if not has_curve:
            continue

        fig.suptitle(f"{method}: Pareto Front By Delay")
        fig.tight_layout()
        save_path = os.path.join(plot_dir, f"{_safe_file_stem(method)}_pareto_front_by_delay.png")
        fig.savefig(save_path, dpi=300)
        outputs[f"{_safe_file_stem(method)}_pareto_front_by_delay"] = save_path

    return outputs


def summarize_selected_test_delay_relative_to_zero(
    delay_static_by_seed_df: pd.DataFrame,
    delay_integral_by_seed_df: pd.DataFrame,
    delays: tuple[float, ...],
    reference_delta: float = 0.0,
    split_name: str = VAL_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_delays = sorted(float(delta) for delta in delays)

    static_rows = []
    static_df = delay_static_by_seed_df.copy()
    if "split" in static_df.columns:
        static_df = static_df.loc[static_df["split"].astype(str) == str(split_name)].copy()

    static_group_cols = ["method", "seed", "weight_key", "run_name", "run_dir"]
    for _, group in static_df.groupby(static_group_cols, dropna=False):
        ref_row = group.loc[np.isclose(group["delta"].astype(float), float(reference_delta))].copy()
        if ref_row.empty:
            continue
        ref_row = ref_row.iloc[0]
        ref_roc = float(ref_row["test_roc_auc"])
        ref_prc = float(ref_row["test_prc_auc"])

        for delta in target_delays:
            cur = group.loc[np.isclose(group["delta"].astype(float), float(delta))].copy()
            if cur.empty:
                continue
            cur = cur.iloc[0]
            static_rows.append({
                "method": cur["method"],
                "seed": cur["seed"],
                "weight_key": cur["weight_key"],
                "run_name": cur["run_name"],
                "run_dir": cur["run_dir"],
                "split": cur["split"],
                "reference_delta": float(reference_delta),
                "delta": float(delta),
                "roc_auc_ratio_to_delay0": (
                    float(cur["test_roc_auc"]) / ref_roc
                    if np.isfinite(ref_roc) and abs(ref_roc) > 1e-12 and np.isfinite(float(cur["test_roc_auc"]))
                    else np.nan
                ),
                "prc_auc_ratio_to_delay0": (
                    float(cur["test_prc_auc"]) / ref_prc
                    if np.isfinite(ref_prc) and abs(ref_prc) > 1e-12 and np.isfinite(float(cur["test_prc_auc"]))
                    else np.nan
                ),
            })

    static_ratio_by_seed_df = pd.DataFrame(static_rows)
    if static_ratio_by_seed_df.empty:
        static_ratio_by_seed_df = pd.DataFrame(columns=[
            "method",
            "seed",
            "weight_key",
            "run_name",
            "run_dir",
            "split",
            "reference_delta",
            "delta",
            "roc_auc_ratio_to_delay0",
            "prc_auc_ratio_to_delay0",
        ])
        static_ratio_agg_df = pd.DataFrame(columns=[
            "method",
            "split",
            "reference_delta",
            "delta",
            "num_runs",
            "mean_roc_auc_ratio_to_delay0",
            "std_roc_auc_ratio_to_delay0",
            "mean_prc_auc_ratio_to_delay0",
            "std_prc_auc_ratio_to_delay0",
        ])
    else:
        agg_rows = []
        for (method, split, delta), group in static_ratio_by_seed_df.groupby(["method", "split", "delta"], dropna=False):
            agg_rows.append({
                "method": method,
                "split": split,
                "reference_delta": float(reference_delta),
                "delta": float(delta),
                "num_runs": int(len(group)),
                "mean_roc_auc_ratio_to_delay0": float(group["roc_auc_ratio_to_delay0"].mean()),
                "std_roc_auc_ratio_to_delay0": float(group["roc_auc_ratio_to_delay0"].std(ddof=0)),
                "mean_prc_auc_ratio_to_delay0": float(group["prc_auc_ratio_to_delay0"].mean()),
                "std_prc_auc_ratio_to_delay0": float(group["prc_auc_ratio_to_delay0"].std(ddof=0)),
            })
        static_ratio_agg_df = pd.DataFrame(agg_rows).sort_values(
            by=["method", "split", "delta"],
        ).reset_index(drop=True)
        static_ratio_by_seed_df = static_ratio_by_seed_df.sort_values(
            by=["method", "seed", "delta", "run_name"],
        ).reset_index(drop=True)

    integral_rows = []
    integral_group_cols = ["method", "seed", "weight_key", "run_name", "run_dir", "mode"]
    for _, group in delay_integral_by_seed_df.groupby(integral_group_cols, dropna=False):
        ref_row = group.loc[np.isclose(group["delta"].astype(float), float(reference_delta))].copy()
        if ref_row.empty:
            continue
        ref_row = ref_row.iloc[0]
        ref_integral = float(ref_row["test_integral_t_at_balacc"])
        for delta in target_delays:
            cur = group.loc[np.isclose(group["delta"].astype(float), float(delta))].copy()
            if cur.empty:
                continue
            cur = cur.iloc[0]
            integral_rows.append({
                "method": cur["method"],
                "seed": cur["seed"],
                "weight_key": cur["weight_key"],
                "run_name": cur["run_name"],
                "run_dir": cur["run_dir"],
                "mode": cur["mode"],
                "reference_delta": float(reference_delta),
                "delta": float(delta),
                "integral_t_at_balacc_ratio_to_delay0": (
                    float(cur["test_integral_t_at_balacc"]) / ref_integral
                    if np.isfinite(ref_integral) and abs(ref_integral) > 1e-12 and np.isfinite(float(cur["test_integral_t_at_balacc"]))
                    else np.nan
                ),
            })

    integral_ratio_by_seed_df = pd.DataFrame(integral_rows)
    if integral_ratio_by_seed_df.empty:
        integral_ratio_by_seed_df = pd.DataFrame(columns=[
            "method",
            "seed",
            "weight_key",
            "run_name",
            "run_dir",
            "mode",
            "reference_delta",
            "delta",
            "integral_t_at_balacc_ratio_to_delay0",
        ])
        integral_ratio_agg_df = pd.DataFrame(columns=[
            "method",
            "mode",
            "reference_delta",
            "delta",
            "num_runs",
            "mean_integral_t_at_balacc_ratio_to_delay0",
            "std_integral_t_at_balacc_ratio_to_delay0",
        ])
    else:
        agg_rows = []
        for (method, mode, delta), group in integral_ratio_by_seed_df.groupby(["method", "mode", "delta"], dropna=False):
            agg_rows.append({
                "method": method,
                "mode": mode,
                "reference_delta": float(reference_delta),
                "delta": float(delta),
                "num_runs": int(len(group)),
                "mean_integral_t_at_balacc_ratio_to_delay0": float(group["integral_t_at_balacc_ratio_to_delay0"].mean()),
                "std_integral_t_at_balacc_ratio_to_delay0": float(group["integral_t_at_balacc_ratio_to_delay0"].std(ddof=0)),
            })
        integral_ratio_agg_df = pd.DataFrame(agg_rows).sort_values(
            by=["method", "mode", "delta"],
        ).reset_index(drop=True)
        integral_ratio_by_seed_df = integral_ratio_by_seed_df.sort_values(
            by=["method", "mode", "seed", "delta", "run_name"],
        ).reset_index(drop=True)

    return static_ratio_by_seed_df, static_ratio_agg_df, integral_ratio_by_seed_df, integral_ratio_agg_df


def _save_delay_ratio_plot(
    df: pd.DataFrame,
    y_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    save_path: str,
    mode: str | None = None,
    split_name: str | None = None,
) -> str | None:
    if df.empty:
        return None

    plot_df = df.copy()
    if split_name is not None:
        if "split" not in plot_df.columns:
            return None
        plot_df = plot_df.loc[plot_df["split"].astype(str) == str(split_name)].copy()
    if mode is not None:
        if "mode" not in plot_df.columns:
            return None
        plot_df = plot_df.loc[plot_df["mode"].astype(str) == str(mode)].copy()
    if plot_df.empty:
        return None

    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    plotted = False
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    y_max = 1.05

    for idx, method in enumerate(sorted(plot_df["method"].dropna().astype(str).unique().tolist())):
        method_df = plot_df.loc[plot_df["method"].astype(str) == method].copy()
        method_df = method_df.sort_values(by="delta")
        x = method_df["delta"].to_numpy(dtype=float)
        y = method_df[y_col].to_numpy(dtype=float)
        if x.size == 0 or y.size == 0 or not np.isfinite(y).any():
            continue
        color = colors[idx % len(colors)]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=5, label=method, color=color)
        if std_col in method_df.columns:
            std = method_df[std_col].to_numpy(dtype=float)
            if std.size == y.size and np.isfinite(std).any():
                lower = y - np.nan_to_num(std, nan=0.0)
                upper = y + np.nan_to_num(std, nan=0.0)
                ax.fill_between(x, lower, upper, color=color, alpha=0.15)
                y_max = max(y_max, float(np.nanmax(upper)) * 1.05)
            else:
                y_max = max(y_max, float(np.nanmax(y)) * 1.05)
        else:
            y_max = max(y_max, float(np.nanmax(y)) * 1.05)
        plotted = True

    if not plotted:
        return None

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Delay")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(np.min(plot_df["delta"])), float(np.max(plot_df["delta"])))
    ax.set_ylim(0.0, max(1.05, min(y_max, 5.0)))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return save_path


def save_delay_ratio_plots(
    save_dir: str,
    static_ratio_agg_df: pd.DataFrame,
    integral_ratio_agg_df: pd.DataFrame,
    split_name: str,
) -> dict[str, str]:
    plot_dir = os.path.join(save_dir, "ratio_plots")
    os.makedirs(plot_dir, exist_ok=True)
    outputs: dict[str, str] = {}
    ratio_specs = [
        (
            static_ratio_agg_df,
            "mean_roc_auc_ratio_to_delay0",
            "std_roc_auc_ratio_to_delay0",
            f"ROC-AUC / ROC-AUC(delay=0) ({split_name})",
            "ROC-AUC Ratio To Delay=0",
            None,
            split_name,
            "roc_auc_ratio_vs_delay",
            os.path.join(plot_dir, f"{_safe_file_stem(split_name)}_roc_auc_ratio_vs_delay.png"),
        ),
        (
            static_ratio_agg_df,
            "mean_prc_auc_ratio_to_delay0",
            "std_prc_auc_ratio_to_delay0",
            f"PRC-AUC / PRC-AUC(delay=0) ({split_name})",
            "PRC-AUC Ratio To Delay=0",
            None,
            split_name,
            "prc_auc_ratio_vs_delay",
            os.path.join(plot_dir, f"{_safe_file_stem(split_name)}_prc_auc_ratio_vs_delay.png"),
        ),
        (
            integral_ratio_agg_df,
            "mean_integral_t_at_balacc_ratio_to_delay0",
            "std_integral_t_at_balacc_ratio_to_delay0",
            "Integral t@bal_acc / Integral(delay=0) (early)",
            "Integral Ratio To Delay=0",
            "early",
            None,
            "integral_t_at_balacc_ratio_vs_delay_early",
            os.path.join(plot_dir, "integral_t_at_balacc_ratio_vs_delay_early.png"),
        ),
        (
            integral_ratio_agg_df,
            "mean_integral_t_at_balacc_ratio_to_delay0",
            "std_integral_t_at_balacc_ratio_to_delay0",
            "Integral t@bal_acc / Integral(delay=0) (last)",
            "Integral Ratio To Delay=0",
            "last",
            None,
            "integral_t_at_balacc_ratio_vs_delay_last",
            os.path.join(plot_dir, "integral_t_at_balacc_ratio_vs_delay_last.png"),
        ),
    ]

    def _filter_ratio_df(df: pd.DataFrame, mode: str | None, split: str | None) -> pd.DataFrame:
        out = df.copy()
        if split is not None:
            if "split" not in out.columns:
                return out.iloc[0:0].copy()
            out = out.loc[out["split"].astype(str) == str(split)].copy()
        if mode is not None:
            if "mode" not in out.columns:
                return out.iloc[0:0].copy()
            out = out.loc[out["mode"].astype(str) == str(mode)].copy()
        return out

    common_x_min = None
    common_x_max = None
    common_y_max = 1.05
    for df, y_col, std_col, _, _, mode, split, _, _ in ratio_specs:
        plot_df = _filter_ratio_df(df, mode, split)
        if plot_df.empty or y_col not in plot_df.columns:
            continue
        x_vals = plot_df["delta"].to_numpy(dtype=float)
        y_vals = plot_df[y_col].to_numpy(dtype=float)
        if x_vals.size == 0 or y_vals.size == 0 or not np.isfinite(y_vals).any():
            continue
        common_x_min = float(np.min(x_vals)) if common_x_min is None else min(common_x_min, float(np.min(x_vals)))
        common_x_max = float(np.max(x_vals)) if common_x_max is None else max(common_x_max, float(np.max(x_vals)))
        upper = y_vals.copy()
        if std_col in plot_df.columns:
            std_vals = plot_df[std_col].to_numpy(dtype=float)
            if std_vals.size == y_vals.size:
                upper = y_vals + np.nan_to_num(std_vals, nan=0.0)
        if np.isfinite(upper).any():
            common_y_max = max(common_y_max, float(np.nanmax(upper)) * 1.05)
    common_y_max = max(1.05, min(common_y_max, 5.0))

    for df, y_col, std_col, title, ylabel, mode, split, output_key, save_path in ratio_specs:
        path = _save_delay_ratio_plot(
            df,
            y_col=y_col,
            std_col=std_col,
            title=title,
            ylabel=ylabel,
            save_path=save_path,
            mode=mode,
            split_name=split,
        )
        if path is not None:
            outputs[output_key] = path

    fig = Figure(figsize=(12, 9))
    axes = fig.subplots(2, 2)
    axes = np.asarray(axes).reshape(-1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    has_any = False
    for ax, (df, y_col, std_col, title, ylabel, mode, split, _, _) in zip(axes, ratio_specs):
        plot_df = _filter_ratio_df(df, mode, split)
        plotted = False
        for idx, method in enumerate(sorted(plot_df.get("method", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())):
            method_df = plot_df.loc[plot_df["method"].astype(str) == method].copy().sort_values(by="delta")
            x = method_df["delta"].to_numpy(dtype=float)
            y = method_df[y_col].to_numpy(dtype=float)
            if x.size == 0 or y.size == 0 or not np.isfinite(y).any():
                continue
            color = colors[idx % len(colors)]
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=5, label=method, color=color)
            if std_col in method_df.columns:
                std = method_df[std_col].to_numpy(dtype=float)
                if std.size == y.size and np.isfinite(std).any():
                    lower = y - np.nan_to_num(std, nan=0.0)
                    upper = y + np.nan_to_num(std, nan=0.0)
                    ax.fill_between(x, lower, upper, color=color, alpha=0.15)
            plotted = True
            has_any = True
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Delay")
        ax.set_ylabel(ylabel)
        if common_x_min is not None and common_x_max is not None:
            ax.set_xlim(common_x_min, common_x_max)
        ax.set_ylim(0.0, common_y_max)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

    if has_any:
        fig.suptitle("Delay Ratio Metrics (Shared Scale)")
        fig.tight_layout()
        combined_path = os.path.join(plot_dir, "ratio_metrics_combined.png")
        fig.savefig(combined_path, dpi=300)
        outputs["ratio_metrics_combined"] = combined_path

    return outputs


def save_delay_metric_plots(
    save_dir: str,
    delay_static_agg_df: pd.DataFrame,
    delay_integral_agg_df: pd.DataFrame,
    delay_pareto_bal_acc_agg_df: pd.DataFrame,
    split_name: str,
) -> dict[str, str]:
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    outputs: dict[str, str] = {}

    static_df = delay_static_agg_df.copy()
    if "split" in static_df.columns:
        static_df = static_df.loc[static_df["split"].astype(str) == str(split_name)].copy()

    roc_path = _save_delay_metric_plot(
        static_df,
        y_col="mean_test_roc_auc",
        std_col="std_test_roc_auc",
        title=f"Test ROC-AUC vs Delay ({split_name})",
        ylabel="ROC-AUC",
        save_path=os.path.join(plot_dir, f"{_safe_file_stem(split_name)}_roc_auc_vs_delay.png"),
    )
    if roc_path is not None:
        outputs["roc_auc_vs_delay"] = roc_path

    prc_path = _save_delay_metric_plot(
        static_df,
        y_col="mean_test_prc_auc",
        std_col="std_test_prc_auc",
        title=f"Test PRC-AUC vs Delay ({split_name})",
        ylabel="PRC-AUC",
        save_path=os.path.join(plot_dir, f"{_safe_file_stem(split_name)}_prc_auc_vs_delay.png"),
    )
    if prc_path is not None:
        outputs["prc_auc_vs_delay"] = prc_path

    for mode in ("early", "last"):
        det_time_path = _save_delay_metric_plot(
            delay_integral_agg_df,
            y_col="mean_test_penalized_mean_t_at_balacc",
            std_col="std_test_penalized_mean_t_at_balacc",
            title=f"Pareto Detection Time vs Delay ({mode})",
            ylabel="Pareto Detection Time",
            save_path=os.path.join(plot_dir, f"pareto_det_time_vs_delay_{mode}.png"),
            mode=mode,
        )
        if det_time_path is not None:
            outputs[f"pareto_det_time_vs_delay_{mode}"] = det_time_path

        bal_acc_path = _save_delay_metric_plot(
            delay_pareto_bal_acc_agg_df,
            y_col="mean_test_pareto_max_bal_acc",
            std_col="std_test_pareto_max_bal_acc",
            title=f"Pareto Max BalAcc vs Delay ({mode})",
            ylabel="Pareto Max BalAcc",
            save_path=os.path.join(plot_dir, f"pareto_max_bal_acc_vs_delay_{mode}.png"),
            mode=mode,
        )
        if bal_acc_path is not None:
            outputs[f"pareto_max_bal_acc_vs_delay_{mode}"] = bal_acc_path

        integral_path = _save_delay_metric_plot(
            delay_integral_agg_df,
            y_col="mean_test_integral_t_at_balacc",
            std_col="std_test_integral_t_at_balacc",
            title=f"Test Integral t@bal_acc vs Delay ({mode})",
            ylabel="Integral t@bal_acc",
            save_path=os.path.join(plot_dir, f"integral_t_at_balacc_vs_delay_{mode}.png"),
            mode=mode,
        )
        if integral_path is not None:
            outputs[f"integral_t_at_balacc_vs_delay_{mode}"] = integral_path

    return outputs


def summarize_delay_mean_over_methods(
    delay_static_agg_df: pd.DataFrame,
    delay_integral_agg_df: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    static_df = delay_static_agg_df.copy()
    if "split" in static_df.columns:
        static_df = static_df.loc[static_df["split"].astype(str) == str(split_name)].copy()

    for delta, group in static_df.groupby("delta", dropna=False):
        rows.append({
            "metric": "roc_auc",
            "mode": "",
            "delta": float(delta),
            "mean_over_methods": float(group["mean_test_roc_auc"].mean()),
            "std_over_methods": float(group["mean_test_roc_auc"].std(ddof=0)),
            "num_methods": int(group["method"].nunique()),
        })
        rows.append({
            "metric": "prc_auc",
            "mode": "",
            "delta": float(delta),
            "mean_over_methods": float(group["mean_test_prc_auc"].mean()),
            "std_over_methods": float(group["mean_test_prc_auc"].std(ddof=0)),
            "num_methods": int(group["method"].nunique()),
        })

    for (mode, delta), group in delay_integral_agg_df.groupby(["mode", "delta"], dropna=False):
        rows.append({
            "metric": "integral_t_at_balacc",
            "mode": str(mode),
            "delta": float(delta),
            "mean_over_methods": float(group["mean_test_integral_t_at_balacc"].mean()),
            "std_over_methods": float(group["mean_test_integral_t_at_balacc"].std(ddof=0)),
            "num_methods": int(group["method"].nunique()),
        })

    mean_df = pd.DataFrame(rows)
    if mean_df.empty:
        mean_df = pd.DataFrame(columns=[
            "metric",
            "mode",
            "delta",
            "mean_over_methods",
            "std_over_methods",
            "num_methods",
        ])
        trend_df = pd.DataFrame(columns=[
            "metric",
            "mode",
            "num_points",
            "delta_min",
            "delta_max",
            "value_at_delta_min",
            "value_at_delta_max",
            "absolute_change",
            "slope_k",
            "intercept",
            "r2",
        ])
        return mean_df, trend_df

    mean_df = mean_df.sort_values(by=["metric", "mode", "delta"]).reset_index(drop=True)

    trend_rows = []
    for (metric, mode), group in mean_df.groupby(["metric", "mode"], dropna=False):
        group = group.sort_values(by="delta")
        x = group["delta"].to_numpy(dtype=float)
        y = group["mean_over_methods"].to_numpy(dtype=float)
        if len(x) < 2 or not np.isfinite(y).all():
            continue
        slope_k, intercept = np.polyfit(x, y, 1)
        y_hat = slope_k * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        trend_rows.append({
            "metric": str(metric),
            "mode": str(mode),
            "num_points": int(len(x)),
            "delta_min": float(np.min(x)),
            "delta_max": float(np.max(x)),
            "value_at_delta_min": float(y[0]),
            "value_at_delta_max": float(y[-1]),
            "absolute_change": float(y[-1] - y[0]),
            "slope_k": float(slope_k),
            "intercept": float(intercept),
            "r2": r2,
        })

    trend_df = pd.DataFrame(trend_rows)
    if trend_df.empty:
        trend_df = pd.DataFrame(columns=[
            "metric",
            "mode",
            "num_points",
            "delta_min",
            "delta_max",
            "value_at_delta_min",
            "value_at_delta_max",
            "absolute_change",
            "slope_k",
            "intercept",
            "r2",
        ])
    else:
        trend_df = trend_df.sort_values(by=["metric", "mode"]).reset_index(drop=True)

    return mean_df, trend_df


def run_test_delay_pipeline(
    logs_dir: str,
    save_dir: str,
    methods: list[str] | None = None,
    force_eval: bool = False,
    delays: tuple[float, ...] = DEFAULT_DELAYS,
) -> dict:
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    methods = _parse_methods(methods)
    os.makedirs(save_dir, exist_ok=True)

    candidates = collect_val_candidates(logs_dir)
    if methods:
        candidates = [row for row in candidates if row["method"] in methods]
    if not candidates:
        raise FileNotFoundError(f"No matching val candidates found under {logs_dir}")

    best_weight_paths = _ensure_val_best_weight_outputs(logs_dir)
    val_ori_df, _ = summarize_val_roc_auc(candidates)
    roc_best_weights_df = _load_best_weights_df(best_weight_paths["roc_auc"])
    if methods and not roc_best_weights_df.empty:
        roc_best_weights_df = roc_best_weights_df.loc[
            roc_best_weights_df["method"].astype(str).isin(methods)
        ].reset_index(drop=True)

    roc_selected_df = _select_ori_best_weights(val_ori_df, roc_best_weights_df)
    if methods and not roc_selected_df.empty:
        roc_selected_df = roc_selected_df.loc[
            roc_selected_df["method"].astype(str).isin(methods)
        ].reset_index(drop=True)
    if roc_selected_df.empty:
        available_methods = sorted(set(val_ori_df.get("method", pd.Series(dtype=object)).astype(str).tolist()))
        raise ValueError(
            "No val ROC-AUC selections matched the requested methods. "
            f"requested={methods or 'ALL'} available={available_methods}"
        )

    selected_pairs = {
        (row.run_dir, row.method): {
            "method": row.method,
            "seed": row.seed,
            "run_name": row.run_name,
            "run_dir": row.run_dir,
        }
        for row in roc_selected_df.itertuples(index=False)
    }

    test_logs_by_key = {}
    eval_rows = []
    eval_failures = []
    selected_items = sorted(selected_pairs.items(), key=lambda kv: (kv[1]["method"], kv[1]["run_name"]))
    pbar = tqdm(selected_items, desc="Loading test delay eval", unit="run")
    for key, item in pbar:
        pbar.set_postfix_str(f"method={item['method']} run={item['run_name']}")
        try:
            eval_status, delay_log_path = ensure_test_delay_eval(item["run_dir"], force_eval=force_eval)
            test_logs_by_key[key] = load_test_delay_logs(item["run_dir"], item["method"])
            eval_rows.append({
                "method": item["method"],
                "seed": item["seed"],
                "run_name": item["run_name"],
                "run_dir": item["run_dir"],
                "eval_status": eval_status,
                "eval_dir": test_logs_by_key[key]["eval_dir"],
                "delay_log_path": delay_log_path,
            })
        except Exception as exc:
            eval_failures.append({
                "method": item["method"],
                "seed": item["seed"],
                "run_name": item["run_name"],
                "run_dir": item["run_dir"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    eval_manifest_df = _build_eval_manifest(eval_rows, roc_selected_df)
    _write_dataframe(eval_manifest_df, os.path.join(save_dir, "selected_test_eval_runs.csv"))
    if eval_failures:
        _write_dataframe(
            pd.DataFrame([{k: v for k, v in row.items() if k != "traceback"} for row in eval_failures]),
            os.path.join(save_dir, "test_eval_failures.csv"),
        )
        _save_payload(os.path.join(save_dir, "test_eval_failures.json"), eval_failures)

    available_selected_df = roc_selected_df.loc[
        [(row.run_dir, row.method) in test_logs_by_key for row in roc_selected_df.itertuples(index=False)]
    ].reset_index(drop=True)

    roc_dir = os.path.join(save_dir, "roc_auc")
    os.makedirs(roc_dir, exist_ok=True)
    _write_dataframe(val_ori_df, os.path.join(roc_dir, "val_candidates_by_seed.csv"))
    _write_dataframe(roc_best_weights_df, os.path.join(roc_dir, "val_best_weights.csv"))
    _write_dataframe(roc_selected_df, os.path.join(roc_dir, "val_selected_by_seed.csv"))

    delay_dir = os.path.join(save_dir, "delay")
    os.makedirs(delay_dir, exist_ok=True)
    delay_static_by_seed_df, delay_static_agg_df = summarize_selected_test_delay_static(
        available_selected_df,
        test_logs_by_key,
        split_name=VAL_SPLIT,
        delays=delays,
    )
    delay_calib_by_seed_df, delay_calib_agg_df = summarize_selected_test_delay_calib(
        available_selected_df,
        test_logs_by_key,
        delays=delays,
    )
    delay_integral_by_seed_df, delay_integral_agg_df = summarize_selected_test_delay_integral(
        available_selected_df,
        test_logs_by_key,
        delays=delays,
    )
    _write_dataframe(delay_static_by_seed_df, os.path.join(delay_dir, "test_delay_static_by_seed.csv"))
    _write_dataframe(delay_static_agg_df, os.path.join(delay_dir, "test_delay_static_aggregate.csv"))
    _write_dataframe(delay_calib_by_seed_df, os.path.join(delay_dir, "test_delay_calib_by_seed.csv"))
    _write_dataframe(delay_calib_agg_df, os.path.join(delay_dir, "test_delay_calib_aggregate.csv"))
    _write_dataframe(delay_integral_by_seed_df, os.path.join(delay_dir, "test_delay_integral_by_seed.csv"))
    _write_dataframe(delay_integral_agg_df, os.path.join(delay_dir, "test_delay_integral_aggregate.csv"))
    delay_pareto_bal_acc_by_seed_df, delay_pareto_bal_acc_agg_df = summarize_selected_test_delay_pareto_max_bal_acc(
        available_selected_df,
        test_logs_by_key,
        delays=delays,
    )
    _write_dataframe(delay_pareto_bal_acc_by_seed_df, os.path.join(delay_dir, "test_delay_pareto_max_bal_acc_by_seed.csv"))
    _write_dataframe(delay_pareto_bal_acc_agg_df, os.path.join(delay_dir, "test_delay_pareto_max_bal_acc_aggregate.csv"))
    ratio_source_delays = tuple(sorted(set(float(delta) for delta in delays) | {0.0}))
    delay_static_ratio_source_by_seed_df, _ = summarize_selected_test_delay_static(
        available_selected_df,
        test_logs_by_key,
        split_name=VAL_SPLIT,
        delays=ratio_source_delays,
    )
    delay_integral_ratio_source_by_seed_df, _ = summarize_selected_test_delay_integral(
        available_selected_df,
        test_logs_by_key,
        delays=ratio_source_delays,
    )
    (
        delay_static_ratio_by_seed_df,
        delay_static_ratio_agg_df,
        delay_integral_ratio_by_seed_df,
        delay_integral_ratio_agg_df,
    ) = summarize_selected_test_delay_relative_to_zero(
        delay_static_ratio_source_by_seed_df,
        delay_integral_ratio_source_by_seed_df,
        delays=delays,
        reference_delta=0.0,
        split_name=VAL_SPLIT,
    )
    _write_dataframe(delay_static_ratio_by_seed_df, os.path.join(delay_dir, "test_delay_static_ratio_to_delay0_by_seed.csv"))
    _write_dataframe(delay_static_ratio_agg_df, os.path.join(delay_dir, "test_delay_static_ratio_to_delay0_aggregate.csv"))
    _write_dataframe(delay_integral_ratio_by_seed_df, os.path.join(delay_dir, "test_delay_integral_ratio_to_delay0_by_seed.csv"))
    _write_dataframe(delay_integral_ratio_agg_df, os.path.join(delay_dir, "test_delay_integral_ratio_to_delay0_aggregate.csv"))
    delay_mean_over_methods_df, delay_trend_summary_df = summarize_delay_mean_over_methods(
        delay_static_agg_df,
        delay_integral_agg_df,
        split_name=VAL_SPLIT,
    )
    _write_dataframe(delay_mean_over_methods_df, os.path.join(delay_dir, "test_delay_mean_over_methods.csv"))
    _write_dataframe(delay_trend_summary_df, os.path.join(delay_dir, "test_delay_trend_summary.csv"))
    plot_paths = save_delay_metric_plots(
        delay_dir,
        delay_static_agg_df,
        delay_integral_agg_df,
        delay_pareto_bal_acc_agg_df,
        split_name=VAL_SPLIT,
    )
    pareto_front_plot_paths = save_delay_pareto_front_plots(
        delay_dir,
        delay_calib_agg_df,
    )
    ratio_plot_paths = save_delay_ratio_plots(
        delay_dir,
        delay_static_ratio_agg_df,
        delay_integral_ratio_agg_df,
        split_name=VAL_SPLIT,
    )

    payload = {
        "logs_dir": logs_dir,
        "save_dir": save_dir,
        "methods": methods,
        "force_eval": bool(force_eval),
        "val_selection_metric": "roc_auc",
        "test_split": VAL_SPLIT,
        "delay_values": [float(delta) for delta in delays],
        "delay_integral_early_bal_acc_min": float(DELAY_PARETO_INTEGRAL_EARLY_BAL_ACC_MIN),
        "delay_integral_early_bal_acc_max": float(DELAY_PARETO_INTEGRAL_EARLY_BAL_ACC_MAX),
        "delay_integral_last_bal_acc_min": float(DELAY_PARETO_INTEGRAL_LAST_BAL_ACC_MIN),
        "delay_integral_last_bal_acc_max": float(DELAY_PARETO_INTEGRAL_LAST_BAL_ACC_MAX),
        "num_val_candidates": int(len(val_ori_df)),
        "num_best_weights": int(len(roc_best_weights_df)),
        "num_selected": int(len(roc_selected_df)),
        "num_selected_with_test_logs": int(len(available_selected_df)),
        "num_eval_failures": int(len(eval_failures)),
        "selected_test_eval_runs": os.path.join(save_dir, "selected_test_eval_runs.csv"),
        "val_best_weights": os.path.join(roc_dir, "val_best_weights.csv"),
        "val_selected_by_seed": os.path.join(roc_dir, "val_selected_by_seed.csv"),
        "test_delay_static_by_seed": os.path.join(delay_dir, "test_delay_static_by_seed.csv"),
        "test_delay_static_aggregate": os.path.join(delay_dir, "test_delay_static_aggregate.csv"),
        "test_delay_calib_by_seed": os.path.join(delay_dir, "test_delay_calib_by_seed.csv"),
        "test_delay_calib_aggregate": os.path.join(delay_dir, "test_delay_calib_aggregate.csv"),
        "test_delay_integral_by_seed": os.path.join(delay_dir, "test_delay_integral_by_seed.csv"),
        "test_delay_integral_aggregate": os.path.join(delay_dir, "test_delay_integral_aggregate.csv"),
        "test_delay_pareto_max_bal_acc_by_seed": os.path.join(delay_dir, "test_delay_pareto_max_bal_acc_by_seed.csv"),
        "test_delay_pareto_max_bal_acc_aggregate": os.path.join(delay_dir, "test_delay_pareto_max_bal_acc_aggregate.csv"),
        "test_delay_static_ratio_to_delay0_by_seed": os.path.join(delay_dir, "test_delay_static_ratio_to_delay0_by_seed.csv"),
        "test_delay_static_ratio_to_delay0_aggregate": os.path.join(delay_dir, "test_delay_static_ratio_to_delay0_aggregate.csv"),
        "test_delay_integral_ratio_to_delay0_by_seed": os.path.join(delay_dir, "test_delay_integral_ratio_to_delay0_by_seed.csv"),
        "test_delay_integral_ratio_to_delay0_aggregate": os.path.join(delay_dir, "test_delay_integral_ratio_to_delay0_aggregate.csv"),
        "test_delay_mean_over_methods": os.path.join(delay_dir, "test_delay_mean_over_methods.csv"),
        "test_delay_trend_summary": os.path.join(delay_dir, "test_delay_trend_summary.csv"),
        "best_weights_records": _to_jsonable_records(roc_best_weights_df),
        **plot_paths,
        **pareto_front_plot_paths,
        **ratio_plot_paths,
    }
    _save_payload(os.path.join(save_dir, "test_delay_summary.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select val-best weights by ROC-AUC, run/refresh test eval if needed, "
            "and summarize delay metrics on the held-out split for delay=0.1..0.6."
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
        help="Directory to save test-delay summaries. Defaults to <logs-dir>/pipeline_test_delay_new.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method name to include. Can be repeated or passed as a comma-separated list.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run test evaluation even if eval/delay_logs.json already exists.",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.1,
        help="Minimum delay value to include in the summary.",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.6,
        help="Maximum delay value to include in the summary.",
    )
    args = parser.parse_args()

    delays = tuple(
        float(delta) for delta in DELAY_DELTAS
        if args.min_delay <= float(delta) <= args.max_delay
    )
    if not delays:
        raise ValueError(
            f"No delay values fall inside [{args.min_delay}, {args.max_delay}] from DELAY_DELTAS={DELAY_DELTAS}"
        )

    save_dir = args.save_dir or os.path.join(os.path.abspath(args.logs_dir), "pipeline_test_delay_new")
    result = run_test_delay_pipeline(
        logs_dir=args.logs_dir,
        save_dir=save_dir,
        methods=args.method,
        force_eval=args.force_eval,
        delays=delays,
    )

    print("Saved test-delay summary to", os.path.abspath(os.path.join(save_dir, "test_delay_summary.json")))
    print(f"val_candidates={result['num_val_candidates']}")
    print(f"selected={result['num_selected']}")
    print(f"selected_with_test_logs={result['num_selected_with_test_logs']}")
    print(f"eval_failures={result['num_eval_failures']}")


if __name__ == "__main__":
    main()
