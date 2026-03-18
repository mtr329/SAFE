import argparse
import json
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from failure_prob.mrefine.const import HANDCRAFTED_METHOD_ALLOWLIST
from failure_prob.mrefine.ref_metrics import REF_TOLERANCE_DELTAS

REF_BASE_METRICS = (
    "point_precision",
    "point_recall",
    "point_f1",
    "point_adjusted_precision",
    "point_adjusted_recall",
    "point_adjusted_f1",
    "segment_precision",
    "segment_recall",
    "segment_f1",
)


def _get_time_tolerant_metric_names() -> list[str]:
    metric_names = []
    for delta in REF_TOLERANCE_DELTAS:
        tau_tag = delta.replace(".", "p")
        metric_names.extend(
            [
                f"time_tolerant_precision_tau{tau_tag}",
                f"time_tolerant_recall_tau{tau_tag}",
                f"time_tolerant_f1_tau{tau_tag}",
            ]
        )
    return metric_names


REF_METRIC_NAMES = (*REF_BASE_METRICS, *_get_time_tolerant_metric_names())
REF_SELECTION_METRICS = (
    "point_f1",
    "point_adjusted_f1",
    "segment_f1",
    "nab_standard",
    "nab_reward_low_fp",
    "nab_reward_low_fn",
    "time_tolerant_f1_tau0p0",
    "time_tolerant_f1_tau0p1",
    "time_tolerant_f1_tau0p2",
    "time_tolerant_f1_tau0p3",
)


def _split_ref_logs_by_method(
    ref_logs: dict,
    fallback_method_name: str,
) -> dict[str, dict]:
    calib_logs = ref_logs.get("calib", {})
    continuous_logs = ref_logs.get("continuous", {})

    if any(mode in calib_logs for mode in ("early", "last")) or any(
        mode in continuous_logs for mode in ("early", "last")
    ):
        return {
            fallback_method_name: {
                "calib": calib_logs,
                "continuous": continuous_logs,
            }
        }

    return {
        method_name: {
            "calib": calib_logs.get(method_name, {}),
            "continuous": continuous_logs.get(method_name, {}),
        }
        for method_name in sorted(set(calib_logs.keys()) | set(continuous_logs.keys()))
    }


def _get_run_meta_from_config(run_dir) -> dict:
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        method_name = os.path.basename(run_dir)
        return {
            "method_name": method_name,
            "is_handcrafted": False,
            "exp_suffix": None,
        }

    cfg = OmegaConf.load(cfg_path)
    method_name = cfg.model.name
    if "distance" in cfg.model:
        method_name += f"_{cfg.model.distance}"

    exp_suffix = getattr(cfg.train, "exp_suffix", None)
    is_handcrafted = bool(getattr(cfg.train, "log_precomputed_only", False))
    return {
        "method_name": method_name,
        "is_handcrafted": is_handcrafted,
        "exp_suffix": exp_suffix,
    }


def _collect_ref_log_paths(logs_dir):
    log_paths = []
    eval_dirs = set()

    direct_eval_dir = os.path.join(logs_dir, "eval")
    if os.path.isdir(direct_eval_dir):
        eval_dirs.add(os.path.abspath(direct_eval_dir))

    for root, _, _ in os.walk(logs_dir):
        if os.path.basename(root) == "eval":
            eval_dirs.add(os.path.abspath(root))

    for eval_dir in sorted(eval_dirs):
        dedicated_path = os.path.join(eval_dir, "ref_logs.json")
        if os.path.isfile(dedicated_path):
            log_paths.append(dedicated_path)
            continue

        legacy_candidates = [
            os.path.join(eval_dir, "my_logs.json"),
            os.path.join(eval_dir, "mylogs.json"),
        ]
        for candidate in legacy_candidates:
            if os.path.isfile(candidate):
                log_paths.append(candidate)
                break

    return log_paths


def _summarize_ref_calib(ref_logs: dict) -> dict:
    summary = {}
    calib_logs = ref_logs.get("calib", {})
    for mode, delta_dict in calib_logs.items():
        summary.setdefault(mode, {})
        for delta, alpha_dicts in delta_dict.items():
            summary[mode].setdefault(delta, {})
            for alpha, metrics_dict in alpha_dicts.items():
                summary[mode][delta].setdefault(alpha, {})
                for metric_name, value in metrics_dict.items():
                    if metric_name == "detect_method":
                        summary[mode][delta][alpha][metric_name] = value
                    else:
                        summary[mode][delta][alpha][metric_name] = float(np.asarray(value).mean())
    return summary


def _summarize_ref_continuous(ref_logs: dict) -> dict:
    summary = {}
    continuous_logs = ref_logs.get("continuous", {})
    for mode, delta_dict in continuous_logs.items():
        summary.setdefault(mode, {})
        for delta, metrics_dict in delta_dict.items():
            summary[mode].setdefault(delta, {})
            for metric_name, value in metrics_dict.items():
                if metric_name == "detect_method":
                    summary[mode][delta][metric_name] = value
                else:
                    summary[mode][delta][metric_name] = float(np.asarray(value).mean())
    return summary


def _merge_ref_summaries(runs_dict: dict[str, dict]) -> dict:
    calib_acc = {}
    continuous_acc = {}
    for ref_logs in runs_dict.values():
        calib_summary = _summarize_ref_calib(ref_logs)
        for mode, delta_dict in calib_summary.items():
            calib_acc.setdefault(mode, {})
            for delta, alpha_dict in delta_dict.items():
                calib_acc[mode].setdefault(delta, {})
                for alpha, metrics in alpha_dict.items():
                    calib_acc[mode][delta].setdefault(alpha, {})
                    for metric_name, value in metrics.items():
                        if metric_name == "detect_method":
                            calib_acc[mode][delta][alpha][metric_name] = value
                        else:
                            calib_acc[mode][delta][alpha].setdefault(metric_name, []).append(value)

        continuous_summary = _summarize_ref_continuous(ref_logs)
        for mode, delta_dict in continuous_summary.items():
            continuous_acc.setdefault(mode, {})
            for delta, metrics in delta_dict.items():
                continuous_acc[mode].setdefault(delta, {})
                for metric_name, value in metrics.items():
                    if metric_name == "detect_method":
                        continuous_acc[mode][delta][metric_name] = value
                    else:
                        continuous_acc[mode][delta].setdefault(metric_name, []).append(value)

    calib_summary = {}
    for mode, delta_dict in calib_acc.items():
        calib_summary.setdefault(mode, {})
        for delta, alpha_dict in delta_dict.items():
            calib_summary[mode].setdefault(delta, {})
            for alpha, metrics in alpha_dict.items():
                calib_summary[mode][delta].setdefault(alpha, {})
                for metric_name, value in metrics.items():
                    if metric_name == "detect_method":
                        calib_summary[mode][delta][alpha][metric_name] = value
                    else:
                        calib_summary[mode][delta][alpha][metric_name] = float(np.mean(value))

    continuous_summary = {}
    for mode, delta_dict in continuous_acc.items():
        continuous_summary.setdefault(mode, {})
        for delta, metrics in delta_dict.items():
            continuous_summary[mode].setdefault(delta, {})
            for metric_name, value in metrics.items():
                if metric_name == "detect_method":
                    continuous_summary[mode][delta][metric_name] = value
                else:
                    continuous_summary[mode][delta][metric_name] = float(np.mean(value))

    return {"calib": calib_summary, "continuous": continuous_summary}


def _ref_calib_summary_to_df(ref_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in ref_summary.items():
        for mode, delta_dict in summary.get("calib", {}).items():
            for delta, alpha_dict in delta_dict.items():
                for alpha, metrics in alpha_dict.items():
                    row = {
                        "method": method_name,
                        "mode": mode,
                        "delta": float(delta),
                        "alpha": float(alpha),
                    }
                    row.update(metrics)
                    rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["method", "mode", "delta", "alpha", *REF_METRIC_NAMES])

    df = pd.DataFrame(rows)
    ordered_columns = ["method", "mode", "delta", "alpha"]
    ordered_columns.extend([name for name in REF_METRIC_NAMES if name in df.columns])
    optional_columns = [name for name in df.columns if name not in ordered_columns and name != "detect_method"]
    if "detect_method" in df.columns:
        ordered_columns.append("detect_method")
    ordered_columns.extend(optional_columns)
    return df[ordered_columns].sort_values(by=["method", "mode", "delta", "alpha"]).reset_index(drop=True)


def _best_rows_by_metric(ref_calib_df: pd.DataFrame, metric_names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    if ref_calib_df.empty:
        return pd.DataFrame(columns=["method", "mode", "delta", "metric", "best_alpha", "best_value"])

    group_cols = ["method", "mode", "delta"]
    for (method_name, mode, delta), group_df in ref_calib_df.groupby(group_cols, sort=True):
        for metric_name in metric_names:
            if metric_name not in group_df.columns:
                continue
            valid_df = group_df.dropna(subset=[metric_name])
            if valid_df.empty:
                best_alpha = np.nan
                best_value = np.nan
            else:
                best_idx = valid_df[metric_name].astype(float).idxmax()
                best_alpha = float(valid_df.loc[best_idx, "alpha"])
                best_value = float(valid_df.loc[best_idx, metric_name])
            rows.append(
                {
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                    "metric": metric_name,
                    "best_alpha": best_alpha,
                    "best_value": best_value,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["method", "mode", "delta", "metric", "best_alpha", "best_value"])

    return pd.DataFrame(rows).sort_values(
        by=["method", "mode", "metric", "delta"]
    ).reset_index(drop=True)


def _ref_continuous_summary_to_df(ref_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in ref_summary.items():
        for mode, delta_dict in summary.get("continuous", {}).items():
            for delta, metrics in delta_dict.items():
                row = {
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                }
                row.update(metrics)
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["method", "mode", "delta", "pate"])

    df = pd.DataFrame(rows)
    ordered_columns = ["method", "mode", "delta"]
    if "pate" in df.columns:
        ordered_columns.append("pate")
    if "detect_method" in df.columns:
        ordered_columns.append("detect_method")
    for column in df.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    return df[ordered_columns].sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)


def summary_ref_metrics(
    logs_dir="logs",
    save_dir=None,
):
    if not save_dir:
        save_dir = os.path.join(logs_dir, "summary")
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    ref_save_dir = os.path.join(save_dir, "ref")
    calib_save_dir = os.path.join(ref_save_dir, "calib")
    continuous_save_dir = os.path.join(ref_save_dir, "continuous")
    best_save_dir = os.path.join(ref_save_dir, "best_over_alpha")
    os.makedirs(calib_save_dir, exist_ok=True)
    os.makedirs(continuous_save_dir, exist_ok=True)
    os.makedirs(best_save_dir, exist_ok=True)

    ref_logs_by_method = {}
    for log_path in _collect_ref_log_paths(logs_dir):
        with open(log_path, "r") as f:
            logs = json.load(f)

        if os.path.basename(log_path) == "ref_logs.json":
            ref_logs = logs
        else:
            if "ref" not in logs:
                continue
            ref_logs = logs["ref"]

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        fallback_method_name = run_meta["method_name"]
        method_logs_by_name = _split_ref_logs_by_method(ref_logs, fallback_method_name)
        for method_name, method_logs in method_logs_by_name.items():
            if (
                run_meta["is_handcrafted"]
                and HANDCRAFTED_METHOD_ALLOWLIST is not None
                and method_name not in HANDCRAFTED_METHOD_ALLOWLIST
            ):
                continue
            ref_logs_by_method.setdefault(method_name, {})
            ref_logs_by_method[method_name][run_name] = method_logs

    ref_summary = {
        method_name: _merge_ref_summaries(runs_dict)
        for method_name, runs_dict in ref_logs_by_method.items()
    }

    with open(os.path.join(ref_save_dir, "ref_summary.json"), "w") as f:
        json.dump(ref_summary, f, indent=2)

    ref_calib_df = _ref_calib_summary_to_df(ref_summary)
    ref_calib_df.to_csv(
        os.path.join(calib_save_dir, "ref_calib_summary.csv"),
        index=False,
    )
    _ref_continuous_summary_to_df(ref_summary).to_csv(
        os.path.join(continuous_save_dir, "ref_continuous_summary.csv"),
        index=False,
    )

    _best_rows_by_metric(ref_calib_df, REF_SELECTION_METRICS).to_csv(
        os.path.join(best_save_dir, "ref_best_over_alpha.csv"),
        index=False,
    )

    return ref_summary


def _resolve_default_logs_dir() -> str:
    env_logs_dir = os.environ.get("REF_METRICS_LOGS_DIR")
    if env_logs_dir:
        return env_logs_dir

    for candidate in ("log_ckpt", "logs"):
        if os.path.isdir(candidate):
            return candidate

    return "log_ckpt"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize reference temporal metrics from evaluation logs.",
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default=None,
        help="Root directory containing evaluation outputs with eval/ref_logs.json or eval/my_logs.json.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional output directory. Defaults to <logs_dir>/summary.",
    )
    args = parser.parse_args()

    summary_ref_metrics(
        logs_dir=args.logs_dir or _resolve_default_logs_dir(),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
