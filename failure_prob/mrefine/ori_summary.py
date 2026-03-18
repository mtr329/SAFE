import argparse
import json
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from failure_prob.mrefine.const import HANDCRAFTED_METHOD_ALLOWLIST


def _get_ori_summary(
    ori_logs,
):
    ori_summary = {}
    for split, split_dict in ori_logs["static"].items():
        ori_summary.setdefault(split, {})
        task_dict = ori_logs["static"][split]["all"]
        ori_summary[split]["roc_auc"] = np.array(task_dict["roc_auc"]).mean()
        ori_summary[split]["prc_auc"] = np.array(task_dict["prc_auc"]).mean()
    return ori_summary


def _split_ori_logs_by_method(
    ori_logs: dict,
    fallback_method_name: str,
) -> dict[str, dict]:
    static_logs = ori_logs.get("static", {})
    calib_logs = ori_logs.get("calib", {})

    if any(split in static_logs for split in ("train", "val_seen", "val_unseen")):
        return {
            fallback_method_name: {
                "static": static_logs,
                "calib": calib_logs,
            }
        }

    method_names = set(static_logs.keys()) | set(calib_logs.keys())
    method_logs_by_name = {}
    for method_name in sorted(method_names):
        method_logs_by_name[method_name] = {
            "static": static_logs.get(method_name, {}),
            "calib": calib_logs.get(method_name, {}),
        }
    return method_logs_by_name


def _summarize_method_runs(runs_dict: dict[str, dict]) -> dict:
    summary_acc = {}
    for ori_logs in runs_dict.values():
        run_summary = _get_ori_summary(ori_logs)
        for split, metrics in run_summary.items():
            summary_acc.setdefault(split, {})
            for metric_name, value in metrics.items():
                summary_acc[split].setdefault(metric_name, []).append(value)

    return {
        split: {
            metric_name: float(np.mean(values))
            for metric_name, values in metrics.items()
        }
        for split, metrics in summary_acc.items()
    }


def _ori_summary_to_df(ori_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, split_metrics in ori_summary.items():
        for split_name, metrics in split_metrics.items():
            row = {
                "method": method_name,
                "split": split_name,
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["method", "split", "roc_auc", "prc_auc"])

    df = pd.DataFrame(rows)
    front_cols = [col for col in ["method", "split", "roc_auc", "prc_auc"] if col in df.columns]
    other_cols = [col for col in df.columns if col not in front_cols]
    return df[front_cols + other_cols]


def _get_ori_figs(
    ori_logs_by_method,
):
    line_alpha = 0.75
    line_width = 2.0
    marker_size = 5
    explicit_method_colors = {
        "lstm": "#c62828",
        "indep": "#ff69b4",
    }
    fallback_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#e377c2",
    ]
    used_colors = set(explicit_method_colors.values())
    fallback_palette = [c for c in fallback_palette if c not in used_colors]

    method_colors = dict(explicit_method_colors)
    fallback_methods = [
        method_name
        for method_name in sorted(ori_logs_by_method)
        if method_name not in method_colors
    ]
    for i, method_name in enumerate(fallback_methods):
        method_colors[method_name] = fallback_palette[i % len(fallback_palette)]

    figs = {}

    for eval_time in ["early", "last"]:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()

        has_curve = False
        for method_name, runs_dict in sorted(ori_logs_by_method.items()):
            alpha_to_avg_det_times = {}
            alpha_to_bal_accs = {}

            for _, ori_logs in sorted(runs_dict.items()):
                calib_logs = ori_logs.get("calib", {}).get(eval_time, {})
                if not calib_logs:
                    continue

                for alpha_str in sorted(calib_logs.keys(), key=float):
                    alpha_dict = calib_logs[alpha_str]
                    if "avg_det_time" not in alpha_dict or "bal_acc" not in alpha_dict:
                        continue
                    alpha_to_avg_det_times.setdefault(alpha_str, []).append(
                        np.asarray(alpha_dict["avg_det_time"]).mean()
                    )
                    alpha_to_bal_accs.setdefault(alpha_str, []).append(
                        np.asarray(alpha_dict["bal_acc"]).mean()
                    )

            avg_det_times = []
            bal_accs = []
            for alpha_str in sorted(alpha_to_avg_det_times.keys(), key=float):
                if alpha_str not in alpha_to_bal_accs:
                    continue
                avg_det_times.append(np.mean(alpha_to_avg_det_times[alpha_str]))
                bal_accs.append(np.mean(alpha_to_bal_accs[alpha_str]))

            if not avg_det_times:
                continue

            has_curve = True
            ax.plot(
                avg_det_times,
                bal_accs,
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label=method_name,
                color=method_colors.get(method_name),
                alpha=line_alpha,
                markeredgewidth=0.0,
            )

        ax.set_xlabel("avg_det_time")
        ax.set_ylabel("bal_acc")
        ax.set_title(f"avg_det_time vs bal_acc ({eval_time})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)
        fig.tight_layout()

        figs[eval_time] = fig

    return figs


def _collect_ori_log_paths(logs_dir):
    log_paths = []
    direct_candidates = [
        os.path.join(logs_dir, "eval", "ori_logs.json"),
        os.path.join(logs_dir, "eval", "my_logs.json"),
        os.path.join(logs_dir, "eval", "mylogs.json"),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            log_paths.append(os.path.abspath(candidate))

    for root, _, files in os.walk(logs_dir):
        if os.path.basename(root) != "eval":
            continue
        for filename in ("ori_logs.json", "my_logs.json", "mylogs.json"):
            if filename in files:
                log_path = os.path.abspath(os.path.join(root, filename))
                if log_path not in log_paths:
                    log_paths.append(log_path)

    return sorted(log_paths)


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


def summary_ori_metrics(
    logs_dir="logs",
    save_dir=None,
):
    if not save_dir:
        save_dir = os.path.join(logs_dir, "summary")
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    ori_save_dir = os.path.join(save_dir, "ori")
    os.makedirs(ori_save_dir, exist_ok=True)

    ori_summary = {}
    ori_logs_by_method = {}
    log_paths = _collect_ori_log_paths(logs_dir)
    for log_path in log_paths:
        with open(log_path, "r") as f:
            logs = json.load(f)

        if os.path.basename(log_path) == "ori_logs.json":
            ori_logs = logs
        else:
            if "ori" not in logs:
                continue
            ori_logs = logs["ori"]

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        fallback_method_name = run_meta["method_name"]
        method_logs_by_name = _split_ori_logs_by_method(ori_logs, fallback_method_name)
        for method_name, method_logs in method_logs_by_name.items():
            if (
                run_meta["is_handcrafted"]
                and HANDCRAFTED_METHOD_ALLOWLIST is not None
                and method_name not in HANDCRAFTED_METHOD_ALLOWLIST
            ):
                continue
            ori_logs_by_method.setdefault(method_name, {})
            ori_logs_by_method[method_name][run_name] = method_logs

    for method_name, runs_dict in ori_logs_by_method.items():
        ori_summary[method_name] = _summarize_method_runs(runs_dict)

    save_path = os.path.join(ori_save_dir, "ori_summary.json")
    with open(save_path, "w") as f:
        json.dump(ori_summary, f, indent=2)

    csv_save_path = os.path.join(ori_save_dir, "ori_summary.csv")
    _ori_summary_to_df(ori_summary).to_csv(csv_save_path, index=False)

    ori_figs = _get_ori_figs(ori_logs_by_method)
    for eval_time, fig in ori_figs.items():
        fig.savefig(os.path.join(ori_save_dir, f"{eval_time}.png"), dpi=400)

    return ori_summary, ori_figs


def _resolve_default_logs_dir() -> str:
    env_logs_dir = os.environ.get("ORI_METRICS_LOGS_DIR")
    if env_logs_dir:
        return env_logs_dir

    for candidate in ("log_ckpt", "logs"):
        if os.path.isdir(candidate):
            return candidate

    return "log_ckpt"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ori metrics from evaluation logs.",
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default=None,
        help="Root directory containing evaluation outputs with eval/ori_logs.json or eval/my_logs.json.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional output directory. Defaults to <logs_dir>/summary.",
    )
    args = parser.parse_args()

    summary_ori_metrics(
        logs_dir=args.logs_dir or _resolve_default_logs_dir(),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
