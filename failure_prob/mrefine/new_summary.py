import argparse
import colorsys
import json
import os

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from failure_prob.mrefine.const import HANDCRAFTED_METHOD_ALLOWLIST

PREFIX_AUC_PLOT_PREFIXES = [f"{p:.2f}" for p in np.arange(0.10, 1.001, 0.10)]
TEMPORAL_LAMBDA_TAGS = ("lam1p0", "lam3p0", "lam5p0")


def _split_new_logs_by_method(
    new_logs: dict,
    fallback_method_name: str,
) -> dict[str, dict]:
    static_logs = new_logs.get("static", {})
    calib_logs = new_logs.get("calib", {})
    soft_logs = new_logs.get("soft", {})

    if any(split in static_logs for split in ("train", "val_seen", "val_unseen")):
        return {
            fallback_method_name: {
                "static": static_logs,
                "calib": calib_logs,
                "soft": soft_logs,
            }
        }

    method_names = set(static_logs.keys()) | set(calib_logs.keys()) | set(soft_logs.keys())
    method_logs_by_name = {}
    for method_name in sorted(method_names):
        method_logs_by_name[method_name] = {
            "static": static_logs.get(method_name, {}),
            "calib": calib_logs.get(method_name, {}),
            "soft": soft_logs.get(method_name, {}),
        }
    return method_logs_by_name


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


def _collect_new_log_paths(logs_dir):
    log_paths = []
    eval_dirs = set()

    direct_eval_dir = os.path.join(logs_dir, "eval")
    if os.path.isdir(direct_eval_dir):
        eval_dirs.add(os.path.abspath(direct_eval_dir))

    for root, _, _ in os.walk(logs_dir):
        if os.path.basename(root) == "eval":
            eval_dirs.add(os.path.abspath(root))

    for eval_dir in sorted(eval_dirs):
        dedicated_path = os.path.join(eval_dir, "new_logs.json")
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


def _summarize_new_static(new_logs: dict) -> dict:
    summary = {}
    static_logs = new_logs.get("static", {})
    for split_name, split_dict in static_logs.items():
        task_dict = split_dict.get("all", {})
        if not task_dict:
            continue

        summary.setdefault(split_name, {})
        for delta, prefix_dict in task_dict.items():
            summary[split_name].setdefault(delta, {})
            for prefix, metrics_dict in prefix_dict.items():
                summary[split_name][delta].setdefault(prefix, {})
                for metric_name in ("roc_auc", "prc_auc"):
                    if metric_name not in metrics_dict:
                        continue
                    summary[split_name][delta][prefix][metric_name] = float(
                        np.asarray(metrics_dict[metric_name]).mean()
                    )
    return summary


def _summarize_new_calib(new_logs: dict) -> dict:
    summary = {}
    calib_logs = new_logs.get("calib", {})
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
                        summary[mode][delta][alpha][metric_name] = float(
                            np.asarray(value).mean()
                        )
    return summary


def _merge_new_summaries(runs_dict: dict[str, dict]) -> dict:
    static_acc = {}
    calib_acc = {}
    soft_acc = {}

    for new_logs in runs_dict.values():
        static_summary = _summarize_new_static(new_logs)
        for split_name, delta_dict in static_summary.items():
            static_acc.setdefault(split_name, {})
            for delta, prefix_dict in delta_dict.items():
                static_acc[split_name].setdefault(delta, {})
                for prefix, metrics in prefix_dict.items():
                    static_acc[split_name][delta].setdefault(prefix, {})
                    for metric_name, value in metrics.items():
                        static_acc[split_name][delta][prefix].setdefault(metric_name, []).append(value)

        calib_summary = _summarize_new_calib(new_logs)
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

        soft_summary = _summarize_new_soft(new_logs)
        for mode, delta_dict in soft_summary.items():
            soft_acc.setdefault(mode, {})
            for delta, metrics in delta_dict.items():
                soft_acc[mode].setdefault(delta, {})
                for metric_name, value in metrics.items():
                    if metric_name == "detect_method":
                        soft_acc[mode][delta][metric_name] = value
                    else:
                        soft_acc[mode][delta].setdefault(metric_name, []).append(value)

    static_summary = {
        split_name: {
            delta: {
                prefix: {
                    metric_name: float(np.mean(values))
                    for metric_name, values in metrics.items()
                }
                for prefix, metrics in prefix_dict.items()
            }
            for delta, prefix_dict in delta_dict.items()
        }
        for split_name, delta_dict in static_acc.items()
    }

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

    soft_summary = {}
    for mode, delta_dict in soft_acc.items():
        soft_summary.setdefault(mode, {})
        for delta, metrics in delta_dict.items():
            soft_summary[mode].setdefault(delta, {})
            for metric_name, value in metrics.items():
                if metric_name == "detect_method":
                    soft_summary[mode][delta][metric_name] = value
                else:
                    soft_summary[mode][delta][metric_name] = float(np.mean(value))

    return {
        "static": static_summary,
        "calib": calib_summary,
        "soft": soft_summary,
    }


def _summarize_new_soft(new_logs: dict) -> dict:
    summary = {}
    soft_logs = new_logs.get("soft", {})
    for mode, delta_dict in soft_logs.items():
        summary.setdefault(mode, {})
        for delta, metrics_dict in delta_dict.items():
            summary[mode].setdefault(delta, {})
            for metric_name, value in metrics_dict.items():
                if metric_name == "detect_method":
                    summary[mode][delta][metric_name] = value
                else:
                    summary[mode][delta][metric_name] = float(np.asarray(value).mean())
    return summary


def _new_calib_summary_to_df(new_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in new_summary.items():
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
        return pd.DataFrame(
            columns=["method", "mode", "delta", "alpha", "avg_det_time", "ttd_auc", "bal_acc"]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by=["method", "mode", "delta", "alpha"]).reset_index(drop=True)


def _new_soft_summary_to_df(new_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in new_summary.items():
        for mode, delta_dict in summary.get("soft", {}).items():
            for delta, metrics in delta_dict.items():
                row = {
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                }
                row.update(metrics)
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["method", "mode", "delta", "soft_avg_det_time", "soft_ttd_auc"]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)


def _get_method_colors(method_names):
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

    method_colors = dict(explicit_method_colors)
    fallback_methods = [
        method_name
        for method_name in sorted(method_names)
        if method_name not in method_colors
    ]
    for i, method_name in enumerate(fallback_methods):
        method_colors[method_name] = fallback_palette[i % len(fallback_palette)]
    return method_colors


def _adjust_color_lightness(color: str, amount: float) -> str:
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, amount))
    return to_hex(colorsys.hls_to_rgb(h, l, s))


def _get_ordered_palette(num_colors: int, cmap_name: str = "turbo") -> list[str]:
    if num_colors <= 0:
        return []

    cmap = colormaps[cmap_name]
    positions = np.linspace(0.05, 0.95, num_colors)
    return [to_hex(cmap(pos)) for pos in positions]


def _interp_det_time_for_bal_acc(alpha_dict: dict, target_bal_acc: float) -> float:
    pairs = []
    for metrics in alpha_dict.values():
        if "bal_acc" not in metrics or "avg_det_time" not in metrics:
            continue
        pairs.append((float(metrics["bal_acc"]), float(metrics["avg_det_time"])))

    if len(pairs) < 2:
        return np.nan

    pairs = sorted(pairs, key=lambda x: x[0])
    bal_accs = np.asarray([p[0] for p in pairs], dtype=float)
    det_times = np.asarray([p[1] for p in pairs], dtype=float)

    uniq_bal_accs, inverse = np.unique(bal_accs, return_inverse=True)
    uniq_det_times = np.zeros_like(uniq_bal_accs)
    counts = np.zeros_like(uniq_bal_accs)
    for idx, det_time in zip(inverse, det_times):
        uniq_det_times[idx] += det_time
        counts[idx] += 1
    uniq_det_times = uniq_det_times / np.maximum(counts, 1)

    if target_bal_acc < uniq_bal_accs.min() or target_bal_acc > uniq_bal_accs.max():
        return np.nan
    return float(np.interp(target_bal_acc, uniq_bal_accs, uniq_det_times))


def _get_new_target_balacc_figs(
    new_summary: dict,
    target_bal_accs: list[float],
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []

    for method_name, summary in sorted(new_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        base_color = "#1f77b4"

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = [float(delta) for delta in deltas]
            has_curve = False

            for i, target_bal_acc in enumerate(target_bal_accs):
                y = []
                for delta in deltas:
                    det_time = _interp_det_time_for_bal_acc(delta_dict[delta], target_bal_acc)
                    y.append(det_time)
                    rows.append({
                        "method": method_name,
                        "mode": mode,
                        "target_bal_acc": target_bal_acc,
                        "delta": float(delta),
                        "min_det_time": det_time,
                    })

                if np.all(np.isnan(y)):
                    continue

                has_curve = True
                color = _adjust_color_lightness(
                    base_color,
                    0.30 + 0.50 * (i + 1) / max(len(target_bal_accs), 1),
                )
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=color,
                    alpha=0.9,
                    label=f"bal_acc={target_bal_acc:.2f}",
                )

            ax.set_xlabel("delay")
            ax.set_ylabel("min_det_time")
            ax.set_title(mode)
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9)

        fig.suptitle(f"{method_name}: min det_time at fixed bal_acc")
        fig.tight_layout()
        figs[f"{method_name}_dettime_at_fixed_balacc"] = fig

    target_df = pd.DataFrame(rows)
    if not target_df.empty:
        target_df = target_df.sort_values(
            by=["method", "mode", "target_bal_acc", "delta"]
        ).reset_index(drop=True)
    else:
        target_df = pd.DataFrame(
            columns=["method", "mode", "target_bal_acc", "delta", "min_det_time"]
        )
    return figs, target_df


def _get_prefix_weight(prefix: str) -> float:
    prefix_value = float(prefix)
    if prefix_value <= 0:
        return 0.0
    return 1.0 / prefix_value


def _get_weighted_metric_over_prefix(prefix_dict: dict, metric_name: str) -> float:
    values = []
    weights = []
    for prefix, metrics in sorted(prefix_dict.items(), key=lambda x: float(x[0])):
        if metric_name not in metrics:
            continue
        weight = _get_prefix_weight(prefix)
        if weight <= 0:
            continue
        values.append(float(metrics[metric_name]))
        weights.append(weight)

    if not values:
        return np.nan
    return float(np.average(values, weights=weights))


def _new_weighted_roc_auc_to_df(new_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in new_summary.items():
        for split_name, delta_dict in summary.get("static", {}).items():
            for delta, prefix_dict in delta_dict.items():
                rows.append({
                    "method": method_name,
                    "split": split_name,
                    "delta": float(delta),
                    "weighted_roc_auc": _get_weighted_metric_over_prefix(prefix_dict, "roc_auc"),
                })

    if not rows:
        return pd.DataFrame(columns=["method", "split", "delta", "weighted_roc_auc"])

    df = pd.DataFrame(rows)
    return df.sort_values(by=["split", "method", "delta"]).reset_index(drop=True)


def _get_new_weighted_roc_auc_figs(new_summary: dict) -> dict[str, Figure]:
    figs = {}
    method_colors = _get_method_colors(new_summary.keys())

    all_splits = sorted({
        split_name
        for summary in new_summary.values()
        for split_name in summary.get("static", {}).keys()
    })
    for split_name in all_splits:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        has_curve = False

        for method_name, summary in sorted(new_summary.items()):
            delta_dict = summary.get("static", {}).get(split_name, {})
            if not delta_dict:
                continue

            deltas = sorted(delta_dict.keys(), key=float)
            x = []
            y = []
            for delta in deltas:
                weighted_roc_auc = _get_weighted_metric_over_prefix(delta_dict[delta], "roc_auc")
                if np.isnan(weighted_roc_auc):
                    continue
                x.append(float(delta))
                y.append(weighted_roc_auc)

            if not x:
                continue

            has_curve = True
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=method_colors[method_name],
                alpha=0.8,
                label=method_name,
            )

        ax.set_xlabel("delay")
        ax.set_ylabel("time_weighted_roc_auc")
        ax.set_title(f"time-weighted roc_auc vs delay ({split_name})")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        fig.tight_layout()
        figs[f"{split_name}_weighted_roc_auc_vs_delay"] = fig

    return figs


def _get_new_prefix_auc_figs(new_summary: dict) -> dict[str, Figure]:
    figs = {}
    palette = _get_ordered_palette(len(PREFIX_AUC_PLOT_PREFIXES), cmap_name="turbo")

    for method_name, summary in sorted(new_summary.items()):
        for split_name, delta_dict in sorted(summary.get("static", {}).items()):
            fig = Figure(figsize=(12, 5))
            axes = fig.subplots(1, 2)
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])

            for ax, metric_name in zip(axes, ["roc_auc", "prc_auc"]):
                deltas = sorted(delta_dict.keys(), key=float)
                has_curve = False

                for i, prefix in enumerate(PREFIX_AUC_PLOT_PREFIXES):
                    x = []
                    y = []
                    for delta in deltas:
                        prefix_metrics = delta_dict[delta].get(prefix, {})
                        metric_value = prefix_metrics.get(metric_name)
                        if metric_value is None:
                            continue
                        x.append(float(delta))
                        y.append(float(metric_value))

                    if not x:
                        continue

                    has_curve = True
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        linewidth=2.0,
                        markersize=4,
                        color=palette[i],
                        alpha=0.9,
                        label=f"prefix={prefix}",
                    )

                ax.set_xlabel("delay")
                ax.set_ylabel(metric_name)
                ax.set_title(metric_name)
                ax.set_xlim(left=0.0)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
                if has_curve:
                    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

            fig.suptitle(f"{method_name}: auc vs delay by prefix ({split_name})")
            fig.tight_layout()
            figs[f"{method_name}_{split_name}_auc_vs_delay_by_prefix"] = fig

    return figs


def _get_new_soft_metric_figs(new_summary: dict) -> dict[str, Figure]:
    figs = {}
    method_colors = _get_method_colors(new_summary.keys())
    metric_specs = (
        ("soft_ttd_auc", "soft_ttd_auc"),
        ("soft_avg_det_time", "soft_avg_det_time"),
    )

    for mode in ["early", "last"]:
        for metric_name, ylabel in metric_specs:
            fig = Figure(figsize=(7, 5))
            ax = fig.subplots()
            has_curve = False

            for method_name, summary in sorted(new_summary.items()):
                delta_dict = summary.get("soft", {}).get(mode, {})
                deltas = sorted(delta_dict.keys(), key=float)
                x = []
                y = []
                for delta in deltas:
                    metric_value = delta_dict[delta].get(metric_name)
                    if metric_value is None:
                        continue
                    x.append(float(delta))
                    y.append(float(metric_value))

                if not x:
                    continue

                has_curve = True
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=5,
                    color=method_colors[method_name],
                    alpha=0.85,
                    label=method_name,
                )

            ax.set_xlabel("delay")
            ax.set_ylabel(ylabel)
            ax.set_title(mode)
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
            fig.tight_layout()
            figs[f"{metric_name}_{mode}"] = fig

    return figs


def _metric_prefers_delay_x(metric_name: str) -> bool:
    if metric_name in ("ttd_auc",):
        return True
    if metric_name.startswith("discounted_ttd_auc_"):
        return True
    if metric_name.startswith("discounted_correct_"):
        return True
    return False


def _format_metric_title(metric_name: str) -> str:
    if metric_name == "ttd_auc":
        return "ttd_auc"
    if metric_name.startswith("discounted_ttd_auc_"):
        lambda_tag = metric_name.replace("discounted_ttd_auc_", "")
        return f"discounted_ttd_auc ({lambda_tag})"
    if metric_name.startswith("discounted_correct_"):
        lambda_tag = metric_name.replace("discounted_correct_", "")
        return f"discounted_correct ({lambda_tag})"
    return metric_name


def _get_new_alpha_metric_figs(new_summary: dict) -> dict[str, Figure]:
    figs = {}
    metric_specs = [
        ("avg_det_time", "avg_det_time"),
        ("ttd_auc", "ttd_auc"),
        ("bal_acc", "bal_acc"),
        ("fpr", "fpr"),
        ("fnr", "fnr"),
    ]
    for lambda_tag in TEMPORAL_LAMBDA_TAGS:
        metric_specs.append((f"discounted_ttd_auc_{lambda_tag}", f"discounted_ttd_auc_{lambda_tag}"))
        metric_specs.append((f"discounted_correct_{lambda_tag}", f"discounted_correct_{lambda_tag}"))

    for method_name, summary in sorted(new_summary.items()):
        for metric_name, ylabel in metric_specs:
            fig = Figure(figsize=(12, 5))
            axes = fig.subplots(1, 2)
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])

            for ax, mode in zip(axes, ["early", "last"]):
                delta_dict = summary.get("calib", {}).get(mode, {})
                deltas = sorted(delta_dict.keys(), key=float)
                has_curve = False

                if _metric_prefers_delay_x(metric_name):
                    all_alphas = sorted({
                        alpha
                        for alpha_dict in delta_dict.values()
                        for alpha in alpha_dict.keys()
                    }, key=float)
                    palette = _get_ordered_palette(len(all_alphas), cmap_name="turbo")

                    for i, alpha in enumerate(all_alphas):
                        x = []
                        y = []
                        for delta in deltas:
                            metrics = delta_dict[delta].get(alpha, {})
                            metric_value = metrics.get(metric_name)
                            if metric_value is None:
                                continue
                            x.append(float(delta))
                            y.append(float(metric_value))

                        if not x:
                            continue

                        has_curve = True
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            linewidth=2.0,
                            markersize=4,
                            color=palette[i % len(palette)],
                            alpha=0.9,
                            label=f"alpha={float(alpha):.2f}",
                        )

                    ax.set_xlabel("delay")
                else:
                    palette = _get_ordered_palette(len(deltas), cmap_name="turbo")

                    for i, delta in enumerate(deltas):
                        alpha_dict = delta_dict[delta]
                        alphas = sorted(alpha_dict.keys(), key=float)
                        x = []
                        y = []
                        for alpha in alphas:
                            metrics = alpha_dict[alpha]
                            metric_value = metrics.get(metric_name)
                            if metric_value is None:
                                continue
                            x.append(float(alpha))
                            y.append(float(metric_value))

                        if not x:
                            continue

                        has_curve = True
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            linewidth=2.0,
                            markersize=4,
                            color=palette[i % len(palette)],
                            alpha=0.9,
                            label=f"delay={delta}",
                        )

                    ax.set_xlabel("alpha")

                ax.set_ylabel(ylabel)
                ax.set_title(mode)
                if _metric_prefers_delay_x(metric_name):
                    ax.set_xlim(left=0.0)
                else:
                    ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
                if has_curve:
                    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

            metric_title = _format_metric_title(metric_name)
            if _metric_prefers_delay_x(metric_name):
                fig.suptitle(f"{method_name}: {metric_title} vs delay by alpha")
                figs[f"{method_name}_{metric_name}_vs_delay_by_alpha"] = fig
            else:
                fig.suptitle(f"{method_name}: {metric_title} vs alpha by delay")
                figs[f"{method_name}_{metric_name}_vs_alpha_by_delay"] = fig
            fig.tight_layout()

    return figs


def _get_alpha_metric_name_from_fig_name(fig_name: str) -> str | None:
    suffixes = (
        "_vs_alpha_by_delay",
        "_vs_delay_by_alpha",
    )
    metric_part = None
    for suffix in suffixes:
        if fig_name.endswith(suffix):
            metric_part = fig_name[: -len(suffix)]
            break
    if metric_part is None:
        return None

    for lambda_tag in TEMPORAL_LAMBDA_TAGS:
        discounted_metric_names = (
            f"discounted_ttd_auc_{lambda_tag}",
            f"discounted_correct_{lambda_tag}",
        )
        for metric_name in discounted_metric_names:
            if metric_part.endswith(f"_{metric_name}"):
                return metric_name

    for metric_name in ("avg_det_time", "ttd_auc", "bal_acc", "fpr", "fnr"):
        if metric_part.endswith(f"_{metric_name}"):
            return metric_name
    return None


def _organize_alpha_metric_root_files(
    alpha_curve_save_dir: str,
    alpha_metric_save_dirs: dict[str, str],
) -> None:
    if not os.path.isdir(alpha_curve_save_dir):
        return

    for filename in os.listdir(alpha_curve_save_dir):
        file_path = os.path.join(alpha_curve_save_dir, filename)
        if not os.path.isfile(file_path) or not filename.endswith(".png"):
            continue

        fig_name = filename[: -len(".png")]
        metric_name = _get_alpha_metric_name_from_fig_name(fig_name)
        if metric_name is None:
            continue

        target_dir = alpha_metric_save_dirs.get(metric_name)
        if target_dir is None:
            continue

        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        os.replace(file_path, target_path)


def summary_new_metrics(
    logs_dir="logs",
    save_dir=None,
):
    if not save_dir:
        save_dir = os.path.join(logs_dir, "summary")
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    new_save_dir = os.path.join(save_dir, "new")
    os.makedirs(new_save_dir, exist_ok=True)
    calib_save_dir = os.path.join(new_save_dir, "calib")
    soft_save_dir = os.path.join(new_save_dir, "soft")
    alpha_curve_save_dir = os.path.join(new_save_dir, "metrics_vs_alpha")
    alpha_metric_save_dirs = {
        metric_name: os.path.join(alpha_curve_save_dir, metric_name)
        for metric_name in (
            "avg_det_time",
            "ttd_auc",
            "bal_acc",
            "fpr",
            "fnr",
            *[f"discounted_ttd_auc_{tag}" for tag in TEMPORAL_LAMBDA_TAGS],
            *[f"discounted_correct_{tag}" for tag in TEMPORAL_LAMBDA_TAGS],
        )
    }
    target_save_dir = os.path.join(new_save_dir, "dettime_at_fixed_balacc")
    weighted_save_dir = os.path.join(new_save_dir, "weighted_roc_auc")
    prefix_auc_save_dir = os.path.join(new_save_dir, "prefix_auc_vs_delay")
    os.makedirs(calib_save_dir, exist_ok=True)
    os.makedirs(soft_save_dir, exist_ok=True)
    os.makedirs(alpha_curve_save_dir, exist_ok=True)
    for metric_save_dir in alpha_metric_save_dirs.values():
        os.makedirs(metric_save_dir, exist_ok=True)
    os.makedirs(target_save_dir, exist_ok=True)
    os.makedirs(weighted_save_dir, exist_ok=True)
    os.makedirs(prefix_auc_save_dir, exist_ok=True)

    new_logs_by_method = {}
    log_paths = _collect_new_log_paths(logs_dir)
    for log_path in log_paths:
        with open(log_path, "r") as f:
            logs = json.load(f)

        if os.path.basename(log_path) == "new_logs.json":
            new_logs = logs
        else:
            if "new" not in logs:
                continue
            new_logs = logs["new"]

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        fallback_method_name = run_meta["method_name"]
        method_logs_by_name = _split_new_logs_by_method(new_logs, fallback_method_name)
        for method_name, method_logs in method_logs_by_name.items():
            if (
                run_meta["is_handcrafted"]
                and HANDCRAFTED_METHOD_ALLOWLIST is not None
                and method_name not in HANDCRAFTED_METHOD_ALLOWLIST
            ):
                continue
            new_logs_by_method.setdefault(method_name, {})
            new_logs_by_method[method_name][run_name] = method_logs

    new_summary = {
        method_name: _merge_new_summaries(runs_dict)
        for method_name, runs_dict in new_logs_by_method.items()
    }

    with open(os.path.join(new_save_dir, "new_summary.json"), "w") as f:
        json.dump(new_summary, f, indent=2)

    _new_calib_summary_to_df(new_summary).to_csv(
        os.path.join(calib_save_dir, "new_calib_summary.csv"),
        index=False,
    )
    _new_soft_summary_to_df(new_summary).to_csv(
        os.path.join(soft_save_dir, "new_soft_summary.csv"),
        index=False,
    )

    soft_metric_figs = _get_new_soft_metric_figs(new_summary)
    for fig_name, fig in soft_metric_figs.items():
        fig.savefig(os.path.join(soft_save_dir, f"{fig_name}.png"), dpi=400)

    alpha_metric_figs = _get_new_alpha_metric_figs(new_summary)
    for fig_name, fig in alpha_metric_figs.items():
        metric_name = _get_alpha_metric_name_from_fig_name(fig_name)
        target_dir = alpha_metric_save_dirs.get(metric_name, alpha_curve_save_dir)
        fig.savefig(os.path.join(target_dir, f"{fig_name}.png"), dpi=400)
    _organize_alpha_metric_root_files(alpha_curve_save_dir, alpha_metric_save_dirs)

    target_bal_accs = [0.7, 0.8, 0.9]
    target_figs, target_df = _get_new_target_balacc_figs(new_summary, target_bal_accs)
    target_df.to_csv(
        os.path.join(target_save_dir, "new_dettime_at_fixed_balacc.csv"),
        index=False,
    )
    for fig_name, fig in target_figs.items():
        fig.savefig(os.path.join(target_save_dir, f"{fig_name}.png"), dpi=400)

    weighted_df = _new_weighted_roc_auc_to_df(new_summary)
    weighted_df.to_csv(
        os.path.join(weighted_save_dir, "new_weighted_roc_auc.csv"),
        index=False,
    )
    weighted_figs = _get_new_weighted_roc_auc_figs(new_summary)
    for fig_name, fig in weighted_figs.items():
        fig.savefig(os.path.join(weighted_save_dir, f"{fig_name}.png"), dpi=400)

    prefix_auc_figs = _get_new_prefix_auc_figs(new_summary)
    for fig_name, fig in prefix_auc_figs.items():
        fig.savefig(os.path.join(prefix_auc_save_dir, f"{fig_name}.png"), dpi=400)

    return new_summary


def _resolve_default_logs_dir() -> str:
    env_logs_dir = os.environ.get("NEW_METRICS_LOGS_DIR")
    if env_logs_dir:
        return env_logs_dir

    for candidate in ("log_ckpt", "logs"):
        if os.path.isdir(candidate):
            return candidate

    return "log_ckpt"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize new metrics from evaluation logs.",
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default=None,
        help="Root directory containing evaluation outputs with eval/new_logs.json or eval/my_logs.json.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional output directory. Defaults to <logs_dir>/summary.",
    )
    args = parser.parse_args()

    summary_new_metrics(
        logs_dir=args.logs_dir or _resolve_default_logs_dir(),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
