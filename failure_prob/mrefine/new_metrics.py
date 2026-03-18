import argparse
import colorsys
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
import numpy as np
import os
import json
import pandas as pd
from omegaconf import OmegaConf
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb, to_hex

from failure_prob.utils.conformal.functional_predictor import (
    RegressionType,
    ModulationType,
    FunctionalPredictor
)

from failure_prob.mrefine.const import DELAY_DELTAS, HANDCRAFTED_METHOD_ALLOWLIST
from failure_prob.mrefine.utils import (
    get_delay_scores,
    get_func_conformal_bands,
)

PREFIX_AUC_PLOT_PREFIXES = [f"{p:.2f}" for p in np.arange(0.10, 1.001, 0.10)]


def _get_new_static_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
    with_delay=True,
):
    delay_deltas = DELAY_DELTAS
    if not with_delay:
        delay_deltas = ["0.0"]

    prefix_rs = [f"{p:.2f}" for p in np.arange(0.05, 1.001, 0.05)]
    
    res_dict.setdefault("static", {})
    res_dict["static"].setdefault(method_name, {})
    static_dict = res_dict["static"][method_name]
    
    for split, rollouts_split in rollouts_by_split_name.items():
        static_dict.setdefault(split, {})

        scores_split = scores_by_split_name[split]
        task_ids = sorted(list(set([rollout.task_id for rollout in rollouts_split])))
        task_ids = ["all"]
        
        for task_id in task_ids:
            static_dict[split].setdefault(f"{task_id}", {})

            with warnings.catch_warnings():
                if task_id != "all":
                    indices_task = [i for i, r in enumerate(rollouts_split) if r.task_id == task_id]
                else:
                    indices_task = range(len(rollouts_split))
                rollouts_task = [rollouts_split[i] for i in indices_task]
                scores_task = [scores_split[i] for i in indices_task]
                labels_task = [1-r.episode_success for r in rollouts_task]

                for delta in delay_deltas:
                    _d = float(delta)
                    task_dict = static_dict[split][f"{task_id}"]
                    task_dict.setdefault(f"{delta}", {})
                    delta_dict = task_dict[f"{delta}"]

                    delayed_scores = [
                        get_delay_scores(s[:r.task_min_step], _d)
                        for s, r in zip(scores_task, rollouts_task)
                    ]
                    # if delta == "0.0" and split == "val_unseen" and method_name == "lstm": breakpoint() 
                    for pref in prefix_rs:
                        delta_dict.setdefault(pref, {})
                        pref_dict = delta_dict[pref]

                        pref_lens = [
                            max(int(round(len(s) * float(pref))), 1)
                            for s in delayed_scores
                        ]
                        pref_scores = [
                            s[:l].max()
                            # s[:l][-1]
                            for s, l in zip(delayed_scores, pref_lens)
                        ]
            
                        fpr, tpr, thresholds = roc_curve(labels_task, pref_scores)
                        roc_auc = auc(fpr, tpr)
                        pre, rec, thresholds = precision_recall_curve(labels_task, pref_scores)
                        prc_auc = auc(rec, pre)
                
                        for key in ["fpr", "tpr", "roc_auc", "pre", "rec", "prc_auc",]:
                            pref_dict.setdefault(key, [])
                        pref_dict["fpr"].append(fpr)
                        pref_dict["tpr"].append(tpr)
                        pref_dict["roc_auc"].append(roc_auc)
                        pref_dict["pre"].append(pre)
                        pref_dict["rec"].append(rec)
                        pref_dict["prc_auc"].append(prc_auc)


def _get_delay_calib_res(
    test_rollouts,
    test_scores_all,
    cp_bands_by_alpha,
    alphas,
    method_name,
    res_dict,
):
    delay_deltas = DELAY_DELTAS

    res_dict.setdefault("calib", {})
    res_dict["calib"].setdefault(method_name, {})
    calib_dict = res_dict["calib"][method_name]

    lower_bound = False
    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts]) # (N,)
    test_labels_all = np.asarray([1-r.episode_success for r in test_rollouts])
    test_scores_all = np.array(test_scores_all) # (N, T)
    n_test_samples = len(test_scores_all)

    for eval_time_mode in ["last", "early"]:
        calib_dict.setdefault(eval_time_mode, {})

        for delta in delay_deltas:
            calib_dict[eval_time_mode].setdefault(f"{delta}", {})
            _d = float(delta)

            for alpha in alphas:
                calib_dict[eval_time_mode][f"{delta}"].setdefault(f"{alpha}", {})
                alpha_dict = calib_dict[eval_time_mode][f"{delta}"][f"{alpha}"]
                cp_band = cp_bands_by_alpha[alpha]

                if eval_time_mode == "last":
                    lengths = test_scores_all.shape[1] # scalar, T
                    delayed_scores_all = [get_delay_scores(s, _d) for s in test_scores_all]
                elif eval_time_mode == "early":
                    lengths = test_earliest_stop # (N,)
                    delayed_scores_all = [
                        get_delay_scores(test_scores_all[i], _d, lengths[i])
                        for i in range(len(test_scores_all))
                    ]
                delayed_scores_all = np.asarray(delayed_scores_all)

                if lower_bound: detection_mask = delayed_scores_all <= cp_band # (N, T)
                else:           detection_mask = delayed_scores_all >= cp_band # (N, T)

                if eval_time_mode == "early":
                    # After the earliest stop, no more detection is possible. 
                    for i in range(len(delayed_scores_all)):
                        detection_mask[i, lengths[i]:] = False

                has_detection = np.any(detection_mask, axis=1) # (N,)
                first_detection = np.argmax(detection_mask, axis=1) # (N,)
                detection_times = np.where(has_detection, first_detection, lengths) # (N,)
                relative_detection_times = detection_times / lengths # (N,)

                # Compute detection time and classification metrics
                pos_mask = test_labels_all == 1 # (N,)
                avg_det_time = np.mean(relative_detection_times[pos_mask])
                predicted = has_detection # (N,)
                tp = (predicted & pos_mask).sum()
                fn = (~predicted & pos_mask).sum()
                fp = (predicted & ~pos_mask).sum()
                tn = (~predicted & ~pos_mask).sum()
                
                # Safe division for metrics
                with np.errstate(divide='ignore', invalid='ignore'):
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    acc = (tp + tn) / n_test_samples
                    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                    bal_acc = (tpr + tnr) / 2
                
                alpha_dict.setdefault("detect_method", method_name)
                for m in ["avg_det_time", "tpr", "tnr", "fpr", "fnr", "acc", "bal_acc", "f1"]:
                    alpha_dict.setdefault(m, [])
                alpha_dict["avg_det_time"].append(avg_det_time)
                alpha_dict["tpr"].append(tpr)
                alpha_dict["tnr"].append(tnr)
                alpha_dict["fpr"].append(fpr)
                alpha_dict["fnr"].append(fnr)
                alpha_dict["acc"].append(acc)
                alpha_dict["bal_acc"].append(bal_acc)
                alpha_dict["f1"].append(f1)


def get_new_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    # static metrics
    _get_new_static_metrics(scores_by_split_name, rollouts_by_split_name, method_name, res_dict)

    # calib
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]

    cal_rollouts, cal_scores_all = [], []
    cal_rollouts.extend(rollouts_by_split_name["val_seen"])
    cal_scores_all.extend(scores_by_split_name["val_seen"])
    test_rollouts, test_scores_all = [], []
    test_rollouts.extend(rollouts_by_split_name["val_unseen"])
    test_scores_all.extend(scores_by_split_name["val_unseen"])

    max_length = max(len(s) for s in cal_scores_all + test_scores_all)
    for i, s in enumerate(cal_scores_all):
        cal_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
    for i, s in enumerate(test_scores_all):
        test_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode='edge')

    cp_bands_by_alpha = get_func_conformal_bands(cal_rollouts, cal_scores_all, alphas)
    _get_delay_calib_res(test_rollouts, test_scores_all, cp_bands_by_alpha, alphas, method_name, res_dict)


def _split_new_logs_by_method(
    new_logs: dict,
    fallback_method_name: str,
) -> dict[str, dict]:
    static_logs = new_logs.get("static", {})
    calib_logs = new_logs.get("calib", {})

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

    return {
        "static": static_summary,
        "calib": calib_summary,
    }


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
    target_save_dir = os.path.join(new_save_dir, "dettime_at_fixed_balacc")
    weighted_save_dir = os.path.join(new_save_dir, "weighted_roc_auc")
    prefix_auc_save_dir = os.path.join(new_save_dir, "prefix_auc_vs_delay")
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


if __name__ == "__main__":
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
