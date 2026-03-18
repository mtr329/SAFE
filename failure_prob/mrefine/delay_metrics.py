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

from failure_prob.mrefine.const import DELAY_DELTAS, HANDCRAFTED_METHOD_ALLOWLIST
from failure_prob.mrefine.utils import (
    get_delay_scores,
    get_func_conformal_bands,
)

PARETO_TARGET_BAL_ACCS = np.round(np.arange(0.55, 0.96, 0.05), 2).tolist()
PARETO_BAL_ACC_RANGE = (0.55, 0.95)


def _get_delay_static_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    delay_deltas = DELAY_DELTAS
    
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

                    scores = [get_delay_scores(s[:r.task_min_step], _d,) for s, r in zip(scores_task, rollouts_task)]
                    scores = [s.max() for s in scores]

                    fpr, tpr, thresholds = roc_curve(labels_task, scores)
                    roc_auc = auc(fpr, tpr)
                    pre, rec, thresholds = precision_recall_curve(labels_task, scores)
                    prc_auc = auc(rec, pre)
                
                    for key in ["fpr", "tpr", "roc_auc", "pre", "rec", "prc_auc",]:
                        delta_dict.setdefault(key, [])
                    delta_dict["fpr"].append(fpr)
                    delta_dict["tpr"].append(tpr)
                    delta_dict["roc_auc"].append(roc_auc)
                    delta_dict["pre"].append(pre)
                    delta_dict["rec"].append(rec)
                    delta_dict["prc_auc"].append(prc_auc)


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


def get_delay_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    # static metrics
    _get_delay_static_metrics(scores_by_split_name, rollouts_by_split_name, method_name, res_dict)

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


def _split_delay_logs_by_method(
    delay_logs: dict,
    fallback_method_name: str,
) -> dict[str, dict]:
    static_logs = delay_logs.get("static", {})
    calib_logs = delay_logs.get("calib", {})

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


def _get_method_name_from_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        return os.path.basename(run_dir)

    cfg = OmegaConf.load(cfg_path)
    method_name = cfg.model.name
    if "distance" in cfg.model:
        method_name += f"_{cfg.model.distance}"
    return method_name


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


def _collect_delay_log_paths(logs_dir):
    log_paths = []
    eval_dirs = set()

    direct_eval_dir = os.path.join(logs_dir, "eval")
    if os.path.isdir(direct_eval_dir):
        eval_dirs.add(os.path.abspath(direct_eval_dir))

    for root, _, _ in os.walk(logs_dir):
        if os.path.basename(root) == "eval":
            eval_dirs.add(os.path.abspath(root))

    for eval_dir in sorted(eval_dirs):
        dedicated_path = os.path.join(eval_dir, "delay_logs.json")
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


def _summarize_delay_static(delay_logs: dict) -> dict:
    summary = {}
    static_logs = delay_logs.get("static", {})
    for split_name, split_dict in static_logs.items():
        task_dict = split_dict.get("all", {})
        if not task_dict:
            continue

        summary.setdefault(split_name, {})
        for delta, delta_dict in task_dict.items():
            summary[split_name].setdefault(delta, {})
            for metric_name in ("roc_auc", "prc_auc"):
                if metric_name not in delta_dict:
                    continue
                summary[split_name][delta][metric_name] = float(
                    np.asarray(delta_dict[metric_name]).mean()
                )
    return summary


def _summarize_delay_calib(delay_logs: dict) -> dict:
    summary = {}
    calib_logs = delay_logs.get("calib", {})
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


def _merge_delay_summaries(runs_dict: dict[str, dict]) -> dict:
    static_acc = {}
    calib_acc = {}

    for delay_logs in runs_dict.values():
        static_summary = _summarize_delay_static(delay_logs)
        for split_name, delta_dict in static_summary.items():
            static_acc.setdefault(split_name, {})
            for delta, metrics in delta_dict.items():
                static_acc[split_name].setdefault(delta, {})
                for metric_name, value in metrics.items():
                    static_acc[split_name][delta].setdefault(metric_name, []).append(value)

        calib_summary = _summarize_delay_calib(delay_logs)
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
                metric_name: float(np.mean(values))
                for metric_name, values in metrics.items()
            }
            for delta, metrics in delta_dict.items()
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


def _delay_static_summary_to_df(delay_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in delay_summary.items():
        for split_name, delta_dict in summary.get("static", {}).items():
            for delta, metrics in delta_dict.items():
                row = {
                    "method": method_name,
                    "split": split_name,
                    "delta": float(delta),
                }
                row.update(metrics)
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["method", "split", "delta", "roc_auc", "prc_auc"])

    df = pd.DataFrame(rows)
    return df.sort_values(by=["split", "method", "delta"]).reset_index(drop=True)


def _delay_calib_summary_to_df(delay_summary: dict) -> pd.DataFrame:
    rows = []
    for method_name, summary in delay_summary.items():
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
            columns=["method", "mode", "delta", "alpha", "avg_det_time", "bal_acc"]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by=["method", "mode", "delta", "alpha"]).reset_index(drop=True)


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


def _interp_min_det_time_for_bal_acc(alpha_dict: dict, target_bal_acc: float) -> float:
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
    uniq_det_times = np.full_like(uniq_bal_accs, np.inf)
    for idx, det_time in zip(inverse, det_times):
        uniq_det_times[idx] = min(uniq_det_times[idx], det_time)

    if target_bal_acc < uniq_bal_accs.min() or target_bal_acc > uniq_bal_accs.max():
        return np.nan
    return float(np.interp(target_bal_acc, uniq_bal_accs, uniq_det_times))


def _delay_target_balacc_to_df(
    delay_summary: dict,
    target_bal_accs: list[float],
) -> pd.DataFrame:
    rows = []
    for method_name, summary in delay_summary.items():
        for mode, delta_dict in summary.get("calib", {}).items():
            for target_bal_acc in target_bal_accs:
                for delta, alpha_dict in delta_dict.items():
                    rows.append({
                        "method": method_name,
                        "mode": mode,
                        "target_bal_acc": target_bal_acc,
                        "delta": float(delta),
                        "avg_det_time": _interp_det_time_for_bal_acc(alpha_dict, target_bal_acc),
                    })

    if not rows:
        return pd.DataFrame(columns=["method", "mode", "target_bal_acc", "delta", "avg_det_time"])

    df = pd.DataFrame(rows)
    return df.sort_values(by=["method", "mode", "target_bal_acc", "delta"]).reset_index(drop=True)


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


def _get_ordered_palette(num_colors: int, cmap_name: str = "cividis") -> list[str]:
    if num_colors <= 0:
        return []

    cmap = colormaps[cmap_name]
    positions = np.linspace(0.15, 0.9, num_colors)
    return [to_hex(cmap(pos)) for pos in positions]


def _get_delay_static_figs(delay_summary: dict) -> dict[str, Figure]:
    figs = {}
    method_colors = _get_method_colors(delay_summary.keys())

    all_splits = sorted({
        split_name
        for summary in delay_summary.values()
        for split_name in summary.get("static", {}).keys()
    })
    for split_name in all_splits:
        for metric_name in ("roc_auc", "prc_auc"):
            fig = Figure(figsize=(7, 5))
            ax = fig.subplots()
            has_curve = False

            for method_name, summary in sorted(delay_summary.items()):
                delta_dict = summary.get("static", {}).get(split_name, {})
                if not delta_dict:
                    continue

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
                    alpha=0.8,
                    label=method_name,
                )

            ax.set_xlabel("delay")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} vs delay ({split_name})")
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
            fig.tight_layout()
            figs[f"{split_name}_{metric_name}_vs_delay"] = fig

    return figs


def _aggregate_interp_curve(
    x_runs: list[np.ndarray],
    y_runs: list[np.ndarray],
    grid: np.ndarray,
) -> np.ndarray | None:
    if not x_runs or not y_runs:
        return None

    interp_runs = []
    for x_vals, y_vals in zip(x_runs, y_runs):
        if x_vals is None or y_vals is None:
            continue

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
            continue

        order = np.argsort(x_arr, kind="stable")
        x_arr = x_arr[order]
        y_arr = y_arr[order]

        uniq_x, inverse = np.unique(x_arr, return_inverse=True)
        uniq_y = np.zeros_like(uniq_x)
        counts = np.zeros_like(uniq_x)
        for idx, y_val in zip(inverse, y_arr):
            uniq_y[idx] += y_val
            counts[idx] += 1
        uniq_y = uniq_y / np.maximum(counts, 1)

        if uniq_x.size == 1:
            interp_runs.append(np.full_like(grid, uniq_y[0], dtype=float))
            continue

        interp_runs.append(np.interp(grid, uniq_x, uniq_y))

    if not interp_runs:
        return None

    return np.mean(np.asarray(interp_runs, dtype=float), axis=0)


def _get_min_min_pareto_frontier(
    x_values: list[float],
    y_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if not x_values or not y_values:
        return np.asarray([]), np.asarray([])

    points = sorted(
        zip(x_values, y_values),
        key=lambda x: (float(x[0]), float(x[1])),
    )
    frontier = []
    best_y = np.inf
    for x_val, y_val in points:
        x_val = float(x_val)
        y_val = float(y_val)
        if y_val < best_y:
            frontier.append((x_val, y_val))
            best_y = y_val

    if not frontier:
        return np.asarray([]), np.asarray([])

    frontier = np.asarray(frontier, dtype=float)
    return frontier[:, 0], frontier[:, 1]


def _get_dettime_balacc_pareto_frontier(
    det_times: list[float],
    bal_accs: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if not det_times or not bal_accs:
        return np.asarray([]), np.asarray([])

    points = sorted(
        zip(det_times, bal_accs),
        key=lambda x: (float(x[0]), -float(x[1])),
    )
    frontier = []
    best_bal_acc = -np.inf
    for det_time, bal_acc in points:
        det_time = float(det_time)
        bal_acc = float(bal_acc)
        if bal_acc > best_bal_acc:
            frontier.append((det_time, bal_acc))
            best_bal_acc = bal_acc

    if not frontier:
        return np.asarray([]), np.asarray([])

    frontier = np.asarray(frontier, dtype=float)
    return frontier[:, 0], frontier[:, 1]


def _get_pareto_curve_from_alpha_dict(alpha_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    det_times = []
    bal_accs = []
    for alpha in sorted(alpha_dict.keys(), key=float):
        metrics = alpha_dict[alpha]
        if "avg_det_time" not in metrics or "bal_acc" not in metrics:
            continue
        det_times.append(float(metrics["avg_det_time"]))
        bal_accs.append(float(metrics["bal_acc"]))

    return _get_dettime_balacc_pareto_frontier(det_times, bal_accs)


def _interp_det_time_on_pareto_for_bal_acc(
    pareto_det_times: np.ndarray,
    pareto_bal_accs: np.ndarray,
    target_bal_acc: float,
) -> float:
    if pareto_det_times.size < 2 or pareto_bal_accs.size < 2:
        return np.nan
    if target_bal_acc < pareto_bal_accs.min() or target_bal_acc > pareto_bal_accs.max():
        return np.nan
    return float(np.interp(target_bal_acc, pareto_bal_accs, pareto_det_times))


def _compute_autc_from_pareto(
    pareto_det_times: np.ndarray,
    pareto_bal_accs: np.ndarray,
) -> float:
    if pareto_det_times.size < 2 or pareto_bal_accs.size < 2:
        return np.nan
    return float(np.trapezoid(pareto_bal_accs, pareto_det_times))


def _compute_pareto_balacc_coverage(
    pareto_bal_accs: np.ndarray,
    bal_acc_min: float,
    bal_acc_max: float,
) -> float:
    if pareto_bal_accs.size == 0 or bal_acc_max <= bal_acc_min:
        return 0.0

    covered_min = max(float(np.min(pareto_bal_accs)), bal_acc_min)
    covered_max = min(float(np.max(pareto_bal_accs)), bal_acc_max)
    covered_span = max(0.0, covered_max - covered_min)
    total_span = bal_acc_max - bal_acc_min
    return float(covered_span / total_span)


def _compute_autc_from_pareto_over_balacc_range(
    pareto_det_times: np.ndarray,
    pareto_bal_accs: np.ndarray,
    bal_acc_min: float,
    bal_acc_max: float,
) -> float:
    if pareto_det_times.size < 2 or pareto_bal_accs.size < 2:
        return np.nan
    if _compute_pareto_balacc_coverage(pareto_bal_accs, bal_acc_min, bal_acc_max) < 1.0:
        return np.nan

    clipped_bal_accs = [bal_acc_min]
    clipped_det_times = [float(np.interp(bal_acc_min, pareto_bal_accs, pareto_det_times))]

    for det_time, bal_acc in zip(pareto_det_times, pareto_bal_accs):
        if bal_acc_min < bal_acc < bal_acc_max:
            clipped_bal_accs.append(float(bal_acc))
            clipped_det_times.append(float(det_time))

    clipped_bal_accs.append(bal_acc_max)
    clipped_det_times.append(float(np.interp(bal_acc_max, pareto_bal_accs, pareto_det_times)))

    return float(np.trapezoid(np.asarray(clipped_bal_accs), np.asarray(clipped_det_times)))


def _compute_max_bal_acc_from_pareto(
    pareto_bal_accs: np.ndarray,
) -> float:
    if pareto_bal_accs.size == 0:
        return np.nan
    return float(np.max(pareto_bal_accs))


def _get_delay_roc_prc_figs(
    delay_logs_by_method: dict[str, dict[str, dict]],
) -> dict[str, Figure]:
    figs = {}
    all_splits = sorted({
        split_name
        for runs_dict in delay_logs_by_method.values()
        for delay_logs in runs_dict.values()
        for split_name in delay_logs.get("static", {}).keys()
    })

    curve_specs = (
        ("roc", "fpr", "tpr", "false_positive_rate", "true_positive_rate", "ROC"),
        ("prc", "rec", "pre", "recall", "precision", "PRC"),
    )
    grid = np.linspace(0.0, 1.0, 201)

    for method_name, runs_dict in sorted(delay_logs_by_method.items()):
        for split_name in all_splits:
            available_deltas = sorted({
                delta
                for delay_logs in runs_dict.values()
                for delta in delay_logs.get("static", {}).get(split_name, {}).get("all", {}).keys()
            }, key=float)
            if not available_deltas:
                continue

            fig = Figure(figsize=(12, 5))
            axes = fig.subplots(1, 2)
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])
            palette = _get_ordered_palette(len(available_deltas), cmap_name="cividis")

            for ax, (_, x_key, y_key, x_label, y_label, title) in zip(axes, curve_specs):
                has_curve = False
                for i, delta in enumerate(available_deltas):
                    x_runs = []
                    y_runs = []
                    auc_values = []

                    for delay_logs in runs_dict.values():
                        delta_logs = delay_logs.get("static", {}).get(split_name, {}).get("all", {}).get(delta, {})
                        x_series = delta_logs.get(x_key, [])
                        y_series = delta_logs.get(y_key, [])
                        auc_key = "roc_auc" if y_key == "tpr" else "prc_auc"
                        auc_series = delta_logs.get(auc_key, [])

                        for x_vals, y_vals in zip(x_series, y_series):
                            x_runs.append(np.asarray(x_vals, dtype=float))
                            y_runs.append(np.asarray(y_vals, dtype=float))
                        for auc_val in auc_series:
                            auc_values.append(float(auc_val))

                    mean_curve = _aggregate_interp_curve(x_runs, y_runs, grid)
                    if mean_curve is None:
                        continue

                    has_curve = True
                    label = f"delay={delta}"
                    if auc_values:
                        label += f" (auc={np.mean(auc_values):.3f})"
                    ax.plot(
                        grid,
                        mean_curve,
                        linewidth=2.0,
                        color=palette[i % len(palette)],
                        alpha=0.95,
                        label=label,
                    )

                if title == "ROC":
                    ax.plot(
                        [0.0, 1.0],
                        [0.0, 1.0],
                        linestyle="--",
                        linewidth=1.0,
                        color="#666666",
                        alpha=0.7,
                    )

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
                if has_curve:
                    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

            fig.suptitle(f"{method_name}: ROC/PRC by delay ({split_name})")
            fig.tight_layout()
            figs[f"{method_name}_{split_name}_roc_prc_by_delay"] = fig

    return figs


def _get_delay_curve_figs(delay_summary: dict) -> dict[str, Figure]:
    figs = {}
    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            palette = _get_ordered_palette(len(deltas), cmap_name="cividis")
            for i, delta in enumerate(deltas):
                alpha_dict = delta_dict[delta]
                alphas = sorted(alpha_dict.keys(), key=float)
                avg_det_times = []
                bal_accs = []
                for alpha in alphas:
                    metrics = alpha_dict[alpha]
                    if "avg_det_time" not in metrics or "bal_acc" not in metrics:
                        continue
                    avg_det_times.append(float(metrics["avg_det_time"]))
                    bal_accs.append(float(metrics["bal_acc"]))

                if not avg_det_times:
                    continue

                color = palette[i % len(palette)]
                ax.plot(
                    avg_det_times,
                    bal_accs,
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=color,
                    alpha=0.9,
                    label=f"delay={delta}",
                )

            ax.set_xlabel("avg_det_time")
            ax.set_ylabel("bal_acc")
            ax.set_title(mode)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if deltas:
                ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

        fig.suptitle(f"{method_name}: bal_acc vs avg_det_time by delay")
        fig.tight_layout()
        figs[f"{method_name}_balacc_vs_dettime_by_delay"] = fig

    return figs


def _get_delay_alpha_metric_figs(delay_summary: dict) -> dict[str, Figure]:
    figs = {}
    metric_specs = (
        ("avg_det_time", "avg_det_time"),
        ("bal_acc", "bal_acc"),
        ("fpr", "fpr"),
        ("fnr", "fnr"),
    )

    for method_name, summary in sorted(delay_summary.items()):
        for metric_name, ylabel in metric_specs:
            fig = Figure(figsize=(12, 5))
            axes = fig.subplots(1, 2)
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])

            for ax, mode in zip(axes, ["early", "last"]):
                delta_dict = summary.get("calib", {}).get(mode, {})
                deltas = sorted(delta_dict.keys(), key=float)
                palette = _get_ordered_palette(len(deltas), cmap_name="cividis")
                has_curve = False

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
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
                if has_curve:
                    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

            fig.suptitle(f"{method_name}: {metric_name} vs alpha by delay")
            fig.tight_layout()
            figs[f"{method_name}_{metric_name}_vs_alpha_by_delay"] = fig

    return figs


def _get_alpha_metric_name_from_fig_name(fig_name: str) -> str | None:
    suffix = "_vs_alpha_by_delay"
    if not fig_name.endswith(suffix):
        return None

    metric_part = fig_name[: -len(suffix)]
    for metric_name in ("avg_det_time", "bal_acc", "fpr", "fnr"):
        if metric_part.endswith(f"_{metric_name}"):
            return metric_name
    return None


def _get_delay_target_balacc_figs(
    delay_summary: dict,
    target_bal_accs: list[float],
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []

    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = [float(delta) for delta in deltas]
            has_curve = False
            palette = _get_ordered_palette(len(target_bal_accs), cmap_name="plasma")

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
                        "avg_det_time": det_time,
                    })

                if np.all(np.isnan(y)):
                    continue

                has_curve = True
                color = palette[i % len(palette)]
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
            ax.set_ylabel("avg_det_time")
            ax.set_title(mode)
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

        fig.suptitle(f"{method_name}: avg_det_time at fixed bal_acc")
        fig.tight_layout()
        figs[f"{method_name}_dettime_at_fixed_balacc"] = fig

    target_df = pd.DataFrame(rows)
    if not target_df.empty:
        target_df = target_df.sort_values(by=["method", "mode", "target_bal_acc", "delta"]).reset_index(drop=True)
    return figs, target_df


def _get_delay_min_target_balacc_figs(
    delay_summary: dict,
    target_bal_accs: list[float],
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []

    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = [float(delta) for delta in deltas]
            has_curve = False
            palette = _get_ordered_palette(len(target_bal_accs), cmap_name="plasma")

            for i, target_bal_acc in enumerate(target_bal_accs):
                y = []
                for delta in deltas:
                    det_time = _interp_min_det_time_for_bal_acc(delta_dict[delta], target_bal_acc)
                    y.append(det_time)
                    rows.append({
                        "method": method_name,
                        "mode": mode,
                        "target_bal_acc": target_bal_acc,
                        "delta": float(delta),
                        "min_avg_det_time": det_time,
                    })

                if np.all(np.isnan(y)):
                    continue

                has_curve = True
                color = palette[i % len(palette)]
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
            ax.set_ylabel("min_avg_det_time")
            ax.set_title(mode)
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

        fig.suptitle(f"{method_name}: min avg_det_time at fixed bal_acc")
        fig.tight_layout()
        figs[f"{method_name}_min_dettime_at_fixed_balacc"] = fig

    target_df = pd.DataFrame(rows)
    if not target_df.empty:
        target_df = target_df.sort_values(
            by=["method", "mode", "target_bal_acc", "delta"]
        ).reset_index(drop=True)
    else:
        target_df = pd.DataFrame(
            columns=["method", "mode", "target_bal_acc", "delta", "min_avg_det_time"]
        )
    return figs, target_df


def _get_delay_min_target_balacc_pareto_figs(
    delay_summary: dict,
    target_bal_accs: list[float],
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []

    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = [float(delta) for delta in deltas]
            has_curve = False
            palette = _get_ordered_palette(len(target_bal_accs), cmap_name="plasma")

            for i, target_bal_acc in enumerate(target_bal_accs):
                y = []
                for delta in deltas:
                    det_time = _interp_min_det_time_for_bal_acc(delta_dict[delta], target_bal_acc)
                    y.append(det_time)
                    rows.append({
                        "method": method_name,
                        "mode": mode,
                        "target_bal_acc": target_bal_acc,
                        "delta": float(delta),
                        "min_avg_det_time": det_time,
                    })

                if np.all(np.isnan(y)):
                    continue

                has_curve = True
                color = palette[i % len(palette)]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=color,
                    alpha=0.35,
                    label=f"bal_acc={target_bal_acc:.2f}",
                )

                valid_x = [xv for xv, yv in zip(x, y) if not np.isnan(yv)]
                valid_y = [yv for yv in y if not np.isnan(yv)]
                pareto_x, pareto_y = _get_min_min_pareto_frontier(valid_x, valid_y)
                if pareto_x.size > 0:
                    ax.plot(
                        pareto_x,
                        pareto_y,
                        linestyle="--",
                        linewidth=2.5,
                        marker="o",
                        markersize=4,
                        color=color,
                        alpha=0.95,
                        markeredgewidth=0.0,
                    )

            ax.set_xlabel("delay")
            ax.set_ylabel("min_avg_det_time")
            ax.set_title(mode)
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

        fig.suptitle(f"{method_name}: min avg_det_time at fixed bal_acc (pareto)")
        fig.tight_layout()
        figs[f"{method_name}_min_dettime_at_fixed_balacc_pareto"] = fig

    target_df = pd.DataFrame(rows)
    if not target_df.empty:
        target_df = target_df.sort_values(
            by=["method", "mode", "target_bal_acc", "delta"]
        ).reset_index(drop=True)
    else:
        target_df = pd.DataFrame(
            columns=["method", "mode", "target_bal_acc", "delta", "min_avg_det_time"]
        )
    return figs, target_df


def _get_delay_pareto_front_figs(
    delay_summary: dict,
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []

    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            palette = _get_ordered_palette(len(deltas), cmap_name="cividis")
            has_curve = False

            for i, delta in enumerate(deltas):
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(delta_dict[delta])
                if pareto_det_times.size == 0:
                    continue

                has_curve = True
                color = palette[i % len(palette)]
                ax.plot(
                    pareto_det_times,
                    pareto_bal_accs,
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=color,
                    alpha=0.95,
                    label=f"delay={delta}",
                )

                for rank, (det_time, bal_acc) in enumerate(zip(pareto_det_times, pareto_bal_accs)):
                    rows.append({
                        "method": method_name,
                        "mode": mode,
                        "delta": float(delta),
                        "pareto_rank": rank,
                        "avg_det_time": float(det_time),
                        "bal_acc": float(bal_acc),
                    })

            ax.set_xlabel("avg_det_time")
            ax.set_ylabel("bal_acc")
            ax.set_title(mode)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

        fig.suptitle(f"{method_name}: pareto front by delay")
        fig.tight_layout()
        figs[f"{method_name}_pareto_front_by_delay"] = fig

    front_df = pd.DataFrame(rows)
    if not front_df.empty:
        front_df = front_df.sort_values(
            by=["method", "mode", "delta", "pareto_rank"]
        ).reset_index(drop=True)
    else:
        front_df = pd.DataFrame(
            columns=["method", "mode", "delta", "pareto_rank", "avg_det_time", "bal_acc"]
        )
    return figs, front_df


def _get_delay_t_at_balacc_from_pareto_figs(
    delay_summary: dict,
    target_bal_accs: list[float],
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []

    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = [float(delta) for delta in deltas]
            palette = _get_ordered_palette(len(target_bal_accs), cmap_name="plasma")
            has_curve = False

            pareto_cache = {
                delta: _get_pareto_curve_from_alpha_dict(alpha_dict)
                for delta, alpha_dict in delta_dict.items()
            }

            for i, target_bal_acc in enumerate(target_bal_accs):
                y = []
                for delta in deltas:
                    pareto_det_times, pareto_bal_accs = pareto_cache[delta]
                    det_time = _interp_det_time_on_pareto_for_bal_acc(
                        pareto_det_times,
                        pareto_bal_accs,
                        target_bal_acc,
                    )
                    y.append(det_time)
                    rows.append({
                        "method": method_name,
                        "mode": mode,
                        "target_bal_acc": target_bal_acc,
                        "delta": float(delta),
                        "pareto_avg_det_time": det_time,
                    })

                if np.all(np.isnan(y)):
                    continue

                has_curve = True
                color = palette[i % len(palette)]
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
            ax.set_ylabel("pareto_avg_det_time")
            ax.set_title(mode)
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if has_curve:
                ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)

        fig.suptitle(f"{method_name}: T@BalAcc from pareto front")
        fig.tight_layout()
        figs[f"{method_name}_t_at_balacc_from_pareto"] = fig

    target_df = pd.DataFrame(rows)
    if not target_df.empty:
        target_df = target_df.sort_values(
            by=["method", "mode", "target_bal_acc", "delta"]
        ).reset_index(drop=True)
    else:
        target_df = pd.DataFrame(
            columns=["method", "mode", "target_bal_acc", "delta", "pareto_avg_det_time"]
        )
    return figs, target_df


def _get_delay_autc_from_pareto_figs(
    delay_summary: dict,
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []
    method_colors = _get_method_colors(delay_summary.keys())

    for mode in ["early", "last"]:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        has_curve = False

        for method_name, summary in sorted(delay_summary.items()):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = []
            y = []

            for delta in deltas:
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(delta_dict[delta])
                autc = _compute_autc_from_pareto(pareto_det_times, pareto_bal_accs)
                x.append(float(delta))
                y.append(autc)
                rows.append({
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                    "pareto_autc": autc,
                })

            valid_pairs = [(dx, autc) for dx, autc in zip(x, y) if not np.isnan(autc)]
            if not valid_pairs:
                continue

            has_curve = True
            ax.plot(
                [pair[0] for pair in valid_pairs],
                [pair[1] for pair in valid_pairs],
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=method_colors[method_name],
                alpha=0.9,
                label=method_name,
            )

        ax.set_xlabel("delay")
        ax.set_ylabel("pareto_autc")
        ax.set_title(f"AUTC from pareto front ({mode})")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        fig.tight_layout()
        figs[f"autc_from_pareto_{mode}"] = fig

    autc_df = pd.DataFrame(rows)
    if not autc_df.empty:
        autc_df = autc_df.sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)
    else:
        autc_df = pd.DataFrame(columns=["method", "mode", "delta", "pareto_autc"])
    return figs, autc_df


def _get_delay_fixed_range_autc_from_pareto_figs(
    delay_summary: dict,
    bal_acc_min: float,
    bal_acc_max: float,
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []
    method_colors = _get_method_colors(delay_summary.keys())

    for mode in ["early", "last"]:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        has_curve = False

        for method_name, summary in sorted(delay_summary.items()):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = []
            y = []

            for delta in deltas:
                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(delta_dict[delta])
                fixed_range_autc = _compute_autc_from_pareto_over_balacc_range(
                    pareto_det_times,
                    pareto_bal_accs,
                    bal_acc_min,
                    bal_acc_max,
                )
                x.append(float(delta))
                y.append(fixed_range_autc)
                rows.append({
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                    "bal_acc_min": bal_acc_min,
                    "bal_acc_max": bal_acc_max,
                    "pareto_fixed_range_autc": fixed_range_autc,
                })

            valid_pairs = [(dx, autc) for dx, autc in zip(x, y) if not np.isnan(autc)]
            if not valid_pairs:
                continue

            has_curve = True
            ax.plot(
                [pair[0] for pair in valid_pairs],
                [pair[1] for pair in valid_pairs],
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=method_colors[method_name],
                alpha=0.9,
                label=method_name,
            )

        ax.set_xlabel("delay")
        ax.set_ylabel("pareto_fixed_range_autc")
        ax.set_title(f"Fixed-range AUTC from pareto ({mode})")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        fig.tight_layout()
        figs[f"fixed_range_autc_from_pareto_{mode}"] = fig

    fixed_range_autc_df = pd.DataFrame(rows)
    if not fixed_range_autc_df.empty:
        fixed_range_autc_df = fixed_range_autc_df.sort_values(
            by=["method", "mode", "delta"]
        ).reset_index(drop=True)
    else:
        fixed_range_autc_df = pd.DataFrame(
            columns=[
                "method", "mode", "delta", "bal_acc_min", "bal_acc_max",
                "pareto_fixed_range_autc",
            ]
        )
    return figs, fixed_range_autc_df


def _get_delay_pareto_coverage_figs(
    delay_summary: dict,
    bal_acc_min: float,
    bal_acc_max: float,
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []
    method_colors = _get_method_colors(delay_summary.keys())

    for mode in ["early", "last"]:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        has_curve = False

        for method_name, summary in sorted(delay_summary.items()):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = []
            y = []

            for delta in deltas:
                _, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(delta_dict[delta])
                coverage = _compute_pareto_balacc_coverage(
                    pareto_bal_accs,
                    bal_acc_min,
                    bal_acc_max,
                )
                x.append(float(delta))
                y.append(coverage)
                rows.append({
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                    "bal_acc_min": bal_acc_min,
                    "bal_acc_max": bal_acc_max,
                    "pareto_coverage": coverage,
                })

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
                alpha=0.9,
                label=method_name,
            )

        ax.set_xlabel("delay")
        ax.set_ylabel("pareto_coverage")
        ax.set_title(f"Pareto coverage ({mode})")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        fig.tight_layout()
        figs[f"pareto_coverage_{mode}"] = fig

    coverage_df = pd.DataFrame(rows)
    if not coverage_df.empty:
        coverage_df = coverage_df.sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)
    else:
        coverage_df = pd.DataFrame(
            columns=["method", "mode", "delta", "bal_acc_min", "bal_acc_max", "pareto_coverage"]
        )
    return figs, coverage_df


def _get_delay_max_bal_acc_from_pareto_figs(
    delay_summary: dict,
) -> tuple[dict[str, Figure], pd.DataFrame]:
    figs = {}
    rows = []
    method_colors = _get_method_colors(delay_summary.keys())

    for mode in ["early", "last"]:
        fig = Figure(figsize=(7, 5))
        ax = fig.subplots()
        has_curve = False

        for method_name, summary in sorted(delay_summary.items()):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
            x = []
            y = []

            for delta in deltas:
                _, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(delta_dict[delta])
                max_bal_acc = _compute_max_bal_acc_from_pareto(pareto_bal_accs)
                x.append(float(delta))
                y.append(max_bal_acc)
                rows.append({
                    "method": method_name,
                    "mode": mode,
                    "delta": float(delta),
                    "pareto_max_bal_acc": max_bal_acc,
                })

            valid_pairs = [(dx, max_bal_acc) for dx, max_bal_acc in zip(x, y) if not np.isnan(max_bal_acc)]
            if not valid_pairs:
                continue

            has_curve = True
            ax.plot(
                [pair[0] for pair in valid_pairs],
                [pair[1] for pair in valid_pairs],
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=method_colors[method_name],
                alpha=0.9,
                label=method_name,
            )

        ax.set_xlabel("delay")
        ax.set_ylabel("pareto_max_bal_acc")
        ax.set_title(f"Max BalAcc from pareto ({mode})")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        fig.tight_layout()
        figs[f"max_bal_acc_from_pareto_{mode}"] = fig

    max_bal_acc_df = pd.DataFrame(rows)
    if not max_bal_acc_df.empty:
        max_bal_acc_df = max_bal_acc_df.sort_values(by=["method", "mode", "delta"]).reset_index(drop=True)
    else:
        max_bal_acc_df = pd.DataFrame(
            columns=["method", "mode", "delta", "pareto_max_bal_acc"]
        )
    return figs, max_bal_acc_df


def summary_delay_metrics(
    logs_dir="logs",
    save_dir=None,
):
    if not save_dir:
        save_dir = os.path.join(logs_dir, "summary")
    logs_dir = os.path.abspath(logs_dir)
    save_dir = os.path.abspath(save_dir)
    delay_save_dir = os.path.join(save_dir, "delay")
    os.makedirs(delay_save_dir, exist_ok=True)
    static_save_dir = os.path.join(delay_save_dir, "static_vs_delay")
    roc_prc_save_dir = os.path.join(delay_save_dir, "roc_prc_by_delay")
    curve_save_dir = os.path.join(delay_save_dir, "balacc_vs_dettime")
    alpha_curve_save_dir = os.path.join(delay_save_dir, "metrics_vs_alpha")
    alpha_metric_save_dirs = {
        metric_name: os.path.join(alpha_curve_save_dir, metric_name)
        for metric_name in ("avg_det_time", "bal_acc", "fpr", "fnr")
    }
    target_save_dir = os.path.join(delay_save_dir, "dettime_at_fixed_balacc")
    min_target_save_dir = os.path.join(delay_save_dir, "min_dettime_at_fixed_balacc")
    min_target_pareto_save_dir = os.path.join(delay_save_dir, "min_dettime_at_fixed_balacc_pareto")
    pareto_front_save_dir = os.path.join(delay_save_dir, "pareto_front_by_delay")
    pareto_t_at_balacc_save_dir = os.path.join(delay_save_dir, "t_at_balacc_from_pareto")
    pareto_autc_save_dir = os.path.join(delay_save_dir, "autc_from_pareto")
    pareto_fixed_range_autc_save_dir = os.path.join(delay_save_dir, "fixed_range_autc_from_pareto")
    pareto_coverage_save_dir = os.path.join(delay_save_dir, "pareto_coverage")
    pareto_max_bal_acc_save_dir = os.path.join(delay_save_dir, "max_bal_acc_from_pareto")
    os.makedirs(static_save_dir, exist_ok=True)
    os.makedirs(roc_prc_save_dir, exist_ok=True)
    os.makedirs(curve_save_dir, exist_ok=True)
    os.makedirs(alpha_curve_save_dir, exist_ok=True)
    for metric_save_dir in alpha_metric_save_dirs.values():
        os.makedirs(metric_save_dir, exist_ok=True)
    os.makedirs(target_save_dir, exist_ok=True)
    os.makedirs(min_target_save_dir, exist_ok=True)
    os.makedirs(min_target_pareto_save_dir, exist_ok=True)
    os.makedirs(pareto_front_save_dir, exist_ok=True)
    os.makedirs(pareto_t_at_balacc_save_dir, exist_ok=True)
    os.makedirs(pareto_autc_save_dir, exist_ok=True)
    os.makedirs(pareto_fixed_range_autc_save_dir, exist_ok=True)
    os.makedirs(pareto_coverage_save_dir, exist_ok=True)
    os.makedirs(pareto_max_bal_acc_save_dir, exist_ok=True)

    delay_logs_by_method = {}
    log_paths = _collect_delay_log_paths(logs_dir)
    for log_path in log_paths:
        with open(log_path, "r") as f:
            logs = json.load(f)

        if os.path.basename(log_path) == "delay_logs.json":
            delay_logs = logs
        else:
            if "delay" not in logs:
                continue
            delay_logs = logs["delay"]

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        fallback_method_name = run_meta["method_name"]
        method_logs_by_name = _split_delay_logs_by_method(delay_logs, fallback_method_name)
        for method_name, method_logs in method_logs_by_name.items():
            if (
                run_meta["is_handcrafted"]
                and HANDCRAFTED_METHOD_ALLOWLIST is not None
                and method_name not in HANDCRAFTED_METHOD_ALLOWLIST
            ):
                continue
            delay_logs_by_method.setdefault(method_name, {})
            delay_logs_by_method[method_name][run_name] = method_logs

    delay_summary = {
        method_name: _merge_delay_summaries(runs_dict)
        for method_name, runs_dict in delay_logs_by_method.items()
    }

    with open(os.path.join(delay_save_dir, "delay_summary.json"), "w") as f:
        json.dump(delay_summary, f, indent=2)

    _delay_static_summary_to_df(delay_summary).to_csv(
        os.path.join(static_save_dir, "delay_static_summary.csv"),
        index=False,
    )
    _delay_calib_summary_to_df(delay_summary).to_csv(
        os.path.join(curve_save_dir, "delay_calib_summary.csv"),
        index=False,
    )

    static_figs = _get_delay_static_figs(delay_summary)
    for fig_name, fig in static_figs.items():
        fig.savefig(os.path.join(static_save_dir, f"{fig_name}.png"), dpi=400)

    roc_prc_figs = _get_delay_roc_prc_figs(delay_logs_by_method)
    for fig_name, fig in roc_prc_figs.items():
        fig.savefig(os.path.join(roc_prc_save_dir, f"{fig_name}.png"), dpi=400)

    curve_figs = _get_delay_curve_figs(delay_summary)
    for fig_name, fig in curve_figs.items():
        fig.savefig(os.path.join(curve_save_dir, f"{fig_name}.png"), dpi=400)

    alpha_metric_figs = _get_delay_alpha_metric_figs(delay_summary)
    for fig_name, fig in alpha_metric_figs.items():
        metric_name = _get_alpha_metric_name_from_fig_name(fig_name)
        target_dir = alpha_metric_save_dirs.get(metric_name, alpha_curve_save_dir)
        fig.savefig(os.path.join(target_dir, f"{fig_name}.png"), dpi=400)

    target_bal_accs = np.round(np.arange(0.5, 0.91, 0.05), 2).tolist()
    target_figs, target_df = _get_delay_target_balacc_figs(delay_summary, target_bal_accs)
    target_df.to_csv(
        os.path.join(target_save_dir, "delay_dettime_at_fixed_balacc.csv"),
        index=False,
    )
    for fig_name, fig in target_figs.items():
        fig.savefig(os.path.join(target_save_dir, f"{fig_name}.png"), dpi=400)

    min_target_bal_accs = np.round(np.arange(0.55, 0.96, 0.05), 2).tolist()
    min_target_figs, min_target_df = _get_delay_min_target_balacc_figs(
        delay_summary,
        min_target_bal_accs,
    )
    min_target_df.to_csv(
        os.path.join(min_target_save_dir, "delay_min_dettime_at_fixed_balacc.csv"),
        index=False,
    )
    for fig_name, fig in min_target_figs.items():
        fig.savefig(os.path.join(min_target_save_dir, f"{fig_name}.png"), dpi=400)

    min_target_pareto_figs, min_target_pareto_df = _get_delay_min_target_balacc_pareto_figs(
        delay_summary,
        min_target_bal_accs,
    )
    min_target_pareto_df.to_csv(
        os.path.join(min_target_pareto_save_dir, "delay_min_dettime_at_fixed_balacc_pareto.csv"),
        index=False,
    )
    for fig_name, fig in min_target_pareto_figs.items():
        fig.savefig(os.path.join(min_target_pareto_save_dir, f"{fig_name}.png"), dpi=400)

    pareto_front_figs, pareto_front_df = _get_delay_pareto_front_figs(delay_summary)
    pareto_front_df.to_csv(
        os.path.join(pareto_front_save_dir, "delay_pareto_front_by_delay.csv"),
        index=False,
    )
    for fig_name, fig in pareto_front_figs.items():
        fig.savefig(os.path.join(pareto_front_save_dir, f"{fig_name}.png"), dpi=400)

    pareto_target_bal_accs = PARETO_TARGET_BAL_ACCS
    pareto_t_figs, pareto_t_df = _get_delay_t_at_balacc_from_pareto_figs(
        delay_summary,
        pareto_target_bal_accs,
    )
    pareto_t_df.to_csv(
        os.path.join(pareto_t_at_balacc_save_dir, "delay_t_at_balacc_from_pareto.csv"),
        index=False,
    )
    for fig_name, fig in pareto_t_figs.items():
        fig.savefig(os.path.join(pareto_t_at_balacc_save_dir, f"{fig_name}.png"), dpi=400)

    pareto_autc_figs, pareto_autc_df = _get_delay_autc_from_pareto_figs(delay_summary)
    pareto_autc_df.to_csv(
        os.path.join(pareto_autc_save_dir, "delay_autc_from_pareto.csv"),
        index=False,
    )
    for fig_name, fig in pareto_autc_figs.items():
        fig.savefig(os.path.join(pareto_autc_save_dir, f"{fig_name}.png"), dpi=400)

    pareto_fixed_range_autc_figs, pareto_fixed_range_autc_df = _get_delay_fixed_range_autc_from_pareto_figs(
        delay_summary,
        PARETO_BAL_ACC_RANGE[0],
        PARETO_BAL_ACC_RANGE[1],
    )
    pareto_fixed_range_autc_df.to_csv(
        os.path.join(
            pareto_fixed_range_autc_save_dir,
            "delay_fixed_range_autc_from_pareto.csv",
        ),
        index=False,
    )
    for fig_name, fig in pareto_fixed_range_autc_figs.items():
        fig.savefig(os.path.join(pareto_fixed_range_autc_save_dir, f"{fig_name}.png"), dpi=400)

    pareto_coverage_figs, pareto_coverage_df = _get_delay_pareto_coverage_figs(
        delay_summary,
        PARETO_BAL_ACC_RANGE[0],
        PARETO_BAL_ACC_RANGE[1],
    )
    pareto_coverage_df.to_csv(
        os.path.join(pareto_coverage_save_dir, "delay_pareto_coverage.csv"),
        index=False,
    )
    for fig_name, fig in pareto_coverage_figs.items():
        fig.savefig(os.path.join(pareto_coverage_save_dir, f"{fig_name}.png"), dpi=400)

    pareto_max_bal_acc_figs, pareto_max_bal_acc_df = _get_delay_max_bal_acc_from_pareto_figs(
        delay_summary,
    )
    pareto_max_bal_acc_df.to_csv(
        os.path.join(pareto_max_bal_acc_save_dir, "delay_max_bal_acc_from_pareto.csv"),
        index=False,
    )
    for fig_name, fig in pareto_max_bal_acc_figs.items():
        fig.savefig(os.path.join(pareto_max_bal_acc_save_dir, f"{fig_name}.png"), dpi=400)

    return delay_summary


def _resolve_default_logs_dir() -> str:
    env_logs_dir = os.environ.get("DELAY_METRICS_LOGS_DIR")
    if env_logs_dir:
        return env_logs_dir

    for candidate in ("log_ckpt", "logs"):
        if os.path.isdir(candidate):
            return candidate

    return "log_ckpt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize delay metrics from evaluation logs.",
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default=None,
        help="Root directory containing evaluation outputs with eval/delay_logs.json or eval/my_logs.json.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional output directory. Defaults to <logs_dir>/summary.",
    )
    args = parser.parse_args()

    summary_delay_metrics(
        logs_dir=args.logs_dir or _resolve_default_logs_dir(),
        save_dir=args.save_dir,
    )
