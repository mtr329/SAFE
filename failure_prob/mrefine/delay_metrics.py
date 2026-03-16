import argparse
import colorsys
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import numpy as np
import os
import json
import pandas as pd
from omegaconf import OmegaConf
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb, to_hex

from failure_prob.utils.conformal.functional_predictor import (
    RegressionType,
    ModulationType,
    FunctionalPredictor
)


HANDCRAFTED_METHOD_ALLOWLIST = [
    "avg_token_prob",
    "avg_token_entropy",
    "max_token_prob",
    "max_token_entropy",
    
    "total_var",
    "pos_var",
    "rot_var",
    "gripper_var",
    "entropy_linkage0.01",
    "entropy_linkage0.05",
    "stac_mmd",
    "stac_single",
]


def _get_func_conformal(
    cal_rollouts,
    cal_scores_all,
    alphas,
):
    # neg
    lower_bound = False
    cal_scores_used = [s for s, r in zip(cal_scores_all, cal_rollouts) if r.episode_success == 1]
    cal_scores_used = np.array(cal_scores_used)
    if len(cal_scores_used) == 1:
        cal_scores_1 = cal_scores_used
        cal_scores_2 = cal_scores_used
    else:
        np.random.shuffle(cal_scores_used)
        n_cal_1 = int(len(cal_scores_used) * 0.3) # 30% according to Chen's implementation
        cal_scores_1 = cal_scores_used[:n_cal_1]
        cal_scores_2 = cal_scores_used[n_cal_1:]

    cp_bands_by_alpha = {}
    for alpha in alphas:
        predictor = FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)
        cp_band = predictor.get_one_sided_prediction_band(
            cal_scores_1, cal_scores_2, alpha, lower_bound=lower_bound)
        
        cp_bands_by_alpha[alpha] = cp_band
    return cp_bands_by_alpha


def get_delay_scores(s, delta, valid_len=None):
    s = np.asarray(s)
    total_len = len(s)
    if total_len == 0:
        return s.copy()

    if valid_len is None:
        valid_len = total_len
    valid_len = max(0, min(int(valid_len), total_len))

    delayed = s.copy()
    if valid_len == 0:
        return delayed

    min_val = s[:valid_len].min()
    n = int(round(valid_len * delta))
    if n <= 0:
        return delayed
    if n >= valid_len:
        delayed[:valid_len] = min_val
        return delayed

    delayed[:n] = min_val
    delayed[n:valid_len] = s[: valid_len - n]
    return delayed


def _get_delay_static_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    delay_deltas = ["0.0", "0.1", "0.2", "0.3", "0.4"]
    
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
    delay_deltas = ["0.0", "0.1", "0.2", "0.3", "0.4"]

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

    cp_bands_by_alpha = _get_func_conformal(cal_rollouts, cal_scores_all, alphas)
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
    direct_candidates = [
        os.path.join(logs_dir, "eval", "my_logs.json"),
        os.path.join(logs_dir, "eval", "mylogs.json"),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            log_paths.append(os.path.abspath(candidate))

    for root, _, files in os.walk(logs_dir):
        if os.path.basename(root) != "eval":
            continue
        for filename in ("my_logs.json", "mylogs.json"):
            if filename in files:
                log_path = os.path.abspath(os.path.join(root, filename))
                if log_path not in log_paths:
                    log_paths.append(log_path)

    return sorted(log_paths)


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


def _get_delay_curve_figs(delay_summary: dict) -> dict[str, Figure]:
    figs = {}
    for method_name, summary in sorted(delay_summary.items()):
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        base_color = "#1f77b4"

        for ax, mode in zip(axes, ["early", "last"]):
            delta_dict = summary.get("calib", {}).get(mode, {})
            deltas = sorted(delta_dict.keys(), key=float)
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

                color = _adjust_color_lightness(base_color, 0.35 + 0.45 * (i + 1) / max(len(deltas), 1))
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
                        "avg_det_time": det_time,
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


def summar_delay_metrics(
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
    curve_save_dir = os.path.join(delay_save_dir, "balacc_vs_dettime")
    target_save_dir = os.path.join(delay_save_dir, "dettime_at_fixed_balacc")
    os.makedirs(static_save_dir, exist_ok=True)
    os.makedirs(curve_save_dir, exist_ok=True)
    os.makedirs(target_save_dir, exist_ok=True)

    delay_logs_by_method = {}
    log_paths = _collect_delay_log_paths(logs_dir)
    for log_path in log_paths:
        with open(log_path, "r") as f:
            logs = json.load(f)

        if "delay" not in logs:
            continue

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        fallback_method_name = run_meta["method_name"]
        method_logs_by_name = _split_delay_logs_by_method(logs["delay"], fallback_method_name)
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

    curve_figs = _get_delay_curve_figs(delay_summary)
    for fig_name, fig in curve_figs.items():
        fig.savefig(os.path.join(curve_save_dir, f"{fig_name}.png"), dpi=400)

    target_bal_accs = np.round(np.arange(0.5, 0.91, 0.05), 2).tolist()
    target_figs, target_df = _get_delay_target_balacc_figs(delay_summary, target_bal_accs)
    target_df.to_csv(
        os.path.join(target_save_dir, "delay_dettime_at_fixed_balacc.csv"),
        index=False,
    )
    for fig_name, fig in target_figs.items():
        fig.savefig(os.path.join(target_save_dir, f"{fig_name}.png"), dpi=400)

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
        help="Root directory containing evaluation outputs with eval/my_logs.json.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional output directory. Defaults to <logs_dir>/summary.",
    )
    args = parser.parse_args()

    summar_delay_metrics(
        logs_dir=args.logs_dir or _resolve_default_logs_dir(),
        save_dir=args.save_dir,
    )
