import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import numpy as np
import os
import json
from omegaconf import OmegaConf
from matplotlib.figure import Figure

from failure_prob.utils.conformal.functional_predictor import (
    RegressionType,
    ModulationType,
    FunctionalPredictor
)


def _get_ori_static_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    res_dict,
):
    res_dict.setdefault("static", {})
    static_dict = res_dict["static"]
    
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

                scores = [s[:r.task_min_step].max() for s, r in zip(scores_task, rollouts_task)]
                fpr, tpr, thresholds = roc_curve(labels_task, scores)
                roc_auc = auc(fpr, tpr)
                pre, rec, thresholds = precision_recall_curve(labels_task, scores)
                prc_auc = auc(rec, pre)

                task_dict = static_dict[split][f"{task_id}"]
                for key in ["fpr", "tpr", "roc_auc", "pre", "rec", "prc_auc",]:
                    task_dict.setdefault(key, [])
                task_dict["fpr"].append(fpr)
                task_dict["tpr"].append(tpr)
                task_dict["roc_auc"].append(roc_auc)
                task_dict["pre"].append(pre)
                task_dict["rec"].append(rec)
                task_dict["prc_auc"].append(prc_auc)


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


def _get_calib_res(
    test_rollouts,
    test_scores_all,
    cp_bands_by_alpha,
    alphas,
    method_name,
    res_dict,
):
    res_dict.setdefault("calib", {})
    calib_dict = res_dict["calib"]

    lower_bound = False
    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts]) # (N,)
    test_labels_all = np.asarray([1-r.episode_success for r in test_rollouts])
    test_scores_all = np.array(test_scores_all) # (N, T)
    n_test_samples = len(test_scores_all)

    for eval_time in ["last", "early"]:
        calib_dict.setdefault(eval_time, {})

        for alpha in alphas:
            calib_dict[eval_time].setdefault(f"{alpha}", {})

            cp_band = cp_bands_by_alpha[alpha]
            if lower_bound: detection_mask = test_scores_all <= cp_band # (N, T)
            else:           detection_mask = test_scores_all >= cp_band # (N, T)

            if eval_time == "last":
                lengths = test_scores_all.shape[1] # scalar, T
            elif eval_time == "early":
                lengths = test_earliest_stop # (N,)
                # After the earliest stop, no more detection is possible. 
                for i in range(len(test_scores_all)):
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
            
            alpha_dict = calib_dict[eval_time][f"{alpha}"]
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
            

def get_ori_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    # static metrics
    _get_ori_static_metrics(scores_by_split_name, rollouts_by_split_name, res_dict)

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
    _get_calib_res(test_rollouts, test_scores_all, cp_bands_by_alpha, alphas, method_name, res_dict)


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


def _get_ori_figs(
    ori_logs_by_method,
):
    explicit_method_colors = {
        "lstm": "#c62828",
        "indep": "#ef9a9a",
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
                linewidth=1.5,
                markersize=4,
                label=method_name,
                color=method_colors.get(method_name),
                alpha=0.8,
            )

        ax.set_xlabel("avg_det_time")
        ax.set_ylabel("bal_acc")
        ax.set_title(f"avg_det_time vs bal_acc ({eval_time})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if has_curve:
            ax.legend(fontsize=8)
        fig.tight_layout()

        figs[eval_time] = fig

    return figs


def _collect_ori_log_paths(logs_dir):
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


def _get_method_name_from_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        return os.path.basename(run_dir)

    cfg = OmegaConf.load(cfg_path)
    method_name = cfg.model.name
    if "distance" in cfg.model:
        method_name += f"_{cfg.model.distance}"
    return method_name


def summar_ori_metrics(
    logs_dir="logs",
    save_dir=None,
):
    if not save_dir: save_dir = os.path.join(logs_dir, "summary")
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

        if "ori" not in logs:
            continue

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        method_name = _get_method_name_from_config(run_dir)
        ori_summary.setdefault(method_name, {})
        ori_logs_by_method.setdefault(method_name, {})
        ori_summary[method_name] = _get_ori_summary(logs["ori"])
        ori_logs_by_method[method_name][run_name] = logs["ori"]

    save_path = os.path.join(ori_save_dir, "ori_summary.json")
    with open(save_path, "w") as f:
        json.dump(ori_summary, f, indent=2)

    ori_figs = _get_ori_figs(ori_logs_by_method)
    for eval_time, fig in ori_figs.items():
        fig.savefig(os.path.join(ori_save_dir, f"{eval_time}.png"))

    return ori_summary, ori_figs


def _resolve_default_logs_dir() -> str:
    env_logs_dir = os.environ.get("ORI_METRICS_LOGS_DIR")
    if env_logs_dir:
        return env_logs_dir

    for candidate in ("log_ckpt", "logs"):
        if os.path.isdir(candidate):
            return candidate

    return "log_ckpt"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize ori metrics from evaluation logs.",
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

    summar_ori_metrics(
        logs_dir=args.logs_dir or _resolve_default_logs_dir(),
        save_dir=args.save_dir,
    )
