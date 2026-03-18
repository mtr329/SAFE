import warnings

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from failure_prob.mrefine.const import DELAY_DELTAS
from failure_prob.mrefine.utils import (
    get_delay_scores,
    get_func_conformal_bands,
)

TEMPORAL_LAMBDAS = (
    ("lam1p0", 1.0),
    ("lam3p0", 3.0),
    ("lam5p0", 5.0),
)


def _compute_ttd_auc(
    relative_detection_times: np.ndarray,
    labels: np.ndarray,
) -> float:
    fail_times = np.asarray(relative_detection_times[labels == 1], dtype=float)
    normal_times = np.asarray(relative_detection_times[labels == 0], dtype=float)

    if fail_times.size == 0 or normal_times.size == 0:
        return np.nan

    normal_times_sorted = np.sort(normal_times)
    normal_count = normal_times_sorted.size
    normal_after_fail = normal_count - np.searchsorted(
        normal_times_sorted,
        fail_times,
        side="right",
    )
    return float(np.mean(normal_after_fail / normal_count))


def _compute_discounted_ttd_auc(
    relative_detection_times: np.ndarray,
    labels: np.ndarray,
    lambda_: float,
) -> float:
    fail_times = np.asarray(relative_detection_times[labels == 1], dtype=float)
    normal_times = np.asarray(relative_detection_times[labels == 0], dtype=float)

    if fail_times.size == 0 or normal_times.size == 0:
        return np.nan

    normal_times_sorted = np.sort(normal_times)
    normal_count = normal_times_sorted.size
    normal_after_fail = normal_count - np.searchsorted(
        normal_times_sorted,
        fail_times,
        side="right",
    )
    pairwise_success = normal_after_fail / normal_count
    discounts = np.exp(-lambda_ * fail_times)
    return float(np.mean(pairwise_success * discounts))


def _compute_discounted_correct_score(
    relative_detection_times: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    lambda_: float,
) -> float:
    correct = (predicted == labels).astype(float)
    discounts = np.exp(-lambda_ * np.asarray(relative_detection_times, dtype=float))
    return float(np.mean(correct * discounts))


def _compute_soft_detection_times(
    delayed_scores_all: np.ndarray,
    lengths,
) -> np.ndarray:
    if np.isscalar(lengths):
        lengths_arr = np.full(delayed_scores_all.shape[0], int(lengths), dtype=int)
    else:
        lengths_arr = np.asarray(lengths, dtype=int)

    soft_detection_times = np.full(delayed_scores_all.shape[0], np.nan, dtype=float)
    for i, length in enumerate(lengths_arr):
        if length <= 0:
            continue

        valid_scores = np.asarray(delayed_scores_all[i, :length], dtype=float)
        if valid_scores.size == 0:
            continue

        shifted_scores = valid_scores - np.max(valid_scores)
        weights = np.exp(shifted_scores)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            continue

        probs = weights / weights_sum
        time_grid = np.arange(valid_scores.size, dtype=float) / float(length)
        soft_detection_times[i] = float(np.sum(time_grid * probs))

    return soft_detection_times


def _get_soft_temporal_res(
    test_rollouts,
    test_scores_all,
    method_name,
    res_dict,
):
    delay_deltas = DELAY_DELTAS

    res_dict.setdefault("soft", {})
    res_dict["soft"].setdefault(method_name, {})
    soft_dict = res_dict["soft"][method_name]

    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts])
    test_labels_all = np.asarray([1 - r.episode_success for r in test_rollouts])
    test_scores_all = np.array(test_scores_all)

    for eval_time_mode in ["last", "early"]:
        soft_dict.setdefault(eval_time_mode, {})

        for delta in delay_deltas:
            soft_dict[eval_time_mode].setdefault(f"{delta}", {})
            delta_dict = soft_dict[eval_time_mode][f"{delta}"]
            _d = float(delta)

            if eval_time_mode == "last":
                lengths = test_scores_all.shape[1]
                delayed_scores_all = np.asarray([
                    get_delay_scores(s, _d)
                    for s in test_scores_all
                ])
            else:
                lengths = test_earliest_stop
                delayed_scores_all = np.asarray([
                    get_delay_scores(test_scores_all[i], _d, lengths[i])
                    for i in range(len(test_scores_all))
                ])

            soft_detection_times = _compute_soft_detection_times(delayed_scores_all, lengths)
            pos_mask = test_labels_all == 1
            delta_dict.setdefault("detect_method", method_name)
            for metric_name in ("soft_avg_det_time", "soft_ttd_auc"):
                delta_dict.setdefault(metric_name, [])

            if np.any(pos_mask):
                delta_dict["soft_avg_det_time"].append(float(np.nanmean(soft_detection_times[pos_mask])))
            else:
                delta_dict["soft_avg_det_time"].append(np.nan)
            delta_dict["soft_ttd_auc"].append(_compute_ttd_auc(soft_detection_times, test_labels_all))


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
                labels_task = [1 - r.episode_success for r in rollouts_task]

                for delta in delay_deltas:
                    _d = float(delta)
                    task_dict = static_dict[split][f"{task_id}"]
                    task_dict.setdefault(f"{delta}", {})
                    delta_dict = task_dict[f"{delta}"]

                    delayed_scores = [
                        get_delay_scores(s[:r.task_min_step], _d)
                        for s, r in zip(scores_task, rollouts_task)
                    ]
                    for pref in prefix_rs:
                        delta_dict.setdefault(pref, {})
                        pref_dict = delta_dict[pref]

                        pref_lens = [
                            max(int(round(len(s) * float(pref))), 1)
                            for s in delayed_scores
                        ]
                        pref_scores = [
                            s[:l].max()
                            for s, l in zip(delayed_scores, pref_lens)
                        ]

                        fpr, tpr, thresholds = roc_curve(labels_task, pref_scores)
                        roc_auc = auc(fpr, tpr)
                        pre, rec, thresholds = precision_recall_curve(labels_task, pref_scores)
                        prc_auc = auc(rec, pre)

                        for key in ["fpr", "tpr", "roc_auc", "pre", "rec", "prc_auc"]:
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
    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts])
    test_labels_all = np.asarray([1 - r.episode_success for r in test_rollouts])
    test_scores_all = np.array(test_scores_all)
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
                    lengths = test_scores_all.shape[1]
                    delayed_scores_all = [get_delay_scores(s, _d) for s in test_scores_all]
                elif eval_time_mode == "early":
                    lengths = test_earliest_stop
                    delayed_scores_all = [
                        get_delay_scores(test_scores_all[i], _d, lengths[i])
                        for i in range(len(test_scores_all))
                    ]
                delayed_scores_all = np.asarray(delayed_scores_all)

                if lower_bound:
                    detection_mask = delayed_scores_all <= cp_band
                else:
                    detection_mask = delayed_scores_all >= cp_band

                if eval_time_mode == "early":
                    for i in range(len(delayed_scores_all)):
                        detection_mask[i, lengths[i]:] = False

                has_detection = np.any(detection_mask, axis=1)
                first_detection = np.argmax(detection_mask, axis=1)
                detection_times = np.where(has_detection, first_detection, lengths)
                relative_detection_times = detection_times / lengths

                pos_mask = test_labels_all == 1
                avg_det_time = np.mean(relative_detection_times[pos_mask])
                ttd_auc = _compute_ttd_auc(relative_detection_times, test_labels_all)
                predicted = has_detection
                tp = (predicted & pos_mask).sum()
                fn = (~predicted & pos_mask).sum()
                fp = (predicted & ~pos_mask).sum()
                tn = (~predicted & ~pos_mask).sum()

                with np.errstate(divide="ignore", invalid="ignore"):
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    acc = (tp + tn) / n_test_samples
                    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                    bal_acc = (tpr + tnr) / 2

                alpha_dict.setdefault("detect_method", method_name)
                for m in ["avg_det_time", "ttd_auc", "tpr", "tnr", "fpr", "fnr", "acc", "bal_acc", "f1"]:
                    alpha_dict.setdefault(m, [])
                for lambda_tag, lambda_value in TEMPORAL_LAMBDAS:
                    alpha_dict.setdefault(f"discounted_ttd_auc_{lambda_tag}", [])
                    alpha_dict.setdefault(f"discounted_correct_{lambda_tag}", [])
                alpha_dict["avg_det_time"].append(avg_det_time)
                alpha_dict["ttd_auc"].append(ttd_auc)
                alpha_dict["tpr"].append(tpr)
                alpha_dict["tnr"].append(tnr)
                alpha_dict["fpr"].append(fpr)
                alpha_dict["fnr"].append(fnr)
                alpha_dict["acc"].append(acc)
                alpha_dict["bal_acc"].append(bal_acc)
                alpha_dict["f1"].append(f1)
                labels_bool = test_labels_all.astype(bool)
                for lambda_tag, lambda_value in TEMPORAL_LAMBDAS:
                    alpha_dict[f"discounted_ttd_auc_{lambda_tag}"].append(
                        _compute_discounted_ttd_auc(
                            relative_detection_times,
                            test_labels_all,
                            lambda_value,
                        )
                    )
                    alpha_dict[f"discounted_correct_{lambda_tag}"].append(
                        _compute_discounted_correct_score(
                            relative_detection_times,
                            labels_bool,
                            predicted,
                            lambda_value,
                        )
                    )


def get_new_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    _get_new_static_metrics(scores_by_split_name, rollouts_by_split_name, method_name, res_dict)

    alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]

    cal_rollouts, cal_scores_all = [], []
    cal_rollouts.extend(rollouts_by_split_name["val_seen"])
    cal_scores_all.extend(scores_by_split_name["val_seen"])
    test_rollouts, test_scores_all = [], []
    test_rollouts.extend(rollouts_by_split_name["val_unseen"])
    test_scores_all.extend(scores_by_split_name["val_unseen"])

    max_length = max(len(s) for s in cal_scores_all + test_scores_all)
    for i, s in enumerate(cal_scores_all):
        cal_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode="edge")
    for i, s in enumerate(test_scores_all):
        test_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode="edge")

    cp_bands_by_alpha = get_func_conformal_bands(cal_rollouts, cal_scores_all, alphas)
    _get_delay_calib_res(test_rollouts, test_scores_all, cp_bands_by_alpha, alphas, method_name, res_dict)
    _get_soft_temporal_res(test_rollouts, test_scores_all, method_name, res_dict)


__all__ = ["get_new_metrics"]
