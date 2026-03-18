import numpy as np

from failure_prob.mrefine.const import DELAY_DELTAS
from failure_prob.mrefine.utils import get_delay_scores, get_func_conformal_bands

REF_TOLERANCE_DELTAS = ("0.0", "0.1", "0.2", "0.3")
NAB_PROFILES = {
    "standard": {"tp_weight": 1.0, "fn_weight": 1.0, "fp_weight": 0.11},
    "reward_low_fp": {"tp_weight": 1.0, "fn_weight": 1.0, "fp_weight": 0.22},
    "reward_low_fn": {"tp_weight": 1.0, "fn_weight": 2.0, "fp_weight": 0.11},
}


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _safe_f1(precision: float, recall: float) -> float:
    den = precision + recall
    if den <= 0:
        return 0.0
    return float(2.0 * precision * recall / den)


def _find_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    padded = np.pad(mask.astype(int), (1, 1), constant_values=0)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _has_overlap(seg_a: tuple[int, int], seg_b: tuple[int, int]) -> bool:
    return seg_a[0] < seg_b[1] and seg_b[0] < seg_a[1]


def _get_failure_onset(rollout, valid_len: int) -> int | None:
    if rollout.episode_success == 1 or valid_len <= 0:
        return None

    # This project only has rollout-level failure labels plus task_min_step.
    # Use task_min_step as a proxy onset so temporal reference metrics stay comparable.
    onset = int(getattr(rollout, "task_min_step", valid_len)) - 1
    return max(0, min(onset, valid_len - 1))


def _build_gt_mask(rollout, valid_len: int) -> np.ndarray:
    gt_mask = np.zeros(valid_len, dtype=bool)
    onset = _get_failure_onset(rollout, valid_len)
    if onset is not None:
        gt_mask[onset:] = True
    return gt_mask


def _scaled_sigmoid(relative_position: float) -> float:
    if relative_position > 3.0:
        return -1.0
    return float(2.0 / (1.0 + np.exp(5.0 * relative_position)) - 1.0)


def _nab_rollout_scores(
    rollout,
    pred_mask: np.ndarray,
    profile: dict[str, float],
) -> tuple[float, float, float]:
    valid_len = int(len(pred_mask))
    onset = _get_failure_onset(rollout, valid_len)
    pred_indices = np.flatnonzero(pred_mask)

    if onset is None:
        raw_score = -profile["fp_weight"] * float(pred_indices.size)
        return raw_score, 0.0, 0.0

    window_start = onset
    window_end = valid_len
    window_width = float(max(window_end - window_start, 1))
    max_tp = _scaled_sigmoid(-1.0)

    raw_score = -profile["fn_weight"]
    for pred_idx in pred_indices:
        if window_start <= pred_idx < window_end:
            relative_position = -float(window_end - pred_idx) / window_width
            score = _scaled_sigmoid(relative_position) * profile["tp_weight"] / max_tp
            raw_score = max(raw_score, score)
        else:
            if pred_idx < window_start:
                distance = float(window_start - pred_idx)
            else:
                distance = float(pred_idx - (window_end - 1))
            position_past_window = distance / float(max(window_width - 1.0, 1.0))
            raw_score += _scaled_sigmoid(position_past_window) * profile["fp_weight"]

    perfect_score = profile["tp_weight"]
    null_score = -profile["fn_weight"]
    return float(raw_score), float(perfect_score), float(null_score)


def _nab_metrics(rollouts, pred_masks: list[np.ndarray]) -> dict[str, float]:
    metrics = {}
    for profile_name, profile in NAB_PROFILES.items():
        raw_total = 0.0
        perfect_total = 0.0
        null_total = 0.0
        for rollout, pred_mask in zip(rollouts, pred_masks):
            raw_score, perfect_score, null_score = _nab_rollout_scores(
                rollout,
                pred_mask,
                profile,
            )
            raw_total += raw_score
            perfect_total += perfect_score
            null_total += null_score

        denom = perfect_total - null_total
        if denom <= 0:
            normalized_score = np.nan
        else:
            normalized_score = (raw_total - null_total) / denom
        metrics[f"nab_{profile_name}"] = float(normalized_score)
    return metrics


def _get_pate_weights(
    rollout,
    valid_len: int,
    pre_buffer: int | None = None,
    post_buffer: int | None = None,
) -> np.ndarray:
    # Adapted PATE-style proximity weights for rollout-level onset labels.
    # We only have a proxy anomaly onset (task_min_step), not dense anomaly spans.
    weights = np.zeros(valid_len, dtype=float)
    onset = _get_failure_onset(rollout, valid_len)
    if onset is None:
        return weights

    anomaly_end = valid_len
    anomaly_len = max(anomaly_end - onset, 1)
    if pre_buffer is None:
        pre_buffer = anomaly_len
    if post_buffer is None:
        post_buffer = anomaly_len
    pre_buffer = max(int(pre_buffer), 0)
    post_buffer = max(int(post_buffer), 0)

    weights[onset:anomaly_end] = 1.0
    if pre_buffer > 0:
        start = max(0, onset - pre_buffer)
        for idx in range(start, onset):
            weights[idx] = max(weights[idx], (idx - start + 1) / float(pre_buffer + 1))
    if post_buffer > 0 and anomaly_end < valid_len:
        stop = min(valid_len, anomaly_end + post_buffer)
        for idx in range(anomaly_end, stop):
            weights[idx] = max(weights[idx], (stop - idx) / float(post_buffer + 1))
    return weights


def _compute_weighted_pr_auc(
    scores: np.ndarray,
    pos_weights: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=float)
    pos_weights = np.asarray(pos_weights, dtype=float)
    if scores.size == 0 or pos_weights.size == 0 or scores.size != pos_weights.size:
        return np.nan

    total_pos = float(np.sum(pos_weights))
    if total_pos <= 0:
        return np.nan

    thresholds = np.unique(scores)[::-1]
    precisions = [1.0]
    recalls = [0.0]

    for threshold in thresholds:
        pred_mask = scores >= threshold
        tp = float(np.sum(pos_weights[pred_mask]))
        fp = float(np.sum((1.0 - pos_weights)[pred_mask]))
        fn = total_pos - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        precisions.append(precision)
        recalls.append(recall)

    precisions.append(0.0)
    recalls.append(1.0)
    return float(np.trapezoid(np.asarray(precisions), np.asarray(recalls)))


def _get_ref_continuous_res(
    test_rollouts,
    test_scores_all,
    method_name,
    res_dict,
):
    res_dict.setdefault("continuous", {})
    res_dict["continuous"].setdefault(method_name, {})
    continuous_dict = res_dict["continuous"][method_name]

    test_earliest_stop = np.asarray([r.task_min_step for r in test_rollouts], dtype=int)
    test_scores_all = np.asarray(test_scores_all)

    for eval_time_mode in ("last", "early"):
        continuous_dict.setdefault(eval_time_mode, {})
        for delta in DELAY_DELTAS:
            delta_float = float(delta)
            continuous_dict[eval_time_mode].setdefault(delta, {})
            delta_dict = continuous_dict[eval_time_mode][delta]

            if eval_time_mode == "last":
                lengths = np.full(len(test_rollouts), test_scores_all.shape[1], dtype=int)
                delayed_scores_all = np.asarray(
                    [get_delay_scores(scores, delta_float) for scores in test_scores_all]
                )
            else:
                lengths = test_earliest_stop
                delayed_scores_all = np.asarray(
                    [
                        get_delay_scores(test_scores_all[i], delta_float, lengths[i])
                        for i in range(len(test_scores_all))
                    ]
                )

            flat_scores = []
            flat_weights = []
            for rollout, delayed_scores, valid_len in zip(test_rollouts, delayed_scores_all, lengths):
                valid_len = int(valid_len)
                flat_scores.append(np.asarray(delayed_scores[:valid_len], dtype=float))
                flat_weights.append(_get_pate_weights(rollout, valid_len))

            if flat_scores:
                all_scores = np.concatenate(flat_scores, axis=0)
                all_weights = np.concatenate(flat_weights, axis=0)
                pate_score = _compute_weighted_pr_auc(all_scores, all_weights)
            else:
                pate_score = np.nan

            delta_dict.setdefault("detect_method", method_name)
            delta_dict.setdefault("pate", [])
            delta_dict["pate"].append(float(pate_score))


def _point_metrics(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        tp += int(np.logical_and(gt_mask, pred_mask).sum())
        fp += int(np.logical_and(~gt_mask, pred_mask).sum())
        fn += int(np.logical_and(gt_mask, ~pred_mask).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": _safe_f1(precision, recall),
    }


def _point_adjusted_metrics(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
) -> dict[str, float]:
    adjusted_preds = []
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        adjusted_pred = np.asarray(pred_mask, dtype=bool).copy()
        for start, end in _find_segments(gt_mask):
            if np.any(pred_mask[start:end]):
                adjusted_pred[start:end] = True
        adjusted_preds.append(adjusted_pred)

    tp = 0
    fp = 0
    fn = 0
    for gt_mask, pred_mask in zip(gt_masks, adjusted_preds):
        tp += int(np.logical_and(gt_mask, pred_mask).sum())
        fp += int(np.logical_and(~gt_mask, pred_mask).sum())
        fn += int(np.logical_and(gt_mask, ~pred_mask).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "point_adjusted_precision": precision,
        "point_adjusted_recall": recall,
        "point_adjusted_f1": _safe_f1(precision, recall),
    }


def _segment_metrics(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> dict[str, float]:
    matched_gt = 0
    total_gt = 0
    matched_pred = 0
    total_pred = 0

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        gt_segments = _find_segments(gt_mask)
        pred_segments = _find_segments(pred_mask)
        total_gt += len(gt_segments)
        total_pred += len(pred_segments)

        for gt_seg in gt_segments:
            if any(_has_overlap(gt_seg, pred_seg) for pred_seg in pred_segments):
                matched_gt += 1
        for pred_seg in pred_segments:
            if any(_has_overlap(pred_seg, gt_seg) for gt_seg in gt_segments):
                matched_pred += 1

    precision = _safe_div(matched_pred, total_pred)
    recall = _safe_div(matched_gt, total_gt)
    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": _safe_f1(precision, recall),
    }


def _time_tolerant_metrics(
    rollouts,
    pred_masks: list[np.ndarray],
    tolerance_delta: float,
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for rollout, pred_mask in zip(rollouts, pred_masks):
        valid_len = len(pred_mask)
        pred_indices = np.flatnonzero(pred_mask)
        has_pred = pred_indices.size > 0
        pred_time = int(pred_indices[0]) if has_pred else None

        onset = _get_failure_onset(rollout, valid_len)
        if onset is None:
            if has_pred:
                fp += 1
            continue

        tolerance = int(round(tolerance_delta * max(valid_len, 1)))
        if has_pred and abs(pred_time - onset) <= tolerance:
            tp += 1
        else:
            fn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    tau_tag = str(f"{tolerance_delta:.1f}").replace(".", "p")
    return {
        f"time_tolerant_precision_tau{tau_tag}": precision,
        f"time_tolerant_recall_tau{tau_tag}": recall,
        f"time_tolerant_f1_tau{tau_tag}": _safe_f1(precision, recall),
    }


def _get_ref_calib_res(
    test_rollouts,
    test_scores_all,
    cp_bands_by_alpha,
    alphas,
    method_name,
    res_dict,
):
    res_dict.setdefault("calib", {})
    res_dict["calib"].setdefault(method_name, {})
    calib_dict = res_dict["calib"][method_name]

    test_earliest_stop = np.asarray([r.task_min_step for r in test_rollouts], dtype=int)
    test_scores_all = np.asarray(test_scores_all)
    lower_bound = False

    for eval_time_mode in ("last", "early"):
        calib_dict.setdefault(eval_time_mode, {})

        for delta in DELAY_DELTAS:
            calib_dict[eval_time_mode].setdefault(delta, {})
            delta_float = float(delta)

            for alpha in alphas:
                cp_band = cp_bands_by_alpha[alpha]
                if eval_time_mode == "last":
                    lengths = np.full(len(test_rollouts), test_scores_all.shape[1], dtype=int)
                    delayed_scores_all = np.asarray(
                        [get_delay_scores(scores, delta_float) for scores in test_scores_all]
                    )
                else:
                    lengths = test_earliest_stop
                    delayed_scores_all = np.asarray(
                        [
                            get_delay_scores(test_scores_all[i], delta_float, lengths[i])
                            for i in range(len(test_scores_all))
                        ]
                    )

                if lower_bound:
                    detection_mask = delayed_scores_all <= cp_band
                else:
                    detection_mask = delayed_scores_all >= cp_band

                if eval_time_mode == "early":
                    for i, valid_len in enumerate(lengths):
                        detection_mask[i, valid_len:] = False

                pred_masks = []
                gt_masks = []
                for rollout, pred_mask_full, valid_len in zip(test_rollouts, detection_mask, lengths):
                    valid_len = int(valid_len)
                    pred_masks.append(np.asarray(pred_mask_full[:valid_len], dtype=bool))
                    gt_masks.append(_build_gt_mask(rollout, valid_len))

                metrics = {}
                metrics.update(_point_metrics(gt_masks, pred_masks))
                metrics.update(_point_adjusted_metrics(gt_masks, pred_masks))
                metrics.update(_segment_metrics(gt_masks, pred_masks))
                metrics.update(_nab_metrics(test_rollouts, pred_masks))
                for tol_delta in REF_TOLERANCE_DELTAS:
                    metrics.update(
                        _time_tolerant_metrics(test_rollouts, pred_masks, float(tol_delta))
                    )

                alpha_dict = calib_dict[eval_time_mode][delta].setdefault(f"{alpha}", {})
                alpha_dict.setdefault("detect_method", method_name)
                for metric_name, metric_value in metrics.items():
                    alpha_dict.setdefault(metric_name, [])
                    alpha_dict[metric_name].append(float(metric_value))


def get_ref_metrics(
    scores_by_split_name,
    rollouts_by_split_name,
    method_name,
    res_dict,
):
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]

    cal_rollouts = list(rollouts_by_split_name["val_seen"])
    cal_scores_all = list(scores_by_split_name["val_seen"])
    test_rollouts = list(rollouts_by_split_name["val_unseen"])
    test_scores_all = list(scores_by_split_name["val_unseen"])

    max_length = max(len(scores) for scores in cal_scores_all + test_scores_all)
    for i, scores in enumerate(cal_scores_all):
        cal_scores_all[i] = np.pad(scores, (0, max_length - len(scores)), mode="edge")
    for i, scores in enumerate(test_scores_all):
        test_scores_all[i] = np.pad(scores, (0, max_length - len(scores)), mode="edge")

    cp_bands_by_alpha = get_func_conformal_bands(cal_rollouts, cal_scores_all, alphas)
    _get_ref_calib_res(
        test_rollouts=test_rollouts,
        test_scores_all=test_scores_all,
        cp_bands_by_alpha=cp_bands_by_alpha,
        alphas=alphas,
        method_name=method_name,
        res_dict=res_dict,
    )
    _get_ref_continuous_res(
        test_rollouts=test_rollouts,
        test_scores_all=test_scores_all,
        method_name=method_name,
        res_dict=res_dict,
    )


def summary_ref_metrics(*args, **kwargs):
    from failure_prob.mrefine.ref_summary import summary_ref_metrics as _summary_ref_metrics

    return _summary_ref_metrics(*args, **kwargs)


def _resolve_default_logs_dir():
    from failure_prob.mrefine.ref_summary import _resolve_default_logs_dir as _resolve

    return _resolve()


def main():
    from failure_prob.mrefine.ref_summary import main as _main

    return _main()

__all__ = [
    "REF_TOLERANCE_DELTAS",
    "get_ref_metrics",
    "summary_ref_metrics",
]


if __name__ == "__main__":
    main()
