from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import numpy as np

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
                for key in ["fpr", "tpr", "roc_auc", "pre", "rec", "prc_auc"]:
                    task_dict.setdefault(key, {})
                    task_dict[key].setdefault(f"{task_id}", [])
                # task_dict["fpr"][f"{task_id}"].append(fpr)
                # task_dict["tpr"][f"{task_id}"].append(tpr)
                task_dict["roc_auc"][f"{task_id}"].append(roc_auc)
                # task_dict["pre"][f"{task_id}"].append(pre)
                # task_dict["rec"][f"{task_id}"].append(rec)
                task_dict["prc_auc"][f"{task_id}"].append(prc_auc)


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

