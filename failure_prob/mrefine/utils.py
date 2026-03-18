import numpy as np

from failure_prob.utils.conformal.functional_predictor import (
    RegressionType,
    ModulationType,
    FunctionalPredictor
)


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


def get_func_conformal_bands(
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
