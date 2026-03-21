import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.mrefine.delay_summary import (
    _compute_max_bal_acc_from_pareto,
    _compute_pareto_balacc_coverage,
    _compute_penalized_mean_t_at_balacc,
    _get_pareto_curve_from_alpha_dict,
)
from failure_prob.pipeline.val_new import (
    _resolve_default_logs_dir,
    _seed_key,
    collect_new_candidates,
)


def _load_val_selection(selection_path: str) -> dict:
    with open(selection_path, "r") as f:
        return json.load(f)


def _index_candidates(candidates: list[dict]) -> tuple[dict, dict]:
    by_run_name = {}
    by_signature = {}
    for candidate in candidates:
        by_run_name[(candidate["method_name"], candidate["seed"], candidate["run_name"])] = candidate
        by_signature.setdefault(
            (candidate["method_name"], candidate["seed"], candidate["config_signature"]),
            [],
        ).append(candidate)
    return by_run_name, by_signature


def _match_candidate(
    selection_record: dict,
    by_run_name: dict,
    by_signature: dict,
) -> tuple[dict | None, str]:
    exact = by_run_name.get((
        selection_record["method"],
        selection_record.get("seed"),
        selection_record["run_name"],
    ))
    if exact is not None:
        return exact, "exact_run_name"

    matched = by_signature.get(
        (
            selection_record["method"],
            selection_record.get("seed"),
            selection_record["config_signature"],
        ),
        [],
    )
    if not matched:
        return None, "missing"

    matched = sorted(matched, key=lambda x: x["run_name"])
    if len(matched) == 1:
        return matched[0], "config_signature"
    return matched[-1], "config_signature_ambiguous_latest"


def _find_alpha_dict_for_delta(delta_dict: dict, target_delta: float) -> dict | None:
    for delta_key, alpha_dict in delta_dict.items():
        if np.isclose(float(delta_key), float(target_delta)):
            return alpha_dict
    return None


def _to_jsonable_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in row.items()
        })
    return records


def summarize_test_scores(
    candidates: list[dict],
    val_selection: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_run_name, by_signature = _index_candidates(candidates)

    rows = []
    missing_rows = []
    penalty_det_time = float(val_selection["penalty_det_time"])
    range_stats = val_selection["range_stats"]
    for selection_record in val_selection["selected"]:
        candidate, match_type = _match_candidate(selection_record, by_run_name, by_signature)
        if candidate is None:
            missing_rows.append({
                "method": selection_record["method"],
                "seed": selection_record.get("seed"),
                "mode": selection_record["mode"],
                "val_run_name": selection_record["run_name"],
                "config_signature": selection_record["config_signature"],
                "reason": match_type,
            })
            continue

        mode = selection_record["mode"]
        target_delta = float(selection_record["delta"])
        alpha_dict = _find_alpha_dict_for_delta(
            candidate["new_summary"]["calib"].get(mode, {}),
            target_delta,
        )
        if alpha_dict is None:
            missing_rows.append({
                "method": selection_record["method"],
                "seed": selection_record.get("seed"),
                "mode": mode,
                "val_run_name": selection_record["run_name"],
                "config_signature": selection_record["config_signature"],
                "reason": f"delta_missing:{target_delta}",
            })
            continue

        stats = range_stats.get(_seed_key(selection_record.get("seed")), {}).get(mode, {})
        range_min = stats.get("bal_acc_min", np.nan)
        range_max = stats.get("bal_acc_max", np.nan)
        pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(alpha_dict)
        score = _compute_penalized_mean_t_at_balacc(
            pareto_det_times,
            pareto_bal_accs,
            range_min,
            range_max,
            penalty_det_time=penalty_det_time,
        )
        coverage = _compute_pareto_balacc_coverage(
            pareto_bal_accs,
            range_min,
            range_max,
        )
        max_bal_acc = _compute_max_bal_acc_from_pareto(pareto_bal_accs)
        rows.append({
            "method": selection_record["method"],
            "seed": selection_record.get("seed"),
            "mode": mode,
            "delta": target_delta,
            "match_type": match_type,
            "val_run_name": selection_record["run_name"],
            "test_run_name": candidate["run_name"],
            "test_run_dir": candidate["run_dir"],
            "config_signature": selection_record["config_signature"],
            "bal_acc_min": range_min,
            "bal_acc_max": range_max,
            "penalty_det_time": penalty_det_time,
            "pareto_points": int(len(pareto_det_times)),
            "pareto_coverage": coverage,
            "pareto_max_bal_acc": max_bal_acc,
            "penalized_mean_t_at_balacc": score,
            "integral_t_at_balacc": (
                score * (range_max - range_min)
                if np.isfinite(score) and np.isfinite(range_min) and np.isfinite(range_max)
                else np.nan
            ),
        })

    score_df = pd.DataFrame(rows)
    if not score_df.empty:
        score_df = score_df.sort_values(
            by=["mode", "seed", "penalized_mean_t_at_balacc", "method"],
            na_position="last",
        ).reset_index(drop=True)

    missing_df = pd.DataFrame(missing_rows)
    if not missing_df.empty:
        missing_df = missing_df.sort_values(by=["mode", "seed", "method", "val_run_name"]).reset_index(drop=True)

    return score_df, missing_df


def summarize_test_aggregate(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "mode",
                "num_runs",
                "mean_penalized_mean_t_at_balacc",
                "std_penalized_mean_t_at_balacc",
                "mean_integral_t_at_balacc",
                "std_integral_t_at_balacc",
            ]
        )

    rows = []
    for (method, mode), group in score_df.groupby(["method", "mode"], dropna=False):
        rows.append({
            "method": method,
            "mode": mode,
            "num_runs": int(len(group)),
            "mean_penalized_mean_t_at_balacc": float(group["penalized_mean_t_at_balacc"].mean()),
            "std_penalized_mean_t_at_balacc": float(group["penalized_mean_t_at_balacc"].std(ddof=0)),
            "mean_integral_t_at_balacc": float(group["integral_t_at_balacc"].mean()),
            "std_integral_t_at_balacc": float(group["integral_t_at_balacc"].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(by=["mode", "mean_penalized_mean_t_at_balacc", "method"]).reset_index(drop=True)


def run_test_pipeline(
    logs_dir: str,
    selection_path: str,
    save_dir: str,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    val_selection = _load_val_selection(selection_path)
    candidates = collect_new_candidates(logs_dir)
    score_df, missing_df = summarize_test_scores(candidates, val_selection)
    aggregate_df = summarize_test_aggregate(score_df)

    score_df.to_csv(os.path.join(save_dir, "test_selected_results_by_seed.csv"), index=False)
    aggregate_df.to_csv(os.path.join(save_dir, "test_selected_results_aggregate.csv"), index=False)
    missing_df.to_csv(os.path.join(save_dir, "test_missing_matches.csv"), index=False)

    payload = {
        "logs_dir": os.path.abspath(logs_dir),
        "selection_path": os.path.abspath(selection_path),
        "save_dir": os.path.abspath(save_dir),
        "results": _to_jsonable_records(score_df),
        "aggregate": _to_jsonable_records(aggregate_df),
        "missing": _to_jsonable_records(missing_df),
    }
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate test logs with the hyper-parameters and BalAcc ranges selected on validation logs.",
    )
    parser.add_argument(
        "--logs-dir",
        default=_resolve_default_logs_dir(),
        help="Root directory that contains test run folders with eval/new_logs.json.",
    )
    parser.add_argument(
        "--selection-path",
        required=True,
        help="Path to val_selection.json produced by failure_prob/pipeline/val_new.py.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save test summaries. Defaults to <logs-dir>/pipeline_test_new.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.abspath(args.logs_dir), "pipeline_test_new")
    result = run_test_pipeline(
        logs_dir=args.logs_dir,
        selection_path=args.selection_path,
        save_dir=save_dir,
    )

    print("Saved test comparison to", os.path.abspath(os.path.join(save_dir, "test_results.json")))
    for record in result["results"]:
        print(
            f"[{record['mode']}] {record['method']}: "
            f"delta={record['delta']:.3f} "
            f"score={record['penalized_mean_t_at_balacc']:.6f} "
            f"test_run={record['test_run_name']}"
        )


if __name__ == "__main__":
    main()
