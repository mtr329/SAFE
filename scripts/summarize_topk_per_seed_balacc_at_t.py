from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.figure import Figure
from failure_prob.mrefine.delay_summary import _compute_penalized_mean_t_at_balacc


METHOD_LABELS = {
    "embed_cosine": "Embed-Cosine",
    "embed_euclid": "Embed-Euclid",
    "embed_mahala": "Embed-Mahala",
    "embed_pca_kmeans": "Embed-PCAKMeans",
    "indep": "MLP",
    "logpZO": "LogPZO",
    "lstm": "LSTM",
    "rnd": "RND",
    "trans": "Trans",
}

SELECTION_SPECS = [
    ("roc_auc", "ROC-AUC", "#1f77b4"),
    ("prc_auc", "PRC-AUC", "#ff7f0e"),
    ("t_over_acc", "Integral", "#2ca02c"),
]

DEFAULT_MODES = ("early", "last")


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], value)


def infer_dataset_label(methods_dir: Path) -> str:
    path_str = str(methods_dir)
    if "/pizero_fast/" in path_str:
        return "PiZero-Fast LIBERO"
    if "/pizero/" in path_str:
        return "PiZero LIBERO"
    if "/openvla/" in path_str:
        return "OpenVLA LIBERO"
    return methods_dir.parent.name.replace("_", " ").title()


def _safe_float_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _iter_method_dirs(methods_dir: Path, exclude_methods: set[str]) -> list[Path]:
    return sorted(
        path for path in methods_dir.iterdir()
        if path.is_dir() and path.name not in exclude_methods and (path / "val_summary.csv").is_file()
    )


def _pareto_curve_from_raw(raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    use_df = raw_df.copy()
    if "is_pareto" in use_df.columns:
        pareto_df = use_df.loc[use_df["is_pareto"].astype(bool)].copy()
        if not pareto_df.empty:
            use_df = pareto_df

    points = (
        use_df[["avg_det_time", "bal_acc"]]
        .assign(
            avg_det_time=lambda x: pd.to_numeric(x["avg_det_time"], errors="coerce"),
            bal_acc=lambda x: pd.to_numeric(x["bal_acc"], errors="coerce"),
        )
        .dropna()
        .sort_values(by=["avg_det_time", "bal_acc"])
        .reset_index(drop=True)
    )
    if points.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    det_times = []
    bal_accs = []
    best_bal_acc = -np.inf
    for row in points.itertuples(index=False):
        det_time = float(row.avg_det_time)
        bal_acc = float(row.bal_acc)
        if bal_acc > best_bal_acc:
            det_times.append(det_time)
            bal_accs.append(bal_acc)
            best_bal_acc = bal_acc
    return np.asarray(det_times, dtype=float), np.asarray(bal_accs, dtype=float)


def _interp_balacc_on_pareto_for_det_time(
    pareto_det_times: np.ndarray,
    pareto_bal_accs: np.ndarray,
    target_det_time: float,
) -> float:
    if pareto_det_times.size < 2 or pareto_bal_accs.size < 2:
        return np.nan
    if target_det_time < pareto_det_times.min() or target_det_time > pareto_det_times.max():
        return np.nan
    return float(np.interp(float(target_det_time), pareto_det_times, pareto_bal_accs))


def _interp_det_time_on_pareto_for_bal_acc(
    pareto_det_times: np.ndarray,
    pareto_bal_accs: np.ndarray,
    target_bal_acc: float,
) -> float:
    if pareto_det_times.size == 0 or pareto_bal_accs.size == 0:
        return np.nan
    if target_bal_acc <= pareto_bal_accs.min():
        return float(pareto_det_times[0])
    if target_bal_acc > pareto_bal_accs.max():
        return np.nan
    return float(np.interp(float(target_bal_acc), pareto_bal_accs, pareto_det_times))


def _compute_pareto_balacc_coverage(
    pareto_bal_accs: np.ndarray,
    bal_acc_min: float,
    bal_acc_max: float,
) -> float:
    if pareto_bal_accs.size == 0 or bal_acc_max <= bal_acc_min:
        return 0.0

    covered_min = max(float(np.min(pareto_bal_accs)), float(bal_acc_min))
    covered_max = min(float(np.max(pareto_bal_accs)), float(bal_acc_max))
    covered_span = max(0.0, covered_max - covered_min)
    total_span = float(bal_acc_max) - float(bal_acc_min)
    return float(covered_span / total_span)


def _compute_t_integral_over_balacc_range(
    pareto_det_times: np.ndarray,
    pareto_bal_accs: np.ndarray,
    bal_acc_min: float,
    bal_acc_max: float,
) -> float:
    if pareto_det_times.size < 2 or pareto_bal_accs.size < 2:
        return np.nan
    if not np.isfinite(bal_acc_min) or not np.isfinite(bal_acc_max) or bal_acc_max <= bal_acc_min:
        return np.nan

    coverage = _compute_pareto_balacc_coverage(pareto_bal_accs, bal_acc_min, bal_acc_max)
    if coverage < 1.0:
        return np.nan

    mean_t = _compute_penalized_mean_t_at_balacc(
        pareto_det_times,
        pareto_bal_accs,
        bal_acc_min,
        bal_acc_max,
    )
    if not np.isfinite(mean_t):
        return np.nan
    return float(mean_t * (bal_acc_max - bal_acc_min))


def _build_run_time_metrics(
    method_dir: Path,
    modes: tuple[str, ...],
    bal_acc_min: float,
    bal_acc_max: float,
    target_det_times: list[float],
    target_bal_accs: list[float],
) -> pd.DataFrame:
    val_summary_df = _load_csv(method_dir / "val_summary.csv")
    rows = []
    for row in val_summary_df.to_dict(orient="records"):
        run_name = row.get("run_name")
        raw_path = row.get("new_raw_path")
        if pd.isna(run_name) or pd.isna(raw_path) or not os.path.isfile(str(raw_path)):
            continue

        raw_df = pd.read_csv(str(raw_path))
        for mode in modes:
            mode_df = raw_df.loc[raw_df["mode"].astype(str) == mode].copy()
            if mode_df.empty:
                continue

            best_t_integral = np.nan
            best_mean_t = np.nan
            best_delta = np.nan
            best_targets: dict[float, float] = {}
            best_max_bal_acc = np.nan
            best_coverage = 0.0
            best_t_at_balacc: dict[float, float] = {float(target): np.nan for target in target_bal_accs}

            for delta in sorted(pd.to_numeric(mode_df["delta"], errors="coerce").dropna().unique().tolist()):
                delta_df = mode_df.loc[np.isclose(pd.to_numeric(mode_df["delta"], errors="coerce"), float(delta))].copy()
                pareto_det_times, pareto_bal_accs = _pareto_curve_from_raw(delta_df)
                if pareto_bal_accs.size:
                    max_bal_acc = float(np.max(pareto_bal_accs))
                    if not np.isfinite(best_max_bal_acc) or max_bal_acc > best_max_bal_acc:
                        best_max_bal_acc = max_bal_acc
                coverage = _compute_pareto_balacc_coverage(
                    pareto_bal_accs,
                    bal_acc_min=bal_acc_min,
                    bal_acc_max=bal_acc_max,
                )
                t_integral = _compute_t_integral_over_balacc_range(
                    pareto_det_times,
                    pareto_bal_accs,
                    bal_acc_min=bal_acc_min,
                    bal_acc_max=bal_acc_max,
                )
                if not np.isfinite(t_integral):
                    continue

                mean_t = t_integral / float(bal_acc_max - bal_acc_min)
                if (not np.isfinite(best_t_integral)) or (t_integral < best_t_integral):
                    best_t_integral = float(t_integral)
                    best_mean_t = float(mean_t)
                    best_delta = float(delta)
                    best_coverage = float(coverage)
                    best_targets = {
                        float(target_det_time): _interp_balacc_on_pareto_for_det_time(
                            pareto_det_times,
                            pareto_bal_accs,
                            float(target_det_time),
                        )
                        for target_det_time in target_det_times
                    }

                for target_bal_acc in target_bal_accs:
                    t_value = _interp_det_time_on_pareto_for_bal_acc(
                        pareto_det_times,
                        pareto_bal_accs,
                        float(target_bal_acc),
                    )
                    if np.isfinite(t_value) and (
                        not np.isfinite(best_t_at_balacc[float(target_bal_acc)])
                        or t_value < best_t_at_balacc[float(target_bal_acc)]
                    ):
                        best_t_at_balacc[float(target_bal_acc)] = float(t_value)

            out_row = {
                "method": method_dir.name,
                "run_name": str(run_name),
                "run_dir": str(row.get("run_dir", "")),
                "mode": mode,
                "val_acc_range_coverage": best_coverage,
                "val_integral_t_over_acc": best_t_integral,
                "val_mean_t_over_acc": best_mean_t,
                "val_acc_range_best_delta": best_delta,
                "best_max_bal_acc": best_max_bal_acc,
            }
            for target_det_time in target_det_times:
                out_row[f"bal_acc_at_t_{_safe_float_tag(target_det_time)}"] = best_targets.get(float(target_det_time), np.nan)
            for target_bal_acc in target_bal_accs:
                out_row[f"best_t_at_balacc_{_safe_float_tag(target_bal_acc)}"] = best_t_at_balacc.get(float(target_bal_acc), np.nan)
            rows.append(out_row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["method", "run_name", "mode"]).reset_index(drop=True)


def _build_base_candidates(method_dir: Path) -> pd.DataFrame:
    roc_df = _load_csv(method_dir / "roc_auc" / "candidates_by_seed.csv")
    prc_df = _load_csv(method_dir / "prc_auc" / "candidates_by_seed.csv")
    base_cols = ["method", "seed", "weight_key", "run_name", "run_dir"]
    roc_use = roc_df[base_cols + ["val_roc_auc_early"]].copy()
    prc_use = prc_df[base_cols + ["val_prc_auc_early"]].copy()
    return roc_use.merge(prc_use, on=base_cols, how="outer").sort_values(
        by=["method", "seed", "weight_key", "run_name"]
    ).reset_index(drop=True)


def build_dataset_candidates(
    methods_dir: Path,
    exclude_methods: set[str],
    modes: tuple[str, ...],
    bal_acc_min: float,
    bal_acc_max: float,
    target_det_times: list[float],
    target_bal_accs: list[float],
) -> pd.DataFrame:
    frames = []
    for method_dir in _iter_method_dirs(methods_dir, exclude_methods):
        base_df = _build_base_candidates(method_dir)
        time_df = _build_run_time_metrics(
            method_dir=method_dir,
            modes=modes,
            bal_acc_min=bal_acc_min,
            bal_acc_max=bal_acc_max,
            target_det_times=target_det_times,
            target_bal_accs=target_bal_accs,
        )
        if base_df.empty or time_df.empty:
            continue
        merged = base_df.merge(
            time_df,
            on=["method", "run_name", "run_dir"],
            how="inner",
        )
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        by=["method", "seed", "weight_key", "run_name", "mode"]
    ).reset_index(drop=True)


def select_topk_per_seed(candidate_df: pd.DataFrame, top_k: int, modes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    if candidate_df.empty:
        return pd.DataFrame()

    ranking_specs = [
        ("roc_auc", "val_roc_auc_early", False),
        ("prc_auc", "val_prc_auc_early", False),
        ("t_over_acc", "val_mean_t_over_acc", True),
    ]
    for mode in modes:
        mode_df = candidate_df.loc[candidate_df["mode"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        for method, method_df in mode_df.groupby("method", dropna=False):
            for seed, seed_df in method_df.groupby("seed", dropna=False):
                for selection_metric, score_col, ascending in ranking_specs:
                    use_df = seed_df.copy()
                    use_df[score_col] = pd.to_numeric(use_df[score_col], errors="coerce")
                    use_df = use_df.loc[np.isfinite(use_df[score_col])].copy()
                    if use_df.empty:
                        continue
                    use_df = use_df.sort_values(
                        by=[score_col, "weight_key", "run_name"],
                        ascending=[ascending, True, True],
                        kind="stable",
                    ).head(int(top_k)).copy()
                    use_df["selection_metric"] = selection_metric
                    use_df["rank_within_seed"] = np.arange(1, len(use_df) + 1)
                    rows.append(use_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        by=["method", "mode", "selection_metric", "seed", "rank_within_seed", "weight_key"]
    ).reset_index(drop=True)


def summarize_selected(
    selected_df: pd.DataFrame,
    target_det_times: list[float],
    target_bal_accs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_method_rows = []
    for (method, mode, selection_metric), group in selected_df.groupby(
        ["method", "mode", "selection_metric"],
        dropna=False,
    ):
        row = {
            "method": method,
            "mode": mode,
            "selection_metric": selection_metric,
            "num_selected_runs": int(len(group)),
            "num_unique_seeds": int(pd.to_numeric(group["seed"], errors="coerce").dropna().nunique()),
            "mean_best_max_bal_acc": float(pd.to_numeric(group["best_max_bal_acc"], errors="coerce").mean()),
            "std_best_max_bal_acc": float(pd.to_numeric(group["best_max_bal_acc"], errors="coerce").std(ddof=0)),
            "mean_val_mean_t_over_acc": float(pd.to_numeric(group["val_mean_t_over_acc"], errors="coerce").mean()),
            "std_val_mean_t_over_acc": float(pd.to_numeric(group["val_mean_t_over_acc"], errors="coerce").std(ddof=0)),
            "mean_val_acc_range_coverage": float(pd.to_numeric(group["val_acc_range_coverage"], errors="coerce").mean()),
            "std_val_acc_range_coverage": float(pd.to_numeric(group["val_acc_range_coverage"], errors="coerce").std(ddof=0)),
        }
        for target_det_time in target_det_times:
            tag = _safe_float_tag(target_det_time)
            series = pd.to_numeric(group[f"bal_acc_at_t_{tag}"], errors="coerce")
            row[f"mean_bal_acc_at_t_{tag}"] = float(series.mean())
            row[f"std_bal_acc_at_t_{tag}"] = float(series.std(ddof=0))
        for target_bal_acc in target_bal_accs:
            tag = _safe_float_tag(target_bal_acc)
            series = pd.to_numeric(group[f"best_t_at_balacc_{tag}"], errors="coerce")
            row[f"mean_best_t_at_balacc_{tag}"] = float(series.mean())
            row[f"std_best_t_at_balacc_{tag}"] = float(series.std(ddof=0))
        by_method_rows.append(row)

    by_method_df = pd.DataFrame(by_method_rows).sort_values(
        by=["method", "mode", "selection_metric"]
    ).reset_index(drop=True)

    overall_rows = []
    for (mode, selection_metric), group in by_method_df.groupby(["mode", "selection_metric"], dropna=False):
        row = {
            "mode": mode,
            "selection_metric": selection_metric,
            "num_methods": int(len(group)),
            "mean_over_methods_best_max_bal_acc": float(pd.to_numeric(group["mean_best_max_bal_acc"], errors="coerce").mean()),
            "std_over_methods_best_max_bal_acc": float(pd.to_numeric(group["mean_best_max_bal_acc"], errors="coerce").std(ddof=0)),
            "mean_over_methods_val_mean_t_over_acc": float(pd.to_numeric(group["mean_val_mean_t_over_acc"], errors="coerce").mean()),
            "std_over_methods_val_mean_t_over_acc": float(pd.to_numeric(group["mean_val_mean_t_over_acc"], errors="coerce").std(ddof=0)),
            "mean_over_methods_val_acc_range_coverage": float(pd.to_numeric(group["mean_val_acc_range_coverage"], errors="coerce").mean()),
            "std_over_methods_val_acc_range_coverage": float(pd.to_numeric(group["mean_val_acc_range_coverage"], errors="coerce").std(ddof=0)),
        }
        for target_det_time in target_det_times:
            tag = _safe_float_tag(target_det_time)
            col = f"mean_bal_acc_at_t_{tag}"
            row[f"mean_over_methods_bal_acc_at_t_{tag}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            row[f"std_over_methods_bal_acc_at_t_{tag}"] = float(pd.to_numeric(group[col], errors="coerce").std(ddof=0))
        for target_bal_acc in target_bal_accs:
            tag = _safe_float_tag(target_bal_acc)
            col = f"mean_best_t_at_balacc_{tag}"
            row[f"mean_over_methods_best_t_at_balacc_{tag}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            row[f"std_over_methods_best_t_at_balacc_{tag}"] = float(pd.to_numeric(group[col], errors="coerce").std(ddof=0))
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows).sort_values(by=["mode", "selection_metric"]).reset_index(drop=True)
    return by_method_df, overall_df


def _save_barplot(
    overall_df: pd.DataFrame,
    target_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    modes: tuple[str, ...],
) -> None:
    if overall_df.empty:
        return
    fig = Figure(figsize=(7.5, 5.2))
    ax = fig.subplots()
    width = 0.24
    x = np.arange(len(modes), dtype=float)

    for idx, (selection_metric, label, color) in enumerate(SELECTION_SPECS):
        offsets = x + (idx - 1) * width
        heights = []
        for mode in modes:
            row = overall_df.loc[
                (overall_df["mode"].astype(str) == mode)
                & (overall_df["selection_metric"].astype(str) == selection_metric)
            ]
            heights.append(float(row[target_col].iloc[0]) if not row.empty else np.nan)
        ax.bar(offsets, heights, width=width, color=color, alpha=0.9, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([mode.title() for mode in modes])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def save_plots(
    overall_df: pd.DataFrame,
    save_dir: Path,
    target_det_times: list[float],
    target_bal_accs: list[float],
    modes: tuple[str, ...],
) -> dict[str, str]:
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if overall_df.empty:
        return outputs

    range_path = plot_dir / "topk_mean_t_over_acc_over_methods.png"
    _save_barplot(
        overall_df=overall_df,
        target_col="mean_over_methods_val_mean_t_over_acc",
        ylabel="Mean t Over Acc Range",
        title="Per-seed Top-k selection: mean t over acc range",
        save_path=range_path,
        modes=modes,
    )
    outputs["plot_mean_t_over_acc"] = str(range_path)

    coverage_path = plot_dir / "topk_acc_range_coverage_over_methods.png"
    _save_barplot(
        overall_df=overall_df,
        target_col="mean_over_methods_val_acc_range_coverage",
        ylabel="Coverage Over Acc Range",
        title="Per-seed Top-k selection: acc-range coverage",
        save_path=coverage_path,
        modes=modes,
    )
    outputs["plot_acc_range_coverage"] = str(coverage_path)

    max_path = plot_dir / "topk_best_max_bal_acc_over_methods.png"
    _save_barplot(
        overall_df=overall_df,
        target_col="mean_over_methods_best_max_bal_acc",
        ylabel="Best Max BalAcc",
        title="Per-seed Top-k selection: best max BalAcc",
        save_path=max_path,
        modes=modes,
    )
    outputs["plot_best_max_bal_acc"] = str(max_path)

    for target_det_time in target_det_times:
        tag = _safe_float_tag(target_det_time)
        save_path = plot_dir / f"topk_bal_acc_at_t_{tag}_over_methods.png"
        _save_barplot(
            overall_df=overall_df,
            target_col=f"mean_over_methods_bal_acc_at_t_{tag}",
            ylabel=f"BalAcc@t={float(target_det_time):.2f}",
            title=f"Per-seed Top-k selection: BalAcc@t={float(target_det_time):.2f}",
            save_path=save_path,
            modes=modes,
        )
        outputs[f"plot_bal_acc_at_t_{tag}"] = str(save_path)

    for target_bal_acc in target_bal_accs:
        tag = _safe_float_tag(target_bal_acc)
        save_path = plot_dir / f"topk_t_at_balacc_{tag}_over_methods.png"
        _save_barplot(
            overall_df=overall_df,
            target_col=f"mean_over_methods_best_t_at_balacc_{tag}",
            ylabel=f"t@BalAcc={float(target_bal_acc):.2f}",
            title=f"Per-seed Top-k selection: t@BalAcc={float(target_bal_acc):.2f}",
            save_path=save_path,
            modes=modes,
        )
        outputs[f"plot_t_at_balacc_{tag}"] = str(save_path)
    return outputs


def build_latex_table(
    overall_df: pd.DataFrame,
    dataset_label: str,
    top_k: int,
    mode: str,
    target_det_times: list[float],
    target_bal_accs: list[float],
    bal_acc_min: float,
    bal_acc_max: float,
) -> str:
    mode_df = overall_df.loc[overall_df["mode"].astype(str) == mode].copy()
    if mode_df.empty:
        return ""

    lines = [
        r"\begin{table*}[htbp]",
        (
            rf"\caption{{{latex_escape(dataset_label)}: mean-over-methods summary for per-seed top-{int(top_k)} "
            rf"selection using a Pareto $t@\mathrm{{BalAcc}}$ integral over $\mathrm{{BalAcc}} \in [{float(bal_acc_min):.2f}, {float(bal_acc_max):.2f}]$. "
            r"Lower is better for mean $t$ over the acc range and $t@\mathrm{BalAcc}$; higher is better for coverage, $\mathrm{BalAcc}@t$, and max BalAcc.}}"
        ),
        r"\begin{center}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|l|" + "c|" * (len(target_bal_accs) + len(target_det_times) + 3) + r"}",
        r"\hline",
        r"\textbf{Selection} & \textbf{Mean $t$ Over Acc Range} & \textbf{Coverage} & " + " & ".join(
            [rf"\textbf{{$t@\mathrm{{BalAcc}}={float(target):.2f}$}}" for target in target_bal_accs] +
            [rf"\textbf{{$\mathrm{{BalAcc}}@t={float(target):.2f}$}}" for target in target_det_times] +
            [r"\textbf{Best Max BalAcc}"]
        ) + r" \\",
        r"\hline",
    ]

    for selection_metric, label, _ in SELECTION_SPECS:
        row = mode_df.loc[mode_df["selection_metric"].astype(str) == selection_metric]
        if row.empty:
            continue
        row = row.iloc[0]
        cells = [
            latex_escape(label),
            "--" if not np.isfinite(row.get("mean_over_methods_val_mean_t_over_acc", np.nan)) else f"{float(row['mean_over_methods_val_mean_t_over_acc']):.4f}",
            "--" if not np.isfinite(row.get("mean_over_methods_val_acc_range_coverage", np.nan)) else f"{float(row['mean_over_methods_val_acc_range_coverage']):.4f}",
        ]
        for target_bal_acc in target_bal_accs:
            tag = _safe_float_tag(target_bal_acc)
            value = row.get(f"mean_over_methods_best_t_at_balacc_{tag}", np.nan)
            cells.append("--" if not np.isfinite(value) else f"{float(value):.4f}")
        for target_det_time in target_det_times:
            tag = _safe_float_tag(target_det_time)
            value = row.get(f"mean_over_methods_bal_acc_at_t_{tag}", np.nan)
            cells.append("--" if not np.isfinite(value) else f"{float(value):.4f}")
        value = row.get("mean_over_methods_best_max_bal_acc", np.nan)
        cells.append("--" if not np.isfinite(value) else f"{float(value):.4f}")
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def process_dataset(
    methods_dir: Path,
    save_root: Path,
    exclude_methods: set[str],
    modes: tuple[str, ...],
    top_k: int,
    bal_acc_min: float,
    bal_acc_max: float,
    target_det_times: list[float],
    target_bal_accs: list[float],
) -> dict:
    dataset_label = infer_dataset_label(methods_dir)
    dataset_tag = _safe_file_stem(dataset_label.lower().replace(" ", "_"))
    save_dir = save_root / dataset_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = build_dataset_candidates(
        methods_dir=methods_dir,
        exclude_methods=exclude_methods,
        modes=modes,
        bal_acc_min=bal_acc_min,
        bal_acc_max=bal_acc_max,
        target_det_times=target_det_times,
        target_bal_accs=target_bal_accs,
    )
    selected_df = select_topk_per_seed(candidate_df, top_k=top_k, modes=modes)
    by_method_df, overall_df = summarize_selected(
        selected_df,
        target_det_times=target_det_times,
        target_bal_accs=target_bal_accs,
    )

    candidate_path = save_dir / "candidate_runs.csv"
    selected_path = save_dir / "selected_topk_per_seed.csv"
    by_method_path = save_dir / "selected_topk_summary_by_method.csv"
    overall_path = save_dir / "selected_topk_summary_mean_over_methods.csv"
    candidate_df.to_csv(candidate_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    by_method_df.to_csv(by_method_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    outputs = {
        "dataset_label": dataset_label,
        "methods_dir": str(methods_dir),
        "save_dir": str(save_dir),
        "candidate_runs_csv": str(candidate_path),
        "selected_topk_per_seed_csv": str(selected_path),
        "selected_topk_summary_by_method_csv": str(by_method_path),
        "selected_topk_summary_mean_over_methods_csv": str(overall_path),
        "num_candidate_rows": int(len(candidate_df)),
        "num_selected_rows": int(len(selected_df)),
    }

    outputs.update(save_plots(overall_df, save_dir, target_det_times, target_bal_accs, modes))
    latex_dir = save_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        latex = build_latex_table(
            overall_df=overall_df,
            dataset_label=dataset_label,
            top_k=top_k,
            mode=mode,
            target_det_times=target_det_times,
            target_bal_accs=target_bal_accs,
            bal_acc_min=bal_acc_min,
            bal_acc_max=bal_acc_max,
        )
        if not latex:
            continue
        latex_path = latex_dir / f"top{int(top_k)}_per_seed_{mode}.tex"
        latex_path.write_text(latex)
        outputs[f"latex_{mode}"] = str(latex_path)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2))
    outputs["summary_json"] = str(summary_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-seed top-k validation selections using a Pareto t@BalAcc integral over an acc range "
            "plus BalAcc@t targets. Outputs are written to log_analysis_outputs/, not back into methods/."
        )
    )
    parser.add_argument(
        "methods_dirs",
        nargs="+",
        help="One or more directories like log_ckpt_new/pizero_fast/pipeline_val_new/methods",
    )
    parser.add_argument(
        "--save-root",
        default="log_analysis_outputs/topk_per_seed_t_over_acc",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many runs to keep per seed for each selection metric.",
    )
    parser.add_argument(
        "--acc-min",
        type=float,
        required=True,
        help="Lower bound of the BalAcc integration window.",
    )
    parser.add_argument(
        "--acc-max",
        type=float,
        required=True,
        help="Upper bound of the BalAcc integration window.",
    )
    parser.add_argument(
        "--target-t",
        nargs="*",
        type=float,
        default=None,
        help="Specific time points for BalAcc@t. Defaults to 0.10, 0.20, 0.30, 0.40.",
    )
    parser.add_argument(
        "--target-bal-acc",
        nargs="*",
        type=float,
        default=[0.60, 0.65, 0.70],
        help="Specific BalAcc targets for t@BalAcc summary.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Mode to include. Can be repeated. Defaults to early and last.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=["trans"],
        help="Method to exclude. Can be repeated. Defaults to trans.",
    )
    args = parser.parse_args()

    bal_acc_min = float(args.acc_min)
    bal_acc_max = float(args.acc_max)
    if bal_acc_max <= bal_acc_min:
        raise SystemExit("--acc-max must be greater than --acc-min")

    if args.target_t:
        target_det_times = sorted({round(float(value), 6) for value in args.target_t})
    else:
        target_det_times = [0.10, 0.20, 0.30, 0.40]
    target_bal_accs = sorted({round(float(value), 6) for value in args.target_bal_acc})

    save_root = Path(args.save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    modes = tuple(dict.fromkeys(args.mode)) if args.mode else DEFAULT_MODES
    exclude_methods = {str(value) for value in args.exclude_method if str(value).strip()}

    outputs = []
    for methods_dir_arg in args.methods_dirs:
        methods_dir = Path(methods_dir_arg).resolve()
        if not methods_dir.is_dir():
            raise SystemExit(f"methods_dir is not a directory: {methods_dir}")
        outputs.append(
            process_dataset(
                methods_dir=methods_dir,
                save_root=save_root,
                exclude_methods=exclude_methods,
                modes=modes,
                top_k=int(args.top_k),
                bal_acc_min=bal_acc_min,
                bal_acc_max=bal_acc_max,
                target_det_times=target_det_times,
                target_bal_accs=target_bal_accs,
            )
        )

    summary = {
        "save_root": str(save_root),
        "top_k": int(args.top_k),
        "acc_min": bal_acc_min,
        "acc_max": bal_acc_max,
        "target_t": [float(value) for value in target_det_times],
        "target_bal_acc": [float(value) for value in target_bal_accs],
        "modes": list(modes),
        "exclude_methods": sorted(exclude_methods),
        "datasets": outputs,
    }
    summary_path = save_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
