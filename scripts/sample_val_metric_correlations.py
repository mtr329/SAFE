from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from matplotlib.figure import Figure


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

METHOD_COLORS = {
    "embed_cosine": "#1f77b4",
    "embed_euclid": "#ff7f0e",
    "embed_mahala": "#2ca02c",
    "embed_pca_kmeans": "#d62728",
    "indep": "#9467bd",
    "logpZO": "#8c564b",
    "lstm": "#e377c2",
    "rnd": "#7f7f7f",
    "trans": "#bcbd22",
}

METRIC_SPECS = [
    ("val_roc_auc_early", "Val ROC-AUC", True),
    ("val_prc_auc_early", "Val PRC-AUC", True),
    ("val_integral_t_at_balacc", "Val Integral", False),
]

DEFAULT_TARGET_BAL_ACCS = (0.60, 0.65, 0.70)
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
    return methods_dir.parent.name.replace("_", " ").title()


def _safe_float_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _parse_seed_runs(value) -> dict[str, str]:
    if pd.isna(value):
        return {}
    data = json.loads(value)
    return {str(k): str(v) for k, v in data.items()}


def _load_method_tables(method_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "val_summary": _load_csv(method_dir / "val_summary.csv"),
        "roc_weights": _load_csv(method_dir / "roc_auc" / "weights_mean_across_seeds.csv"),
        "prc_weights": _load_csv(method_dir / "prc_auc" / "weights_mean_across_seeds.csv"),
        "pareto_weights": _load_csv(method_dir / "pareto" / "weights_mean_across_seeds.csv"),
    }


def _select_best_pareto_rows(pareto_df: pd.DataFrame, min_seeds: int, modes: tuple[str, ...]) -> pd.DataFrame:
    if pareto_df.empty:
        return pd.DataFrame(columns=[
            "method",
            "mode",
            "weight_key",
            "delta",
            "num_seeds",
            "seed_runs",
            "mean_val_integral_t_at_balacc",
            "mean_val_penalized_mean_t_at_balacc",
        ])

    use_df = pareto_df.copy()
    use_df = use_df.loc[use_df["num_seeds"].fillna(0).astype(int) >= int(min_seeds)].copy()
    use_df = use_df.loc[use_df["mode"].astype(str).isin(modes)].copy()
    if use_df.empty:
        return use_df

    use_df["missing_score"] = ~np.isfinite(pd.to_numeric(use_df["mean_val_integral_t_at_balacc"], errors="coerce"))
    use_df = use_df.sort_values(
        by=[
            "method",
            "mode",
            "weight_key",
            "missing_score",
            "mean_val_integral_t_at_balacc",
            "mean_val_penalized_mean_t_at_balacc",
            "delta",
        ],
        ascending=[True, True, True, True, True, True, True],
        na_position="last",
        kind="stable",
    )
    return use_df.drop_duplicates(subset=["method", "mode", "weight_key"], keep="first").drop(
        columns=["missing_score"]
    ).reset_index(drop=True)


def _build_candidate_rows(
    methods_dir: Path,
    exclude_methods: set[str],
    min_seeds: int,
    modes: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for method_dir in sorted(
        path for path in methods_dir.iterdir()
        if path.is_dir() and (path / "val_summary.csv").is_file()
    ):
        method = method_dir.name
        if method in exclude_methods:
            continue

        tables = _load_method_tables(method_dir)
        roc_df = tables["roc_weights"].copy()
        prc_df = tables["prc_weights"].copy()
        pareto_df = _select_best_pareto_rows(tables["pareto_weights"], min_seeds=min_seeds, modes=modes)
        if roc_df.empty or prc_df.empty or pareto_df.empty:
            continue

        roc_df = roc_df.loc[roc_df["num_seeds"].fillna(0).astype(int) >= int(min_seeds)].copy()
        prc_df = prc_df.loc[prc_df["num_seeds"].fillna(0).astype(int) >= int(min_seeds)].copy()
        if roc_df.empty or prc_df.empty:
            continue

        roc_map = roc_df.set_index("weight_key")
        prc_map = prc_df.set_index("weight_key")
        pareto_by_mode = {
            mode: mode_df.set_index("weight_key")
            for mode, mode_df in pareto_df.groupby("mode", dropna=False)
        }
        shared_weight_keys = set(roc_map.index.astype(str)) & set(prc_map.index.astype(str))
        for mode in modes:
            mode_df = pareto_by_mode.get(mode)
            if mode_df is None or mode_df.empty:
                shared_weight_keys = set()
                break
            shared_weight_keys &= set(mode_df.index.astype(str))

        for weight_key in sorted(shared_weight_keys):
            roc_row = roc_map.loc[weight_key]
            prc_row = prc_map.loc[weight_key]
            for mode in modes:
                pareto_row = pareto_by_mode[mode].loc[weight_key]
                rows.append({
                    "method": method,
                    "weight_key": str(weight_key),
                    "mode": mode,
                    "num_seeds": int(min(
                        pd.to_numeric(pd.Series([roc_row["num_seeds"]]), errors="coerce").iloc[0],
                        pd.to_numeric(pd.Series([prc_row["num_seeds"]]), errors="coerce").iloc[0],
                        pd.to_numeric(pd.Series([pareto_row["num_seeds"]]), errors="coerce").iloc[0],
                    )),
                    "seed_runs": pareto_row["seed_runs"],
                    "delta": float(pareto_row["delta"]),
                    "val_roc_auc_early": float(roc_row["mean_val_roc_auc_early"]),
                    "val_prc_auc_early": float(prc_row["mean_val_prc_auc_early"]),
                    "val_integral_t_at_balacc": float(pareto_row["mean_val_integral_t_at_balacc"]),
                    "val_penalized_mean_t_at_balacc": float(pareto_row["mean_val_penalized_mean_t_at_balacc"]),
                    "val_curve_fraction_achieved": float(pareto_row["mean_val_curve_fraction_achieved"]),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "method",
            "weight_key",
            "mode",
            "num_seeds",
            "seed_runs",
            "delta",
            "val_roc_auc_early",
            "val_prc_auc_early",
            "val_integral_t_at_balacc",
            "val_penalized_mean_t_at_balacc",
            "val_curve_fraction_achieved",
        ])

    return pd.DataFrame(rows).sort_values(by=["method", "weight_key", "mode"]).reset_index(drop=True)


def _sample_candidates(candidate_df: pd.DataFrame, samples_per_method: int, random_seed: int) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()

    rng = np.random.default_rng(int(random_seed))
    sampled_weight_keys: list[tuple[str, str]] = []
    for method, method_df in candidate_df.groupby("method", dropna=False):
        weight_keys = sorted(method_df["weight_key"].dropna().astype(str).unique().tolist())
        if not weight_keys:
            continue
        sample_size = min(int(samples_per_method), len(weight_keys))
        chosen = rng.choice(np.asarray(weight_keys, dtype=object), size=sample_size, replace=False)
        sampled_weight_keys.extend((str(method), str(weight_key)) for weight_key in chosen.tolist())

    sampled_keys_df = pd.DataFrame(sampled_weight_keys, columns=["method", "weight_key"]).drop_duplicates()
    return candidate_df.merge(sampled_keys_df, on=["method", "weight_key"], how="inner").sort_values(
        by=["method", "weight_key", "mode"]
    ).reset_index(drop=True)


def _interp_t_at_balacc_from_raw(raw_df: pd.DataFrame, target_bal_acc: float) -> float:
    if raw_df.empty:
        return np.nan
    use_df = raw_df.copy()
    if "is_pareto" in use_df.columns:
        pareto_df = use_df.loc[use_df["is_pareto"].astype(bool)].copy()
        if not pareto_df.empty:
            use_df = pareto_df

    pairs = use_df[["bal_acc", "avg_det_time"]].copy()
    pairs["bal_acc"] = pd.to_numeric(pairs["bal_acc"], errors="coerce")
    pairs["avg_det_time"] = pd.to_numeric(pairs["avg_det_time"], errors="coerce")
    pairs = pairs.dropna().sort_values(by=["bal_acc", "avg_det_time"]).reset_index(drop=True)
    if pairs.empty or len(pairs) < 2:
        return np.nan

    grouped = pairs.groupby("bal_acc", as_index=False)["avg_det_time"].min().sort_values(by="bal_acc")
    bal_accs = grouped["bal_acc"].to_numpy(dtype=float)
    det_times = grouped["avg_det_time"].to_numpy(dtype=float)
    if target_bal_acc < bal_accs.min() or target_bal_acc > bal_accs.max():
        return np.nan
    return float(np.interp(float(target_bal_acc), bal_accs, det_times))


def _load_run_to_raw_path(method_dir: Path) -> dict[str, str]:
    val_summary_df = _load_csv(method_dir / "val_summary.csv")
    mapping = {}
    for row in val_summary_df.to_dict(orient="records"):
        run_name = row.get("run_name")
        raw_path = row.get("new_raw_path")
        if pd.notna(run_name) and pd.notna(raw_path):
            mapping[str(run_name)] = str(raw_path)
    return mapping


def compute_sample_targets(
    sampled_df: pd.DataFrame,
    methods_dir: Path,
    target_bal_accs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sampled_df.empty:
        empty = sampled_df.copy()
        for target in target_bal_accs:
            empty[f"mean_t_at_balacc_{_safe_float_tag(target)}"] = np.nan
            empty[f"std_t_at_balacc_{_safe_float_tag(target)}"] = np.nan
            empty[f"num_valid_t_at_balacc_{_safe_float_tag(target)}"] = 0
        return empty, pd.DataFrame(columns=[
            "method", "weight_key", "mode", "seed_label", "seed_run", "delta", "target_bal_acc", "t_at_balacc"
        ])

    run_path_cache: dict[str, dict[str, str]] = {}
    raw_cache: dict[tuple[str, str, float], pd.DataFrame] = {}
    seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for row in sampled_df.to_dict(orient="records"):
        method = str(row["method"])
        method_dir = methods_dir / method
        if method not in run_path_cache:
            run_path_cache[method] = _load_run_to_raw_path(method_dir)
        run_to_raw_path = run_path_cache[method]
        seed_runs = _parse_seed_runs(row["seed_runs"])
        per_target_values = {float(target): [] for target in target_bal_accs}

        for seed_label, seed_run in sorted(seed_runs.items()):
            raw_path = run_to_raw_path.get(str(seed_run))
            if raw_path is None or not os.path.isfile(raw_path):
                continue

            cache_key = (raw_path, str(row["mode"]), float(row["delta"]))
            if cache_key not in raw_cache:
                raw_df = pd.read_csv(raw_path)
                raw_df = raw_df.loc[raw_df["mode"].astype(str) == str(row["mode"])].copy()
                raw_df = raw_df.loc[np.isclose(pd.to_numeric(raw_df["delta"], errors="coerce"), float(row["delta"]))].copy()
                raw_cache[cache_key] = raw_df
            raw_df = raw_cache[cache_key]

            for target_bal_acc in target_bal_accs:
                t_value = _interp_t_at_balacc_from_raw(raw_df, float(target_bal_acc))
                seed_rows.append({
                    "method": method,
                    "weight_key": str(row["weight_key"]),
                    "mode": str(row["mode"]),
                    "seed_label": str(seed_label),
                    "seed_run": str(seed_run),
                    "delta": float(row["delta"]),
                    "target_bal_acc": float(target_bal_acc),
                    "t_at_balacc": t_value,
                })
                if np.isfinite(t_value):
                    per_target_values[float(target_bal_acc)].append(float(t_value))

        out_row = dict(row)
        for target_bal_acc in target_bal_accs:
            values = np.asarray(per_target_values[float(target_bal_acc)], dtype=float)
            out_row[f"mean_t_at_balacc_{_safe_float_tag(target_bal_acc)}"] = (
                float(values.mean()) if values.size else np.nan
            )
            out_row[f"std_t_at_balacc_{_safe_float_tag(target_bal_acc)}"] = (
                float(values.std(ddof=0)) if values.size else np.nan
            )
            out_row[f"num_valid_t_at_balacc_{_safe_float_tag(target_bal_acc)}"] = int(values.size)
        aggregate_rows.append(out_row)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(by=["method", "weight_key", "mode"]).reset_index(drop=True)
    seed_df = pd.DataFrame(seed_rows)
    if not seed_df.empty:
        seed_df = seed_df.sort_values(by=["mode", "target_bal_acc", "method", "weight_key", "seed_label"]).reset_index(drop=True)
    return aggregate_df, seed_df


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method="pearson"))


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    xr = frame.iloc[:, 0].rank(method="average")
    yr = frame.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def summarize_metric_correlations(
    sampled_targets_df: pd.DataFrame,
    target_bal_accs: list[float],
) -> pd.DataFrame:
    rows = []
    for mode in DEFAULT_MODES:
        mode_df = sampled_targets_df.loc[sampled_targets_df["mode"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        for target_bal_acc in target_bal_accs:
            target_col = f"mean_t_at_balacc_{_safe_float_tag(target_bal_acc)}"
            if target_col not in mode_df.columns:
                continue
            target_series = pd.to_numeric(mode_df[target_col], errors="coerce")
            target_quality = -target_series
            for metric_col, metric_label, higher_is_better in METRIC_SPECS:
                metric_series = pd.to_numeric(mode_df[metric_col], errors="coerce")
                metric_quality = metric_series if higher_is_better else -metric_series
                valid = pd.concat([metric_quality, target_quality], axis=1).dropna()
                rows.append({
                    "mode": mode,
                    "target_bal_acc": float(target_bal_acc),
                    "metric": metric_col,
                    "metric_label": metric_label,
                    "num_points": int(len(valid)),
                    "spearman": _spearman_corr(metric_quality, target_quality),
                    "pearson": _pearson_corr(metric_quality, target_quality),
                })

    if not rows:
        return pd.DataFrame(columns=["mode", "target_bal_acc", "metric", "metric_label", "num_points", "spearman", "pearson"])
    return pd.DataFrame(rows).sort_values(by=["mode", "target_bal_acc", "metric"]).reset_index(drop=True)


def _plot_metric_scatter_grid(
    plot_df: pd.DataFrame,
    target_bal_acc: float,
    mode: str,
    save_path: Path,
) -> None:
    fig = Figure(figsize=(15, 4.5))
    axes = fig.subplots(1, 3)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    target_col = f"mean_t_at_balacc_{_safe_float_tag(target_bal_acc)}"
    plot_df = plot_df.loc[np.isfinite(pd.to_numeric(plot_df[target_col], errors="coerce"))].copy()
    if plot_df.empty:
        return

    for ax, (metric_col, metric_label, _) in zip(axes, METRIC_SPECS):
        metric_values = pd.to_numeric(plot_df[metric_col], errors="coerce")
        target_values = pd.to_numeric(plot_df[target_col], errors="coerce")
        valid_df = plot_df.loc[np.isfinite(metric_values) & np.isfinite(target_values)].copy()
        if valid_df.empty:
            ax.set_title(metric_label)
            ax.grid(True, alpha=0.25)
            continue

        for method, method_df in valid_df.groupby("method", dropna=False):
            ax.scatter(
                pd.to_numeric(method_df[metric_col], errors="coerce"),
                pd.to_numeric(method_df[target_col], errors="coerce"),
                s=38,
                alpha=0.85,
                label=METHOD_LABELS.get(str(method), str(method)),
                color=METHOD_COLORS.get(str(method), "#1f77b4"),
                edgecolors="white",
                linewidths=0.5,
            )

        ax.set_title(metric_label)
        ax.set_xlabel(metric_label)
        ax.set_ylabel(f"Val t@BalAcc={float(target_bal_acc):.2f}")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(f"Randomly sampled 3-seed groups: {mode}, target BalAcc={float(target_bal_acc):.2f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def save_scatter_plots(
    sampled_targets_df: pd.DataFrame,
    save_dir: Path,
    target_bal_accs: list[float],
) -> dict[str, str]:
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if sampled_targets_df.empty:
        return outputs

    for mode in DEFAULT_MODES:
        mode_df = sampled_targets_df.loc[sampled_targets_df["mode"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        for target_bal_acc in target_bal_accs:
            save_path = plot_dir / f"sampled_metric_scatter_{mode}_{_safe_float_tag(target_bal_acc)}.png"
            _plot_metric_scatter_grid(
                plot_df=mode_df,
                target_bal_acc=float(target_bal_acc),
                mode=mode,
                save_path=save_path,
            )
            outputs[f"scatter_{mode}_{_safe_float_tag(target_bal_acc)}"] = str(save_path)
    return outputs


def build_correlation_latex(corr_df: pd.DataFrame, dataset_label: str) -> str:
    if corr_df.empty:
        return ""

    targets = sorted(corr_df["target_bal_acc"].dropna().astype(float).unique().tolist())
    lines = [
        r"\begin{table*}[htbp]",
        (
            rf"\caption{{Spearman correlation between validation metrics and validation $t@\mathrm{{BalAcc}}$ "
            rf"on randomly sampled 3-seed groups from {latex_escape(dataset_label)}. Larger is better because both "
            r"metric and target are sign-aligned to represent better early detection.}}"
        ),
        r"\begin{center}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|l|l|" + "c|" * len(targets) + r"}",
        r"\hline",
        r"\textbf{Mode} & \textbf{Metric} & " + " & ".join(
            rf"\textbf{{$t@\mathrm{{BalAcc}}={float(target):.2f}$}}" for target in targets
        ) + r" \\",
        r"\hline",
    ]

    for mode in DEFAULT_MODES:
        mode_df = corr_df.loc[corr_df["mode"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        for metric_col, metric_label, _ in METRIC_SPECS:
            row_df = mode_df.loc[mode_df["metric"].astype(str) == metric_col].copy()
            cells = [latex_escape(mode), latex_escape(metric_label)]
            for target in targets:
                hit = row_df.loc[np.isclose(row_df["target_bal_acc"].astype(float), float(target))]
                if hit.empty or not np.isfinite(float(hit["spearman"].iloc[0])):
                    cells.append("--")
                else:
                    cells.append(f"{float(hit['spearman'].iloc[0]):.4f}")
            lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample complete 3-seed weight groups from pipeline_val_new/methods, "
            "then compare val ROC/PRC/Integral against fixed val t@bal_acc targets."
        )
    )
    parser.add_argument(
        "methods_dir",
        help="Directory like log_ckpt_new/pizero_fast/pipeline_val_new/methods",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory. Defaults to <methods_dir>/sampled_metric_eval.",
    )
    parser.add_argument(
        "--samples-per-method",
        type=int,
        default=8,
        help="How many weight_key groups to sample per method.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for sampling weight groups.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=3,
        help="Minimum number of seeds required for a sampled weight group.",
    )
    parser.add_argument(
        "--target-bal-acc",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_BAL_ACCS),
        help="Fixed BalAcc targets to evaluate.",
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
        help="Method name to exclude. Can be repeated. Defaults to trans.",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Optional dataset label for captions.",
    )
    args = parser.parse_args()

    methods_dir = Path(args.methods_dir).resolve()
    if not methods_dir.is_dir():
        raise SystemExit(f"methods_dir is not a directory: {methods_dir}")

    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir
        else (methods_dir / "sampled_metric_eval").resolve()
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    target_bal_accs = sorted({round(float(value), 6) for value in args.target_bal_acc})
    modes = tuple(dict.fromkeys(args.mode)) if args.mode else DEFAULT_MODES
    exclude_methods = {str(value) for value in args.exclude_method if str(value).strip()}
    dataset_label = args.dataset_label or infer_dataset_label(methods_dir)

    candidate_df = _build_candidate_rows(
        methods_dir=methods_dir,
        exclude_methods=exclude_methods,
        min_seeds=int(args.min_seeds),
        modes=modes,
    )
    sampled_df = _sample_candidates(
        candidate_df=candidate_df,
        samples_per_method=int(args.samples_per_method),
        random_seed=int(args.random_seed),
    )
    sampled_targets_df, seed_target_df = compute_sample_targets(
        sampled_df=sampled_df,
        methods_dir=methods_dir,
        target_bal_accs=target_bal_accs,
    )
    corr_df = summarize_metric_correlations(
        sampled_targets_df=sampled_targets_df,
        target_bal_accs=target_bal_accs,
    )

    candidate_path = save_dir / "eligible_candidates.csv"
    sampled_path = save_dir / "sampled_candidates.csv"
    sampled_targets_path = save_dir / "sampled_candidates_with_targets.csv"
    seed_target_path = save_dir / "sampled_seed_targets.csv"
    corr_path = save_dir / "metric_target_correlations.csv"
    candidate_df.to_csv(candidate_path, index=False)
    sampled_df.to_csv(sampled_path, index=False)
    sampled_targets_df.to_csv(sampled_targets_path, index=False)
    seed_target_df.to_csv(seed_target_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    outputs = {
        "methods_dir": str(methods_dir),
        "save_dir": str(save_dir),
        "dataset_label": dataset_label,
        "exclude_methods": sorted(exclude_methods),
        "modes": list(modes),
        "target_bal_accs": [float(value) for value in target_bal_accs],
        "samples_per_method": int(args.samples_per_method),
        "random_seed": int(args.random_seed),
        "min_seeds": int(args.min_seeds),
        "eligible_candidates_csv": str(candidate_path),
        "sampled_candidates_csv": str(sampled_path),
        "sampled_candidates_with_targets_csv": str(sampled_targets_path),
        "sampled_seed_targets_csv": str(seed_target_path),
        "metric_target_correlations_csv": str(corr_path),
        "num_eligible_rows": int(len(candidate_df)),
        "num_sampled_rows": int(len(sampled_df)),
    }

    outputs.update(save_scatter_plots(sampled_targets_df, save_dir, target_bal_accs))
    latex = build_correlation_latex(corr_df, dataset_label)
    if latex:
        latex_path = save_dir / "metric_target_correlations.tex"
        latex_path.write_text(latex)
        outputs["metric_target_correlations_tex"] = str(latex_path)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
