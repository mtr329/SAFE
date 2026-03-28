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

from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from failure_prob.mrefine.delay_summary import (  # noqa: E402
    _get_pareto_curve_from_alpha_dict,
    _interp_det_time_on_pareto_for_bal_acc,
)
from failure_prob.mrefine.new_summary import _split_new_logs_by_method  # noqa: E402


SELECTION_STRATEGIES = [
    ("roc_auc", "Val ROC", "#1f77b4"),
    ("prc_auc", "Val PRC", "#ff7f0e"),
    ("pareto", "Val Integral", "#2ca02c"),
]
SELECTION_LABELS = {key: label for key, label, _ in SELECTION_STRATEGIES}
SELECTION_COLORS = {key: color for key, _, color in SELECTION_STRATEGIES}

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

DEFAULT_TARGET_BAL_ACCS = (0.60, 0.65, 0.70)
DEFAULT_MODES = ("early", "last")


def infer_dataset_label(summary_dir: Path) -> str:
    path_str = str(summary_dir)
    if "/pizero_fast/" in path_str:
        return "PiZero-Fast LIBERO"
    if "/pizero/" in path_str:
        return "PiZero LIBERO"
    return summary_dir.parent.name.replace("_", " ").title()


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


def _safe_float_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _metric_file(summary_dir: Path, strategy: str) -> Path:
    return summary_dir / strategy / "best_test_metrics_by_seed.csv"


def _load_strategy_frames(summary_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for strategy, _, _ in SELECTION_STRATEGIES:
        csv_path = _metric_file(summary_dir, strategy)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing strategy metrics file: {csv_path}")
        frames[strategy] = pd.read_csv(csv_path)
    return frames


def _find_float_key(data: dict, target: float) -> tuple[str | None, dict | None]:
    for key, value in data.items():
        if np.isclose(float(key), float(target)):
            return key, value
    return None, None


def _mean_alpha_dict(alpha_dict: dict) -> dict:
    mean_alpha_dict = {}
    for alpha, metrics in alpha_dict.items():
        mean_alpha_dict[alpha] = {}
        for key, value in metrics.items():
            if key == "detect_method":
                mean_alpha_dict[alpha][key] = value
                continue
            arr = np.asarray(value, dtype=float)
            mean_alpha_dict[alpha][key] = float(arr.mean()) if arr.size else np.nan
    return mean_alpha_dict


def _load_method_new_logs(run_dir: str, method: str, cache: dict[tuple[str, str], dict]) -> dict:
    cache_key = (run_dir, method)
    if cache_key in cache:
        return cache[cache_key]

    eval_dir = Path(run_dir) / "eval"
    candidates = [
        eval_dir / "new_logs.json",
        eval_dir / "my_logs.json",
        eval_dir / "mylogs.json",
    ]
    log_path = next((path for path in candidates if path.is_file()), None)
    if log_path is None:
        raise FileNotFoundError(f"Missing new eval logs under {eval_dir}")

    with log_path.open("r") as f:
        logs = json.load(f)
    new_logs = logs if log_path.name == "new_logs.json" else logs.get("new")
    if not isinstance(new_logs, dict):
        raise KeyError(f"Missing 'new' logs in {log_path}")

    split_logs = _split_new_logs_by_method(new_logs, fallback_method_name=method)
    if method not in split_logs:
        raise KeyError(f"Method {method} missing from {log_path}")

    cache[cache_key] = split_logs[method]
    return cache[cache_key]


def _iter_selected_records(strategy: str, df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    if strategy in {"roc_auc", "prc_auc"}:
        for row in df.itertuples(index=False):
            for mode in DEFAULT_MODES:
                delta_col = f"test_delta_{mode}"
                delta = getattr(row, delta_col, np.nan)
                records.append({
                    "selection_strategy": strategy,
                    "method": getattr(row, "method"),
                    "seed": getattr(row, "seed", np.nan),
                    "run_name": getattr(row, "run_name", ""),
                    "run_dir": getattr(row, "run_dir", ""),
                    "mode": mode,
                    "delta": delta,
                })
        return records

    for row in df.itertuples(index=False):
        records.append({
            "selection_strategy": strategy,
            "method": getattr(row, "method"),
            "seed": getattr(row, "seed", np.nan),
            "run_name": getattr(row, "run_name", ""),
            "run_dir": getattr(row, "run_dir", ""),
            "mode": getattr(row, "mode", ""),
            "delta": getattr(row, "delta", np.nan),
        })
    return records


def compute_t_at_balacc_summary(
    summary_dir: Path,
    target_bal_accs: list[float],
    modes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = _load_strategy_frames(summary_dir)
    log_cache: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    failures: list[dict] = []

    for strategy, df in frames.items():
        for record in _iter_selected_records(strategy, df):
            mode = str(record["mode"])
            if mode not in modes:
                continue

            delta = pd.to_numeric(pd.Series([record["delta"]]), errors="coerce").iloc[0]
            if not np.isfinite(delta):
                for target_bal_acc in target_bal_accs:
                    rows.append({
                        **record,
                        "target_bal_acc": float(target_bal_acc),
                        "matched_delta_key": None,
                        "num_pareto_points": 0,
                        "t_at_balacc": np.nan,
                        "missing_curve": True,
                    })
                continue

            try:
                method_logs = _load_method_new_logs(str(record["run_dir"]), str(record["method"]), log_cache)
                calib_logs = method_logs.get("calib", {}).get(mode, {})
                matched_delta_key, alpha_dict = _find_float_key(calib_logs, float(delta))
                if alpha_dict is None:
                    raise KeyError(
                        f"delta={float(delta):.4f} missing for mode={mode} in run={record['run_dir']}"
                    )

                pareto_det_times, pareto_bal_accs = _get_pareto_curve_from_alpha_dict(
                    _mean_alpha_dict(alpha_dict)
                )
                for target_bal_acc in target_bal_accs:
                    rows.append({
                        **record,
                        "target_bal_acc": float(target_bal_acc),
                        "matched_delta_key": matched_delta_key,
                        "num_pareto_points": int(len(pareto_det_times)),
                        "t_at_balacc": _interp_det_time_on_pareto_for_bal_acc(
                            pareto_det_times,
                            pareto_bal_accs,
                            float(target_bal_acc),
                        ),
                        "missing_curve": False,
                    })
            except Exception as exc:
                failures.append({
                    **record,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                for target_bal_acc in target_bal_accs:
                    rows.append({
                        **record,
                        "target_bal_acc": float(target_bal_acc),
                        "matched_delta_key": None,
                        "num_pareto_points": 0,
                        "t_at_balacc": np.nan,
                        "missing_curve": True,
                    })

    by_seed_df = pd.DataFrame(rows)
    if by_seed_df.empty:
        empty = pd.DataFrame(columns=[
            "selection_strategy",
            "method",
            "seed",
            "run_name",
            "run_dir",
            "mode",
            "delta",
            "target_bal_acc",
            "matched_delta_key",
            "num_pareto_points",
            "t_at_balacc",
            "missing_curve",
        ])
        failure_df = pd.DataFrame(columns=[
            "selection_strategy", "method", "seed", "run_name", "run_dir", "mode", "delta", "error_type", "error",
        ])
        agg_df = pd.DataFrame(columns=[
            "selection_strategy", "method", "mode", "target_bal_acc", "num_runs", "mean_t_at_balacc", "std_t_at_balacc",
        ])
        return empty, agg_df, failure_df

    by_seed_df = by_seed_df.sort_values(
        by=["mode", "target_bal_acc", "method", "selection_strategy", "seed", "run_name"],
        na_position="last",
    ).reset_index(drop=True)

    agg_rows = []
    for (strategy, method, mode, target_bal_acc), group in by_seed_df.groupby(
        ["selection_strategy", "method", "mode", "target_bal_acc"],
        dropna=False,
    ):
        values = pd.to_numeric(group["t_at_balacc"], errors="coerce")
        agg_rows.append({
            "selection_strategy": strategy,
            "method": method,
            "mode": mode,
            "target_bal_acc": float(target_bal_acc),
            "num_runs": int(values.notna().sum()),
            "mean_t_at_balacc": float(values.mean()),
            "std_t_at_balacc": float(values.std(ddof=0)),
        })

    agg_df = pd.DataFrame(agg_rows).sort_values(
        by=["mode", "target_bal_acc", "method", "selection_strategy"],
    ).reset_index(drop=True)
    failure_df = pd.DataFrame(failures)
    if not failure_df.empty:
        failure_df = failure_df.sort_values(
            by=["mode", "method", "selection_strategy", "seed", "run_name"],
            na_position="last",
        ).reset_index(drop=True)
    return by_seed_df, agg_df, failure_df


def _save_grouped_barplot(
    plot_df: pd.DataFrame,
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    methods = sorted(plot_df["method"].dropna().astype(str).unique().tolist())
    if not methods:
        return

    width = 0.24
    x = np.arange(len(methods), dtype=float)
    fig_width = max(9.0, 1.0 * len(methods) + 2.0)
    fig = Figure(figsize=(fig_width, 5.5))
    ax = fig.subplots()

    for idx, (strategy, label, color) in enumerate(SELECTION_STRATEGIES):
        offsets = x + (idx - 1) * width
        heights = []
        errors = []
        positions = []
        for method, xpos in zip(methods, offsets):
            row = plot_df.loc[
                (plot_df["method"].astype(str) == method)
                & (plot_df["selection_strategy"].astype(str) == strategy)
            ]
            if row.empty:
                continue
            mean = float(row["mean_t_at_balacc"].iloc[0])
            if not np.isfinite(mean):
                continue
            std = float(row["std_t_at_balacc"].iloc[0]) if "std_t_at_balacc" in row.columns else 0.0
            heights.append(mean)
            errors.append(0.0 if not np.isfinite(std) else std)
            positions.append(xpos)

        if positions:
            ax.bar(
                positions,
                heights,
                width=width,
                label=label,
                color=color,
                alpha=0.90,
                yerr=errors,
                capsize=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in methods], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def save_plots(
    agg_df: pd.DataFrame,
    save_dir: Path,
) -> dict[str, str]:
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if agg_df.empty:
        return outputs

    for mode in DEFAULT_MODES:
        mode_df = agg_df.loc[agg_df["mode"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        targets = sorted(mode_df["target_bal_acc"].dropna().astype(float).unique().tolist())
        for target_bal_acc in targets:
            target_df = mode_df.loc[np.isclose(mode_df["target_bal_acc"].astype(float), float(target_bal_acc))].copy()
            if target_df.empty:
                continue
            save_path = plot_dir / f"t_at_balacc_{_safe_float_tag(target_bal_acc)}_{mode}_by_val_selection.png"
            _save_grouped_barplot(
                target_df,
                save_path=save_path,
                title=f"Test T@BalAcc={float(target_bal_acc):.2f} By Val Selection ({mode})",
                ylabel="Test T@BalAcc",
            )
            outputs[f"t_at_balacc_{_safe_float_tag(target_bal_acc)}_{mode}"] = str(save_path)

    return outputs


def format_value(mean: float, highlight: str | None) -> str:
    if not np.isfinite(mean):
        return "--"
    mean_text = f"{mean:.4f}"
    if highlight == "best":
        return rf"\textcolor{{red}}{{\textbf{{{mean_text}}}\textsuperscript{{*}}}}"
    if highlight == "second":
        return rf"\textcolor[rgb]{{1.0,0.5,0.0}}{{\textbf{{{mean_text}}}\textsuperscript{{**}}}}"
    return mean_text


def build_mode_latex_table(
    agg_df: pd.DataFrame,
    mode: str,
    dataset_label: str,
    targets: list[float],
) -> str:
    mode_df = agg_df.loc[agg_df["mode"].astype(str) == str(mode)].copy()
    methods = sorted(mode_df["method"].dropna().astype(str).unique().tolist())
    safe_dataset_tag = re.sub(r"[^a-z0-9]+", "_", dataset_label.lower()).strip("_")
    latex_label = f"tab:t_at_balacc_{mode}_{safe_dataset_tag}"

    lines = [
        r"\begin{table*}[htbp]",
        (
            rf"\caption{{Test-set $t@\mathrm{{BalAcc}}$ on {latex_escape(dataset_label)} "
            rf"for the {latex_escape(mode)} setting. Lower is better. Within each method and target BalAcc block, "
            r"red bold values with \textsuperscript{*} denote the best validation-based selection strategy; "
            r"orange bold values with \textsuperscript{**} denote the second-best.}}"
        ),
        rf"\label{{{latex_label}}}",
        r"\begin{center}",
        r"\resizebox{\textwidth}{!}{%",
    ]

    col_spec = "|l|" + "|".join("c" * len(SELECTION_STRATEGIES) for _ in targets) + "|"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    header_1 = [r"\textbf{Method}"]
    for target in targets:
        header_1.append(
            rf"\multicolumn{{3}}{{|c|}}{{\textbf{{$t@\mathrm{{BalAcc}}={float(target):.2f}$}}}}"
        )
    lines.append(" & ".join(header_1) + r" \\")
    num_cols = 1 + 3 * len(targets)
    lines.append(rf"\cline{{2-{num_cols}}}")
    header_2 = [" "]
    for _ in targets:
        for strategy, label, _ in SELECTION_STRATEGIES:
            del strategy
            header_2.append(rf"\textbf{{\textit{{{latex_escape(label)}}}}}")
    lines.append(" & ".join(header_2) + r" \\")
    lines.append(r"\hline")

    for method in methods:
        method_df = mode_df.loc[mode_df["method"].astype(str) == method].copy()
        cells = [latex_escape(METHOD_LABELS.get(method, method))]
        for target in targets:
            target_df = method_df.loc[np.isclose(method_df["target_bal_acc"].astype(float), float(target))].copy()
            ranked = []
            strategy_to_mean = {}
            for strategy, _, _ in SELECTION_STRATEGIES:
                row = target_df.loc[target_df["selection_strategy"].astype(str) == strategy]
                mean = float(row["mean_t_at_balacc"].iloc[0]) if not row.empty else np.nan
                strategy_to_mean[strategy] = mean
                if np.isfinite(mean):
                    ranked.append(mean)
            ranked = sorted(set(ranked))
            for strategy, _, _ in SELECTION_STRATEGIES:
                mean = strategy_to_mean.get(strategy, np.nan)
                highlight = None
                if np.isfinite(mean) and ranked:
                    if abs(mean - ranked[0]) < 1e-12:
                        highlight = "best"
                    elif len(ranked) > 1 and abs(mean - ranked[1]) < 1e-12:
                        highlight = "second"
                cells.append(format_value(mean, highlight))
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def save_latex_tables(
    agg_df: pd.DataFrame,
    save_dir: Path,
    dataset_label: str,
) -> dict[str, str]:
    latex_dir = save_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if agg_df.empty:
        return outputs

    for mode in DEFAULT_MODES:
        mode_df = agg_df.loc[agg_df["mode"].astype(str) == mode].copy()
        if mode_df.empty:
            continue
        targets = sorted(mode_df["target_bal_acc"].dropna().astype(float).unique().tolist())
        if not targets:
            continue
        latex = build_mode_latex_table(
            agg_df=agg_df,
            mode=mode,
            dataset_label=dataset_label,
            targets=targets,
        )
        save_path = latex_dir / f"selection_t_at_balacc_{mode}.tex"
        save_path.write_text(latex)
        outputs[f"latex_{mode}"] = str(save_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and visualize fixed-BalAcc t@bal_acc metrics for pipeline_test_new "
            "selection strategies (Val ROC / Val PRC / Val Integral)."
        )
    )
    parser.add_argument(
        "summary_dir",
        help="Path to pipeline_test_new output directory.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory. Defaults to <summary_dir>/selection_t_at_balacc.",
    )
    parser.add_argument(
        "--target-bal-acc",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_BAL_ACCS),
        help="Fixed BalAcc targets to evaluate. Example: --target-bal-acc 0.6 0.65 0.7",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Mode to keep. Can be repeated. Defaults to early and last.",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Optional dataset label for LaTeX captions.",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    if not summary_dir.is_dir():
        raise SystemExit(f"summary_dir is not a directory: {summary_dir}")
    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir
        else (summary_dir / "selection_t_at_balacc").resolve()
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    target_bal_accs = sorted({round(float(value), 6) for value in args.target_bal_acc})
    modes = tuple(dict.fromkeys(args.mode)) if args.mode else DEFAULT_MODES
    dataset_label = args.dataset_label or infer_dataset_label(summary_dir)

    by_seed_df, agg_df, failure_df = compute_t_at_balacc_summary(
        summary_dir=summary_dir,
        target_bal_accs=target_bal_accs,
        modes=modes,
    )

    by_seed_path = save_dir / "selection_t_at_balacc_by_seed.csv"
    agg_path = save_dir / "selection_t_at_balacc_summary.csv"
    by_seed_df.to_csv(by_seed_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    outputs = {
        "summary_dir": str(summary_dir),
        "save_dir": str(save_dir),
        "dataset_label": dataset_label,
        "target_bal_accs": [float(value) for value in target_bal_accs],
        "modes": list(modes),
        "by_seed_csv": str(by_seed_path),
        "summary_csv": str(agg_path),
        "num_rows": int(len(by_seed_df)),
    }

    if not failure_df.empty:
        failure_path = save_dir / "selection_t_at_balacc_failures.csv"
        failure_df.to_csv(failure_path, index=False)
        outputs["failures_csv"] = str(failure_path)
        outputs["num_failures"] = int(len(failure_df))
    else:
        outputs["num_failures"] = 0

    outputs.update(save_plots(agg_df, save_dir))
    outputs.update(save_latex_tables(agg_df, save_dir, dataset_label))

    summary_json_path = save_dir / "summary.json"
    summary_json_path.write_text(json.dumps(outputs, indent=2))
    print(summary_json_path)


if __name__ == "__main__":
    main()
