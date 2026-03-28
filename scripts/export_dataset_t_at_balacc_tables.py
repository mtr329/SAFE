from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_topk_per_seed_balacc_at_t import (
    SELECTION_SPECS,
    _safe_float_tag,
    latex_escape,
)


DATASET_LABELS = {
    "pizero-fast_libero": "PiZero-Fast LIBERO",
    "pizero_libero": "PiZero LIBERO",
}


def infer_dataset_label(dataset_dir: Path) -> str:
    return DATASET_LABELS.get(dataset_dir.name, dataset_dir.name.replace("_", " ").title())


def load_overall_summary(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "selected_topk_summary_mean_over_methods.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    return pd.read_csv(path)


def format_time(value: float, is_best: bool) -> str:
    if not np.isfinite(value):
        return "--"
    text = f"{float(value):.4f}"
    return rf"\textbf{{{text}}}" if is_best else text


def build_dataset_table(
    overall_df: pd.DataFrame,
    dataset_label: str,
    top_k: int,
    target_bal_accs: list[float],
) -> str:
    value_map: dict[tuple[str, str, float], float] = {}
    for mode in ("early", "last"):
        mode_df = overall_df.loc[overall_df["mode"].astype(str) == mode].copy()
        for selection_metric, _, _ in SELECTION_SPECS:
            row = mode_df.loc[mode_df["selection_metric"].astype(str) == selection_metric]
            if row.empty:
                continue
            row = row.iloc[0]
            for target in target_bal_accs:
                tag = _safe_float_tag(target)
                value_map[(mode, selection_metric, target)] = float(
                    row.get(f"mean_over_methods_best_t_at_balacc_{tag}", np.nan)
                )

    best_map: dict[tuple[str, float], float] = {}
    for mode in ("early", "last"):
        for target in target_bal_accs:
            candidates = [
                value_map.get((mode, selection_metric, target), np.nan)
                for selection_metric, _, _ in SELECTION_SPECS
            ]
            finite = [value for value in candidates if np.isfinite(value)]
            best_map[(mode, target)] = min(finite) if finite else np.nan

    column_spec = "|l|" + "c|" * (2 * len(target_bal_accs))
    lines = [
        r"\begin{table}[htbp]",
        (
            rf"\caption{{{latex_escape(dataset_label)}: mean detection time to reach target BalAcc, "
            rf"averaged over seeds and then averaged over methods for per-seed top-{int(top_k)} selections. "
            r"Lower is better. Bold indicates the best value in each column.}"
        ),
        r"\begin{center}",
        r"\resizebox{0.92\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        r"\textbf{Selection Metric} & \multicolumn{3}{c|}{\textbf{Early}} & \multicolumn{3}{c|}{\textbf{Last}} \\",
        r"\cline{2-7}",
        r" & " + " & ".join(
            [rf"\textbf{{$t@\mathrm{{BalAcc}}={float(target):.2f}$}}" for target in target_bal_accs] * 2
        ) + r" \\",
        r"\hline",
    ]

    for selection_metric, label, _ in SELECTION_SPECS:
        cells = [latex_escape(label)]
        for mode in ("early", "last"):
            for target in target_bal_accs:
                value = value_map.get((mode, selection_metric, target), np.nan)
                best_value = best_map.get((mode, target), np.nan)
                is_best = np.isfinite(value) and np.isfinite(best_value) and abs(value - best_value) <= 1e-12
                cells.append(format_time(value, is_best=is_best))
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export one LaTeX table per dataset for t@BalAcc summaries. "
            "Each table has Early and Last column groups and bolds the best value per column."
        )
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        help="Dataset output dirs under log_analysis_outputs/topk_per_seed_t_over_acc, e.g. pizero-fast_libero",
    )
    parser.add_argument(
        "--target-bal-acc",
        nargs="*",
        type=float,
        default=[0.60, 0.65, 0.70],
        help="Target BalAcc values to include.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k value used in the upstream summary, for caption text only.",
    )
    parser.add_argument(
        "--save-dir",
        default="log_analysis_outputs/topk_per_seed_t_over_acc/dataset_tables",
        help="Directory for combined dataset-level LaTeX tables.",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for dataset_arg in args.dataset_dirs:
        dataset_dir = Path(dataset_arg)
        overall_df = load_overall_summary(dataset_dir)
        dataset_label = infer_dataset_label(dataset_dir)
        latex = build_dataset_table(
            overall_df=overall_df,
            dataset_label=dataset_label,
            top_k=args.top_k,
            target_bal_accs=[float(x) for x in args.target_bal_acc],
        )
        out_path = save_dir / f"{dataset_dir.name}_t_at_balacc.tex"
        out_path.write_text(latex)
        print(out_path)


if __name__ == "__main__":
    main()
