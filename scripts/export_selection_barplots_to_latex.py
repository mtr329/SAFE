from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = [
    ("test_roc_auc", "ROC-AUC", True),
    ("test_prc_auc", "PRC-AUC", True),
    ("test_integral_t_at_balacc_early", "Integral@Early", False),
    ("test_integral_t_at_balacc_last", "Integral@Last", False),
]

SELECTION_STRATEGIES = [
    ("roc_auc", "Val ROC"),
    ("prc_auc", "Val PRC"),
    ("pareto", "Val Integral"),
]

SELECTION_LABELS = dict(SELECTION_STRATEGIES)

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


def infer_dataset_label(csv_path: Path) -> str:
    path_str = str(csv_path)
    if "/pizero_fast/" in path_str:
        return "PiZero-Fast LIBERO"
    if "/pizero/" in path_str:
        return "PiZero LIBERO"
    return csv_path.parent.parent.parent.name.replace("_", " ").title()


def build_method_strategy_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for method, method_df in df.groupby("method", sort=True):
        row: dict[str, object] = {"method": method}
        for metric_key, metric_label, _ in METRICS:
            metric_df = method_df[method_df["metric"] == metric_key].copy()
            metric_df["mean"] = pd.to_numeric(metric_df["mean"], errors="coerce")
            metric_df["std"] = pd.to_numeric(metric_df["std"], errors="coerce")
            for strategy_key, _ in SELECTION_STRATEGIES:
                strategy_df = metric_df[metric_df["selection_strategy"] == strategy_key]
                if strategy_df.empty:
                    row[f"{metric_label} {strategy_key} mean"] = float("nan")
                    row[f"{metric_label} {strategy_key} std"] = float("nan")
                    continue
                strategy_row = strategy_df.iloc[0]
                row[f"{metric_label} {strategy_key} mean"] = (
                    float(strategy_row["mean"]) if pd.notna(strategy_row["mean"]) else float("nan")
                )
                row[f"{metric_label} {strategy_key} std"] = (
                    float(strategy_row["std"]) if pd.notna(strategy_row["std"]) else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def format_value(mean: float, std: float, highlight: str | None) -> str:
    if pd.isna(mean):
        return "--"
    mean_text = f"{mean:.4f}"
    if highlight == "best":
        return rf"\textcolor{{red}}{{\textbf{{{mean_text}}}\textsuperscript{{*}}}}"
    if highlight == "second":
        return rf"\textcolor[rgb]{{1.0,0.5,0.0}}{{\textbf{{{mean_text}}}\textsuperscript{{**}}}}"
    return mean_text


def build_ieee_table_latex(csv_path: Path, dataset_label: str | None = None, table_label: str | None = None) -> str:
    table = build_method_strategy_table(csv_path)
    dataset_name = dataset_label or infer_dataset_label(csv_path)
    safe_dataset_tag = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    latex_label = table_label or f"tab:selection_summary_{safe_dataset_tag}"

    ranked_values_by_column: dict[tuple[str, str], list[float]] = {}
    for _, metric_label, higher_is_better in METRICS:
        for strategy_key, _ in SELECTION_STRATEGIES:
            series = pd.to_numeric(table[f"{metric_label} {strategy_key} mean"], errors="coerce").dropna()
            if series.empty:
                continue
            unique_values = sorted({float(value) for value in series.tolist()}, reverse=higher_is_better)
            ranked_values_by_column[(metric_label, strategy_key)] = unique_values

    lines = [
        r"\begin{table*}[htbp]",
        rf"\caption{{Test-set summary on {latex_escape(dataset_name)}. Each entry shows the mean under three validation-based model-selection criteria. Red bold values with \textsuperscript{{*}} denote the best result in each column; orange bold values with \textsuperscript{{**}} denote the second-best.}}",
        rf"\label{{{latex_label}}}",
        r"\begin{center}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Method} & \multicolumn{3}{|c|}{\textbf{ROC-AUC $\uparrow$}} & \multicolumn{3}{|c|}{\textbf{PRC-AUC $\uparrow$}} & \multicolumn{3}{|c|}{\textbf{Integral@Early $\downarrow$}} & \multicolumn{3}{|c|}{\textbf{Integral@Last $\downarrow$}} \\",
        r"\cline{2-13}",
        r" & \textbf{\textit{Val ROC}} & \textbf{\textit{Val PRC}} & \textbf{\textit{Val Integral}} & \textbf{\textit{Val ROC}} & \textbf{\textit{Val PRC}} & \textbf{\textit{Val Integral}} & \textbf{\textit{Val ROC}} & \textbf{\textit{Val PRC}} & \textbf{\textit{Val Integral}} & \textbf{\textit{Val ROC}} & \textbf{\textit{Val PRC}} & \textbf{\textit{Val Integral}} \\",
        r"\hline",
    ]

    for _, row in table.iterrows():
        cells = [latex_escape(METHOD_LABELS.get(str(row["method"]), str(row["method"])))]
        for _, metric_label, _ in METRICS:
            for strategy_key, _ in SELECTION_STRATEGIES:
                mean = row[f"{metric_label} {strategy_key} mean"]
                std = row[f"{metric_label} {strategy_key} std"]
                ranked_values = ranked_values_by_column.get((metric_label, strategy_key), [])
                highlight = None
                if pd.notna(mean) and ranked_values:
                    mean_value = float(mean)
                    if abs(mean_value - ranked_values[0]) < 1e-12:
                        highlight = "best"
                    elif len(ranked_values) > 1 and abs(mean_value - ranked_values[1]) < 1e-12:
                        highlight = "second"
                cells.append(format_value(float(mean), float(std), highlight=highlight))
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{center}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def export_table(csv_path: Path, out_path: Path, dataset_label: str | None = None) -> None:
    latex = build_ieee_table_latex(csv_path, dataset_label=dataset_label)
    out_path.write_text(latex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+", help="selection_barplot_summary.csv paths")
    parser.add_argument(
        "--dataset-labels",
        nargs="*",
        default=None,
        help="Optional dataset labels aligned with csv_paths.",
    )
    args = parser.parse_args()

    if args.dataset_labels is not None and len(args.dataset_labels) not in (0, len(args.csv_paths)):
        raise SystemExit("--dataset-labels must be omitted or have the same length as csv_paths")

    labels = args.dataset_labels or [None] * len(args.csv_paths)
    for csv_arg, dataset_label in zip(args.csv_paths, labels):
        csv_path = Path(csv_arg)
        out_path = csv_path.with_name("selection_barplot_ieee_table.tex")
        export_table(csv_path, out_path, dataset_label=dataset_label)
        print(out_path)


if __name__ == "__main__":
    main()
