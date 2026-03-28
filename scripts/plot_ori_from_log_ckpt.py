from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from failure_prob.mrefine.ori_summary import (
    _collect_ori_log_paths,
    _get_ori_figs,
    _get_run_meta_from_config,
    _ori_summary_to_df,
    _split_ori_logs_by_method,
    _summarize_method_runs,
)


TABLE_LABEL_TO_METHOD = {
    "Embed-Cosine": "embed_cosine",
    "Embed-Euclid": "embed_euclid",
    "Embed-Mahala": "embed_mahala",
    "Embed-PCAKMeans": "embed_pca_kmeans",
    "MLP": "indep",
    "LogPZO": "logpZO",
    "LSTM": "lstm",
    "RND": "rnd",
    "Trans": "trans",
}


def parse_methods_from_table(table_tex: Path) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for raw_line in table_tex.read_text().splitlines():
        line = raw_line.strip()
        if "&" not in line:
            continue
        first_cell = line.split("&", 1)[0].strip()
        if not first_cell or first_cell.startswith("\\"):
            continue
        method_name = TABLE_LABEL_TO_METHOD.get(first_cell)
        if method_name is None or method_name in seen:
            continue
        seen.add(method_name)
        methods.append(method_name)
    return methods


def summarize_filtered_ori_metrics(
    logs_dir: Path,
    save_dir: Path,
    include_methods: set[str],
    exclude_methods: set[str],
) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    include_methods = set(include_methods) - set(exclude_methods)
    ori_summary: dict[str, dict] = {}
    ori_logs_by_method: dict[str, dict[str, dict]] = {}

    for log_path_str in _collect_ori_log_paths(str(logs_dir)):
        log_path = Path(log_path_str)
        with log_path.open("r") as f:
            logs = json.load(f)

        ori_logs = logs if log_path.name == "ori_logs.json" else logs.get("ori")
        if not ori_logs:
            continue

        run_dir = log_path.parent.parent
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(str(run_dir))
        fallback_method_name = run_meta["method_name"]
        method_logs_by_name = _split_ori_logs_by_method(ori_logs, fallback_method_name)

        for method_name, method_logs in method_logs_by_name.items():
            if method_name not in include_methods:
                continue
            ori_logs_by_method.setdefault(method_name, {})
            ori_logs_by_method[method_name][run_name] = method_logs

    for method_name, runs_dict in ori_logs_by_method.items():
        ori_summary[method_name] = _summarize_method_runs(runs_dict)

    (save_dir / "selected_methods.json").write_text(json.dumps(sorted(include_methods), indent=2))
    (save_dir / "ori_summary.json").write_text(json.dumps(ori_summary, indent=2))
    _ori_summary_to_df(ori_summary).to_csv(save_dir / "ori_summary.csv", index=False)

    ori_figs = _get_ori_figs(ori_logs_by_method)
    for eval_time, fig in ori_figs.items():
        fig.savefig(save_dir / f"{eval_time}.png", dpi=400)

    return ori_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ori figures from log_ckpt using methods listed in a LaTeX selection table.",
    )
    parser.add_argument(
        "--logs-dir",
        default="log_ckpt",
        help="Root directory containing eval/ori_logs.json runs.",
    )
    parser.add_argument(
        "--table-tex",
        required=True,
        help="Path to selection_barplot_ieee_table.tex used to choose methods.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory. Defaults to <logs_dir>/summary/ori_from_table_methods.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=["trans"],
        help="Method name to exclude after parsing the table. Can be passed multiple times.",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    table_tex = Path(args.table_tex).resolve()
    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir
        else (logs_dir / "summary" / "ori_from_table_methods").resolve()
    )

    include_methods = set(parse_methods_from_table(table_tex))
    exclude_methods = set(args.exclude_method)
    kept_methods = sorted(include_methods - exclude_methods)
    print("Methods from table:", sorted(include_methods))
    print("Excluded methods:", sorted(exclude_methods))
    print("Kept methods:", kept_methods)

    summarize_filtered_ori_metrics(
        logs_dir=logs_dir,
        save_dir=save_dir,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )
    print("Saved outputs to", save_dir)


if __name__ == "__main__":
    main()
