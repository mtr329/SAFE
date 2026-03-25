import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _load_seed_curve(raw_path: str, mode: str, delta: float) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.loc[df["mode"] == mode].copy()
    if "delta" in df.columns:
        df = df.loc[np.isclose(df["delta"].astype(float), float(delta))].copy()
    if df.empty:
        return df
    return df.sort_values(by=["alpha", "avg_det_time", "bal_acc"]).reset_index(drop=True)


def _aggregate_seed_curves(seed_curve_rows: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(seed_curve_rows, ignore_index=True)
    grouped = (
        merged.groupby("alpha", as_index=False)
        .agg(
            mean_avg_det_time=("avg_det_time", "mean"),
            std_avg_det_time=("avg_det_time", "std"),
            mean_bal_acc=("bal_acc", "mean"),
            std_bal_acc=("bal_acc", "std"),
            num_seeds=("seed_run", "nunique"),
        )
        .sort_values(by=["alpha"])
        .reset_index(drop=True)
    )
    for col in ("std_avg_det_time", "std_bal_acc"):
        grouped[col] = grouped[col].fillna(0.0)
    return grouped


def _plot_method_curve(
    method: str,
    mode: str,
    seed_df: pd.DataFrame,
    mean_df: pd.DataFrame,
    output_path: Path,
    delta: float,
    weight_key: str,
) -> None:
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()

    for _, curve_df in seed_df.groupby("seed_run", sort=True):
        ax.plot(
            curve_df["avg_det_time"].to_numpy(dtype=float),
            curve_df["bal_acc"].to_numpy(dtype=float),
            color="#bdbdbd",
            linewidth=1.25,
            alpha=0.7,
        )

    ax.plot(
        mean_df["mean_avg_det_time"].to_numpy(dtype=float),
        mean_df["mean_bal_acc"].to_numpy(dtype=float),
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
    )

    ax.set_xlabel("Average Detection Time")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{method}: val bal_acc vs det_time ({mode})")
    ax.text(
        0.02,
        0.02,
        f"delta={delta:.3g}\nweight={weight_key}",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def _plot_combined_curves(mode: str, mean_rows: list[pd.DataFrame], output_path: Path) -> None:
    if not mean_rows:
        return

    fig = Figure(figsize=(8, 5.5))
    ax = fig.subplots()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, mean_df in enumerate(sorted(mean_rows, key=lambda df: str(df["method"].iloc[0]))):
        method = str(mean_df["method"].iloc[0])
        color = colors[idx % len(colors)]
        ax.plot(
            mean_df["mean_avg_det_time"].to_numpy(dtype=float),
            mean_df["mean_bal_acc"].to_numpy(dtype=float),
            label=method,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )

    ax.set_xlabel("Average Detection Time")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Validation bal_acc vs det_time ({mode})")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def _collect_method_curves(method_dir: Path) -> tuple[list[dict], list[str]]:
    val_summary_path = method_dir / "val_summary.csv"
    best_weights_path = method_dir / "pareto" / "best_weights.csv"
    issues: list[str] = []
    if not val_summary_path.is_file():
        issues.append(f"missing {val_summary_path}")
        return [], issues
    if not best_weights_path.is_file():
        issues.append(f"missing {best_weights_path}")
        return [], issues

    val_df = pd.read_csv(val_summary_path)
    if val_df.empty or "run_name" not in val_df.columns or "new_raw_path" not in val_df.columns:
        issues.append(f"invalid {val_summary_path}")
        return [], issues

    raw_path_by_run = {
        str(run_name): str(raw_path)
        for run_name, raw_path in zip(val_df["run_name"], val_df["new_raw_path"])
        if pd.notna(run_name) and pd.notna(raw_path)
    }

    best_df = pd.read_csv(best_weights_path)
    outputs = []
    for row in best_df.to_dict(orient="records"):
        mode = str(row.get("mode", "early"))
        weight_key = str(row.get("weight_key", ""))
        delta = float(row.get("delta", 0.0))
        seed_runs_raw = row.get("seed_runs")
        if pd.isna(seed_runs_raw):
            issues.append(f"{method_dir.name} {mode}: missing seed_runs")
            continue

        seed_run_map = json.loads(seed_runs_raw)
        seed_curve_rows = []
        for seed_label, run_name in sorted(seed_run_map.items()):
            raw_path = raw_path_by_run.get(str(run_name))
            if raw_path is None:
                issues.append(f"{method_dir.name} {mode}: run_name not found in val_summary.csv: {run_name}")
                continue
            if not os.path.isfile(raw_path):
                issues.append(f"{method_dir.name} {mode}: missing raw curve file: {raw_path}")
                continue

            curve_df = _load_seed_curve(raw_path, mode=mode, delta=delta)
            if curve_df.empty:
                issues.append(f"{method_dir.name} {mode}: empty curve for {run_name} delta={delta}")
                continue

            curve_df = curve_df[["alpha", "avg_det_time", "bal_acc"]].copy()
            curve_df["method"] = method_dir.name
            curve_df["mode"] = mode
            curve_df["seed_label"] = str(seed_label)
            curve_df["seed_run"] = str(run_name)
            curve_df["weight_key"] = weight_key
            curve_df["delta"] = delta
            seed_curve_rows.append(curve_df)

        if not seed_curve_rows:
            continue

        seed_df = pd.concat(seed_curve_rows, ignore_index=True)
        mean_df = _aggregate_seed_curves(seed_curve_rows)
        mean_df["method"] = method_dir.name
        mean_df["mode"] = mode
        mean_df["weight_key"] = weight_key
        mean_df["delta"] = delta
        outputs.append(
            {
                "method": method_dir.name,
                "mode": mode,
                "weight_key": weight_key,
                "delta": delta,
                "seed_df": seed_df,
                "mean_df": mean_df,
            }
        )

    return outputs, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot validation balanced-accuracy vs detection-time curves from pipeline_val_new/methods.",
    )
    parser.add_argument(
        "--methods-dir",
        required=True,
        help="Directory like log_ckpt_new/pizero_fast/pipeline_val_new/methods",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots and CSV summaries. Defaults to <methods-dir>/balacc_dettime_curves",
    )
    args = parser.parse_args()

    methods_dir = Path(args.methods_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else methods_dir / "balacc_dettime_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_method_dir = output_dir / "per_method"
    per_method_dir.mkdir(parents=True, exist_ok=True)

    all_seed_rows = []
    all_mean_rows = []
    issues = []

    entries_by_mode: dict[str, list[pd.DataFrame]] = {"early": [], "last": []}
    for method_dir in sorted(
        path
        for path in methods_dir.iterdir()
        if path.is_dir() and (path / "val_summary.csv").is_file()
    ):
        entries, method_issues = _collect_method_curves(method_dir)
        issues.extend(method_issues)
        for entry in entries:
            method = entry["method"]
            mode = entry["mode"]
            seed_df = entry["seed_df"]
            mean_df = entry["mean_df"]
            delta = float(entry["delta"])
            weight_key = str(entry["weight_key"])

            per_method_path = per_method_dir / f"{_safe_file_stem(method)}_balacc_dettime_{mode}.png"
            _plot_method_curve(
                method=method,
                mode=mode,
                seed_df=seed_df,
                mean_df=mean_df,
                output_path=per_method_path,
                delta=delta,
                weight_key=weight_key,
            )

            all_seed_rows.append(seed_df.copy())
            all_mean_rows.append(mean_df.copy())
            if mode in entries_by_mode:
                entries_by_mode[mode].append(mean_df.copy())

    if all_seed_rows:
        pd.concat(all_seed_rows, ignore_index=True).sort_values(
            by=["method", "mode", "seed_run", "alpha"]
        ).to_csv(output_dir / "curve_points_by_seed.csv", index=False)
    if all_mean_rows:
        pd.concat(all_mean_rows, ignore_index=True).sort_values(
            by=["method", "mode", "alpha"]
        ).to_csv(output_dir / "curve_points_mean.csv", index=False)

    for mode, mean_rows in entries_by_mode.items():
        if mean_rows:
            _plot_combined_curves(mode, mean_rows, output_dir / f"all_methods_balacc_dettime_{mode}.png")

    summary = {
        "methods_dir": str(methods_dir),
        "output_dir": str(output_dir),
        "num_methods": len({df["method"].iloc[0] for df in all_mean_rows}) if all_mean_rows else 0,
        "num_curves": len(all_mean_rows),
        "issues": issues,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
