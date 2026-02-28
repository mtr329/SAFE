# Process metrics from local W&B logs (no online API access)
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise ImportError("PyYAML is required to parse local W&B config.yaml") from e


WANDB_META_V2 = {
    "pi0fast_libero_v4": [
        # Pi0-FAST on the LIBERO benchmark
        {
            "project_name": "local",
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["lstm", "mlp", "trans"],
            "ablated_configs": [
                "model.name",
                "dataset.feat_name",
                "dataset.token_idx_rel",
                "model.lr",
                "model.lambda_reg",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["embed"],
            "ablated_configs": [
                "model.name",
                "dataset.feat_name",
                "dataset.token_idx_rel",
                "model.distance",
                "model.topk",
                "model.cumsum",
                "model.pca_dim",
                "model.n_clusters",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["chen"],
            "ablated_configs": [
                "model.name",
                "dataset.feat_name",
                "dataset.token_idx_rel",
                "model.use_success_only",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
    ],
    "openvla_libero_v2": [
        # OpenVLA on the LIBERO benchmark
        {
            "project_name": "local",
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["lstm", "mlp", "trans"],
            "ablated_configs": [
                "model.name",
                "dataset.token_idx_rel",
                "model.lr",
                "model.lambda_reg",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["embed"],
            "ablated_configs": [
                "model.name",
                "dataset.token_idx_rel",
                "model.distance",
                "model.topk",
                "model.cumsum",
                "model.pca_dim",
                "model.n_clusters",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["chen"],
            "ablated_configs": ["model.name", "dataset.token_idx_rel"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
    ],
    "pi0diff_libero_v1": [
        # Pi0 (diffusion version) on LIBERO
        {
            "project_name": "local",
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["lstm", "mlp", "trans"],
            "ablated_configs": [
                "model.name",
                "dataset.horizon_idx_rel",
                "dataset.diff_idx_rel",
                "model.lr",
                "model.lambda_reg",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["embed"],
            "ablated_configs": [
                "model.name",
                "dataset.horizon_idx_rel",
                "dataset.diff_idx_rel",
                "model.distance",
                "model.topk",
                "model.cumsum",
                "model.pca_dim",
                "model.n_clusters",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["chen"],
            "ablated_configs": [
                "model.name",
                "dataset.horizon_idx_rel",
                "dataset.diff_idx_rel",
            ],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
    ],
    "opi0_simpler_v1": [
        # open-pi-zero on SimplerEnv
        {
            "project_name": "local",
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["lstm", "mlp", "trans"],
            "ablated_configs": [
                "model.name",
                "dataset.horizon_idx_rel",
                "dataset.diff_idx_rel",
                "model.lr",
                "model.lambda_reg",
            ],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["embed"],
            "ablated_configs": [
                "model.name",
                "dataset.horizon_idx_rel",
                "dataset.diff_idx_rel",
                "model.distance",
                "model.topk",
                "model.cumsum",
                "model.pca_dim",
                "model.n_clusters",
            ],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["chen"],
            "ablated_configs": [
                "model.name",
                "dataset.horizon_idx_rel",
                "dataset.diff_idx_rel",
            ],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": "local",
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
    ],
}

META_MAP = {
    "v2": WANDB_META_V2,
}

METRIC_MAP = {
    "v2": "falert_early_roc_auc",
}

HANDCRAFTED_METRICS_SINGLE = [
    "max_token_prob",
    "avg_token_prob",
    "max_token_entropy",
    "avg_token_entropy",
    "stac_single",
]

HANDCRAFTED_METRICS_MULTI = [
    "total_var",
    "pos_var",
    "rot_var",
    "gripper_var",
    "entropy_linkage",
    "stac_mmd",
]

MODEL_NAME_ORDER = [
    "max_token_prob",
    "avg_token_prob",
    "max_token_entropy",
    "avg_token_entropy",
    "embed-mahala",
    "embed-euclid",
    "embed-cosine",
    "embed-pca_kmeans",
    "rnd",
    "logpZO",
    "total_var",
    "pos_var",
    "rot_var",
    "gripper_var",
    "entropy_linkage",
    "stac_mmd",
    "stac_single",
    "lstm",
    "indep",
]


def unwrap_value(node: Any) -> Any:
    if isinstance(node, dict) and "value" in node and len(node) == 1:
        return unwrap_value(node["value"])
    if isinstance(node, dict):
        return {k: unwrap_value(v) for k, v in node.items()}
    if isinstance(node, list):
        return [unwrap_value(v) for v in node]
    return node


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_simple_value(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool)) or v is None


def parse_run_id(run_dir_name: str) -> str:
    # run-YYYYMMDD_HHMMSS-<id>
    parts = run_dir_name.split("-")
    return parts[-1] if len(parts) >= 3 else run_dir_name


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any] | None) -> pd.DataFrame:
    if not filters:
        return df

    for key, cond in filters.items():
        if key == "group":
            col = "group"
        elif key.startswith("config."):
            col = key[len("config.") :]
        else:
            col = key

        if col not in df.columns:
            print(f"[warn] filter column not found: {col} (from {key})")
            continue

        if isinstance(cond, dict) and "$in" in cond:
            df = df[df[col].isin(cond["$in"])].copy()
        else:
            df = df[df[col] == cond].copy()
    return df


def load_local_runs_df(log_root: str, filters: Dict[str, Any] | None = None) -> pd.DataFrame:
    log_root = os.path.abspath(log_root)
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"log_root not found: {log_root}")

    rows = []
    for run_dir in sorted(Path(log_root).glob("run-*/")):
        files_dir = run_dir / "files"
        summary_path = files_dir / "wandb-summary.json"
        config_path = files_dir / "config.yaml"

        if not summary_path.exists() or not config_path.exists():
            continue

        with summary_path.open("r") as f:
            summary = json.load(f)
        with config_path.open("r") as f:
            config_raw = yaml.safe_load(f)

        config_unwrapped = unwrap_value(config_raw or {})
        if isinstance(config_unwrapped, dict):
            config_unwrapped.pop("_wandb", None)
        config_flat = flatten_dict(config_unwrapped if isinstance(config_unwrapped, dict) else {})

        run_dir_name = run_dir.name
        run_id = parse_run_id(run_dir_name)

        row: Dict[str, Any] = {
            "name": run_dir_name,
            "_id": run_id,
            "group": config_flat.get("train.wandb_group_name"),
            "_project": config_flat.get("train.wandb_project"),
        }

        for k, v in summary.items():
            if is_simple_value(v):
                row[k] = v

        row.update(config_flat)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = apply_filters(df, filters)
    return df


def parse_runs_df_to_split_df_v2(
    runs_df: pd.DataFrame,
    config_compared: list[str],
) -> pd.DataFrame:
    if runs_df.empty:
        cols = ["split", "metric", "value"] + config_compared
        return pd.DataFrame(columns=cols)
    for config in config_compared:
        assert config in runs_df.columns, f"Column '{config}' not found in DataFrame."

    split_data = []
    for col in runs_df.columns:
        for split in ["train", "val_seen", "val_unseen"]:
            if not col.endswith(f"_{split}"):
                continue
            metric = col[: -len(f"_{split}")]
            for i, value in enumerate(runs_df[col]):
                new_row = {
                    "split": split,
                    "metric": metric,
                    "value": value,
                }
                for config in config_compared:
                    new_row[config] = runs_df[config].iloc[i]
                split_data.append(new_row)

    split_df = pd.DataFrame(split_data)
    if split_df.empty:
        cols = ["split", "metric", "value"] + config_compared
        return pd.DataFrame(columns=cols)
    split_df["value"] = split_df["value"].astype(float) * 100
    return split_df


def df_group_mean_except(
    df: pd.DataFrame,
    mean_over_cols: list[str],
    mean_value_cols: list[str],
    mean_std: bool = False,
) -> pd.DataFrame:
    if isinstance(mean_over_cols, str):
        mean_over_cols = [mean_over_cols]
    if isinstance(mean_value_cols, str):
        mean_value_cols = [mean_value_cols]

    group_cols = [col for col in df.columns if col not in (mean_over_cols + mean_value_cols)]

    def agg_func(x):
        if all(isinstance(i, str) for i in x):
            return x.iloc[0]
        if all(
            (isinstance(i, (float, int)) or (isinstance(i, str) and i.replace(".", "", 1).isdigit()))
            for i in x
        ):
            vals = x.astype(float)
            if mean_std:
                m = vals.mean()
                s = vals.std()
                return f"\\mstd{{{m:.2f}}}{{{s:.2f}}}"
            return vals.mean()
        raise ValueError("Mixed types in group: cannot aggregate.")

    agg = {col: agg_func for col in mean_value_cols}
    df = df.groupby(group_cols).agg(agg).reset_index()
    return df


def pull_metrics_from_group_v2_local(
    log_root: str,
    group_names: list[str],
    ablated_configs: list[str],
    group_configs: list[str],
    filters: dict | None = None,
    return_wandb_info: bool = False,
) -> pd.DataFrame:
    assert isinstance(group_configs, list)

    info_configs: list[str] = []
    if return_wandb_info:
        info_configs = ["_project", "_id"]

    runs_df = load_local_runs_df(log_root, filters)
    print(f"Loaded {len(runs_df)} runs from {log_root}")
    if runs_df.empty:
        print("No runs found; skipping metrics aggregation.")
        return pd.DataFrame()

    split_df = parse_runs_df_to_split_df_v2(
        runs_df,
        group_configs + ablated_configs + info_configs,
    )

    for col in group_configs:
        split_df[col] = split_df[col].astype(str)

    compare_df = split_df[split_df["metric"].str.contains("falert")]

    compare_df[["metric", "method"]] = compare_df["metric"].str.split("/", expand=True)
    cols = compare_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("metric")))
    cols.insert(1, cols.pop(cols.index("method")))
    compare_df = compare_df[cols]

    compare_df = compare_df.sort_values(by=group_configs + ["split"])

    for col in ablated_configs:
        if col == "model.pca_dim":
            compare_df[col] = compare_df[col].fillna(64)
        elif col == "model.n_clusters":
            compare_df[col] = compare_df[col].fillna(16)
        else:
            compare_df[col] = compare_df[col].fillna("default")

    if group_configs:
        compare_df_suite_mean = df_group_mean_except(compare_df, group_configs, ["value"])
        for col in group_configs:
            compare_df_suite_mean[col] = "avg"
        compare_df = pd.concat([compare_df_suite_mean, compare_df])

        compare_df_suite_mean = df_group_mean_except(compare_df, group_configs, ["value"], mean_std=True)
        for col in group_configs:
            compare_df_suite_mean[col] = "avg_mstd"
        compare_df = pd.concat([compare_df_suite_mean, compare_df])

    pivot_df = compare_df.pivot_table(
        index=["metric", "method"] + ablated_configs,
        columns=group_configs + ["split"],
        values="value",
        aggfunc="first",
    ).reset_index()

    if len(group_configs) == 0:
        pivot_df.columns = [a if len(b) == 0 else b for a, b in pivot_df.columns]
    elif len(group_configs) == 1:
        pivot_df.columns = [a if len(b) == 0 else f"{a}-{b}" for a, b in pivot_df.columns]
    elif len(group_configs) == 2:
        pivot_df.columns = [a if len(b) == 0 else f"{a}-{b}-{c}" for a, b, c in pivot_df.columns]

    return pivot_df


def main(args: argparse.Namespace):
    metric = METRIC_MAP[args.meta]

    if args.meta not in META_MAP:
        raise ValueError(f"Invalid meta version: {args.meta}")

    save_root = args.save_root or f"./scripts/wandb_metrics_batch_local_{args.meta}"
    os.makedirs(save_root, exist_ok=True)

    if args.benchmark == "all":
        benchmark_names = list(META_MAP[args.meta].keys())
    else:
        assert args.benchmark in META_MAP[args.meta], f"Invalid benchmark name: {args.benchmark}"
        benchmark_names = [args.benchmark]

    for benchmark_name in benchmark_names:
        benchmark = META_MAP[args.meta][benchmark_name]
        print(f"Processing {benchmark_name}...")
        benchmark_best = []

        for group in benchmark:
            group_names = group["group_names"]
            exp_suffixes = group["exp_suffixes"]
            ablated_configs = group["ablated_configs"]
            group_configs = group["group_configs"]
            extra_filters = group["extra_filters"]

            print(f"Processing {group_names} {exp_suffixes}")

            filters = {
                "group": {"$in": group_names},
                "config.train.exp_suffix": {"$in": exp_suffixes},
                **extra_filters,
            }

            compare_df = pull_metrics_from_group_v2_local(
                args.log_root,
                group_names,
                ablated_configs,
                group_configs,
                filters,
            )

            if compare_df.empty:
                print("[warn] no runs matched; skipping")
                continue

            if "model.lr" in compare_df.columns:
                compare_df["model.lr"] = compare_df["model.lr"].astype(str)
            if "model.lambda_reg" in compare_df.columns:
                compare_df["model.lambda_reg"] = compare_df["model.lambda_reg"].astype(str)

            if "extra_name" in group:
                save_path = f"{save_root}/{group_names[0]}-{'_'.join(exp_suffixes)}-{group['extra_name']}.csv"
            else:
                save_path = f"{save_root}/{group_names[0]}-{'_'.join(exp_suffixes)}.csv"
            print(f"Saving results to {save_path}.\n")

            rename_map = {
                "avg-train": "train",
                "avg-val_seen": "val_seen",
                "avg-val_unseen": "val_unseen",
                "avg-avg-train": "train",
                "avg-avg-val_seen": "val_seen",
                "avg-avg-val_unseen": "val_unseen",
            }
            compare_df.rename(columns=rename_map, inplace=True)

            compare_df.to_csv(save_path, index=False)

            if "handcrafted" in exp_suffixes[0]:
                assert len(exp_suffixes) == 1
                metrics = HANDCRAFTED_METRICS_SINGLE
                if "handcrafted_multi" in exp_suffixes[0]:
                    metrics = HANDCRAFTED_METRICS_MULTI
                compare_df["method_full"] = compare_df["method"]
                compare_df["method"] = "handcrafted"
                compare_df["model.name"] = "handcrafted"
                to_drop = []
                for i, row in compare_df.iterrows():
                    match = False
                    for metric_name in metrics:
                        if metric_name in row["method_full"]:
                            compare_df.at[i, "model.name"] = metric_name
                            match = True
                            break
                    if not match:
                        to_drop.append(i)
                compare_df.drop(to_drop, inplace=True)
                compare_df.reset_index(drop=True, inplace=True)

            if compare_df["model.name"].astype(str).str.contains("embed").any():
                for i, row in compare_df.iterrows():
                    if row["model.name"] == "embed":
                        compare_df.at[i, "model.name"] = row["model.name"] + "-" + str(row["model.distance"])

            compare_df = compare_df[compare_df["metric"] == metric]
            idx = compare_df.groupby(["model.name", "method"])["val_seen"].idxmax()
            df_max = compare_df.loc[idx].reset_index(drop=True)
            benchmark_best.append(df_max)

        if not benchmark_best:
            print(f"[warn] no data produced for {benchmark_name}\n")
            continue

        benchmark_best = pd.concat(benchmark_best, ignore_index=True)

        cols = benchmark_best.columns.tolist()
        cols.insert(0, cols.pop(cols.index("method")))
        cols.insert(1, cols.pop(cols.index("model.name")))
        cols.insert(2, cols.pop(cols.index("train")))
        cols.insert(3, cols.pop(cols.index("val_seen")))
        cols.insert(4, cols.pop(cols.index("val_unseen")))
        benchmark_best = benchmark_best[cols]

        benchmark_best["model.name"] = pd.Categorical(
            benchmark_best["model.name"], categories=MODEL_NAME_ORDER, ordered=True
        )
        benchmark_best = benchmark_best.sort_values(by=["model.name"])

        benchmark_save_path = f"{save_root}/{benchmark_name}.csv"
        print(f"Saving benchmark best runs to {benchmark_save_path}.\n")
        benchmark_best.to_csv(benchmark_save_path, index=False, float_format="%.2f")
        print(f"Finished processing {benchmark_name}.\n")
        print("=" * 50)
        print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="v2", help="The meta version to use")
    parser.add_argument("--benchmark", type=str, default="all", help="The benchmark to process")
    parser.add_argument(
        "--log_root",
        type=str,
        default="wandb_trans/wandb",
        help="Local W&B log root that contains run-* directories",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="log_my_csv/",
        help="Optional output directory for CSVs",
    )
    args = parser.parse_args()
    main(args)
