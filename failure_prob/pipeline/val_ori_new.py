import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.mrefine.ori_summary import (
    _collect_ori_log_paths,
    _get_run_meta_from_config,
    _split_ori_logs_by_method,
)
from failure_prob.pipeline.val_new import (
    _extract_seed,
    _resolve_default_logs_dir,
    build_candidate_signature,
)


def collect_ori_candidates(logs_dir: str) -> list[dict]:
    logs_dir = os.path.abspath(logs_dir)
    candidates = []
    for log_path in _collect_ori_log_paths(logs_dir):
        with open(log_path, 'r') as f:
            raw_logs = json.load(f)

        if os.path.basename(log_path) == 'ori_logs.json':
            ori_logs = raw_logs
        else:
            if 'ori' not in raw_logs:
                continue
            ori_logs = raw_logs['ori']

        run_dir = os.path.dirname(os.path.dirname(log_path))
        run_name = os.path.relpath(run_dir, logs_dir)
        run_meta = _get_run_meta_from_config(run_dir)
        method_logs_by_name = _split_ori_logs_by_method(ori_logs, run_meta['method_name'])

        cfg = None
        cfg_path = os.path.join(run_dir, 'config.yaml')
        if os.path.isfile(cfg_path):
            cfg = OmegaConf.load(cfg_path)
        seed = _extract_seed(cfg, run_name)

        for method_name, method_logs in sorted(method_logs_by_name.items()):
            if cfg is not None:
                config_signature, signature_payload = build_candidate_signature(cfg, method_name)
                exp_suffix = getattr(cfg.train, 'exp_suffix', None)
            else:
                config_signature = hashlib.md5(f'{method_name}:{run_name}'.encode('utf-8')).hexdigest()
                signature_payload = {'method_name': method_name, 'run_name': run_name}
                exp_suffix = None

            candidate_id = f'{method_name}::{run_name}'
            candidates.append({
                'candidate_id': candidate_id,
                'method_name': method_name,
                'run_name': run_name,
                'run_dir': os.path.abspath(run_dir),
                'log_path': os.path.abspath(log_path),
                'seed': seed,
                'exp_suffix': exp_suffix,
                'config_signature': config_signature,
                'signature_payload': signature_payload,
                'ori_summary': {
                    'static': method_logs.get('static', {}),
                    'calib': method_logs.get('calib', {}),
                },
            })

    return candidates


def summarize_ori_static(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        static_logs = candidate['ori_summary'].get('static', {})
        for split_name, split_dict in static_logs.items():
            task_dict = split_dict.get('all', {})
            if not task_dict:
                continue
            roc_values = np.asarray(task_dict.get('roc_auc', []), dtype=float)
            prc_values = np.asarray(task_dict.get('prc_auc', []), dtype=float)
            rows.append({
                'candidate_id': candidate['candidate_id'],
                'method': candidate['method_name'],
                'seed': candidate['seed'],
                'split': split_name,
                'run_name': candidate['run_name'],
                'run_dir': candidate['run_dir'],
                'log_path': candidate['log_path'],
                'exp_suffix': candidate['exp_suffix'],
                'config_signature': candidate['config_signature'],
                'num_points_roc': int(roc_values.size),
                'num_points_prc': int(prc_values.size),
                'roc_auc': float(roc_values.mean()) if roc_values.size else np.nan,
                'prc_auc': float(prc_values.mean()) if prc_values.size else np.nan,
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                'candidate_id', 'method', 'seed', 'split', 'run_name', 'run_dir', 'log_path',
                'exp_suffix', 'config_signature', 'num_points_roc', 'num_points_prc', 'roc_auc', 'prc_auc',
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=['method', 'seed', 'split', 'run_name'],
        na_position='last',
    ).reset_index(drop=True)


def _mark_pareto_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    marked_rows = []
    best_bal_acc = -np.inf
    pareto_rank = 0
    for row in sorted(rows, key=lambda x: (x['avg_det_time'], -x['bal_acc'], x['alpha'])):
        row = dict(row)
        is_pareto = row['bal_acc'] > best_bal_acc
        row['is_pareto'] = bool(is_pareto)
        if is_pareto:
            row['pareto_rank'] = pareto_rank
            pareto_rank += 1
            best_bal_acc = row['bal_acc']
        else:
            row['pareto_rank'] = np.nan
        marked_rows.append(row)
    return marked_rows


def summarize_ori_alpha_points(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        calib_logs = candidate['ori_summary'].get('calib', {})
        for mode in ('early', 'last'):
            mode_rows = []
            for alpha, metrics in sorted(calib_logs.get(mode, {}).items(), key=lambda x: float(x[0])):
                if 'avg_det_time' not in metrics or 'bal_acc' not in metrics:
                    continue
                mode_rows.append({
                    'candidate_id': candidate['candidate_id'],
                    'method': candidate['method_name'],
                    'seed': candidate['seed'],
                    'mode': mode,
                    'alpha': float(alpha),
                    'run_name': candidate['run_name'],
                    'run_dir': candidate['run_dir'],
                    'log_path': candidate['log_path'],
                    'exp_suffix': candidate['exp_suffix'],
                    'config_signature': candidate['config_signature'],
                    'detect_method': metrics.get('detect_method'),
                    'avg_det_time': float(np.asarray(metrics['avg_det_time']).mean()),
                    'bal_acc': float(np.asarray(metrics['bal_acc']).mean()),
                    'acc': float(np.asarray(metrics['acc']).mean()) if 'acc' in metrics else np.nan,
                    'f1': float(np.asarray(metrics['f1']).mean()) if 'f1' in metrics else np.nan,
                    'tpr': float(np.asarray(metrics['tpr']).mean()) if 'tpr' in metrics else np.nan,
                    'tnr': float(np.asarray(metrics['tnr']).mean()) if 'tnr' in metrics else np.nan,
                    'fpr': float(np.asarray(metrics['fpr']).mean()) if 'fpr' in metrics else np.nan,
                    'fnr': float(np.asarray(metrics['fnr']).mean()) if 'fnr' in metrics else np.nan,
                })
            rows.extend(_mark_pareto_rows(mode_rows))

    if not rows:
        return pd.DataFrame(
            columns=[
                'candidate_id', 'method', 'seed', 'mode', 'alpha', 'run_name', 'run_dir', 'log_path',
                'exp_suffix', 'config_signature', 'detect_method', 'avg_det_time', 'bal_acc', 'acc', 'f1',
                'tpr', 'tnr', 'fpr', 'fnr', 'is_pareto', 'pareto_rank',
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=['method', 'seed', 'mode', 'run_name', 'avg_det_time', 'alpha'],
        na_position='last',
    ).reset_index(drop=True)


def summarize_ori_static_aggregate(static_df: pd.DataFrame) -> pd.DataFrame:
    if static_df.empty:
        return pd.DataFrame(
            columns=['method', 'seed', 'split', 'num_runs', 'mean_roc_auc', 'std_roc_auc', 'mean_prc_auc', 'std_prc_auc']
        )

    rows = []
    for (method, seed, split_name), group in static_df.groupby(['method', 'seed', 'split'], dropna=False):
        rows.append({
            'method': method,
            'seed': seed,
            'split': split_name,
            'num_runs': int(len(group)),
            'mean_roc_auc': float(group['roc_auc'].mean()),
            'std_roc_auc': float(group['roc_auc'].std(ddof=0)),
            'mean_prc_auc': float(group['prc_auc'].mean()),
            'std_prc_auc': float(group['prc_auc'].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(by=['method', 'seed', 'split']).reset_index(drop=True)


def run_validation_ori_pipeline(logs_dir: str, save_dir: str) -> dict:
    os.makedirs(save_dir, exist_ok=True)

    candidates = collect_ori_candidates(logs_dir)
    static_df = summarize_ori_static(candidates)
    static_agg_df = summarize_ori_static_aggregate(static_df)
    alpha_points_df = summarize_ori_alpha_points(candidates)

    static_df.to_csv(os.path.join(save_dir, 'val_ori_static_by_run.csv'), index=False)
    static_agg_df.to_csv(os.path.join(save_dir, 'val_ori_static_aggregate_by_seed.csv'), index=False)
    alpha_points_df.to_csv(os.path.join(save_dir, 'val_ori_alpha_points_by_run.csv'), index=False)
    alpha_points_df[alpha_points_df['is_pareto']].to_csv(
        os.path.join(save_dir, 'val_ori_pareto_alpha_points_by_run.csv'),
        index=False,
    )

    payload = {
        'logs_dir': os.path.abspath(logs_dir),
        'save_dir': os.path.abspath(save_dir),
        'num_candidates': int(len(candidates)),
        'num_static_rows': int(len(static_df)),
        'num_alpha_rows': int(len(alpha_points_df)),
    }
    with open(os.path.join(save_dir, 'val_ori_summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Summarize validation-set ori metrics and alpha Pareto fronts for evaluated checkpoints.',
    )
    parser.add_argument(
        '--logs-dir',
        default=_resolve_default_logs_dir(),
        help='Root directory that contains run folders with eval/ori_logs.json.',
    )
    parser.add_argument(
        '--save-dir',
        default=None,
        help='Directory to save ori validation summaries. Defaults to <logs-dir>/pipeline_val_ori_new.',
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.abspath(args.logs_dir), 'pipeline_val_ori_new')
    result = run_validation_ori_pipeline(
        logs_dir=args.logs_dir,
        save_dir=save_dir,
    )

    print('Saved ori validation summary to', os.path.abspath(os.path.join(save_dir, 'val_ori_summary.json')))
    print(
        f"Candidates={result['num_candidates']} static_rows={result['num_static_rows']} alpha_rows={result['num_alpha_rows']}"
    )


if __name__ == '__main__':
    main()
