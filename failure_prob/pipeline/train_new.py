import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from failure_prob.conf import Config, process_cfg
from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import RolloutDataset, normalize_rollouts_hidden_states
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.utils.constants import EVAL_TIME_QUANTILES, MANUAL_METRICS
from failure_prob.utils.random import seed_everything
from failure_prob.utils.routines import (
    eval_metrics_and_log,
    eval_model_and_log,
    eval_save_timing_plots,
)
from failure_prob.utils.split_io import save_split_signature
from failure_prob.utils.timer import Timer
from failure_prob.utils.video import eval_save_videos, eval_save_videos_functional_cp


DEFAULT_LOG_ROOT = "./log_ckpt_new"


def _cli_has_override(prefix: str) -> bool:
    return any(arg.startswith(prefix) for arg in sys.argv[1:])


def _parse_seeds(seed_cfg: str | int) -> list[int]:
    if isinstance(seed_cfg, int):
        return [seed_cfg]
    if seed_cfg.isnumeric() and "-" not in seed_cfg:
        return [int(seed_cfg)]

    assert all(s.isdigit() for s in seed_cfg.split("-")), (
        "All seeds must be integers separated by '-'"
    )
    return [int(s) for s in seed_cfg.split("-")]


def _get_model_dir_name(cfg: Config) -> str:
    parts = [str(cfg.model.name)]
    if "distance" in cfg.model:
        parts.append(str(cfg.model.distance))

    exp_suffix = getattr(cfg.train, "exp_suffix", None)
    if exp_suffix:
        exp_suffix = str(exp_suffix)
        if exp_suffix not in parts:
            parts.append(exp_suffix)
    return "-".join(parts)


def _get_seed_log_dir(
    cfg: Config,
    seed: int,
    launch_date: str,
    launch_time: str,
) -> str:
    if _cli_has_override("train.logs_save_path="):
        return str(cfg.train.logs_save_path)

    root = str(cfg.train.logs_save_root or DEFAULT_LOG_ROOT)
    dataset_name = str(cfg.dataset.name)
    model_dir = _get_model_dir_name(cfg)
    return os.path.join(
        root,
        dataset_name,
        f"seed{seed}",
        model_dir,
        launch_date,
        launch_time,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    if not _cli_has_override("train.logs_save_root=") and not _cli_has_override("train.logs_save_path="):
        cfg.train.logs_save_root = DEFAULT_LOG_ROOT

    cfg = process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    if getattr(cfg.dataset, "use_cache", False):
        if getattr(cfg.dataset, "cache_path", None):
            cache_target = str(cfg.dataset.cache_path)
        else:
            cache_target = str(cfg.dataset.cache_dir)
        print(
            "Dataset cache enabled:",
            f"target={cache_target}",
            f"refresh={bool(getattr(cfg.dataset, 'refresh_cache', False))}",
        )
    else:
        print("Dataset cache disabled.")

    launch_date = datetime.utcnow().strftime("%Y%m%d")
    launch_time = datetime.utcnow().strftime("%H%M%S")
    # breakpoint()
    seed_everything(0)
    with Timer("Loading rollouts"):
        all_rollouts = load_rollouts(cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        
        if cfg.dataset.load_to_cuda:
            all_rollouts = [r.to("cuda") for r in all_rollouts]
    
    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")

    if cfg.dataset.normalize_hidden_states:
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)

    seeds = _parse_seeds(cfg.train.seed)
    for seed in seeds:
        print(f"Running seed {seed}")
        cfg.train.seed = seed
        cfg.train.logs_save_path = _get_seed_log_dir(cfg, seed, launch_date, launch_time)
        seed_everything(seed)

        if (
            cfg.train.eval_save_logs
            or cfg.train.eval_save_video
            or cfg.train.eval_save_ckpt
            or cfg.train.eval_save_video_functional
            or cfg.train.eval_save_timing_plots
        ):
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            with open(os.path.join(cfg.train.logs_save_path, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        wandb.init(
            project=cfg.train.wandb_project,
            dir=cfg.train.wandb_dir,
            name=cfg.train.exp_name,
            group=cfg.train.wandb_group_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        
        rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
        split_save_path = os.path.join(
            cfg.train.logs_save_path,
            f"split_seed{seed}.json",
        )
        split_signature = save_split_signature(split_save_path, cfg, rollouts_by_split_name)
        print(
            "Saving split signature to",
            os.path.abspath(split_save_path),
            f"(md5={split_signature['split_md5']})",
        )

        train_rollouts = rollouts_by_split_name["train"]
        input_dim = train_rollouts[0].hidden_states.shape[-1]
        print(
            f"hidden_feature: {train_rollouts[0].hidden_states.shape} "
            f"{train_rollouts[0].hidden_states.dtype}"
        )
        print("Saving outputs under", os.path.abspath(cfg.train.logs_save_path))

        dataset_by_split_name = {
            split_name: RolloutDataset(cfg, rollouts)
            for split_name, rollouts in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            split_name: DataLoader(
                dataset,
                batch_size=cfg.model.batch_size,
                shuffle="train" in split_name,
                num_workers=0,
            )
            for split_name, dataset in dataset_by_split_name.items()
        }

        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            to_be_logged = eval_metrics_and_log(
                cfg,
                rollouts_by_split_name,
                MANUAL_METRICS[cfg.dataset.name],
                EVAL_TIME_QUANTILES[cfg.dataset.name],
            )
            to_be_logged["epoch"] = 0
            wandb.log(to_be_logged)
        
        if cfg.train.log_precomputed_only:
            wandb.finish(quiet=True)
            continue

        model: BaseModel = get_model(cfg, input_dim)
        print(model)
        model.to("cuda")

        optimizer, lr_scheduler = model.get_optimizer()
        n_epochs = cfg.model.n_epochs
        pbar = trange(n_epochs)
        for epoch in pbar:
            to_be_logged = {"epoch": epoch + 1}
            
            model.train()
            avg_loss = model.train_epoch(optimizer, dataloader_by_split_name["train"])
            pbar.set_description(f"Avg Loss: {avg_loss:.4f}")
            to_be_logged["train_loss"] = avg_loss

            if lr_scheduler is not None:
                lr_scheduler.step()
                to_be_logged["learning_rate"] = optimizer.param_groups[0]["lr"]

            model.eval()
            if epoch % cfg.train.roc_every == 0 or epoch == n_epochs - 1:
                eval_logs = eval_model_and_log(
                    cfg,
                    model,
                    rollouts_by_split_name,
                    dataloader_by_split_name,
                    EVAL_TIME_QUANTILES[cfg.dataset.name],
                    plot_score_curves=epoch == n_epochs - 1,
                    plot_auc_curves=epoch == n_epochs - 1,
                    log_classification_metrics=epoch == n_epochs - 1,
                )
                to_be_logged.update(eval_logs)

            wandb.log(to_be_logged)

        if cfg.train.eval_save_video:
            for split, dataloader in dataloader_by_split_name.items():
                if split == "train":
                    continue
                video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_{split}")
                print("Saving videos to", os.path.abspath(video_save_folder))
                os.makedirs(video_save_folder, exist_ok=True)
                eval_save_videos(dataloader, model, cfg, video_save_folder)

        if cfg.train.eval_save_video_functional:
            video_save_folder = os.path.join(cfg.train.logs_save_path, "videos_functional")
            os.makedirs(video_save_folder, exist_ok=True)
            eval_save_videos_functional_cp(
                cfg,
                model,
                rollouts_by_split_name,
                dataloader_by_split_name,
                video_save_folder,
                alpha=cfg.train.eval_cp_alpha,
            )

        if cfg.train.eval_save_timing_plots:
            plot_save_folder = os.path.join(cfg.train.logs_save_path, "timing_plots")
            os.makedirs(plot_save_folder, exist_ok=True)
            eval_save_timing_plots(
                cfg,
                model,
                rollouts_by_split_name,
                dataloader_by_split_name,
                plot_save_folder,
                alpha=cfg.train.eval_cp_alpha,
            )

        if cfg.train.eval_save_ckpt:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            ckpt_save_path = os.path.join(
                cfg.train.logs_save_path,
                f"model_seed{seed}.ckpt",
            )
            print("Saving model checkpoint to", os.path.abspath(ckpt_save_path))
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": n_epochs,
                },
                ckpt_save_path,
            )

        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
