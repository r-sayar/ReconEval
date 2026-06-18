from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
import scvi

from hydra.utils import instantiate
from lightning.pytorch import seed_everything

'''
Main training script for deep learning model
'''

@hydra.main(config_path='../configs', config_name="base_train", version_base=None)
def train(cfg: DictConfig) -> None:

    save_path = os.path.normpath(os.path.join(cfg.model.save.dir, cfg.model.save.filename))
    scvi.settings.logging_dir = save_path
    
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.precision)
    print("cfg.model:", cfg.model)

    model = instantiate(cfg.model.model_args)
    print("model:", model)

    dm = instantiate(cfg.model.dm_args)
    logger = instantiate(cfg.model.logging.wandb) if "logging" in cfg.model else None
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.model.training.callbacks] if "callbacks" in cfg.model.training else []
    print("callbacks:", callbacks)

    train_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    train_kwargs.update({
        "datamodule": dm,
        "logger": logger,
        "callbacks": callbacks,
    })

    if hasattr(cfg.model.training, 'plan_kwargs'):
        train_kwargs["plan_kwargs"] = cfg.model.training.plan_kwargs
    
    print("train_kwargs", train_kwargs)

    model.train(**train_kwargs)
    wandb.finish()


if __name__ == "__main__":
    train()