import hydra
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from helpers.callbacks import get_callbacks
from helpers.dataset import get_dataloaders
from helpers.diffusion import get_diffusion
from helpers.graph_denoiser import GraphDenoiser
from helpers.pl_module import WeightDenoiser
from helpers.model import WeightDiffusionTransformer

import wandb

@hydra.main(config_path="configs", config_name="graph_conditional_gram_baseline.yaml", version_base="1.1")
def train(cfg: DictConfig) -> None:
    if cfg.diffusion.type == "DDIM" and cfg.model.nca_loss_every_n_steps > 0:
        raise ValueError("NCA loss is not supported for DDIM diffusion (only EDM). Set nca_loss_every_n_steps to 0.")

    print("-"*50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-"*50)

    L.seed_everything(cfg.seed)

    if not cfg.debug:
        tags = cfg.logger.tags
        if cfg.model.use_cross_attention and 'cross_attention' not in tags:
            tags.append("cross_attention")
        if cfg.model.conditioning and 'conditioning' not in tags:
            tags.append("conditioning")
        if cfg.diffusion.type not in tags:
            tags.append(cfg.diffusion.type)
        logger = WandbLogger(
            project=cfg.logger.project, 
            save_dir=f"/scratch/izar/{cfg.username}/outputs/",
            log_model="all", 
            tags=tags,
            entity=cfg.logger.entity
            )
    else:
        logger = None

    trn_dataloader, val_dataloader, normaliser = get_dataloaders(cfg)


    encoder = None
    if cfg.model.conditioning:
        encoder = hydra.utils.instantiate(cfg.texture_encoder)
        if "CLIP" in cfg.texture_encoder._target_:
            encoder = encoder.to("cuda")
        test_inp = next(iter(trn_dataloader))["nca_image"]
        cond_dim = encoder(test_inp).shape[-1]
        cfg.model.cond_dim = cond_dim
        print(f"Updated cond_dim: {cfg.model.cond_dim}")

    diffusion = get_diffusion(cfg)
    if cfg.model.type == "graph":
        model = GraphDenoiser(cfg)
    else:
        model = WeightDiffusionTransformer(cfg)
    pl_module = WeightDenoiser(cfg=cfg, model=model, diffusion=diffusion, normaliser=normaliser,encoder=encoder)
    callbacks = get_callbacks(cfg)

    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=callbacks,
        deterministic=False, # CLIP fails with deterministic behaviour
        precision=cfg.trainer.precision,
        num_sanity_val_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader)

if __name__ == "__main__":
    train()
