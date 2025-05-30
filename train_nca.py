import torch
from helpers.nca.nca_pl_module import NCA_pl
from helpers.nca.nca_data import TextureDataset,TextureImageDataset
from helpers.nca.nca_model import NCA
from helpers.nca.nca_loss import TextureLoss
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from helpers.model import WeightDiffusionTransformer
from helpers.diffusion import get_diffusion
from helpers.dataset import get_dataloaders
from helpers.pl_module import WeightDenoiser
from helpers.utils import create_ckpt_from_weight_samples
from helpers.callbacks import TimingCallback
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything
from datetime import datetime
from tqdm import tqdm
import wandb
import json
from pathlib import Path

@hydra.main(config_path="configs", config_name="train_nca.yaml", version_base="1.1",)
def train(cfg:DictConfig) -> None:
    if cfg.seed:
        seed_everything(cfg.seed)

    if cfg.model.use_diffusion_sampled_weights and cfg.model.use_bubbly_weights:
        raise ValueError("Cannot use both diffusion sampled weights and bubbly weights at the same time.")

    texture_dataset = TextureDataset(data_dir=cfg.data.data_path)
    if cfg.training.train_bubbly:
        cfg.data.pattern = 'bubbly_0101'
        texture_dataset.filter_images(pattern=cfg.data.pattern)
        # if cfg.data.random_sample > 0:
        #     texture_dataset.random_sample(n=cfg.data.random_sample)
    else:
    # Load subset_data.json
        with open('subset_data.json', 'r') as f:
            subset_data = json.load(f)
        subset_data = [Path(img_path) for img_path in subset_data][0:cfg.data.random_sample]
        texture_dataset.images = subset_data

    # Current date and time
    group_name_prefix = datetime.now().strftime("%Y-%m-%d_%H_%M")
    if cfg.model.use_diffusion_sampled_weights:
        group_name = f"training_from_diffusion_sampled_weights"
    elif cfg.model.use_bubbly_weights:
        group_name = f"traing_from_bubbly_weights"
    else:
        group_name = f"nca_training_from_random"
    group_name = f"{group_name}_{group_name_prefix}"

    if cfg.model.use_diffusion_sampled_weights:
        print("Using diffusion sampled weights")
        diffusion_cfg = OmegaConf.load('configs/train.yaml')
        diffusion_cfg.model.conditioning = False
        diffusion = get_diffusion(diffusion_cfg)
        model = WeightDiffusionTransformer(diffusion_cfg)
        _, _, normaliser = get_dataloaders(diffusion_cfg)
        api = wandb.Api()
        artifact = api.artifact(cfg.model.diffusion_weights_ckpt)
        ckpt = artifact.download()

        checkpoint = torch.load(str(list(Path(ckpt).glob('*.ckpt'))[0]),weights_only=False)
        pl_module = WeightDenoiser(
            cfg=diffusion_cfg,
            diffusion=diffusion,
            model=model,
            normaliser=normaliser
        )
        pl_module.model.load_state_dict({k.replace('model.',''):v for (k,v) in checkpoint['state_dict'].items()})
        pl_module.to('cuda:0')

    elif cfg.model.use_bubbly_weights:
        print("Using bubbly weights")
        api = wandb.Api()
        artifact = api.artifact(cfg.model.bubbly_weights_ckpt)
        ckpt = artifact.download()
        checkpoint = torch.load(str(list(Path(ckpt).glob('*.ckpt'))[0]),weights_only=False)
    else:
        print("Using random weights")


    for img_path in tqdm(texture_dataset.images):
        dataset = TextureImageDataset(img_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4)
        nca = NCA()
        criterion = TextureLoss()
        nca_pl = NCA_pl(nca=nca, criterion=criterion)

        if not cfg.debug:
            logger = WandbLogger(
                project=cfg.logger.project, 
                save_dir=f"/scratch/izar/{cfg.username}/outputs/",
                log_model="checkpoint", 
                tags=cfg.logger.tags,
                entity=cfg.logger.entity,
                name=img_path.stem if not cfg.training.train_bubbly else None,
                group=group_name if not cfg.training.train_bubbly else None,
                )
            
            logger.experiment.log({"target_image": [wandb.Image(str(img_path))]})

            # Save the cfg to wandb
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))


        else:
            logger = None

        if cfg.model.use_diffusion_sampled_weights:
            x_denoised = pl_module.sample(num_steps=cfg.model.num_steps)
            ckpt = create_ckpt_from_weight_samples(x_denoised)
            nca_pl.nca.load_state_dict(ckpt['state_dict'])
        elif cfg.model.use_bubbly_weights:
            nca_pl.nca.load_state_dict({k.replace('nca.',''):v for (k,v) in checkpoint['state_dict'].items() if 'nca' in k})


        callbacks = [
            ModelCheckpoint(
                monitor="train_loss",
                save_top_k=1,
                mode="min",
                save_last=True,
                every_n_epochs=5
            ),
            TimingCallback(),
            LearningRateMonitor(logging_interval="epoch")
            
        ]
        trainer = L.Trainer(max_epochs=cfg.training.num_epochs, accelerator='gpu', devices=-1,log_every_n_steps=1,logger=logger,callbacks=callbacks)
        trainer.fit(nca_pl, dataloader)

        # Close wandb run
        if not cfg.debug:
            artifact = wandb.Artifact(f'model-{logger.experiment.id}', type='model')
            artifact.add_file(trainer.checkpoint_callback.last_model_path)
            logger.experiment.log_artifact(artifact)
            logger.experiment.finish()



if __name__ == '__main__':
    train()