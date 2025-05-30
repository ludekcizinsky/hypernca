import torch
import lightning as L
from helpers.optimizer import CosineWithWarmup
from torch.optim.lr_scheduler import LambdaLR
from helpers.utils import get_weight_grad_norm
from typing_extensions import Self
from helpers.generator import Generator
from torch import Tensor
from helpers.utils import create_ckpt_from_weight_samples
import numpy as np
import torch.nn.functional as F
import random
import wandb

class WeightDenoiser(L.LightningModule):
    def __init__(self, cfg, diffusion, model, normaliser, encoder:torch.nn.Module=None) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)  # Logs config to W&B etc.
        self.model = model
        self.diffusion = diffusion
        self.normaliser = normaliser
        self.encoder = encoder
        self.generator = Generator()
        self.generate_interval = cfg.trainer.log_fid_every_n_epoch
        self.conditional_diffusion = cfg.model.conditioning
        self.cfg_prob = 0.1 if cfg.model.use_cfg else 0.0

    def setup(self, stage: str) -> None:
        if self.logger is not None:
            metric = wandb.run.define_metric(name="val-images/*", step_metric="epoch")
            print(f"Defined metric: {metric}")

    def forward(self, x):
        return self.model(x)
    
    def nca_loss_step(self,Dx:Tensor,images:Tensor,B:int) -> None:
        x_hat = self.normaliser.inverse_transform_with_grad(Dx)
        steps = np.random.randint(1, 5)
        self.generator.generate(samples=x_hat, steps=steps)
        generated_images = self.generator.generated_images
        self.generator.generated_images = []
        loss = F.mse_loss(torch.stack(generated_images,dim=0), images,reduction='mean')
        self.log("train/nca_loss", loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=B)
        return loss


    def training_step(self, batch, batch_idx):
        x = batch['weights']
        if self.hparams.model.cond_on_nca:
            cond = batch['nca_image']
        else:
            cond = batch['gt_image']
        B = x.shape[0]

        x_normalized = self.normaliser.transform(x)
        if self.encoder is not None and random.random() >= self.cfg_prob:
            cond = self.encoder(cond)
        else:
            cond = None

        denoising_loss, _, _, _, bins = self.diffusion.get_loss(self.model, x_normalized, cond, return_aux=True)

        if bins is not None:
            batch_bin_loss = torch.mean(denoising_loss, dim=(1, 2)).detach().cpu().numpy()
            for bin, bin_loss in zip(bins, batch_bin_loss):
                self.log(f"train/bin_loss_{bin}", bin_loss, on_step=True, on_epoch=False,batch_size=B)
            
            denoising_loss = denoising_loss.mean()
        self.log("train/denoising_loss", denoising_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=B)

        return denoising_loss

    def on_train_epoch_end(self) -> None:
        # Log gradient and weight norms
        weight_norm, _ = get_weight_grad_norm(self.model)
        self.log("optim/weight_norm", weight_norm, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx) -> None:

        x = batch['weights']
        if self.hparams.model.cond_on_nca:
            cond = batch['nca_image']
        else:
            cond = batch['gt_image']
        B = x.shape[0]
        x_normalized = self.normaliser.transform(x)

        if self.encoder is not None:
            cond = self.encoder(cond)
        else:
            cond = None

        denoising_loss, _, _, _, bins = self.diffusion.get_loss(self.model, x_normalized, cond, return_aux=True)

        if bins is not None:
            batch_bin_loss = torch.mean(denoising_loss, dim=(1, 2)).detach().cpu().numpy()
            for bin, bin_loss in zip(bins, batch_bin_loss):
                self.log(f"val/bin_loss_{bin}", bin_loss, on_step=False, on_epoch=True,batch_size=B)
            
                denoising_loss = denoising_loss.mean()

        self.log("val/denoising_loss", denoising_loss, on_step=False, on_epoch=True,batch_size=B)


        if self.current_epoch % self.generate_interval == 0 and self.current_epoch > 0 or self.current_epoch==999:# or last epoch
            cond = batch.get('gt_image', None)
            if self.encoder is not None:
                cond = self.encoder(cond)
            else:
                cond = None

            samples = self.sample(num_steps=100, cond=cond, seed=batch_idx,num_samples=B)
            self.generator.generate_and_log(samples=samples, gt_images=batch.get('gt_image', None), epoch=self.current_epoch, logger=self.logger,texture_names=batch.get('texture', None),bidx=batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.generate_interval == 0 and self.current_epoch > 0 or self.current_epoch==999:
            results = self.generator.log_metrics()
            for metric, value in results.items():
                self.log(f"val/{metric}", value, on_step=False, on_epoch=True)

    @torch.no_grad()
    def sample(self,num_samples:int=10,num_steps:int=18,cond:torch.Tensor=None,seed=None) -> torch.Tensor:
        """
        Sample from the model.

        Args:
            num_samples (int): Number of samples to generate.
            cond (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Generated samples.
        """
        num_samples = num_samples if cond is None else cond.shape[0]
        # Sample from the diffusion model
        x_denoised = self.diffusion.sample(self.model, num_steps=num_steps, cond=cond,seed=seed, num_samples=num_samples)
        # Inverse transform the samples
        x_denoised = self.normaliser.inverse_transform(x_denoised)
        return x_denoised


    def predict_step(self, batch, batch_idx):

        B = batch['weights'].shape[0]
        cond = batch.get('gt_image', None)
        cond = self.encoder(cond)

        samples = self.sample(num_steps=100, cond=cond, seed=batch_idx,num_samples=B)
        gen_imgs = self.generator.generate_and_log(
            samples=samples, 
            gt_images=batch.get('gt_image', None), 
            epoch=self.current_epoch, 
            logger=self.logger, 
            texture_names=batch.get('texture', None), 
            bidx=batch_idx, 
            return_images=True
        )
        return {
            'gen_imgs': gen_imgs,
            'gt_imgs': batch.get('gt_image', None),
            "nca_imgs": batch.get('nca_image', None),
            "tex_names": batch.get('texture', None)
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, *args, **kwargs) -> Self:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu",weights_only=False)
        # Get the model state dict
        model_state_dict = checkpoint["state_dict"]
        # Create the model instance
        model = cls(*args, **kwargs)
        # Load the state dict into the model
        model.load_state_dict(model_state_dict, strict=False)
        return model


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters() if self.encoder is None else list(self.model.parameters()) + list(self.encoder.parameters()), lr=self.hparams.model.lr,weight_decay=self.hparams.model.weight_decay)

        self.steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        total_steps = self.trainer.max_epochs * self.steps_per_epoch

        scheduler = LambdaLR(
            opt,
            lr_lambda=CosineWithWarmup(
                warmup_steps=self.steps_per_epoch*self.hparams.model.warmup_epochs,
                total_steps=total_steps,
                warmup_factor=self.hparams.model.warmup_factor,
            ),
        )

        return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
    
