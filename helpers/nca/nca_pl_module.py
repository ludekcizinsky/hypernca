import torch
import lightning as L
import numpy as np
from torch import nn, Tensor
from PIL import Image
import imageio
import wandb
from typing_extensions import Self


class NCA_pl(L.LightningModule):
    def __init__(self, nca:nn.Module,criterion:nn.Module) -> None:
        """
        Lightning wrapper for NCA model.

        Args:
            nca (nn.Module) : NCA model to be wrapped.
        """
        super().__init__()
        self.nca = nca
        self.criterion = criterion
        for param in self.criterion.parameters():
            param.requires_grad = False
        self.criterion.eval()


        self.log_interval = 25
        self.pool_size = 256
        self.batch_size = 4
        self.inject_seed_step = 8

        self.automatic_optimization = False


    def forward(self, x:Tensor) -> Tensor:
        """
        Forward pass through the NCA model.

        Args:
            x (Tensor) : Input tensor.

        Returns:
            Tensor
        """
        return self.nca(x)
    
    @classmethod
    def load_from_checkpoint(cls,checkpoint_path:str,map_location=None,nca:nn.Module=None,criterion:nn.Module=None,strict:bool=True)-> Self:
        """
        Override standard load_from_checkpoint to load NCA model,
        as criterion contains parameters that we don't expect to be in the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # Load the model
        model = cls(nca=nca, criterion=criterion)
        model.nca.load_state_dict(checkpoint, strict=strict)

        return model

    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000,2000],gamma=0.5)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': lr_sched,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    

    def training_step(self, batch:tuple[Tensor,str], batch_idx:int) -> Tensor:
        """
        Training step for the NCA model.

        Args:
            batch (tuple[Tensor,str]) : tgt image, image path
            batch_idx (int) : Batch index.

        Returns:
            Tensor
        """
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        if self.current_epoch == 0:
            self.pool = self.nca.seed(256).to(self.device)

        target_image,image_path = batch


        with torch.no_grad():
            batch_idx = np.random.choice(self.pool_size, self.batch_size, replace=False)
            x = self.pool[batch_idx]
            if self.current_epoch % self.inject_seed_step == 0:
                    x[:1] = self.nca.seed(1)

        step_n = np.random.randint(32, 128)

        for _ in range(step_n):
            x = self.nca(x)

        overflow_loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
        texture_loss, texture_loss_per_img = self.criterion(target_image, self.nca.to_rgb(x))
        loss = texture_loss + 10.0 * overflow_loss

        with torch.no_grad():
            loss.backward()
            for p in self.nca.parameters():
                p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients
            opt.step()
            opt.zero_grad()
            lr_sched.step()

            self.pool[batch_idx] = x


        self.log('train_overflow_loss', overflow_loss, on_step=False, on_epoch=True, prog_bar=False,batch_size=1)
        self.log('train_texture_loss', texture_loss, on_step=False, on_epoch=True, prog_bar=False,batch_size=1)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=1)


        if self.current_epoch % self.log_interval == 0:
            imgs = self.nca.to_rgb(x[:4]).permute([0, 2, 3, 1]).detach().cpu().numpy()
            imgs = np.hstack((np.clip(imgs, 0, 1) * 255.0).astype(np.uint8))
            # Log image
            if self.logger.experiment.__class__.__name__ == 'ExperimentWriter':
                Image.fromarray(imgs).save(f'{self.logger.log_dir}/epoch-{self.current_epoch}.png')
            elif self.logger.__class__.__name__ == "WandbLogger":
                self.logger.experiment.log({
                f"train/generated_images_epoch={self.current_epoch}": [wandb.Image(imgs)],
                "global_step": self.global_step
            })

            img = self.generate(steps=50, size=128,generate_video=False)
            # Log image
            if self.logger.experiment.__class__.__name__ == 'ExperimentWriter':
                Image.fromarray(img).save(f'{self.logger.log_dir}/epoch-{self.current_epoch}_generated.png')
            elif self.logger.__class__.__name__ == "WandbLogger":
                self.logger.experiment.log({
                f"train/generated_image_epoch={self.current_epoch}": [wandb.Image(img)],
                "global_step": self.global_step
            })

        return loss

    #@torch.no_grad()
    def generate(self,steps:int=500,size:int=128,video_path:str=None,generate_video:bool=False,to_uint8:bool=True) -> None:
        """
        Generate images using the NCA model.
        Args:
            steps (int) : Number of steps to generate.
        """
        if generate_video and not to_uint8:
            raise ValueError("to_uint8 must be True when generate_video is True")
        s = self.nca.seed(1, size=size).to(self.device)
        step_n = 8

        if video_path is None and generate_video:
            # create random video name
            from secrets import token_hex
            video_path = f"video_{token_hex(8)}.gif"

        frames = []

        for _ in range(steps):
            for _ in range(step_n):
                s[:] = self.nca(s)
            img = self.nca.to_rgb(s[0])
            if to_uint8:
                img = img.permute(1, 2, 0).detach().cpu().numpy()
                img = (img * 255.0).astype(np.uint8)
            if generate_video:
                frames.append(img)

        if generate_video:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved GIF at: {video_path}")

        return img

if __name__ == "__main__":
    from helpers.nca.nca_data import TextureDataset,TextureImageDataset
    from helpers.nca.utils import weights_to_ckpt,nca_weights
    from helpers.nca.nca_model import NCA
    from helpers.nca.nca_loss import TextureLoss
    from torch.utils.data import DataLoader
    texture_dataset = TextureDataset(data_dir='hypernca/images/flickr+dtd_128')
    texture_dataset.filter_images('banded_0002')
    for img_path in texture_dataset.images:
        dataset = TextureImageDataset(img_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        nca = NCA()
        criterion = TextureLoss()

        ckpt = weights_to_ckpt(nca_weights()['banded_0002'])
        torch.save(ckpt, 'test.ckpt')
        nca_lit =  NCA_pl.load_from_checkpoint('test.ckpt',nca=nca,criterion=criterion)
        nca_lit.to('cuda:0')

        nca_lit.generate(steps=100, size=128, video_path=f"video_{img_path.stem}.gif")