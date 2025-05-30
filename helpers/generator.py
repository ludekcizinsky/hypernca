from re import I
from torch import Tensor
from helpers.nca.nca_pl_module import NCA_pl
from helpers.nca.nca_model import NCA
from helpers.nca.nca_loss import TextureLoss
from helpers.utils import create_ckpt_from_weight_samples
from helpers.evaluator import Evaluator
import torch
from PIL import Image
import numpy as np
import wandb
import secrets
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

import wandb

def plot_comparison(gt_images:Tensor, generated_images:Tensor,generated_images2:Tensor=None,titles:list=None,max_plots_per_batch:int=None) -> None:
    assert len(gt_images) == len(generated_images)
    if generated_images2 is not None:
        assert len(generated_images) == len(generated_images2)

    for i in range(len(gt_images)):
        if max_plots_per_batch is not None and i >= max_plots_per_batch:
            break
        if generated_images2 is not None:    
            img = torch.cat(tensors=(gt_images[i],generated_images[i],generated_images2[i]), dim=2)
        else:
            img = torch.cat(tensors=(gt_images[i],generated_images[i]), dim=2)
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255.0).astype(np.uint8)
        Image.fromarray(img).save(f'{titles[i]}.png')

class Generator:
    def __init__(self):
        nca = NCA()
        criterion = TextureLoss()
        self.nca_pl = NCA_pl(nca=nca, criterion=criterion)
        for param in self.nca_pl.parameters():
            param.requires_grad = False
        self.nca_pl.eval()
        self.nca_pl.to('cuda:0')
        self.evaluator = Evaluator()

        self.generated_images = []

    
    def generate(self, samples:Tensor,steps:int=50):
        """
        Args: 
            samples (Tensor): Input tensor of shape (batch_size, 61,96).
        """
        for i in range(samples.shape[0]):
            ckpt = create_ckpt_from_weight_samples(samples[i].unsqueeze(0))
            self.nca_pl.nca.load_state_dict(ckpt['state_dict'])
            img = self.nca_pl.generate(steps=steps, size=128, generate_video=False,to_uint8=False)
            img = torch.clip(img, 0, 1)
            self.generated_images.append(img)

    def log_generated_images(self,gt_images,epoch,texture_names,logger=None, bidx=0) -> None:
        if logger is not None and logger.__class__.__name__ == "WandbLogger":
            images_with_captions = []
            for i in range(len(self.generated_images[:3])):
                gt_img = to_pil_image(gt_images[i].detach().cpu())
                gen_img = to_pil_image(self.generated_images[i].detach().cpu())

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
                axs[0].imshow(gt_img)
                axs[0].set_axis_off()
                axs[0].set_title('GT', fontsize=12)
                axs[1].imshow(gen_img)
                axs[1].set_axis_off()
                axs[1].set_title('Predicted', fontsize=12)
                caption = texture_names[i]

                wandb_image = wandb.Image(fig, caption=caption)
                images_with_captions.append(wandb_image)

            dict_to_log = dict()
            for i, image in enumerate(images_with_captions):
                dict_to_log[f'val-images/bidx_{bidx}-sample_{i}'] = image
            dict_to_log['epoch'] = epoch
            logger.experiment.log(dict_to_log)

        # for i in range(len(self.generated_images)):
        #     if i >= 3:
        #         break
        #     if gt_images is not None:
        #         img = torch.cat((gt_images[i],self.generated_images[i]), dim=2)
        #         if logger is None:
        #             img = img.permute(1, 2, 0).detach().cpu().numpy()
        #             img = (img * 255.0).astype(np.uint8)
        #             random_name = secrets.token_hex(8)
        #             Image.fromarray(img).save(f'generated_{random_name}.png')
        #         elif logger.experiment.__class__.__name__ == 'SummaryWriter':
        #             img = img.permute(1, 2, 0).detach().cpu().numpy()
        #             img = (img * 255.0).astype(np.uint8)
        #             Image.fromarray(img).save(f'{logger.log_dir}/epoch-{epoch}_generated_{texture_names[i]}.png')
        #         elif logger.__class__.__name__ == "WandbLogger":
        #             img = img.permute(1, 2, 0).detach().cpu().numpy()
        #             logger.experiment.log({
        #                 f"generated_image_epoch={epoch}_{texture_names[i]}": [wandb.Image(img)],
        #             })

        #     else:
        #         img = self.generated_images[i].permute(1, 2, 0).detach().cpu().numpy()
        #         if logger is None:
        #             img = (img * 255.0).astype(np.uint8)
        #             random_name = secrets.token_hex(8) if texture_names is None else texture_names[i]
        #             Image.fromarray(img).save(f'generated_{random_name}.png')
        #         elif logger.experiment.__class__.__name__ == 'SummaryWriter':
        #             img = (img * 255.0).astype(np.uint8)
        #             Image.fromarray(img).save(f'{logger.log_dir}/epoch-{epoch}_generated_{texture_names[i]}.png')
        #         elif logger.__class__.__name__ == "WandbLogger":
        #             logger.experiment.log({
        #                 f"generated_image_epoch={epoch}_{texture_names[i]}": [wandb.Image(img)],
        #             })

     

    def compute_metrics(self, gt_images:Tensor,generated_images:torch.Tensor=None) -> None:
        """
        Args:
            gt_images (Tensor): Ground truth images of shape (batch_size, 3, 128, 128).
        """
        if generated_images is None:
            generated_images = torch.stack(self.generated_images).to(gt_images.device)

        generated_images = torch.nan_to_num(generated_images, nan=0.0, posinf=1.0, neginf=0.0)
        generated_images = torch.clamp(generated_images, 0, 1)

        self.evaluator.update(gt_images, generated_images)


    def generate_and_log(self, samples:Tensor, gt_images:Tensor, epoch:int,texture_names:list, logger=None,bidx=0, return_images=False) -> None:
        """
        Args:
            samples (Tensor): Input tensor of shape (batch_size, 61,96).
            gt_images (Tensor): Ground truth images of shape (batch_size, 3, 128, 128).
            epoch (int): Current epoch.
        """
        self.generate(samples=samples)
        self.log_generated_images(gt_images=gt_images,epoch=epoch,logger=logger,texture_names=texture_names,bidx=bidx)
        self.compute_metrics(gt_images=gt_images)
        if return_images:
            imgs = self.generated_images
        else:
            imgs = None
        self.generated_images = []  # Clear the generated images after logging

        return imgs

    def log_metrics(self) -> None:
        """
        Args:
            epoch (int): Current epoch.
        """
        self.evaluator.compute()
        results = self.evaluator.log_results()
        return results



