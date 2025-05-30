from pydoc import text
import torch
import wandb
from helpers.pl_module import WeightDenoiser
from omegaconf import OmegaConf
from helpers.model import WeightDiffusionTransformer
from helpers.diffusion import get_diffusion
from helpers.dataset import get_dataloaders
from helpers.nca.nca_pl_module import NCA_pl
from helpers.nca.nca_model import NCA
from helpers.nca.nca_loss import TextureLoss
from pathlib import Path
from helpers.utils import create_ckpt_from_weight_samples
from helpers.utils import weights_to_ckpt
import numpy as np
from tqdm import tqdm

ckpt_path = Path('ludekcizinsky/hypernca/model-a5yb2dgd:v3')
if not ckpt_path.exists():
    api = wandb.Api()
    artifact = api.artifact("ludekcizinsky/hypernca/model-a5yb2dgd:v3")
    artifact_dir = artifact.download()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--true_nca_weights',type=bool, default=False)
    args = parser.parse_args()
    num_samples = args.num_samples
    true_nca_weights = args.true_nca_weights
    print(f'num_samples: {num_samples}')
    print(f'true_nca_weights: {true_nca_weights}')



    if true_nca_weights:
        b1 = '/scratch/izar/theilgaa/hypernca/pretrained_nca/Flickr+DTD_NCA/b1.npy'
        w1 = '/scratch/izar/theilgaa/hypernca/pretrained_nca/Flickr+DTD_NCA/w1.npy'
        w2 = '/scratch/izar/theilgaa/hypernca/pretrained_nca/Flickr+DTD_NCA/w2.npy'
        texture_names = '/scratch/izar/theilgaa/hypernca/pretrained_nca/Flickr+DTD_NCA/texture_names.npy'


        w1 = torch.from_numpy(np.load(w1))
        b1 = torch.from_numpy(np.load(b1))
        w2 = torch.from_numpy(np.load(w2))
        texture_names = np.load(texture_names)
        texture_names = dict(zip(texture_names,list(range(len(texture_names)))))

        # random textures 
        textures = np.random.sample(list(texture_names.keys()), num_samples)


        for texture_name in tqdm(textures):
            texture_name = texture_names[texture_name]
            w1 = w1[texture_name]
            b1 = b1[texture_name]
            w2 = w2[texture_name]

            weights = {
                "w1.weight": w1,
                "w1.bias": b1,
                "w2.weight": w2
            }

            ckpt = weights_to_ckpt(weights=weights)

            # Load the NCA model
            nca_pl = NCA_pl(nca=NCA(),criterion=TextureLoss())
            nca_pl.nca.load_state_dict(ckpt['state_dict'])
            nca_pl.to('cuda:0')
            # Generate images
            nca_pl.generate(steps=100, size=128, video_path='video.gif', generate_video=True)

    else:
        ckpt = 'artifacts/model-a5yb2dgd:v3/model.ckpt'
        checkpoint = torch.load(ckpt,weights_only=False)

        # Load the WeightDiffusionTransformer model
        cfg = OmegaConf.load('configs/train.yaml')
        diffusion = get_diffusion(cfg)
        model = WeightDiffusionTransformer(cfg)
        trn_dataloader, val_dataloader, normaliser = get_dataloaders(cfg)
        pl_module = WeightDenoiser.load_from_checkpoint(checkpoint_path=ckpt,cfg=cfg,diffusion=diffusion,model=model,normaliser=normaliser)
        pl_module.to('cuda:0')


        # Sample weights and convert to checkpoint
        for i in tqdm(range(num_samples)):
            seed = np.random.randint(0, 1000000)
            x_denoised = pl_module.sample(num_steps=50,seed=seed)
            ckpt = create_ckpt_from_weight_samples(x_denoised)

            # Load the NCA model
            nca_pl = NCA_pl(nca=NCA(),criterion=TextureLoss())
            nca_pl.nca.load_state_dict(ckpt['state_dict'])
            nca_pl.to('cuda:0')
            # Generate images
            nca_pl.generate(steps=100, size=128, video_path=f'video_{i}.gif', generate_video=True)

