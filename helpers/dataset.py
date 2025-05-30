import os

import PIL
import numpy as np
import torch
from torch import utils
from PIL import Image
from helpers.tokenisation import mixed_tokenize, mixed_untokenize
from helpers.utils import flatten_params, get_image_tensor


def get_dataloaders(cfg):
    dataset = ParamDataset(cfg)
    dataset.weight_only_mode = True

    train_size = int(cfg.data.trn_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_tokens = torch.stack([train_dataset[i]['weights'] for i in range(len(train_dataset))])  # shape: (N, 61, 96)
    train_mean, train_std = train_tokens.mean(dim=0, keepdim=True), train_tokens.std(dim=0, keepdim=True)
    normaliser = Normalizer(train_mean, train_std, cfg.diffusion.sigma_data)

    dataset.weight_only_mode = False
    train_dataset.weight_only_mode = False
    val_dataset.weight_only_mode = False

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    return train_dataloader, val_dataloader, normaliser

class Normalizer:
    def __init__(self, mean, std, sigma_data=0.5) -> None:
        self.mean = mean
        self.std = std
        self.sigma_data = sigma_data

    @torch.no_grad()
    def transform(self, x):
        # Set std to 0.5 to match the EDM config
        device = x.device
        return (x - self.mean.to(device)) * self.sigma_data / (self.std.to(device) + 1e-8)

    @torch.no_grad()
    def inverse_transform(self, x):
        device = x.device
        return x * (self.std.to(device) + 1e-8) / self.sigma_data + self.mean.to(device)

    def transform_with_grad(self, x):
        device = x.device
        return (x - self.mean.to(device)) * self.sigma_data / (self.std.to(device) + 1e-8)

    def inverse_transform_with_grad(self, x):
        device = x.device
        return x * (self.std.to(device) + 1e-8) / self.sigma_data + self.mean.to(device)

class ParamDataset(utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.weight_only_mode:bool = False

        param_path = cfg.data.nca_weights_path
        self.base_path = param_path.split('hypernca')[0]
        self.w1 = torch.from_numpy(np.load(os.path.join(param_path, f"w1.npy")))
        self.b1 = torch.from_numpy(np.load(os.path.join(param_path, f"b1.npy")))
        self.w2 = torch.from_numpy(np.load(os.path.join(param_path, f"w2.npy")))

        self.texture_names = np.load(os.path.join(param_path, "texture_names.npy"))

        self.w_token = mixed_tokenize(self.w1, self.b1, self.w2) # -> N x 61 x 96


    def __getitem__(self, index:int) -> None:

        # Load the weights according to the graph encoder config
        if self.cfg.data.uses_graph_encoder:
            weights = flatten_params(self.w1[index], self.b1[index], self.w2[index]) # -> (5856,)
        else:
            weights = self.w_token[index] # -> (61, 96)

        # Load the texture img
        if not self.weight_only_mode:
            texture_name = self.texture_names[index]
            nca_img_path = f"hypernca/images/nca_flickr+dtd_128/{texture_name}.jpg"
            nca_img_full_path = os.path.join(self.base_path, nca_img_path)
            nca_img = get_image_tensor(nca_img_full_path)

            gt_img_path = f"hypernca/images/flickr+dtd_128/{texture_name}.jpg"  
            gt_img_full_path = os.path.join(self.base_path, gt_img_path)
            gt_img = get_image_tensor(gt_img_full_path)
            return {'weights':weights,'nca_image':nca_img,'gt_image':gt_img,'texture':texture_name}
        return {'weights':weights}

    def __len__(self):
        return len(self.w_token)
    

if __name__ == "__main__":

    if False:
        param_path = "/scratch/izar/cizinsky/hypernca/pretrained_nca/Flickr+DTD_NCA"

        dataset = ParamDataset(param_path)
        print(len(dataset))
        total = 0
        for i in range(len(dataset)):
            item = dataset[i]
            print(item["texture_name"])
            print(item["w1"].shape)
            print(item["b1"].shape)
            print(item["w2"].shape)
            print()

            total += 1

            if total > 10:
                break
    

    from omegaconf import OmegaConf
    config_path = "/home/cizinsky/x-to-nif/configs/train.yaml"
    cfg = OmegaConf.load(config_path)

    trn_dataloader, val_dataloader = get_dataloaders(cfg)


    for i, batch in enumerate(trn_dataloader):
        print(batch["texture_name"])
        print(batch["w1"].shape)
        print(batch["b1"].shape)
        print(batch["w2"].shape)
        print()

        break