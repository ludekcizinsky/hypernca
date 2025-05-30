import os
from argparse import ArgumentParser
import torch
from omegaconf import OmegaConf
import wandb
import json

from PIL import Image
from tqdm import tqdm

from helpers.dataset import get_dataloaders
from helpers.diffusion import get_diffusion
from helpers.graph_denoiser import GraphDenoiser
from helpers.pl_module import WeightDenoiser
from helpers.texture_encoding import GramEncoder, CLIP, VisionTransformer

import lightning as L

from torchmetrics.image import FrechetInceptionDistance, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity

DOWNLOAD_DIR = '/scratch/izar/cizinsky/x-to-nif/tmp'
RESULTS_DIR = '/home/cizinsky/x-to-nif/results_stored'
IMAGES_DIR = '/scratch/izar/cizinsky/x-to-nif/final-report/'

def download_ckpt(model_id, download_dir) -> str:
    artifact = wandb.Api().artifact(f'ludekcizinsky/hypernca/model-{model_id}:latest')
    artifact.download(download_dir)

def get_pl_module(model_id, artifact_dir, username='cizinsky', model_type='newest'):

    # download ckpt from wandb based on model_id
    download_ckpt(model_id, artifact_dir)

    # Get first path to checkpoint
    path_to_ckpt = os.path.join(artifact_dir, "model.ckpt")

    # Load config
    ckpt = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['hyper_parameters']
    if model_type == 'baseline':
        default_cfg = OmegaConf.load('configs/graph_conditional_gram_baseline.yaml')
        cfg = OmegaConf.merge(default_cfg, cfg)

    # Adjust the config
    OmegaConf.set_struct(cfg, False)
    cfg.data.nca_weights_path = f'/scratch/izar/{username}/hypernca/pretrained_nca/Flickr+DTD_NCA'
    cfg.model.type = model_type
    cfg.data.num_workers = 10
    if model_type == 'baseline':
        cfg.model.use_cross_attention = False

    # Load all the other components
    _, val_dataloader, normaliser = get_dataloaders(cfg)

    diffusion = get_diffusion(cfg)
    model = GraphDenoiser(cfg)

    if "Gram" in cfg.texture_encoder._target_:
        encoder = GramEncoder(hidden_size=cfg.texture_encoder.hidden_size, normalize=cfg.texture_encoder.normalize)
    elif "CLIP" in cfg.texture_encoder._target_:
        proj_dim = cfg.texture_encoder.proj_dim if cfg.texture_encoder.proj_dim is not None else None
        encoder = CLIP(proj_dim=proj_dim)
    elif "VisionTransformer" in cfg.texture_encoder._target_:
        encoder = VisionTransformer(
            pretrained=cfg.texture_encoder.pretrained,
            trainable=cfg.texture_encoder.trainable,
            num_hidden_layers=cfg.texture_encoder.num_hidden_layers,
            patch_size=cfg.texture_encoder.patch_size,
            hidden_dim=cfg.texture_encoder.hidden_dim,
            num_layers=cfg.texture_encoder.num_layers,
            num_heads=cfg.texture_encoder.num_heads,
            mlp_dim=cfg.texture_encoder.mlp_dim,
            image_size=cfg.texture_encoder.image_size,
        )

    pl_module = WeightDenoiser(cfg=cfg, model=model, diffusion=diffusion, normaliser=normaliser,encoder=encoder).to('cuda')
    pl_module.load_state_dict(ckpt['state_dict'], strict=True)

    return pl_module, val_dataloader


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a 3xHxW image tensor to a PIL image, scaling values from [min, max] to [0, 255].

    Args:
        tensor (torch.Tensor): A 3xHxW image tensor with arbitrary value range.

    Returns:
        PIL.Image.Image: The converted image.
    """
    assert tensor.ndim == 3 and tensor.shape[0] == 3, "Expected tensor of shape [3, H, W]"
    
    # Normalize to [0, 1]
    tensor = tensor.clone()
    tensor -= tensor.min()
    tensor /= tensor.max().clamp(min=1e-8)

    # Convert to [0, 255] and uint8
    tensor = (tensor * 255).clamp(0, 255).byte()

    # Convert to CHW -> HWC
    array = tensor.permute(1, 2, 0).cpu().numpy()

    # Create PIL image
    return Image.fromarray(array)


def compute_metrics(preds_list, gt_list, batch_size=180):
    assert len(preds_list) == len(gt_list), "Prediction and GT list lengths must match"

    device = preds_list[0].device

    # Initialize metrics
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=False).to(device)

    for i in tqdm(range(0, len(preds_list), batch_size)):
        pred_batch = torch.stack(preds_list[i:i + batch_size]).to(device)
        target_batch = torch.stack(gt_list[i:i + batch_size]).to(device)

        psnr_metric.update(pred_batch, target_batch)
        lpips_metric.update(pred_batch, target_batch)
        fid_metric.update(pred_batch, real=False)
        fid_metric.update(target_batch, real=True)

    # Compute final results
    psnr = psnr_metric.compute()
    lpips = lpips_metric.compute()
    fid = fid_metric.compute()

    return {
        'psnr': psnr.item(),
        'lpips': lpips.item(),
        'fid': fid.item()
    }

def evaluate_post_train(run_id, limit_predict_batches=None):
    L.seed_everything(2025)

    pl_module, val_dataloader = get_pl_module(run_id, DOWNLOAD_DIR, model_type='baseline')
    if limit_predict_batches is not None:
        trainer = L.Trainer(accelerator="gpu", devices=1, logger=False, callbacks=[], limit_predict_batches=limit_predict_batches)
    else:
        trainer = L.Trainer(accelerator="gpu", devices=1, logger=False, callbacks=[])
    outputs = trainer.predict(pl_module, val_dataloader)


    all_gen_imgs, all_gt_imgs, all_nca_imgs = [], [], []
    for output in outputs:
        for i in range(len(output['gen_imgs'])):
            gen_img = output['gen_imgs'][i]
            if torch.isnan(gen_img).any() or torch.isinf(gen_img).any():
                continue
            all_gen_imgs.append(gen_img)
            all_gt_imgs.append(output['gt_imgs'][i])
            all_nca_imgs.append(output['nca_imgs'][i])

    assert len(all_gen_imgs) == len(all_gt_imgs) == len(all_nca_imgs), f"Lengths are not equal: {len(all_gen_imgs)}, {len(all_gt_imgs)}, {len(all_nca_imgs)}"

    # save images
    save_images_run_dir = os.path.join(IMAGES_DIR, f'{run_id}_images')
    os.makedirs(save_images_run_dir, exist_ok=True)
    torch.save(all_gen_imgs, os.path.join(save_images_run_dir, 'all_gen_imgs.pt'))
    torch.save(all_gt_imgs, os.path.join(save_images_run_dir, 'all_gt_imgs.pt'))
    torch.save(all_nca_imgs, os.path.join(save_images_run_dir, 'all_nca_imgs.pt'))


    # save metrics as json
    metrics_against_gt = compute_metrics(all_gen_imgs, all_gt_imgs)
    metrics_against_nca = compute_metrics(all_gen_imgs, all_nca_imgs)
    run_metrics = {
        'metrics_against_gt': metrics_against_gt,
        'metrics_against_nca': metrics_against_nca
    }
    with open(os.path.join(RESULTS_DIR, f'{run_id}_final_report.json'), 'w') as f:
        json.dump(run_metrics, f)

if __name__ == "__main__":

    # parse args
    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    evaluate_post_train(args.run_id)