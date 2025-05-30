import wandb
from helpers import nca
from helpers.pl_module import WeightDenoiser
from omegaconf import DictConfig 
import hydra
from helpers.dataset import get_dataloaders
from helpers.diffusion import get_diffusion
from helpers.model import WeightDiffusionTransformer
from hydra import initialize, compose
import torch
from tqdm import tqdm
from helpers.evaluator import Evaluator
from helpers.generator import plot_comparison

models ={
    #'baseline_conditional_gram':['ludekcizinsky/hypernca/model-sdnv2k1z:v3','conditional_gram_baseline.yaml'],

    # 'conditional_clip': ['ludekcizinsky/hypernca/model-4kkhlc27:v3','conditional_clip.yaml'],

    # 'conditional_gram_cross_attn': ['ludekcizinsky/hypernca/model-abpao9wn:v3','conditional_gram_cross_attn.yaml'],

    # 'conditional_clip_cross_attn': ['ludekcizinsky/hypernca/model-orel1kfr:v3','conditional_clip_cross_attn.yaml'],

    # 'conditional_vit': ['ludekcizinsky/hypernca/model-gmgakqz4:v3','conditional_vit.yaml'],

    # 'conditional_vit_cross_attn': ['ludekcizinsky/hypernca/model-mxuzzmgt:v3','conditional_vit_cross_attn.yaml'],

    'baseline_ddim': ['ludekcizinsky/hypernca/model-qpnb6ln6:v3','conditional_ddim_gram_baseline.yaml'],
}


def download_ckpt(artifact_path:str) -> str:
    """
    Downloads the checkpoint from the given artifact path.
    """
    artifact = wandb.Api().artifact(artifact_path)
    artifact_dir = artifact.download()
    return f"{artifact_dir}/model.ckpt"


def list2tensor(tensor_list:list) -> torch.Tensor:
    tensor_list = torch.stack(tensor_list)
    tensor_list = torch.nan_to_num(tensor_list, nan=0.0, posinf=1.0, neginf=0.0)
    tensor_list = torch.clamp(tensor_list, 0, 1)
    return tensor_list.to('cuda:0')


def get_model_type_and_ckpt_path(model_type:str) -> tuple[DictConfig,str]:
    model_type,cfg_path = models[model_type]

    with initialize(config_path="configs"):
        cfg = compose(config_name=cfg_path)

    return cfg, model_type

def setup_model(cfg:DictConfig,trn_dataloader,normaliser) -> WeightDenoiser:
    encoder = None
    if cfg.model.conditioning:
        encoder = hydra.utils.instantiate(cfg.texture_encoder)
        test_inp = next(iter(trn_dataloader))["image"]
        cond_dim = encoder(test_inp).shape[-1]
        cfg.model.cond_dim = cond_dim
        print(f"Updated cond_dim: {cfg.model.cond_dim}")

    diffusion = get_diffusion(cfg)
    model = WeightDiffusionTransformer(cfg)

    pl_module = WeightDenoiser(cfg=cfg, model=model, diffusion=diffusion, normaliser=normaliser,encoder=encoder)
    pl_module.to('cuda:0')
    return pl_module    


def eval(model_type:str,num_diffusion_steps,num_steps,compare2nca) -> None:
    """
    Evaluates the model on the given dataset.
    """
    # Load the model and data
    cfg, model_type_ = get_model_type_and_ckpt_path(model_type)
    ckpt = download_ckpt(model_type_)
    trn_dataloader, val_dataloader, normaliser = get_dataloaders(cfg)
    pl_module = setup_model(cfg,trn_dataloader,normaliser)


    w = torch.load(ckpt, map_location='cuda',weights_only=False)
    pl_module.load_state_dict(w['state_dict'],strict=True)

    eval_gt2nca = Evaluator()
    eval_nca2diff = Evaluator()

    for batch in tqdm(val_dataloader):
        weights, cond = batch['weights'],batch.get('image', None)
        B = weights.shape[0]
        cond = cond.to(pl_module.device)
        if pl_module.encoder is not None:
            cond = pl_module.encoder(cond)
        else:
            cond = None
        gt_images = batch.get('image', None)

        if gt_images is not None:
            gt_images = gt_images.to(pl_module.device)

        # Diffusion Weights To RGB predictions
        samples = pl_module.sample(num_steps=num_diffusion_steps, cond=cond)
        pl_module.generator.generate(samples=samples,steps=num_steps)
        # Diffusion Log metrics compared to GT
        pl_module.generator.compute_metrics(gt_images=gt_images)
        diffusion_generated_images = pl_module.generator.generated_images
        diffusion_generated_images = list2tensor(diffusion_generated_images)
        pl_module.generator.generated_images = []

        # GET NCA Weights To RGB predictions
        nca_generated_images = None
        if compare2nca:
            pl_module.generator.generate(samples=weights,steps=num_steps)
            # NCA Log metrics compared to GT
            nca_generated_images = pl_module.generator.generated_images
            nca_generated_images = list2tensor(nca_generated_images)
            pl_module.generator.generated_images = []
            eval_gt2nca.update(gt_images,nca_generated_images)

            # Compare Diffusion and NCA generated images
            eval_nca2diff.update(nca_generated_images,diffusion_generated_images)

        titles = batch.get('texture',None)
        if titles is not None:
            titles = [f"{title}_{model_type}_diff_steps_{num_diffusion_steps}_steps_{num_steps}" for title in titles]
        else:
            titles = [f"sample_{i}" for i in range(B)]

        # ORDER OF IMAGES: GT | DIFFUSION | NCA
        plot_comparison(gt_images=gt_images,generated_images=diffusion_generated_images,generated_images2=nca_generated_images,titles=titles
        ,max_plots_per_batch=3)


    pl_module.generator.evaluator.compute()
    gt_diff_results = pl_module.generator.evaluator.get_results()
    pl_module.generator.evaluator.reset()

    eval_nca2diff.compute()
    nca_diff_results = eval_nca2diff.get_results()
    eval_nca2diff.reset()

    eval_gt2nca.compute()
    gt_nca_results = eval_gt2nca.get_results()
    eval_gt2nca.reset()

    return gt_diff_results,nca_diff_results,gt_nca_results

if __name__ == "__main__":
    import lightning as L
    import pandas as pd
    from pathlib import Path

    for MT in models:
        for diff_steps in [100]:
            for steps in [50]:
                MODEL_TYPE = MT
                NUM_DIFFUSION_STEPS = diff_steps # Number of diffusion steps for sampling
                NUM_STEPS = steps # Number of steps for NCA generation
                COMPARE2NCA = True # Compare to NCA generated images in addition to GT
                # Extend by add variables for noise levels here 
                # ....
                # ....
                L.seed_everything(2025)

                cond_diff_results_path = Path("cond_diff_results.json")

                gt_diff_results,nca_diff_results,gt_nca_results = eval(
                    model_type=MODEL_TYPE,
                    num_diffusion_steps=NUM_DIFFUSION_STEPS,
                    num_steps=NUM_STEPS,
                    compare2nca=COMPARE2NCA
                )

                if cond_diff_results_path.exists():
                    df = pd.read_json(cond_diff_results_path)

                else:
                    df = pd.DataFrame({
                        'model_type': [],
                        'compare_type':[],
                        'num_diffusion_steps': [],
                        'num_steps': [],
                        'fid': [],
                        'psnr': [],
                        'lpips': [],
                        'kid_mean': [],
                        'kid_std': [],
                    })

                # GT vs Diffusion
                df = pd.concat([df, pd.DataFrame({
                    'model_type': [MODEL_TYPE],
                    'compare_type': ['gt_diffusion'],
                    'num_diffusion_steps': [NUM_DIFFUSION_STEPS],
                    'num_steps': [NUM_STEPS],
                    'fid': [gt_diff_results['fid']],
                    'psnr': [gt_diff_results['psnr']],
                    'lpips': [gt_diff_results['lpips']],
                    'kid_mean': [gt_diff_results['kid_mean']],
                    'kid_std': [gt_diff_results['kid_std']],
                })], ignore_index=True)


                # GT vs NCA
                df = pd.concat([df, pd.DataFrame({
                    'model_type': [MODEL_TYPE],
                    'compare_type': ['gt_nca'],
                    'num_diffusion_steps': [NUM_DIFFUSION_STEPS],
                    'num_steps': [NUM_STEPS],
                    'fid': [gt_nca_results['fid']],
                    'psnr': [gt_nca_results['psnr']],
                    'lpips': [gt_nca_results['lpips']],
                    'kid_mean': [gt_nca_results['kid_mean']],
                    'kid_std': [gt_nca_results['kid_std']],
                })], ignore_index=True)

                # NCA vs Diffusion
                df = pd.concat([df, pd.DataFrame({
                    'model_type': [MODEL_TYPE],
                    'compare_type': ['nca_diffusion'],
                    'num_diffusion_steps': [NUM_DIFFUSION_STEPS],
                    'num_steps': [NUM_STEPS],
                    'fid': [nca_diff_results['fid']],
                    'psnr': [nca_diff_results['psnr']],
                    'lpips': [nca_diff_results['lpips']],
                    'kid_mean': [nca_diff_results['kid_mean']],
                    'kid_std': [nca_diff_results['kid_std']],
                })], ignore_index=True)

                df.to_json(cond_diff_results_path)





