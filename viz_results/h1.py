"""Get Table Results For H1 Experiments"""


from pathlib import Path
import shutil
import re
import wandb
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from helpers.evaluator import Evaluator


def custom_sort_key(path):
    filename = path.split('/')[-1] 
    match = re.match(r'([a-zA-Z_]+)(\d+)\.jpg', filename)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (filename, 0)


def download_content(epoch:int,group_runs) -> None:
    download_dir = Path(f"results/{group_name}/epoch_{epoch}")
    download_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"generated_image_epoch={epoch}(?!\d)")

    for run in tqdm(group_runs):
        for file in run.files():
            match_generated = pattern.search(file.name)
            match_target = 'target_image_0' in file.name
            if match_generated or match_target:
                # Check if file already exists
                if Path(download_dir/f"predicted_{run.name}.jpg").exists() and Path(download_dir/f"gt_{run.name}.jpg").exists():
                    print(f"File {run.name} already exists, skipping download.")
                    continue
                    
                downloaded_path = Path(file.download(root=download_dir, replace=True).name)

                if match_generated:
                    new_name = f"predicted_{run.name}"
                elif match_target:
                    new_name = f"gt_{run.name}"

                new_path = f"{download_dir / new_name}.jpg"
                shutil.move(str(downloaded_path), str(new_path))

    media_path = download_dir/'media'
    if media_path.exists():
        shutil.rmtree(media_path)

    print(f"Downloaded files to {download_dir}")

def load_image_arrays(group_name,epoch:int,device='cuda:0') -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(f"results/{group_name}/epoch_{epoch}")
    gt_files = list(path.glob("gt_*.jpg"))
    pred_files = list(path.glob("predicted_*.jpg"))

    gt_files = [str(file) for file in gt_files]
    pred_files = [str(file) for file in pred_files]


    gt_files = sorted(gt_files, key=custom_sort_key)
    pred_files = sorted(pred_files, key=custom_sort_key)

    # Convert to numpy -> Tensor
    gt_tensor = []
    for file in gt_files:
        img = Image.open(file).convert('RGB')
        img.thumbnail(size=(128, 128),resample=Image.LANCZOS)
        img = np.float32(img) / 255.0
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        gt_tensor.append(img.to(device))
    gt_tensor = torch.stack(gt_tensor)

    pred_tensor = []
    for file in pred_files:
        img = Image.open(file).convert('RGB')
        img.thumbnail(size=(128, 128),resample=Image.LANCZOS)
        img = np.float32(img) / 255.0
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        pred_tensor.append(img.to(device))
    pred_tensor = torch.stack(pred_tensor)

    return gt_tensor, pred_tensor



if __name__ == "__main__":
    import pandas as pd
    api = wandb.Api()
    group_names = {
        'nca_random':'nca_training_from_random_2025-05-03_23',
        'nca_bubbly':'traing_from_bubbly_weights_2025-05-04_13',
        'nca_diffusion_step_50':'training_from_diffusion_sampled_weights_2025-05-04_13',
        'nca_diffusion_step_500':'training_from_diffusion_sampled_weights_2025-05-04_13_26',
        'nca_diffusion_step_50_cond_baseline':'training_from_diffusion_sampled_weights_2025-05-18_12_16'

    }

    for group_name in group_names:
        group_name = group_names[group_name]
        #if group_name == "nca_training_from_random_2025-05-03_23": continue
        #if group_name == "traing_from_bubbly_weights_2025-05-04_13": continue

        filters = {"group": group_name}

        group_runs = api.runs("ludekcizinsky/hypernca", filters=filters)

        for epoch in list(range(0, 1000, 25)):


            # Download content
            download_content(epoch, group_runs)
            gt_tensor, pred_tensor = load_image_arrays(group_name,epoch=epoch)
            evaluator = Evaluator()
            evaluator.update(gt_tensor, pred_tensor)
            evaluator.compute()
            results = evaluator.get_results()
    
            iter_df = pd.DataFrame({'fid': [results['fid']],
                                    'psnr': [results['psnr']],
                                    'lpips': [results['lpips']],
                                    'kid_mean': [results['kid_mean']],
                                    'kid_std': [results['kid_std']],
                                    'epoch': [epoch],
                                    'group_name': [group_name],
                                    })
    
            save_path = Path("h1_df_real.json")
            if save_path.exists():
                df = pd.read_json(save_path)
                df = pd.concat([df, iter_df], ignore_index=True)
            else:
                df = iter_df
            df.to_json(save_path)
    


