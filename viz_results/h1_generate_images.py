"""Plot images for H1 Experiments"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import lightning as L

if not Path("results").exists():
    raise FileNotFoundError("Results directory does not exist. Please run the training python viz_results/h1.py first")



SEED = 123


L.seed_everything(SEED)


all_samples = Path("results/nca_training_from_random_2025-05-03_23/epoch_0")

all_samples = list(all_samples.glob("predicted_*.jpg"))
all_samples = [str(file) for file in all_samples]
all_samples = sorted(all_samples, key=lambda x: int(x.split("_")[-1].split(".")[0]))
all_samples = [x.split("/")[-1] for x in all_samples]

num_samples = 5
epoch = 975


subset_samples = np.random.choice(all_samples, num_samples, replace=False)

group1 = "nca_training_from_random_2025-05-03_23"
group2 = "traing_from_bubbly_weights_2025-05-04_13"
group3 = "training_from_diffusion_sampled_weights_2025-05-04_13"
group4 = "training_from_diffusion_sampled_weights_2025-05-04_13_26"

group_name_dict = {
    'nca_training_from_random_2025-05-03_23':'Random Weights',
    'traing_from_bubbly_weights_2025-05-04_13':'Bubbly Weights',
    'training_from_diffusion_sampled_weights_2025-05-04_13':'UD (T=50)',
    'training_from_diffusion_sampled_weights_2025-05-04_13_26':'UD (T=500)',
    'GT':'GT',
}


groups = [group1, group2, group3, group4]
use_gt:bool = True

def find_samples(group,gt:bool=False):
    path = Path(f"results/{group}/epoch_{epoch}")
    path = list(path.glob("predicted_*.jpg"))
    path = [str(file) for file in path if file.name in subset_samples]
    path = sorted(path, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if gt:
        path = [x.replace("predicted", "gt") for x in path]
    path = [x.split("/")[-1] for x in path]
    img_ = []
    for p in path:
        img = Image.open(f"results/{group}/epoch_{epoch}/{p}").convert('RGB')
        img.thumbnail(size=(128, 128), resample=Image.LANCZOS)
        img = np.float32(img) / 255.0
        img = np.array(img)
        img_.append(img)
    return img_
    

N,M = len(groups)+1 if use_gt else len(groups), num_samples
fig,axs = plt.subplots(M, N, figsize=(M*2, N*2), dpi=300,gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

for i in range(N):
    if use_gt and i == 0:
        group = groups[i]
        images = find_samples(group, gt=use_gt)
        group = "GT"
    else:
        group = groups[i-1] if use_gt else groups[i]
        images = find_samples(group, gt=False)

    for j, img in enumerate(images):
        axs[j,i].imshow(img)
        axs[j,i].axis('off')
        if j == 0:
            axs[j,i].set_title(group_name_dict[group], fontsize=14, fontweight='bold')
        axs[j,i].set_xticks([])
        axs[j,i].set_yticks([])
        axs[j,i].set_yticklabels([])
        axs[j,i].set_xticklabels([])
        axs[j,i].set_aspect('equal')
        axs[j,i].set_frame_on(False)

#plt.subplots_adjust(wspace=0.001, hspace=0.01)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.savefig(f"epoch_{epoch}_samples_{SEED}.png", dpi=300, bbox_inches='tight', pad_inches=0.02)




















