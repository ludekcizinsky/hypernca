"""Plot Images for H1 Experiments Over Time (epochs)"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import lightning as L


if not Path("results").exists():
    raise FileNotFoundError("Results directory does not exist. Please run the training python viz_results/h1.py first")

for SEED in range(50, 100):


    L.seed_everything(SEED)


    all_samples = Path("results/nca_training_from_random_2025-05-03_23/epoch_0")

    all_samples = list(all_samples.glob("predicted_*.jpg"))
    all_samples = [str(file) for file in all_samples]
    all_samples = sorted(all_samples, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    all_samples = [x.split("/")[-1] for x in all_samples]

    epochs = [25,150,275,300]


    subset_samples = np.random.choice(all_samples, 1, replace=False)

    group1 = "nca_training_from_random_2025-05-03_23"
    group2 = "traing_from_bubbly_weights_2025-05-04_13"
    group3 = "training_from_diffusion_sampled_weights_2025-05-04_13"
    group4 = "training_from_diffusion_sampled_weights_2025-05-04_13_26"
    group5 = "training_from_diffusion_sampled_weights_2025-05-18_12_16"

    group_name_dict = {
        'nca_training_from_random_2025-05-03_23':'Random Weights',
        'traing_from_bubbly_weights_2025-05-04_13':'Bubbly Weights',
        'training_from_diffusion_sampled_weights_2025-05-04_13':'UD (T=50)',
        #'training_from_diffusion_sampled_weights_2025-05-04_13_26':'UD (T=500)',
        'training_from_diffusion_sampled_weights_2025-05-18_12_16':'CD (T=50)',
        'GT':'GT',
    }


    groups = [group1, group2, group3, group5]
    use_gt:bool = True

    def find_samples(group,epoch:int,gt:bool=False):
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



    N,M = len(groups)+1 if use_gt else len(groups), len(epochs)
    fig,axs = plt.subplots(N, M, figsize=(M*2, N*2), dpi=300,gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    for i in range(N):
        for j,epoch in enumerate(epochs):
            if use_gt and i == 0 and j == 0:
                group = groups[i]
                img = find_samples(group, gt=use_gt,epoch=epoch)[0]
                group = "GT"

                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                if j == 0:
                    # Set vertical y label
                    axs[i,j].text(-0.1, 0.5, group_name_dict[group],
                                    fontsize=12, fontweight='bold', va='center',
                                    ha='right', rotation=90, transform=axs[i,j].transAxes)
                else:
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    axs[i,j].set_yticklabels([])
                    axs[i,j].set_xticklabels([])
                if i == 0:
                    # Set horizontal x label
                    axs[i,j].set_title(f"Epoch {epoch}", fontsize=12, fontweight='bold')
                axs[i,j].set_aspect('equal')
                axs[i,j].set_frame_on(False)
            elif use_gt and i==0 and j >0:
                # Plot empty image
                axs[i,j].imshow(np.ones((128, 128, 3)))
                axs[i,j].axis('off')
                axs[i,j].set_title(f"Epoch {epoch}", fontsize=12, fontweight='bold')
            else:
                group = groups[i-1] if use_gt else groups[i]
                img = find_samples(group, gt=False,epoch=epoch)[0]

                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                if j == 0:
                    # Set vertical y label
                    axs[i,j].text(-0.1, 0.5, group_name_dict[group],
                                    fontsize=12, fontweight='bold', va='center',
                                    ha='right', rotation=90, transform=axs[i,j].transAxes)
                else:
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    axs[i,j].set_yticklabels([])
                    axs[i,j].set_xticklabels([])
                if i == 0:
                    # Set horizontal x label
                    axs[i,j].set_title(f"Epoch {epoch}", fontsize=12, fontweight='bold')
                axs[i,j].set_aspect('equal')
                axs[i,j].set_frame_on(False)

    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"time_samples_{SEED}_{epochs}.png", dpi=300, pad_inches=0.1, bbox_inches='tight')



















