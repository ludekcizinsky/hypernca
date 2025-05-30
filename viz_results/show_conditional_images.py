import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

# img_path = 'flickr_016519_conditional_gram_cross_attn_diff_steps_100_steps_50.png'

diff_steps = 100
steps = 50


#images = ['flickr_016519','honeycombed_0080','lacelike_0003']
images = ['flickr_024119','flickr_023172','flickr_020575']
#images = ['flickr_019915','flickr_014438','flickr_007171']
#images = ['flickr_024207']

img_list = list(Path('images').glob("*.png"))
img_list = [x.name for x in img_list]
models = [x for x in img_list if x.startswith(images[0]) and x.endswith('.png') and str(diff_steps) in x and str(steps) in x] 


model_translator = {
    'conditional_clip_diff': 'CLIP',
    'baseline_conditional_gram_diff': 'Sequential',
    'conditional_vit_diff': 'ViT',
    'conditional_clip_cross_attn_diff': 'CLIP + CA',
    'conditional_gram_cross_attn_diff': 'Gram + CA',
    'conditional_vit_cross_attn_diff': 'ViT + CA',
    #'baseline_ddim_diff': 'DDIM',
    #'graph_diff': 'Graph',

}


M = len(model_translator)
N = len(images)

TEXTSIZE = 10
fig, axs = plt.subplots(N,M+2,figsize=(M*1.5, N*0.75), dpi=300,gridspec_kw={'wspace': 0.00, 'hspace': 0.0})

for j,image in enumerate(images):
    mod_list = list(Path('images').glob("*.png"))
    mod_list = [x.name for x in mod_list]

    models = [x for x in mod_list if x.startswith(image) and x.endswith('.png') and str(diff_steps) in x and str(steps) in x] 


    # If any of keys of model_translator not in models then remove it from models
    drop_indices = []
    for i in range(len(models)):
        is_present = False
        for key in model_translator.keys():
            if key in models[i]:
                is_present = True
                break
        if not is_present:
            drop_indices.append(i)
            
    # Remove the indices from the list
    for i in sorted(drop_indices, reverse=True):
        models.pop(i)

    row_iter = 0
    for i,model_path in enumerate(models):
        split =model_path.split('_')
        idx = [x for x in range(len(split)) if split[x] == 'steps'][0]
        model = ('_'.join(split[:idx]))[len(image)+1:]

        img = Image.open(f"images/{model_path}")
        img = img.convert('RGB')
        img = np.array(img)
        gt = img[:,:128,:]
        diff = img[:,128:256,:]
        nca = img[:,256:384,:]

        if N==1:
            if i == 0:

                axs[row_iter].imshow(gt)
                axs[row_iter].set_xticks([])
                axs[row_iter].set_yticks([])
                axs[row_iter].set_yticklabels([])
                axs[row_iter].set_xticklabels([])
                axs[row_iter].set_aspect('equal')
                axs[row_iter].set_frame_on(False)
                axs[row_iter].set_title('GT', fontsize=TEXTSIZE, fontweight='bold')
                row_iter += 1
                axs[row_iter].imshow(nca)
                axs[row_iter].set_xticks([])
                axs[row_iter].set_yticks([])
                axs[row_iter].set_yticklabels([])
                axs[row_iter].set_xticklabels([])
                axs[row_iter].set_aspect('equal')
                axs[row_iter].set_frame_on(False)
                axs[row_iter].set_title('NCA', fontsize=TEXTSIZE, fontweight='bold')

                row_iter += 1
            axs[row_iter].imshow(diff)
            axs[row_iter].set_xticks([])
            axs[row_iter].set_yticks([])
            axs[row_iter].set_yticklabels([])
            axs[row_iter].set_xticklabels([])
            axs[row_iter].set_aspect('equal')
            axs[row_iter].set_frame_on(False)
            axs[row_iter].set_title(model_translator[model], fontsize=TEXTSIZE, fontweight='bold')

            row_iter += 1
        else:
            if i == 0:
                axs[j,row_iter].imshow(gt)
                axs[j,row_iter].set_xticks([])
                axs[j,row_iter].set_yticks([])
                axs[j,row_iter].set_yticklabels([])
                axs[j,row_iter].set_xticklabels([])
                axs[j,row_iter].set_aspect('equal')
                axs[j,row_iter].set_frame_on(False)
                if j == 0:
                    axs[j,row_iter].set_title('GT', fontsize=TEXTSIZE, fontweight='bold')
                row_iter += 1
                axs[j,row_iter].imshow(nca)
                axs[j,row_iter].set_xticks([])
                axs[j,row_iter].set_yticks([])
                axs[j,row_iter].set_yticklabels([])
                axs[j,row_iter].set_xticklabels([])
                axs[j,row_iter].set_aspect('equal')
                axs[j,row_iter].set_frame_on(False)
                if j == 0:
                    axs[j,row_iter].set_title('NCA', fontsize=TEXTSIZE, fontweight='bold')
                row_iter += 1
            axs[j,row_iter].imshow(diff)
            axs[j,row_iter].set_xticks([])
            axs[j,row_iter].set_yticks([])
            axs[j,row_iter].set_yticklabels([])
            axs[j,row_iter].set_xticklabels([])
            axs[j,row_iter].set_aspect('equal')
            axs[j,row_iter].set_frame_on(False)
            if j == 0:
                axs[j,row_iter].set_title(model_translator[model], fontsize=TEXTSIZE, fontweight='bold')
            row_iter += 1

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.00, hspace=0.00)

plt.savefig('comparison_2.png',dpi=300, bbox_inches='tight', pad_inches=0.02)


# img = Image.open(img_path)
# img = img.convert('RGB')
# img = np.array(img)

# gt = img[:,:128,:]
# diff = img[:,128:256,:]
# nca = img[:,256:384,:]
# plt.imshow(gt)
# plt.savefig('gt.png')