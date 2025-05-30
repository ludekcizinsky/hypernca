import numpy as np
import torch

def nca_weights(path:str='hypernca/pretrained_nca/Flickr+DTD_NCA') -> dict:
    """
    Load NCA weights from pretrained directory.
    """
    w1 = np.load(f'{path}/w1.npy') # (30775,96,48)
    w2 = np.load(f'{path}/w2.npy') # (30775,12,96)
    b1 = np.load(f'{path}/b1.npy') # (30775,96)
    names = np.load(f'{path}/texture_names.npy') # (30775,)
    weight_dict = {
        name: {
            "w1.weight": torch.from_numpy(w1[i]).float(),
            "w2.weight": torch.from_numpy(w2[i]).float(),
            "w1.bias": torch.from_numpy(b1[i]).float()
        }
        for i, name in enumerate(names)
    }

    return weight_dict


def weights_to_ckpt(weights:dict) -> dict:
    """
    Convert weights to checkpoint format.

    Args:
        weights (dict) : Weights dictionary.

    Returns:
        dict
    """
    state_dict = {}
    for k, v in weights.items():
        if k in ['w1.weight','w2.weight']:
            state_dict[f'{k}'] = v[...,None,None]
        elif k == 'w1.bias':
            state_dict[f'{k}'] = v

    ckpt = {
        'state_dict': state_dict,
    }
    return ckpt
