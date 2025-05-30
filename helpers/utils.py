import torch
import torch.nn as nn
from PIL import Image
from helpers.nca.utils import weights_to_ckpt
from helpers.tokenisation import mixed_untokenize
import numpy as np

from typing import Tuple

def get_weight_grad_norm(model:torch.nn.Module) -> tuple:
    weight_norm = 0
    grad_norm = 0

    total_params = 0

    for p in model.parameters():
        if p.grad is not None:
            grad_norm += torch.norm(p.grad).item()

        weight_norm += torch.norm(p).item()
        total_params += p.numel()

    weight_norm = weight_norm / total_params
    grad_norm = grad_norm / total_params
    return weight_norm, grad_norm


def create_ckpt_from_weight_samples(x_denoised):
    """
    Create a checkpoint from the weight samples.

    Args:
        x_denoised (torch.Tensor): Denoised weights.

    Returns:
        dict: Checkpoint dictionary.
    """
    w1, b1, w2 = unflatten_params(x_denoised, 48, 96, 12)
    weights = {
        "w1.weight": w1[0],
        "w1.bias": b1[0],
        "w2.weight": w2[0]
    }
    ckpt = weights_to_ckpt(weights=weights)
    return ckpt


def flatten_params(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor
) -> torch.Tensor:
    """
    Packs (w1, b1, w2) into a flat vector.
    
    Args:
      w1: (..., H, I) or (H, I)
      b1: (..., H)    or (H,)
      w2: (..., O, H) or (O, H)
    
    Returns:
      flat: (..., H*I + H + O*H)
    """
    # detect & normalize batch‐dim
    batched = (w1.dim() == 3)
    if not batched:
        w1 = w1.unsqueeze(0)
        b1 = b1.unsqueeze(0)
        w2 = w2.unsqueeze(0)

    bsz, H, I = w1.shape
    bsz2, H2   = b1.shape
    bsz3, O, H3 = w2.shape
    if not (bsz==bsz2==bsz3 and H==H2==H3):
        raise ValueError(f"Shape mismatch, got w1{w1.shape}, b1{b1.shape}, w2{w2.shape}")

    flat_w1 = w1.reshape(bsz, -1)      # (bsz, H*I)
    flat_b1 = b1.reshape(bsz, -1)      # (bsz, H)
    flat_w2 = w2.reshape(bsz, -1)      # (bsz, O*H)
    flat    = torch.cat([flat_w1, flat_b1, flat_w2], dim=1)

    return flat if batched else flat.squeeze(0)


def unflatten_params(
    flat: torch.Tensor,
    in_dim: int,
    hidden_dim: int,
    out_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverts flatten_params.
    
    Args:
      flat:  (bsz, H*I + H + O*H) or (H*I + H + O*H)
      in_dim:   I
      hidden_dim: H
      out_dim:  O
    
    Returns:
      w1: (bsz, H, I) or (H, I)
      b1: (bsz, H)    or (H,)
      w2: (bsz, O, H) or (O, H)
    """
    # detect & normalize batch‐dim
    batched = (flat.dim() == 2)
    if not batched:
        flat = flat.unsqueeze(0)

    bsz, total = flat.shape
    n_w1 = hidden_dim * in_dim
    n_b1 = hidden_dim
    n_w2 = out_dim * hidden_dim
    if total != (n_w1 + n_b1 + n_w2):
        raise ValueError(f"Flat vector has length {total}, expected {n_w1+n_b1+n_w2}")

    w1_flat = flat[:, :n_w1]
    b1_flat = flat[:, n_w1:n_w1 + n_b1]
    w2_flat = flat[:, n_w1 + n_b1:]

    w1 = w1_flat.reshape(bsz, hidden_dim, in_dim)
    b1 = b1_flat.reshape(bsz, hidden_dim)
    w2 = w2_flat.reshape(bsz, out_dim, hidden_dim)

    if not batched:
        w1 = w1.squeeze(0)
        b1 = b1.squeeze(0)
        w2 = w2.squeeze(0)

    return w1, b1, w2


def get_pretrained_sequential(w1, b1, w2):

    # Model definition
    mlp = nn.Sequential(
        nn.Linear(48, 96, bias=True),
        nn.ReLU(),
        nn.Linear(96, 12, bias=False),
    )

    # Load weights into model
    with torch.no_grad():
        mlp[0].weight.copy_(w1)
        mlp[0].bias.copy_(b1)
        mlp[2].weight.copy_(w2)

    return mlp

def get_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(size=(128, 128),resample=Image.LANCZOS)
    img = np.float32(img) / 255.0
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    return img