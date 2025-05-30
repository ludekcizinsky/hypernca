"""
Inspired from https://github.com/wpeebles/G.pt/blob/main/Gpt/models/transformer.py
"""

import torch
import torch.nn as nn
import math
from torch import Tensor


@torch.no_grad()
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = torch.unsqueeze(timesteps, dim=0)
    if timesteps.dim() == 1:  # (N,)
        timesteps = timesteps.unsqueeze(1)  # (N, D) where D=1

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )

    #     print(timestep.shape, freqs.shape)
    angle = timesteps.float() * freqs[None, :]
    embedding = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class EncoderLayer(nn.Module):
    """
    A single (vanilla) transformer encoder layer.
    """

    def __init__(self, embed_dim:int, n_head:int,use_cross_attention:bool=False) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention


        self.attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)

        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
            self.ln_cross = nn.LayerNorm(embed_dim)
            self.kv_norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x:Tensor,cond:Tensor=None) -> Tensor:
        ln1_x = self.ln1(x)
        attention_output = self.attn(ln1_x, ln1_x, ln1_x, need_weights=False)[0]
        x = x + attention_output

        if self.use_cross_attention and cond is not None:
            ln_cross_x = self.ln_cross(x)
            kv_norm = self.kv_norm(cond)
            x = x + self.cross_attn(ln_cross_x, kv_norm, kv_norm, need_weights=False)[0]


        ln2_x = self.ln2(x)
        mlp_output = self.mlp(ln2_x)
        x = x + mlp_output

        return x


class WeightDiffusionTransformer(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        if cfg.model.use_cross_attention and not cfg.model.conditioning:
            raise ValueError("Cross attention is enabled but no conditioning tensor is provided. Please set cfg.model.conditioning to True.")


        # Input encoding

        # Timestep embedding
        self.time_projector = self._get_projector(
            cfg.model.temb_dim, 2*cfg.model.hidden_dim, cfg.model.hidden_dim
        )

        # Weight embedding
        self.weight_projector = self._get_projector(
            cfg.model.weight_dim, 2*cfg.model.hidden_dim, cfg.model.hidden_dim
        )

        # (Optional) condition embedding
        if cfg.model.conditioning:
            self.cond_projector = self._get_projector(
                cfg.model.cond_dim, 2*cfg.model.hidden_dim, cfg.model.hidden_dim
            )

        else:
            self.cond_projector = None

        self.num_tokens = cfg.model.num_weight_tokens
        if cfg.model.type == 'baseline':
            self.pos_emb = nn.Embedding(self.num_tokens+2, cfg.model.hidden_dim)
            self.num_tokens += 2
        else:
            self.pos_emb = nn.Embedding(self.num_tokens, cfg.model.hidden_dim)
        self.ln_in = nn.LayerNorm(cfg.model.hidden_dim)


        if cfg.model.type != 'baseline':
            self.weight_emb = nn.Embedding(3, cfg.model.hidden_dim)
            self.register_buffer('weight_ids', torch.cat([
                        torch.full((48,), 0, dtype=torch.long),
                        torch.full((1,), 1, dtype=torch.long),
                        torch.full((12,), 2, dtype=torch.long)
                        ])
            )

        self.total_tokens = self.num_tokens + 2 if self.cond_projector is not None else self.num_tokens +1

        num_token_types = 3 if not self.cfg.model.use_cross_attention else 2

        if cfg.model.type != 'baseline': 
            if self.cfg.model.use_cross_attention or self.cond_projector is None: 
                self.register_buffer(
                    "token_type_ids",torch.cat([
                        torch.full((self.num_tokens,), 0, dtype=torch.long),
                        torch.full((1,), 1, dtype=torch.long)
                        ]))
            else:
                self.register_buffer(
                    "token_type_ids",torch.cat([
                        torch.full((self.num_tokens,), 0, dtype=torch.long),
                        torch.full((1,), 1, dtype=torch.long),
                        torch.full((1,), 2, dtype=torch.long)
                        ]))
            self.token_type_emb = nn.Embedding(num_token_types, cfg.model.hidden_dim)

        # Vanilla transformer
        self.blocks = nn.ModuleList(
            [
                EncoderLayer(cfg.model.hidden_dim, cfg.model.num_heads,use_cross_attention=cfg.model.use_cross_attention)
                for _ in range(cfg.model.num_layers)
            ]
        )

        # Output decoding
        self.ln_out = nn.LayerNorm(cfg.model.hidden_dim)
        self.out_proj = self._get_projector(
            cfg.model.hidden_dim, cfg.model.hidden_dim, 1
        )

        # TODO: honestly do not know how much this matters...
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _get_projector(in_dim, hidden_dim, out_dim):

        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )


    def forward(self, x_w:Tensor, t:Tensor, cond:Tensor=None) -> Tensor:
        """
        Perform a forward pass through the GPT model.

        Args:
            x_w (torch.Tensor): The input tensor of shape B x Nw x Dw
            t (torch.Tensor): The timestep tensor of shape B x Dt
            cond (torch.Tensor): The condition tensor of shape B x Nc x Dc
        """        
        # Input encoding

        # Timestep embedding
        t_emb = timestep_embedding(t, self.cfg.model.temb_dim).to(x_w.device) # -> B x Dt
        t_emb = self.time_projector(t_emb).unsqueeze(1) # -> B x 1 x D

        # Weight embedding + positional embedding
        pos_emb = self.pos_emb(torch.arange(self.num_tokens,device=x_w.device))
        if self.cfg.model.type != 'baseline':
            weight_emb = self.weight_emb(self.weight_ids) # -> 1 x Nw x D
            x_w_emb = [
                ((self.weight_projector[i](x_w[:, i]))+pos_emb[i]+weight_emb[i]).unsqueeze(1) for i in range(self.cfg.model.num_weight_tokens)
            ] # -> B x Nw x D
        else:
            x_w_emb = self.weight_projector(x_w) # -> B x Nw x D

       # (Optional) condition embedding
        cond_emb = torch.zeros_like(t_emb,device=x_w.device)
        if self.cond_projector is not None and cond is not None:
            cond_emb = self.cond_projector(cond) # -> B x 1 x HiddenDim
        if self.cfg.model.use_cross_attention:
            if self.cfg.model.type != 'baseline':
                x_in = torch.cat(x_w_emb + [t_emb], dim=1)
            else:
                weights_seq_len = x_w_emb.shape[1]
                x_in = torch.cat([x_w_emb, t_emb], dim=1) 
                x_in = x_in + pos_emb.unsqueeze(0)[:,:x_in.shape[1],:]
        else:
            if self.cfg.model.type != 'baseline':
                x_in = torch.cat(x_w_emb + [t_emb, cond_emb], dim=1)
            else:
                weights_seq_len = x_w_emb.shape[1]
                x_in = torch.cat([x_w_emb, t_emb, cond_emb], dim=1)
                x_in = x_in + pos_emb.unsqueeze(0)[:,:x_in.shape[1],:]

        # Add token type embedding
        if self.cfg.model.type != 'baseline':
            x_in = self.token_type_emb(self.token_type_ids).unsqueeze(0) + x_in

        # Layer normalization
        x_in = self.ln_in(x_in)
            
        # Vanilla transformer encoder
        for block in self.blocks:
            x_in = block(x_in,cond=None if not self.cfg.model.use_cross_attention or cond is None else cond_emb)
        

        # Output decoding
        x_out = self.out_proj(x_in)[:,:weights_seq_len,:].squeeze(-1) # -> B x Nw x 1 -> B x Nw

        return x_out


if __name__ == "__main__":

    from omegaconf import OmegaConf
    config_path = "configs/train.yaml"
    cfg = OmegaConf.load(config_path)

    model = WeightDiffusionTransformer(cfg)
    B = 32
    x_w = torch.randn(B, 61, 96)  
    t = torch.randint(0, 1000, (B,))  # Random timesteps
    if cfg.model.conditioning:
        cond = torch.randn(B, 1, 256)  # Random conditioning tensor
    else:
        cond = None

    print(f"x_w shape: {x_w.shape}")
    print(f"t shape: {t.shape}")
    if cond is not None:
        print(f"cond shape: {cond.shape}")
    else:
        print("No conditioning tensor provided.")

    output = model(x_w, t, cond=cond)
    print(output.shape)  # Should be (32, 4, 512)