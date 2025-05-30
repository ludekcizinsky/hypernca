import torch
import torch.nn as nn

from helpers.gmn.model_arch_graph import sequential_to_arch, arch_to_graph
from helpers.gmn.graph_models import EdgeMPNN
from helpers.model import timestep_embedding
from helpers.utils import unflatten_params, get_pretrained_sequential
from torch_geometric.data import Data, Batch


class GraphDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        edge_in_dim = 6 + cfg.model.temb_dim 
        self.gnn = EdgeMPNN(
            node_in_dim=3, 
            edge_in_dim=edge_in_dim, 
            hidden_dim=cfg.model.graph_encoder.hidden_dim, 
            node_out_dim=cfg.model.graph_encoder.node_out_dim, 
            edge_out_dim=cfg.model.graph_encoder.edge_out_dim, 
            num_layers=cfg.model.graph_encoder.num_layers, 
            dropout=cfg.model.graph_encoder.dropout, 
            reduce=cfg.model.graph_encoder.reduce,
            global_in_dim=cfg.model.cond_dim
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.model.graph_encoder.edge_out_dim, cfg.model.graph_encoder.edge_out_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.model.graph_encoder.edge_out_dim // 2, 1)
        )


        # self.weight_encoder = WeightDiffusionTransformer(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z_t, t, cond): 
        # time embedding
        t_emb = timestep_embedding(t, self.cfg.model.temb_dim) # -> B x Dt

        # conditioning embedding
        graph_cond = cond.squeeze(1)

        data_list = []
        for i in range(len(z_t)):

            # construct graph
            flat_weights = z_t[i]
            w1, b1, w2 = unflatten_params(flat_weights, 48, 96, 12)
            mlp = get_pretrained_sequential(w1, b1, w2)
            arch = sequential_to_arch(mlp)
            graph = arch_to_graph(arch)
            x, edge_index, edge_attr = graph

            # add time embedding to each edge
            t_emb_i = t_emb[i] # (D, )
            t_emb_i_per_edge = t_emb_i.repeat(edge_attr.size(0), 1).to(edge_attr.device)
            edge_attr = torch.cat([edge_attr, t_emb_i_per_edge], dim=1)
            u = graph_cond[i].to(edge_attr.device).unsqueeze(0)
            batch = torch.zeros(x.size(0), dtype=torch.long).to(edge_attr.device)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)



        # pass through GNN
        _, edge_attr_out = self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr.to(self.device), batch.u.to(self.device), batch.batch.to(self.device))

        # reshape back to (B, E, out_dim):
        B = len(data_list)
        E = data_list[0].edge_index.size(1)   # number of edges per graph
        D = edge_attr_out.size(1)             # out_dim
        edge_attr_batched = edge_attr_out.view(B, E, D)

        result = self.head(edge_attr_batched).squeeze(-1)

        return result