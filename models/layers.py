from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops


class ACM_GCN_Filter(MessagePassing, ABC):
    def __init__(
            self, in_channels: int, out_channels: int, filterbank: str,
            add_self_loops: bool = True, normalize: bool = True,
            **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(ACM_GCN_Filter, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filterbank = filterbank
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.linear = nn.Linear(in_channels, out_channels)

        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None):
        num_nodes = x.size(self.node_dim)
        fill_value = 1
        if isinstance(edge_index, Tensor):
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
            if self.add_self_loops:
                edge_index, tmp_edge_weight = add_remaining_self_loops(
                    edge_index, edge_weight, fill_value, num_nodes)
                assert tmp_edge_weight is not None
                edge_weight = tmp_edge_weight

            row, col = edge_index[0], edge_index[1]
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)

            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        else:
            raise TypeError("Invalid edge_weight type")

        x = self.linear(x)
        if self.filterbank == "I":
            return x

        aggr = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.filterbank == "HP":
            return x - aggr
        elif self.filterbank == "LP":
            return aggr

        raise ValueError("Invalid Filterbank Name")


class ACM_GCN_Framework(nn.Module):
    def __init__(
            self, in_dim, out_dim, hidden_dim: int = 64,
            mix: bool = False, temprature: float = 3.0
    ):
        super().__init__()
        self.HP = ACM_GCN_Filter(in_dim, out_dim, filterbank="HP")
        self.LP = ACM_GCN_Filter(in_dim, out_dim, filterbank="LP")
        self.I = ACM_GCN_Filter(in_dim, out_dim, filterbank="I")

        self.mix = mix
        self.T = temprature
        if mix:
            self.lin_mix = nn.Linear(3, 3)

        self.lin_h = nn.Linear(out_dim, 1)
        self.lin_l = nn.Linear(out_dim, 1)
        self.lin_i = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        H_hp = F.relu(self.HP(x, edge_index))
        H_lp = F.relu(self.LP(x, edge_index))
        H_i = F.relu(self.I(x, edge_index))

        alpha_h = torch.sigmoid(self.lin_h(H_hp))
        alpha_l = torch.sigmoid(self.lin_l(H_lp))
        alpha_i = torch.sigmoid(self.lin_i(H_i))

        if self.mix:
            hli = torch.cat([alpha_h, alpha_l, alpha_i], dim=1)
            hli = F.softmax(self.lin_mix(hli) / self.T, dim=1)
            alpha_h, alpha_l, alpha_i = hli[:, 0], hli[:, 1], hli[:, 2]
        else:
            alpha_h, alpha_l, alpha_i = alpha_h[:, 0], alpha_l[:, 0], alpha_i[:, 0]

        out = torch.matmul(torch.diag(alpha_h), H_hp) + \
              torch.matmul(torch.diag(alpha_l), H_lp) + \
              torch.matmul(torch.diag(alpha_i), H_i)

        return out
