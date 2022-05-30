import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops


class ACM_Framework(MessagePassing):
    def __init__(
                    self, in_channels: int, out_channels: int, filterbank: str,
                    add_self_loops: bool = True, normalize: bool = True,
                    **kwargs
                ):
        kwargs.setdefault("aggr", "add")
        super(ACM_Framework, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filterbank = filterbank
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None):
        num_nodes = x.size(self.node_dim)
        fill_value = 1
        if isinstance(edge_index, Tensor):
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
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

        x = self.linear(x)

        aggr = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.filterbank == "HP":
            return x - aggr
        elif self.filterbank == "LP":
            return aggr
        elif self.filterbank == "I":
            return x

        raise ValueError("Invalid Filterbank Name")


