import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ACM_Framework


class ACM_GCN_Single(nn.Module):
    def __init__(self, in_dim, out_dim, mix: bool = False):
        super().__init__()
        self.HP = ACM_Framework(in_dim, out_dim, filterbank="HP")
        self.LP = ACM_Framework(in_dim, out_dim, filterbank="LP")
        self.I = ACM_Framework(in_dim, out_dim, filterbank="I")
        self.lin_h = nn.Linear(out_dim, 1)
        self.lin_l = nn.Linear(out_dim, 1)
        self.lin_i = nn.Linear(out_dim, 1)
        self.mix = mix

    def forward(self, x, edge_index):
        H_hp = F.relu(self.HP(x, edge_index))
        H_lp = F.relu(self.LP(x, edge_index))
        H_i = F.relu(self.I(x, edge_index))
        sigma_h = self.lin_h(H_hp)
        sigma_l = self.lin_l(H_lp)
        sigma_i = self.lin_i(H_i)
        out = torch.diag(sigma_h) * H_hp + torch.diag(sigma_l) * H_lp + torch.diag(sigma_i) * H_i

        return F.log_softmax(out, dim=1)


class ACM_GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout: float = 0.5):
        super().__init__()
        self.acm1 = ACM_GCN_Single(in_dim, hidden_dim)
        self.acm2 = ACM_GCN_Single(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.acm1(x, edge_index)
        x = self.acm2(x, edge_index)
        return F.log_softmax(x, dim=1)

