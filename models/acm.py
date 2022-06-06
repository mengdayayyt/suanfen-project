import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ACM_GCN_Filter


class ACM_Framework(nn.Module):
    def __init__(
            self, in_dim, out_dim, ACM_Filter: type,
            mix: bool = False, temprature: float = 3.0
    ):
        super().__init__()
        self.HP = ACM_Filter(in_dim, out_dim, filterbank="HP")
        self.LP = ACM_Filter(in_dim, out_dim, filterbank="LP")
        self.I = ACM_Filter(in_dim, out_dim, filterbank="I")

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


class HighOrder_ACM_Framework(nn.Module):
    def __init__(
            self, in_dim, out_dim, ACM_Filter: type,
            mix: bool = False, temprature: float = 3.0
    ):
        super().__init__()
        self.HP = ACM_Filter(in_dim, out_dim, filterbank="HP")
        self.LP = ACM_Filter(in_dim, out_dim, filterbank="LP")
        self.I = ACM_Filter(in_dim, out_dim, filterbank="I")
        self.HP2 = ACM_Filter(in_dim, out_dim, filterbank="HP2")
        self.LP2 = ACM_Filter(in_dim, out_dim, filterbank="LP2")

        self.mix = mix
        self.T = temprature
        if mix:
            self.lin_mix = nn.Linear(5, 5)

        self.lin_h = nn.Linear(out_dim, 1)
        self.lin_l = nn.Linear(out_dim, 1)
        self.lin_h2 = nn.Linear(out_dim, 1)
        self.lin_l2 = nn.Linear(out_dim, 1)
        self.lin_i = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        H_hp = F.relu(self.HP(x, edge_index))
        H_lp = F.relu(self.LP(x, edge_index))
        H_hp2 = F.relu(self.HP2(x, edge_index))
        H_lp2 = F.relu(self.LP2(x, edge_index))
        H_i = F.relu(self.I(x, edge_index))

        alpha_h = torch.sigmoid(self.lin_h(H_hp))
        alpha_l = torch.sigmoid(self.lin_l(H_lp))
        alpha_h2 = torch.sigmoid(self.lin_h2(H_hp2))
        alpha_l2 = torch.sigmoid(self.lin_l2(H_lp2))
        alpha_i = torch.sigmoid(self.lin_i(H_i))

        if self.mix:
            hli = torch.cat([alpha_i, alpha_h, alpha_l, alpha_h2, alpha_l2], dim=1)
            hli = F.softmax(self.lin_mix(hli) / self.T, dim=1)
            alpha_i, alpha_h, alpha_l, alpha_h2, alpha_l2 = hli[:, 0], hli[:, 1], hli[:, 2], hli[:, 3], hli[:, 4]
        else:
            alpha_h, alpha_l, alpha_i, alpha_h2, alpha_l2 = \
                alpha_h[:, 0], alpha_l[:, 0], alpha_i[:, 0], alpha_h2[:, 0], alpha_l2[:, 0]

        out = torch.matmul(torch.diag(alpha_h), H_hp) + \
              torch.matmul(torch.diag(alpha_l), H_lp) + \
              torch.matmul(torch.diag(alpha_h2), H_hp2) + \
              torch.matmul(torch.diag(alpha_l2), H_lp2) + \
              torch.matmul(torch.diag(alpha_i), H_i)

        return out


class ACM_GNN(nn.Module):
    def __init__(
                    self, in_dim, out_dim, hidden_dim: int = 64,
                    ACM_Framework: type = ACM_Framework, ACM_Filter: type = ACM_GCN_Filter,
                    dropout: float = 0.5, improve: bool = False, mix: bool = False
                ):
        super().__init__()
        self.acm, self.acm2 = None, None

        if improve:
            self.acm = ACM_Framework(in_dim=in_dim, out_dim=hidden_dim, ACM_Filter=ACM_Filter, mix=mix)
            self.acm2 = ACM_Framework(in_dim=hidden_dim, out_dim=out_dim, ACM_Filter=ACM_Filter, mix=mix)
        else:
            self.acm = ACM_Framework(in_dim, out_dim, ACM_Filter=ACM_Filter, mix=mix)
        assert self.acm is not None

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.acm(x, edge_index)
        if self.acm2 is not None:
            x = F.relu(x)
            x = self.acm2(x, edge_index)
        return F.log_softmax(x, dim=1)


