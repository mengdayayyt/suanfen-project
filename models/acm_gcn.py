import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ACM_GCN_Framework, ACM_GAT_Framework


class ACM_GCN(nn.Module):
    def __init__(
                    self, in_dim, out_dim, hidden_dim: int = 64,
                    dropout: float = 0.5, improve: bool = False, mix: bool = False
                ):
        super().__init__()
        self.acm, self.acm2 = None, None

        if improve:
            self.acm = ACM_GCN_Framework(in_dim=in_dim, out_dim=hidden_dim, mix=mix)
            self.acm2 = ACM_GCN_Framework(in_dim=hidden_dim, out_dim=out_dim, mix=mix)
        else:
            self.acm = ACM_GCN_Framework(in_dim, out_dim, mix=mix)
        assert self.acm is not None

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.acm(x, edge_index)
        if self.acm2 is not None:
            x = F.relu(x)
            x = self.acm2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ACM_GAT(nn.Module):
    def __init__(
                    self, in_dim, out_dim, hidden_dim: int = 64,
                    dropout: float = 0.5, improve: bool = False, mix: bool = False
                ):
        super().__init__()
        self.acm, self.acm2 = None, None

        if improve:
            self.acm = ACM_GAT_Framework(in_dim=in_dim, out_dim=hidden_dim, mix=mix)
            self.acm2 = ACM_GAT_Framework(in_dim=hidden_dim, out_dim=out_dim, mix=mix)
        else:
            self.acm = ACM_GAT_Framework(in_dim, out_dim, mix=mix)
        assert self.acm is not None

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.acm(x, edge_index)
        if self.acm2 is not None:
            x = F.relu(x)
            x = self.acm2(x, edge_index)
        return F.log_softmax(x, dim=1)