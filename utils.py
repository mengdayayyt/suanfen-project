import torch
import numpy as np


# Force the random seed to be some int.
def set_seed_global(seed: int, force_deter=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if force_deter:
        torch.use_deterministic_algorithms(True)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def cal_accuracy(pred, label, mask):
    return (pred[mask] == label[mask]).sum() / mask.sum()