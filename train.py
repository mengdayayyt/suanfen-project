import torch
import torch.nn as nn
import torch_geometric.transforms as T

from tqdm import tqdm

from configs import args
from utils import set_seed_global, cal_accuracy


def train_model(model, data, train_mask, val_mask, test_mask):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    loss = nn.CrossEntropyLoss()
    # Train procedure
    best_epoch = 0
    best_val_acc = 0
    best_val_test_acc = 0
    with tqdm(range(args.epoch)) as tq:
        for epoch in tq:
            model.train()
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss_t = loss(output[train_mask], data.y[train_mask])
            loss_t.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            train_acc = cal_accuracy(pred, data.y, train_mask)
            with torch.no_grad():
                model.eval()
                output = model(data.x, data.edge_index)
                loss_t = loss(output[val_mask], data.y[val_mask])
                val_loss = loss_t
                pred = output.argmax(dim=1)
                val_acc = cal_accuracy(pred, data.y, val_mask)
                test_acc = cal_accuracy(pred, data.y, test_mask)

            # Print infos
            infos = {
                'Epoch': epoch,
                'TrainAcc': '{:.3}'.format(train_acc.item()),
                'ValAcc': '{:.3}'.format(val_acc.item()),
                'TestAcc': '{:.3}'.format(test_acc.item())
            }

            tq.set_postfix(infos)
            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                best_val_test_acc = test_acc

    print('Best performance at epoch {} with test acc {:.3f}'.format(
        best_epoch, best_val_test_acc))

    return best_val_test_acc