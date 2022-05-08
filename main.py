import torch
import torch.nn as nn
import torch_geometric.transforms as T

from tqdm import tqdm
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork

import pdb
from configs import args
from models.gcn import GCNNet
from utils import set_seed_global, cal_accuracy

if __name__ == '__main__':

    print(args)
    set_seed_global(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Fetch datasets, data, and masks
    if args.dataset in ['cora']:
        dataset = Planetoid(root='./datasets',
                            name=args.dataset,
                            transform=None)
    elif args.dataset in ['texas']:
        dataset = WebKB(root='./datasets', name=args.dataset, transform=None)
    elif args.dataset in ['chameleon']:
        dataset = WikipediaNetwork(root='./datasets',
                                   name=args.dataset,
                                   transform=None)
    else:
        raise NotImplementedError
    data = dataset[0].to(device)
    # Print these masks to see what they are
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    # Create model, optimizer, loss function,
    model = GCNNet(in_dim=data.num_node_features,
                   hidden_dim=args.hidden_dim,
                   out_dim=dataset.num_classes,
                   dropout=args.dp).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    loss = nn.CrossEntropyLoss()

    # Train procudure
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
