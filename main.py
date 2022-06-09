import torch
import pandas as pd

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from collections import defaultdict
from matplotlib import pyplot as plt

import pdb
from utils import set_seed_global
from configs import args
from train import train_model, draw_learning_curve
from models.gcn import GCNNet
from models.acm import ACM_Framework, HighOrder_ACM_Framework, ACM_GNN
from models.layers import ACM_GCN_Filter, ACM_GAT_Filter, HighOrder_ACM_GCN_Filter


def reset_model(_model):
    for layer in _model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


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

    # Default model
    model = ACM_GNN(in_dim=data.num_node_features,
                    out_dim=dataset.num_classes,
                    hidden_dim=args.hidden_dim,
                    ACM_Framework=HighOrder_ACM_Framework,
                    ACM_Filter=HighOrder_ACM_GCN_Filter,
                    mix=True,
                    improve=False,
                    dropout=args.dp).to(device)

    # Create & Train model
    if args.experiment == 0:
        if args.dataset in ["texas", "chameleon"]:
            reset_model(model)
            val_test_acc = []
            for i in range(10):
                print("{}: Training on mask {}.".format(args.dataset.capitalize(), i + 1))
                train_mask = data.train_mask[:, i]
                val_mask = data.val_mask[:, i]
                test_mask = data.test_mask[:, i]
                val_test_acc.append(train_model(model, data, train_mask, val_mask, test_mask))
            print("{}: Average test_acc {:.3}".format(args.dataset.capitalize(), sum(val_test_acc) / 10))
        else:
            train_model(model, data, data.train_mask, data.val_mask, data.test_mask)

    elif args.experiment == 1:
        models = [
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, name="ACM-GCN").to(device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, mix=False, name="ACM-GCN-no_mix").to(
                device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, improve=True, name="ACM-GCN-2Layer").to(
                device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, ACM_Framework=HighOrder_ACM_Framework,
                    ACM_Filter=HighOrder_ACM_GCN_Filter, name="HOACM-GCN").to(device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, mix=False,
                    ACM_Framework=HighOrder_ACM_Framework,
                    ACM_Filter=HighOrder_ACM_GCN_Filter, name="HOACM-GCN-no_mix").to(device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, ACM_Filter=ACM_GAT_Filter,
                    name="ACM-GAT").to(device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, ACM_Filter=ACM_GAT_Filter,
                    name="ACM-GAT-no_mix").to(device),
            ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, improve=True, name="ACM-GAT-2Layer").to(
                device)
        ]
        result_acc = defaultdict(list)
        for model in models:
            if args.dataset in ["texas", "chameleon"]:
                reset_model(model)
                val_test_acc = []
                for i in range(10):
                    print("{}: Training on mask {}.".format(args.dataset.capitalize(), i + 1))
                    train_mask = data.train_mask[:, i]
                    val_mask = data.val_mask[:, i]
                    test_mask = data.test_mask[:, i]
                    val_test_acc.append(train_model(model, data, train_mask, val_mask, test_mask))
                print("{}: Average test_acc {:.3}".format(args.dataset.capitalize(), sum(val_test_acc) / 10))
                result_acc[model.name].append(float((sum(val_test_acc) / 10)))
            else:
                val_test_acc = train_model(model, data, data.train_mask, data.val_mask, data.test_mask)
                result_acc[model.name].append(float(val_test_acc))
        result = pd.DataFrame(result_acc)
        result.to_csv("./results/acc", index=False, mode="a")

    elif args.experiment == 2:
        GCN_model = GCNNet(in_dim=data.num_node_features, hidden_dim=args.hidden_dim,
                           out_dim=dataset.num_classes, dropout=args.dp).to(device)
        model = ACM_GNN(in_dim=data.num_node_features, out_dim=dataset.num_classes, name="ACM-GCN").to(device)
        if args.dataset in ["texas", "chameleon"]:
            train_mask = data.train_mask[:, 7]
            val_mask = data.val_mask[:, 7]
            test_mask = data.test_mask[:, 7]
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        plt.figure()
        plt.title("Performance of ACM-GCN and GCN on {}".format(args.dataset))
        p1, p2, p3, p4 = train_model(GCN_model, data, train_mask, val_mask, test_mask)
        draw_learning_curve(args.epoch, p1, p2, p3, p4, model_name="GCN")
        p1, p2, p3, p4 = train_model(model, data, train_mask, val_mask, test_mask)
        draw_learning_curve(args.epoch, p1, p2, p3, p4, model_name="ACM-GCN", color_mod=1)
        plt.savefig("./results/ACMGCN_and_GCN_on_{}.png".format(args.dataset))
