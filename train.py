import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from matplotlib import pyplot as plt

from configs import args
from utils import cal_accuracy


def train_model(model, data, train_mask, val_mask, test_mask):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    loss = nn.CrossEntropyLoss()
    # Train procedure
    best_epoch = 0
    best_val_acc = 0
    best_val_test_acc = 0
    epoch_train_acc = []
    epoch_val_acc = []
    epoch_test_acc = []
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

            # Train infos
            infos = {
                'Epoch': epoch,
                'TrainAcc': '{:.3}'.format(train_acc.item()),
                'ValAcc': '{:.3}'.format(val_acc.item()),
                'TestAcc': '{:.3}'.format(test_acc.item())
            }

            tq.set_postfix(infos)
            epoch_train_acc.append(float(train_acc))
            epoch_val_acc.append(float(val_acc))
            epoch_test_acc.append(float(test_acc))
            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                best_val_test_acc = test_acc

    print('Best performance at epoch {} with test acc {:.3f}'.format(
        best_epoch, best_val_test_acc))

    if args.experiment == 2:
        return best_epoch, epoch_train_acc, epoch_val_acc, epoch_test_acc
    return best_val_test_acc


def draw_learning_curve(max_epoch, best_epoch, train_acc, val_acc, test_acc, model_name, color_mod: int = 0):
    epochs = np.array([_ for _ in range(max_epoch)])
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    if color_mod == 0:
        plt.plot(epochs, train_acc, color="green", label="{}:train acc".format(model_name))
        plt.plot(epochs, val_acc, color="blue", label="{}:val acc".format(model_name))
        plt.plot(epochs, test_acc, color="orange", label="{}:test acc".format(model_name))
        plt.axvline(best_epoch, linestyle='--', color="red")
    elif color_mod == 1:
        plt.plot(epochs, train_acc, color="lightgreen", label="{}:train acc".format(model_name))
        plt.plot(epochs, val_acc, color="lightblue", label="{}:val acc".format(model_name))
        plt.plot(epochs, test_acc, color="yellow", label="{}:test acc".format(model_name))
        plt.axvline(best_epoch, linestyle='--', color="pink")
    plt.legend(loc=1)
