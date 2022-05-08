import argparse

arg_parser = argparse.ArgumentParser()

# Dataset to use
arg_parser.add_argument('--dataset',
                        type=str,
                        default='cora',
                        choices=['cora', 'chameleon', 'texas'])
# Random seed
arg_parser.add_argument('--seed', type=int, default=42)
# Hidden dimension of neural networks
arg_parser.add_argument('--hidden_dim', type=int, default=128)
# Epochs to train
arg_parser.add_argument('--epoch', type=int, default=200)
# Learning rate
arg_parser.add_argument('--lr', type=float, default=1e-3)
# Weight decay
arg_parser.add_argument('--wd', type=float, default=1e-5)
# Dropout probability
arg_parser.add_argument('--dp', type=float, default=0.5)

args = arg_parser.parse_args()