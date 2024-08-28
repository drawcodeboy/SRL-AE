import numpy as np
from matplotlib import pyplot as plt

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--model", default="LSTM-AE", help="Select Model")
    
    return parser

def main(args):
    train_loss_path = f"saved/losses/train_loss_{args.model}.npy"
    val_loss_path = f"saved/losses/val_loss_{args.model}.npy"
    
    train_loss = np.load(train_loss_path)
    val_loss = np.load(val_loss_path)
    
    x = np.arange(1, len(train_loss)+1)
    
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Loss', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)