import os, sys
sys.path.append(os.getcwd()) # Execute 

from dataloader import load_dataset
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--index", type=int, default=0)
    
    return parser

def cls_mapping(target):
    cls_dict = {
        0: "Normal",
        1: "R-on-T Premature Ventricular Contraction",
        2: "Premature Ventricular Contraction", # --index=1356
        3: "Supraventricular Premature beat",
        4: "Unknown Beat"
    }
    return cls_dict[target]

def main(args):
    ds = load_dataset(preprocessing=False,
                      mode='test')
    sample, target = ds[args.index]
    
    x = np.arange(0, len(sample))
    label = cls_mapping(target)
    
    plt.plot(x, sample, color='r')
    plt.title(label)
    
    plt.gca().axes.xaxis.set_visible(False)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize ECG", parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)