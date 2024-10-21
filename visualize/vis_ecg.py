import os, sys
sys.path.append(os.getcwd()) # Execute 

from dataloader import load_dataset
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--index1", type=int, default=0)
    parser.add_argument("--index2", type=int, default=1356)
    parser.add_argument("--savefig", action='store_true')
    
    return parser

def cls_mapping(target):
    cls_dict = {
        0: "Normal", # --index=0
        1: "R-on-T Premature Ventricular Contraction",
        2: "Premature Ventricular Contraction", # --index=1356
        3: "Supraventricular Premature beat",
        4: "Unknown Beat"
    }
    return cls_dict[target]

def transform_to_df(ds, label=0):
    df = pd.DataFrame({
        'timestep': [],
        'mV': []
    })
    
    for sample, cls in ds:
        if cls == label:
            df_temp = pd.DataFrame({
                'timestep': np.arange(0, len(sample)),
                'mV': sample
            })
            df = pd.concat([df, df_temp])
            
    return df
    
def main(args):
    ds = load_dataset(preprocessing=False,
                      mode='test')
    sample_normal, target_normal = ds[args.index1]
    sample_anomaly, target_anomaly = ds[args.index2]
    
    x = np.arange(0, len(sample_normal))
    
    label_normal = cls_mapping(target_normal)
    label_anomaly = cls_mapping(target_anomaly)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_xticks([i * 35 for i in range(0, 5)])
    axes[0].set_title(f"Normal", fontsize=18, pad=10)
    axes[0].set_ylim(-5, 2)
    axes[0].set_xlabel('Time step', fontsize=15)
    axes[0].set_ylabel('mV', fontsize=15)
    axes[0].grid()
    
    axes[1].set_xticks([i * 35 for i in range(0, 5)])
    axes[1].set_title(f"Anomaly (PVC)", fontsize=18, pad=10)
    axes[1].set_ylim(-5, 2)
    axes[1].set_xlabel('Time step', fontsize=15)
    axes[1].set_ylabel('mV', fontsize=15)
    axes[1].grid()
    
    data_normal = transform_to_df(ds, label=0)
    data_anomaly = transform_to_df(ds, label=2)
    
    sns.lineplot(data=data_normal, x="timestep", y="mV", errorbar='sd', ax=axes[0], color='g') # 
    sns.lineplot(data=data_anomaly, x="timestep", y="mV", errorbar='sd', ax=axes[1], color='r') # 
    sns.despine()
    
    plt.tight_layout()
    
    if args.savefig:
        plt.savefig(f"figures/ECG_visualization.jpg", dpi=300)
    else:
        plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize ECG", parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)