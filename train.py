import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import argparse
from typing import Optional

from models import load_model
from dataloader import PTB_XL_Dataset

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--use-cuda", action='store_true')
    
    parser.add_argument("--model", default='LSTM-AE')
    parser.add_argument("--num-attn-heads", type=Optional[int], default=None)
    
    parser.add_argument("--freq", type=int, default=500)
    parser.add_argument("--seconds", type=int, default=2)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
        
    return parser

def print_setup(device, args):
    print("########[Settings]########\n")
    print(f"  [device]: {device}")
    print(f"  [model]: {args.model}")
    print(f"  [num-attn-heads]: {args.num_attn_heads}")
    print(f"  [epochs]: {args.epochs}")
    print(f"  [lr]: {args.lr}")
    print(f"  [batch size]: {args.batch_size}")
    print("\n##########################")
    
def main(args):
    device = 'cpu'
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        
    print_setup(device, args)
    
    # Load Model
    model = load_model(model_name=args.model,
                       num_attn_heads=args.num_attn_heads)
    
    # Load Dataset
    train_ds = PTB_XL_Dataset(data_dir='data/PTB-XL',
                              metadata_path='data/PTB-XL/ptbxl_database.csv',
                              mode='train',
                              freq=args.freq,
                              seconds=args.seconds)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    
    sample = next(iter(train_dl))
    print(sample.shape)  
    # Loss Function (Reconstruction Loss: MAE Loss)
    # loss_fn = nn.MAELoss()
    
    # Optimizer
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ECG Anomaly Detection', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)