import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import time
import os

from models import load_model, MAELoss
from dataloader import load_dataset
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # GPU
    parser.add_argument("--use-cuda", action='store_true')
    
    # Model
    parser.add_argument("--model", default='LSTM-AE')
    
    # Dataset
    parser.add_argument("--dataset", default="ECG5000")
    parser.add_argument("--data-root-dir", default="data/ECG5000")
    parser.add_argument("--freq", type=int, default=500)
    parser.add_argument("--seconds", type=int, default=2)
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    
    # Save Paths
    parser.add_argument("--save-weights-dir", default="saved/weights")
    parser.add_argument("--save-losses-dir", default="saved/losses")
    
    return parser

def print_setup(device, args):
    print("=======================[Settings]========================")
    print(f"\n  [GPU]")
    print(f"  |-[device]: {device}")
    print(f"\n  [MODEL]")
    print(f"  |-[model]: {args.model}")
    print(f"\n  [DATA]")
    print(f"  |-[dataset(ALL)]: {args.dataset}")
    print(f"  |-[data-root-dir(ALL)]: {args.data_root_dir}")
    print(f"  |-[freq(PTB-XL)]: {args.freq}")
    print(f"  |-[seconds(PTB-XL)]: {args.seconds}")
    print(f"\n  [HYPERPARAMETERS]")
    print(f"  |-[epochs]: {args.epochs}")
    print(f"  |-[lr]: {args.lr}")
    print(f"  |-[weight decay]: {args.weight_decay}")
    print(f"  |-[batch size]: {args.batch_size}")
    print(f"\n [SAVE PATHS]")
    print(f"  |-[SAVE WEIGHTS DIR]: {args.save_weights_dir}")
    print(f"  |-[SAVE LOSSES DIR]: {args.save_losses_dir}")
    print("\n=======================================================")
    
def main(args):
    device = 'cpu'
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        
    print_setup(device, args)
    
    # Load Model
    model = load_model(model_name=args.model).to(device)
    
    # Load Dataset
    train_ds = load_dataset(dataset=args.dataset,
                            data_dir=args.data_root_dir,
                            metadata_path=os.path.join(args.data_root_dir, "ptbxl_database.csv"),
                            mode='train',
                            freq=args.freq,
                            seconds=args.seconds)
    print(f"train samples: {len(train_ds)}")
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    
    val_ds = load_dataset(dataset=args.dataset,
                          data_dir=args.data_root_dir,
                          metadata_path=os.path.join(args.data_root_dir, "ptbxl_database.csv"),
                          mode='val',
                          freq=args.freq,
                          seconds=args.seconds)
    print(f"validation samples: {len(val_ds)}")
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=args.batch_size)
    
    # Loss Function (Reconstruction Loss: MAE Loss)
    loss_fn = MAELoss().to(device)
    
    # Optimizer
    p = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer,
                                 mode='min',
                                 factor=0.5,
                                 patience=5,
                                 min_lr=1e-7)
    
    total_train_loss = []
    total_val_loss = []
    
    min_val_loss = 10000.
    
    for current_epoch in range(0, args.epochs):
        current_epoch += 1
        print("======================================================")
        print(f"Epoch: [{current_epoch:03d}/{args.epochs:03d}]")
        print()
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss = train_one_epoch(args.model, current_epoch, model, train_dl, optimizer, loss_fn, scheduler, device)
        train_time = int(time.time() - start_time)
        print(f"Training Time: {train_time//60:02d}m {train_time%60:02d}s")
        print()
        
        # Validation
        start_time = int(time.time())
        val_loss, _, _ = validate(model, val_dl, loss_fn, scheduler, device) # loss의 mean, std 값, threshold 리턴
        val_time = int(time.time()) - start_time
        print(f"Validation Reconstruction Loss: {val_loss:.6f}")
        print(f"Validation Time: {val_time//60:02d}m {val_time%60:02d}s")
        print()
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model_ckpt(model, args.model, current_epoch, args.save_weights_dir)
        
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)
        save_loss_ckpt(args.model, current_epoch, total_train_loss, total_val_loss, args.save_losses_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ECG Anomaly Detection Train', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)