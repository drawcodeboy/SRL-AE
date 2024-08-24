from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import argparse
import time
from typing import Optional
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
    parser.add_argument("--num-attn-heads", type=int, default=None)
    
    # Dataset
    parser.add_argument("--dataset", default="ECG5000")
    parser.add_argument("--data-root-dir", default="data/ECG5000")
    parser.add_argument("--freq", type=int, default=500)
    parser.add_argument("--seconds", type=int, default=2)
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=8)
    
    # Paths
    parser.add_argument("--weights-filename", default="LSTM-AE_001.pth")
    parser.add_argument("--savefig", action='store_true')
    
    return parser

def print_setup(device, args):
    print("=======================[Settings]========================")
    print(f"\n  [GPU]")
    print(f"  |-[device]: {device}")
    print(f"\n  [MODEL]")
    print(f"  |-[model]: {args.model}")
    print(f"  |-[num-attn-heads]: {args.num_attn_heads}")
    print(f"\n  [DATA]")
    print(f"  |-[dataset(ALL)]: {args.dataset}")
    print(f"  |-[data-root-dir(ALL)]: {args.data_root_dir}")
    print(f"  |-[freq(PTB-XL)]: {args.freq}")
    print(f"  |-[seconds(PTB-XL)]: {args.seconds}")
    print(f"\n  [HYPERPARAMETERS]")
    print(f"  |-[batch size]: {args.batch_size}")
    print(f"\n [PATHS]")
    print(f"  |-[SAVE FIG]: {args.savefig}")
    print(f"  |-[WEIGHTS FILENAME]: {args.weights_filename}")
    print("\n=========================================================")
    
def print_metrics(metrics_dict:dict):
    print()
    for key, value in metrics_dict.items():
        print(f"    [{key}]: {value:.8f}")
    print()
    
def main(args):
    device = 'cpu'
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        
    print_setup(device, args)
    
    # Load Model
    model = load_model(model_name=args.model,
                       num_attn_heads=args.num_attn_heads).to(device)
    ckpt = torch.load(os.path.join('saved/weights', args.weights_filename),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"It was trained {ckpt['epochs']} EPOCHS")
    
    # Load Dataset
    # train for threshold
    train_ds = load_dataset(dataset=args.dataset,
                            data_dir=args.data_root_dir,
                            metadata_path=os.path.join(args.data_root_dir, "ptbxl_database.csv"),
                            mode='train',
                            freq=args.freq,
                            seconds=args.seconds)
    print(f"train samples: {len(train_ds)}")
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    
    test_ds = load_dataset(dataset=args.dataset,
                           data_dir=args.data_root_dir,
                           metadata_path=os.path.join(args.data_root_dir, "ptbxl_database.csv"),
                           mode='test',
                           freq=args.freq,
                           seconds=args.seconds)
    print(f"test samples: {len(test_ds)}")
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size)
    
    # Loss Function (Reconstruction Loss: MAE Loss)
    loss_fn = MAELoss().to(device)
    
    # Get Threshold
    print("======================================================")
    start_time = int(time.time())
    loss_mean, loss_std, threshold = validate(model, train_dl, loss_fn, None, device)
    threshold_time = int(time.time() - start_time)
    print(f"Getting Threshold Time: {threshold_time//60:02d}m {threshold_time%60:02d}s")
    print(f"loss mean: {loss_mean:.6f}, loss std: {loss_std:.6f}")
    print(f"<<Threshold: {threshold:.6f}>>")
        
    # Test
    start_time = int(time.time())
    metrics_dict, outputs, targets, losses = evaluate(model, test_dl, loss_fn, threshold, device)
    test_time = int(time.time()) - start_time
    print(f"Test Time: {test_time//60:02d}m {test_time%60:02d}s")
    print_metrics(metrics_dict)
    
    outputs = outputs.reshape(-1)
    targets = targets.reshape(-1)
    
    losses = np.stack((np.array(losses), targets), axis=-1)
    
    loss_normal = []
    loss_abnormal = []
    
    for loss, label in losses:
        if label == 0:
            loss_normal.append(loss)
        else:
            loss_abnormal.append(loss)

    plt.figure(figsize=(12, 4))
    plt.xlim(0, 50)
    bins=300
    plt.hist(loss_normal, bins=bins)
    plt.hist(loss_abnormal, bins=bins)
    plt.title(f"{args.model} reconstruction loss")
    
    plt.text(35, 95, f"train loss threshold: {threshold:.6f}")
    plt.text(35, 85, f"train loss mean: {loss_mean:.4f}")
    plt.text(35, 75, f"train loss std: {loss_std:.4f}")
    
    if args.savefig == True:
        plt.savefig(f"figures/{args.model}_reconstruction.jpg", dpi=300)
    else:
        plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ECG Anomaly Detection Test', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)