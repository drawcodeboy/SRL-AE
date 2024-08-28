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
    print("=========================================================")
    start_time = int(time.time())
    loss_mean, loss_std, init_threshold = validate(model, train_dl, loss_fn, None, device)
    threshold_time = int(time.time() - start_time)
    print(f"Getting Threshold Time: {threshold_time//60:02d}m {threshold_time%60:02d}s")
    print(f"loss mean: {loss_mean:.6f}, loss std: {loss_std:.6f}")
    print(f"<<Init Threshold: {init_threshold:.6f}>>")
        
    # Test
    print("=========================================================")
    start_time = int(time.time())
    metrics_dict, losses, opt_threshold, norm_loss, abnorm_loss = evaluate(model, test_dl, loss_fn, init_threshold, device)
    test_time = int(time.time()) - start_time
    print(f"Test Time: {test_time//60:02d}m {test_time%60:02d}s")
    print(f"<<Optimized Threshold: {opt_threshold:.6f}>>")
    print_metrics(metrics_dict)
    
    norm_loss2 = norm_loss[norm_loss <= opt_threshold]
    abnorm_loss2 = abnorm_loss[abnorm_loss > opt_threshold]

    # Settings
    plt.figure(figsize=(12, 4))
    plt.xlim(0, 50)
    plt.ylim(0, 120)
    bins=300
    plt.hist(norm_loss, bins=bins)
    plt.hist(abnorm_loss, bins=bins)
    plt.title(f"{args.model} reconstruction loss")
    
    # Threshold
    plt.axvline(x=opt_threshold, color='r')
    
    # Mean Distance
    plt.plot([np.mean(norm_loss), np.mean(abnorm_loss)], [30, 30], marker='o', color='g')
    plt.plot([np.mean(norm_loss2), 
              np.mean(abnorm_loss2)], [20, 20], marker='o', color='m')
    
    # Metrics
    plt.text(35, 95, f"Optimized Loss Threshold: {opt_threshold:.6f}")
    plt.text(35, 85, f"Normal Loss Mean: {np.mean(norm_loss):.4f}")
    plt.text(35, 75, f"Abormal Loss Mean: {np.mean(abnorm_loss):.4f}")
    plt.text(35, 65, f"Mean Distance(all): {abs(np.mean(abnorm_loss)-np.mean(norm_loss)):.4f}")
    plt.text(35, 55, f"Mean Distance(threshold): {abs(np.mean(abnorm_loss2)-np.mean(norm_loss2)):.4f}")
    
    plt.text(35, 45, f"Acc: {metrics_dict['Accuracy']*100:.4f}%")
    plt.text(35, 35, f"F1-Score: {metrics_dict['F1-Score']*100:.4f}")
    
    if args.savefig == True:
        plt.savefig(f"figures/{args.model}_reconstruction.jpg", dpi=300)
    else:
        plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ECG Anomaly Detection Test', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)