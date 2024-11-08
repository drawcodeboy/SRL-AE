from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import argparse
import time
import os

from models import load_model
from dataloader import load_dataset
from utils import *
from utils.engine import get_latent
from utils import tsne_generator

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
    parser.add_argument("--batch-size", type=int, default=8)
    
    # Paths
    parser.add_argument("--weights-filename")
    parser.add_argument("--savefig", action='store_true')
    
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
    model = load_model(model_name=args.model).to(device)
    ckpt = torch.load(os.path.join('saved/weights', args.weights_filename),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"It was trained {ckpt['epochs']} EPOCHS")
    
    # Load Dataset
    test_ds = load_dataset(dataset=args.dataset,
                           data_dir=args.data_root_dir,
                           metadata_path=os.path.join(args.data_root_dir, "ptbxl_database.csv"),
                           mode='test',
                           freq=args.freq,
                           seconds=args.seconds)
    print(f"test samples: {len(test_ds)}")
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size)
        
    # Get Latent Variables
    print("=========================================================")
    start_time = int(time.time())
    latents, targets = get_latent(model, test_dl, device)
    test_time = int(time.time()) - start_time
    print(f"Extract Latent Time: {test_time//60:02d}m {test_time%60:02d}s")
    
    start_time = int(time.time())
    tsne_generator(latents, targets, args.model)
    test_time = int(time.time()) - start_time
    print(f"T-SNE Time: {test_time//60:02d}m {test_time%60:02d}s")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Latent Variables Extraction', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)