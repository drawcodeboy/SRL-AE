import torch
import numpy as np
from .metrics import *

def train_one_epoch(epoch, model, dataloader, optimizer, loss_fn, scheduler, device):
    model.train()
    
    total_loss = []
    for batch_idx, (batch, targets) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        batch = batch.to(device)
        
        outputs = model(batch)
        
        # Reconstruction Loss
        loss = loss_fn(outputs, batch)
        
        total_loss.extend(loss.tolist())
        
        # get mean(reduce loss tensors dim) for back propagation
        loss = torch.mean(loss)
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.8f}", end="")
    print()
        
    return sum(total_loss)/len(total_loss) # One Epoch Mean Loss

@torch.no_grad()
def validate(model, dataloader, loss_fn, scheduler, device, threshold_method=1):
    '''
        - 2가지 용도로 사용 예정
            - (1) validation: 평균 loss 값 리턴 (Overfitting 확인용) 및 스케줄러 step
            - (2) threhosld: loss의 mean, std를 통해서 참고한 논문처럼 threshold 구해서 리턴
    '''
    model.eval()
    
    total_loss = []
    for batch_idx, (batch, targets) in enumerate(dataloader, start=1):
        batch = batch.to(device)
        
        outputs = model(batch)
        
        loss = loss_fn(outputs, batch)
        
        total_loss.extend(loss.tolist())
        print(f"\rValidate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    total_loss = np.array(total_loss)
    
    loss_mean, loss_std = np.mean(total_loss), np.std(total_loss)
    
    if scheduler is not None:
        # Threshold를 구하는 게 아닌 경우
        scheduler.step(loss_mean)
    
    if threshold_method == 1:
        # Threshold Method (1) (https://www.mdpi.com/1999-4893/16/3/152)
        loss_threshold = loss_mean + 2*loss_std
        
    elif threshold_method == 2:
        # Threshold Method (2) (https://arxiv.org/abs/2204.06701)
        loss_threshold = np.max(total_loss)
    
    return loss_mean, loss_std, loss_threshold

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, threshold, device):
    '''
        Normal = 0
        Others = 1, 2, 3, 4
    '''
    model.eval()
    
    total_outputs, total_targets = [], []
    
    for batch_idx, (batch, targets) in enumerate(dataloader, start=1):
        batch = batch.to(device)
        
        outputs = model(batch)
        
        loss = loss_fn(outputs, batch)
        
        # thereshold보다 작으면 0(Normal), 아니면 1(Abnormal)
        outputs = torch.where((loss <= threshold), 0, 1).view(loss.shape[0], -1)
        
        # 다른 label(1, 2, 3, 4)를 1로 intergrate
        targets = torch.where((targets == 0), 0, 1).view(targets.shape[0], -1)
        
        total_outputs.extend(outputs.tolist())
        total_targets.extend(targets.tolist())
        
    total_outputs = np.array(total_outputs)
    total_targets = np.array(total_targets)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 
               'Sensitivity', 'Specificity', 'F1-Score']
    metrics_dict = get_metrics(total_outputs, total_targets, metrics)
    
    return metrics_dict