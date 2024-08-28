import torch
import numpy as np
from .metrics import *

__all__ = ['train_one_epoch', 'validate', 'evaluate', 'opt_threshold']

def train_one_epoch(model_name, epoch, model, dataloader, optimizer, loss_fn, scheduler, device):
    model.train()
    
    total_loss = []
    for batch_idx, (batch, targets) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        batch = batch.to(device)
        
        outputs = model(batch)
        
        # Reconstruction Loss
        loss = loss_fn(outputs, batch)
        
        # Regularization term is model use Sparsity parameter
        if model_name[:4] == 'Spar':
            loss += model.enc_sparsity_loss
        
        # debug
        # print(loss)
        
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
def evaluate(model, dataloader, loss_fn, threshold, device, threshold_method=3):
    '''
        Normal = 0
        Others = 1, 2, 3, 4
    '''
    model.eval()
    
    total_outputs, total_targets = [], []
    total_loss = []
    
    for batch_idx, (batch, targets) in enumerate(dataloader, start=1):
        batch = batch.to(device)
        
        outputs = model(batch)
        
        loss = loss_fn(outputs, batch)
        
        total_loss.extend(loss.tolist())
        
        # thereshold보다 작으면 0(Normal), 아니면 1(Abnormal)
        outputs = torch.where((loss <= threshold), 0, 1).view(loss.shape[0], -1)
        
        # 다른 label(1, 2, 3, 4)를 1로 intergrate
        targets = torch.where((targets == 0), 0, 1).view(targets.shape[0], -1)
        
        total_outputs.extend(outputs.tolist())
        total_targets.extend(targets.tolist())
        
        print(f"\rTest: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    total_outputs = np.array(total_outputs)
    total_targets = np.array(total_targets)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 
               'Sensitivity', 'Specificity', 'F1-Score']
    
    metrics_dict = None
    normal_loss, abnormal_loss = None, None
    
    if threshold_method != 3:
        # Don't Use!
        metrics_dict = get_metrics(metrics, total_outputs, total_targets)
    else:
        metrics_dict, threshold, normal_loss, abnormal_loss = opt_threshold(threshold, np.array(total_loss), total_targets.reshape(-1))
    
    return metrics_dict, total_loss, threshold, normal_loss, abnormal_loss

def opt_threshold(init_threshold, total_loss, total_targets, interval:float=1):
    total_info = np.stack((total_loss, total_targets)) # (N, ), (N, ) -> (2, N)
    
    total_info = total_info[:, total_info[0].argsort()] # sort according to threshold
    
    normal_loss = total_info[0, (total_info[1] == 0)]
    abnormal_loss = total_info[0, (total_info[1] == 1)]
    
    min_thd = total_info[0, (total_info[0]) < (init_threshold-interval)][-1]
    max_thd = total_info[0, (total_info[0]) > (init_threshold+interval)][0]
    
    # Threshold candidates 구하기
    threshold_indices = np.where(((total_info[0] >= min_thd) & (total_info[0] <= max_thd)))
    thresholds = total_info[0][threshold_indices]
    
    confusion = dict()
    metrics = ['Accuracy', 'Precision', 'Recall', 
               'Sensitivity', 'Specificity', 'F1-Score']
    
    opt_metrics_dict = None
    opt_thd = None
    
    for thd in thresholds:
        confusion['TP'] = (normal_loss <= thd).sum()
        confusion['FN'] = (normal_loss > thd).sum()
        confusion['TN'] = (abnormal_loss > thd).sum()
        confusion['FP'] = (abnormal_loss <= thd).sum()
        
        metrics_dict = get_metrics(metrics, confusion=confusion)
        
        if opt_metrics_dict is None: # init
            opt_metrics_dict = metrics_dict
            opt_thd = thd
            
        elif metrics_dict['Accuracy'] > opt_metrics_dict['Accuracy']:
            opt_metrics_dict = metrics_dict
            opt_thd = thd
    
    return opt_metrics_dict, opt_thd, normal_loss, abnormal_loss