
def train_one_epoch(epoch, model, dataloader, optimizer, loss_fn, scheduler, device):
    model.train()
    
    total_loss = 0.
    for batch_idx, (batch, targets) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        batch = batch.to(device)
        
        outputs = model(batch)
        
        # Reconstruction Loss
        loss = loss_fn(outputs, batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.8f}", end="")
    print()
        
    return (total_loss/len(dataloader)).detach().cpu().numpy() # One Epoch Mean Loss

@torch.no_grad()
def validate():
    '''
        - 2가지 용도로 사용 예정
            - (1) validation: 평균 loss 값 리턴 (Overfitting 확인용)
            - (2) threhosld: loss의 mean, std를 통해서 참고한 논문처럼 threshold 구해서 리턴
    '''
    model.eval()

@torch.no_grad()
def evaludate():
    model.eval()