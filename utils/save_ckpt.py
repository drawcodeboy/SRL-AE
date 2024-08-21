
def save_model_ckpt(model, model_name, current_epoch, save_weights_dir):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = current_epoch
    
    save_name = f"{model_name}-{current_epoch:03d}.pth"
    
    try:
        torch.save(ckpt, save_weights_dir)
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")

def save_loss_ckpt(model_name, current_epoch, train_loss, val_loss, save_losses_dir):
    try:
        np.save(os.path.join(save_losses_dir, f'train_loss_{model_name}_{current_epoch:03d}.npy'), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss')
    
    try:
        np.save(os.path.join(save_losses_dir, f'val_loss_{model_name}_{current_epoch:03d}.npy'), np.array(val_loss))
        print('Save Validation Loss')
    except:
        print('Can\'t Save Validation Loss')