import torch

def save_model(model, optimizer, scheduler, epoch, filepath):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, model, optimizer=None, scheduler=None):

    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully.")
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded successfully.")

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded successfully.")
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Resumed training from epoch {epoch}")
    return epoch
