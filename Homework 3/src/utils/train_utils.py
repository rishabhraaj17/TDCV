import torch
from datetime import datetime
import shutil


def save_checkpoint(state, is_best=False, checkpoint_dir=None, best_model_dir=None):
    f_path = checkpoint_dir + str(state['current_model']) + datetime.now().strftime("%m-%d-%Y") + '_checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_f_path = best_model_dir / 'best_model.pt'
        shutil.copyfile(f_path, best_f_path)


def load_checkpoint(checkpoint_f_path, model, optimizer):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_f_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']