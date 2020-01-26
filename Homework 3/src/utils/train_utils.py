import torch
from datetime import datetime
import shutil

from torch.utils.data import SubsetRandomSampler

from datasets import TripletDataset


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


def get_train_valid_loader(batch_size,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=True):
    if 0 <= valid_size <= 1:
        assert valid_size, "dataset split is incorrect"

    dataset = TripletDataset(data_dir='../../dataset/', train=False, online=False)

    # Validation and Train Split
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(valid_size * num_samples)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    return train_loader, valid_loader


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True):
    dataset = TripletDataset(data_dir='../../dataset/', train=False, online=False)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


def get_train_loader(batch_size,
                     shuffle=True,
                     num_workers=0,
                     pin_memory=True):
    dataset = TripletDataset(data_dir='../../dataset/', train=True, online=False)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader
