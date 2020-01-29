import torch
import torch.nn as nn

from datetime import datetime


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class DescriptorNetwork(nn.Module):
    def __init__(self):
        super(DescriptorNetwork, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(in_channels=16, out_channels=7, kernel_size=5, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     Lambda(lambda x: x.view(x.size(0), -1)),
                                     nn.Linear(in_features=1008, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(in_features=256, out_features=16))
        self.created_at = f'model_{datetime.now().strftime("%m-%d-%Y_T_%H-%M-%S")}'

    def forward(self, x):
        return self.network(x)

    def save(self, path, save_state_dict=True):
        if save_state_dict:
            torch.save(self.state_dict(), path + self.created_at + '_state_dict.pt')
        else:
            torch.save(self, path + self.created_at + '.pt')
