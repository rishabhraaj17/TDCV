import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from datasets import TripletDataset
import numpy as np

# data = TripletDataset(data_dir='../dataset/', train=True, online=False)
# data_loader = DataLoader(dataset=data, batch_size=8, shuffle=True, num_workers=0)
# for d in data_loader:
#     print(d)
# # plt.imshow(data[5000][0])
# # plt.show()
# print('done')
x = torch.randn(size=(8, 3, 64, 64))
c1 = nn.Conv2d(3, 16, 8, stride=1, padding=0)
m1 = nn.MaxPool2d(2, stride=2)
c2 = nn.Conv2d(16, 7, 5, stride=1, padding=0)
m2 = nn.MaxPool2d(2, stride=2)
l1 = nn.Linear(1008, 256)
l2 = nn.Linear(256, 16)
out = c1(x)
out = m1(out)
out = c2(out)
out = m2(out)
out = out.view(out.size(0), -1)
out = l2(l1(out))
print(out.size())

