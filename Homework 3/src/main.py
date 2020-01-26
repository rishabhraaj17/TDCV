import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import DescriptorNetwork
from loss import TripletAndPairLoss

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
x = torch.randn(size=(9, 3, 64, 64))
m = DescriptorNetwork()
l = TripletAndPairLoss(batch_size=9)
out = m(x)
loss = l(out)
print(out.size())
print(loss)

