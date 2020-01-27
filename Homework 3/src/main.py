import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DescriptorNetwork
from loss import TripletAndPairLoss

import matplotlib.pyplot as plt
from datasets import TripletDataset, TemplateDataset
from utils.train_utils import get_template_loader
import numpy as np

# TODO: try LR schedulers

# data = TripletDataset(data_dir='../dataset/', train=True, online=False)
# data_loader = DataLoader(dataset=data, batch_size=8, shuffle=True, num_workers=0)
# m = DescriptorNetwork()
# l = TripletAndPairLoss(batch_size=8)
# for d in data_loader:
#     anchor, puller, pusher = d
#
#     X = torch.cat([anchor[0].float(), puller[0].float(), pusher[0].float()]).permute(dims=[0, 3, 1, 2])
#     y = m(X)
#     loss = l(y)
#     print(loss)

# plt.imshow(data[5000][0])
# plt.show()
db = get_template_loader()
a = next(iter(db))
c, d, e = a[0].float(), a[1].float(), torch.tensor(a[2:]).float()
print('done')


# x = torch.randn(size=(96, 3, 64, 64))
# m = DescriptorNetwork()
# l = TripletAndPairLoss(batch_size=96)
# out = m(x)
# loss = l(out)
# print(out.size())
# print(loss)

