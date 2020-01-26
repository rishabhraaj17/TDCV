import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from datasets import TripletDataset
import numpy as np

data = TripletDataset(data_dir='../dataset/', train=True, online=False)
data_loader = DataLoader(dataset=data, batch_size=8, shuffle=True, num_workers=0)
for d in data_loader:
    print(d)
# plt.imshow(data[5000][0])
# plt.show()
print('done')
