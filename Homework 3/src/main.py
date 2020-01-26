import torch
import matplotlib.pyplot as plt
from datasets import TripletDataset
import numpy as np

data = TripletDataset(data_dir='../dataset/', train=True, online=False)
a = data[0]
# plt.imshow(data[5000][0])
# plt.show()
print('done')
