import torch
import matplotlib.pyplot as plt
from datasets import DummyDataset
from torchvision import transforms
from utils.data_utils import get_train_mean_and_std
import numpy as np


data_dir = "../dataset/"
classes = ("ape", "benchvise", "cam", "cat", "duck")
m, std = get_train_mean_and_std(data_dir, classes)
print(m, std)

normalize = transforms.Normalize(mean=m, std=std)

# define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
db = torch.load('../dataset/S_train.pt')
# plt.imshow(db[5000][0])
# plt.show()
print(np.max(db[0][0]))

data = DummyDataset(data_dir='../dataset/', dataset_name='S_train.pt', transform=transform)
# plt.imshow(data[5000][0])
# plt.show()
print(np.max(data[0][0]))
print('done')
