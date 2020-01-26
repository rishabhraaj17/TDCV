import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset


class DummyDataset(VisionDataset):
    def __init__(self, data_dir: str, dataset_name: str, transform: transforms = None):
        super(DummyDataset, self).__init__(root=data_dir + dataset_name, transform=transform)
        self.data_dir = data_dir
        self.dataset = torch.load(data_dir + dataset_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
