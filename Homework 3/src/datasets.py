import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from utils.data_utils import del_theta_quaternion, get_train_mean_and_std, get_normalized_datasets


class DataSample(object):
    def __init__(self, image: torch.Tensor, label: int, pose: torch.Tensor):
        super(DataSample, self).__init__()
        self.image = image
        self.label = label
        self.pose = pose


class Triplet(object):
    def __init__(self, anchor: DataSample, puller: DataSample, pusher: DataSample):
        super(Triplet, self).__init__()
        self.anchor = anchor
        self.puller = puller
        self.pusher = pusher


class TripletDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, online: bool = False):
        super(TripletDataset, self).__init__()
        self.data_dir = data_dir
        if online:
            classes = ("ape", "benchvise", "cam", "cat", "duck")
            m, std = get_train_mean_and_std(data_dir, classes)
            db, train, test = get_normalized_datasets(data_dir, classes, m, std)
            if train:
                self.dataset = train
            else:
                self.dataset = test
            self.db = db
        else:
            if train:
                self.dataset = torch.load(data_dir + 'S_train.pt')
            else:
                self.dataset = torch.load(data_dir + 'S_test.pt')
            self.db = torch.load(data_dir + 'S_db.pt')

    def get_puller(self, sample: list):
        puller = None
        min_distance = 50000
        for i in range(len(self.db)):
            if sample[1] == self.db[i][1]:
                distance = del_theta_quaternion(q1=sample[2:6], q2=self.db[i][2:6])
                if distance < min_distance:
                    min_distance = distance
                    puller = self.db[i]
        return puller

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        anchor = self.dataset[item]
        puller = self.get_puller(anchor)
        pusher_idx = torch.randint(len(self.db), (1,)).item()
        pusher = self.db[pusher_idx]
        # anchor = DataSample(image=torch.from_numpy(anchor[0]), label=anchor[1], pose=torch.tensor(anchor[2:]))
        # puller = DataSample(image=torch.from_numpy(puller[0]), label=puller[1], pose=torch.tensor(puller[2:]))
        # pusher = DataSample(image=torch.from_numpy(pusher[0]), label=pusher[1], pose=torch.tensor(pusher[2:]))
        # return Triplet(anchor, puller, pusher)
        return anchor, puller, pusher


class DummyDataset(VisionDataset):
    def __init__(self, data_dir: str, dataset_name: str, transform: transforms = None):
        super(DummyDataset, self).__init__(root=data_dir + dataset_name, transform=transform)
        self.data_dir = data_dir
        self.dataset = torch.load(data_dir + dataset_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class TemplateDataset(Dataset):
    def __init__(self, data_dir: str):
        super(TemplateDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = torch.load(data_dir + 'S_db.pt')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class TestDataset(Dataset):
    def __init__(self, data_dir: str):
        super(TestDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = torch.load(data_dir + 'S_test.pt')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class TrainDataset(Dataset):
    def __init__(self, data_dir: str):
        super(TrainDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = torch.load(data_dir + 'S_train.pt')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
