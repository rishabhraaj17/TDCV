import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from model import DescriptorNetwork
from loss import TripletAndPairLoss
from train import Solver
from evaluate import Evaluator

import matplotlib.pyplot as plt
from utils.data_utils import get_train_mean_and_std
from utils.train_utils import get_template_loader, get_test_loader, get_train_loader, get_train_valid_loader


data_dir = "../../dataset/"
classes = ("ape", "benchvise", "cam", "cat", "duck")
batch_size = 32
k_neighbour_count = 5
epochs = 10

save_path = '../models/'
checkpoint_save_path = '../models/checkpoints/'
adam_args = {
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 0
}

optim = torch.optim.Adam
scheduler = ReduceLROnPlateau
writer_train = None
writer_val = None
writer_template = None
writer_test = None

template_loader = get_template_loader(batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = get_train_loader(batch_size=batch_size, shuffle=True, num_workers=0)
_, valid_loader = get_train_valid_loader(batch_size=32, shuffle=True, num_workers=0)

m, std = get_train_mean_and_std(data_dir, classes)

model = DescriptorNetwork()
loss_function = TripletAndPairLoss(batch_size=batch_size)


def train():
    solver = Solver(loss_function=loss_function, optim_args=adam_args, optimizer=optim, scheduler=scheduler, dataset_mean=m,
                    dataset_dev=std, writer_train=writer_train, writer_val=writer_val, writer_descriptor=writer_template, k_neighbour_count=k_neighbour_count)

    solver.solve(model=model, train_loader=train_loader, val_loader=valid_loader, template_loader=template_loader, num_epochs=epochs, save_path=save_path, save_model=False,
                 log_checkpoint=False, checkpoint_dir=checkpoint_save_path, plot_normalized_confusion_mat=True, resume_training=False, resume_checkpoint_file=None)


def test(net, template_descriptor_path):
    test_loader = get_test_loader()
    evaluator = Evaluator(dataset_mean=m, dataset_dev=std, writer_test=writer_test, k_neighbour_count=k_neighbour_count)
    evaluator.evaluate(model=net, test_loader=test_loader, template_descriptor_path=template_descriptor_path, plot_normalized_confusion_mat=True, save_path=save_path)


if __name__ == '__main__':
    date_time = 'dd-mm-yyyy'
    hours_sec = 'HH-SS'
    epoch_to_pick = 0
    template_descriptor_path = f'{save_path}/{date_time}/template_descriptor_epoch_{epoch_to_pick}_{hours_sec}.pt'
    is_training = True
    if is_training:
        train()
    else:
        test(net=model, template_descriptor_path=template_descriptor_path)