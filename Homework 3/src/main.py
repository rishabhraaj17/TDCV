import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from model import DescriptorNetwork
from loss import TripletAndPairLoss
from train import Solver
from evaluate import Evaluator

from utils.data_utils import get_train_mean_and_std
from utils.train_utils import get_template_loader, get_train_loader, get_test_valid_loader


data_dir = "../dataset/"
classes = ("ape", "benchvise", "cam", "cat", "duck")
batch_size = 32
k_neighbour_count = 7
epochs = 25

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
writer = SummaryWriter()

template_loader = get_template_loader(batch_size=1, shuffle=True, num_workers=8)
train_loader = get_train_loader(batch_size=batch_size, shuffle=True, num_workers=8)
test_loader, valid_loader = get_test_valid_loader(batch_size=1, shuffle=True, num_workers=8)

# m, std = get_train_mean_and_std(data_dir, classes)
m, std = [37.11641652, 41.60517811, 48.8279312], [52.97441827, 57.09825236, 67.82412182]

model = DescriptorNetwork()
loss_function = TripletAndPairLoss(batch_size=batch_size)


def train():
    solver = Solver(loss_function=loss_function, optim_args=adam_args, optimizer=optim, scheduler=scheduler, dataset_mean=m,
                    dataset_dev=std, writer=writer, k_neighbour_count=k_neighbour_count)

    solver.solve(model=model, train_loader=train_loader, val_loader=valid_loader, template_loader=template_loader, num_epochs=epochs, save_path=save_path, save_model=True, save_state_dict=True,
                 log_checkpoint=False, checkpoint_dir=checkpoint_save_path, plot_normalized_confusion_mat=False, resume_training=False, resume_checkpoint_file=None)


def test(net, template_descriptor_pth):
    evaluator = Evaluator(dataset_mean=m, dataset_dev=std, writer=writer, k_neighbour_count=k_neighbour_count)
    evaluator.evaluate(model=net, test_loader=test_loader, template_descriptor_path=template_descriptor_pth, plot_normalized_confusion_mat=False, save_path=save_path)


if __name__ == '__main__':
    template_descriptor_path = f'../models/01-29-2020_T_21/template_descriptor_epoch_24_21-53.pt'
    is_training = False

    if is_training:
        train()
    else:
        state_dict_path = f'../models/model_01-29-2020_T_21-10-55_state_dict.pt'
        state_dict = torch.load(state_dict_path)
        model: torch.nn.Module = DescriptorNetwork()
        model.load_state_dict(state_dict)
        test(net=model, template_descriptor_pth=template_descriptor_path)