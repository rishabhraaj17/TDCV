import torch
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils.data_utils import del_theta_quaternion
from utils.train_utils import save_checkpoint, load_checkpoint
from utils.vis_utils import plot_confusion_matrix


class Solver(object):
    def __init__(self, loss_function, optim_args, dataset_mean, dataset_dev, optimizer=optim.Adam, writer_train=None, writer_val=None, writer_descriptor=None):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.optim_args = optim_args
        self.dataset_dev = dataset_dev
        self.dataset_mean = dataset_mean
        self.writer_val = writer_val
        self.writer_train = writer_train
        self.writer_descriptor = writer_descriptor

    def train_step(self, model, optimizer, train_loader, current_epoch, device):
        model.train()

        train_loss = 0

        for batch, data in enumerate(train_loader):
            anchor, puller, pusher = data
            anchor, puller, pusher = anchor.float().to(device), puller.float().to(device), pusher.float().to(device)

            X = torch.cat([anchor[0], puller[0], pusher[0]]).permute(dims=[0, 3, 1, 2])

            optimizer.zero_grad()
            y_pred = model(X)
            triplet_loss, pair_loss = self.loss_function(y_pred)
            loss = triplet_loss + pair_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch % 10 == 0 and self.writer_train is not None:
                self.writer_train.add_scalar('Train/Triplet_Loss', triplet_loss.item(), batch)
                self.writer_train.add_scalar('Train/Pair_Loss', pair_loss.item(), batch)

        train_loss /= len(train_loader.dataset)

        if self.writer_train is not None:
            self.writer_train.add_scalar('Training Loss', train_loss, current_epoch)

        print(f'Epoch: {current_epoch}, Training Loss: {train_loss}')

    def validation_step(self, model, val_loader, nearest_neighbours, knn_dataset_label, knn_dataset_pose, current_epoch, device):
        model.eval()

        val_accuracy = 0
        angular_differences = []
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch, data in enumerate(val_loader):
                image, label, pose = data[0].permute(dims=[0, 3, 1, 2]).float(), data[1].float(), torch.tensor(data[2:]).float()
                image, label, pose = image.to(device), label.to(device), pose.to(device)
                val_descriptor = model(image)

                prediction_distance, prediction_idx = nearest_neighbours.kneighbors(val_descriptor.numpy())
                prediction_label = int(np.round(knn_dataset_label[prediction_idx[0][0]]))
                y_pred.append(prediction_label)
                y_true.append(label.item())
                if prediction_label == label.item():
                    val_accuracy += 1
                    angular_difference = del_theta_quaternion(knn_dataset_pose[prediction_idx[0][0]],
                                                              pose.numpy())
                    angular_differences.append(angular_difference)

        val_accuracy /= len(val_loader.dataset)

        if self.writer_val is not None:
            self.writer_val.add_scalar('Validation Accuracy', val_accuracy, current_epoch)

        print(f'Epoch : {current_epoch}, Validation Accuracy: {val_accuracy}')
        return val_accuracy, angular_differences, y_true, y_pred

    def build_template_space_prep(self, model, template_loader, current_epoch, device, save_path=None):
        knn_dataset = np.ndarray([len(template_loader.dataset), 16 + 1 + 4], dtype=np.float64)
        template_embeddings = np.ndarray([0, 16], dtype=np.float64)
        tensorboard_labels = []
        tensorboard_images = torch.zeros(size=(0, 3, 64, 64))
        with torch.no_grad():
            for batch, data in enumerate(template_loader):
                image, label, pose = data[0].permute(dims=[0, 3, 1, 2]).float(), data[1].float(), torch.tensor(data[2:]).float()
                image, label, pose = image.to(device), label.to(device), pose.to(device)
                descriptor = model(image)
                knn_dataset[batch, :] = np.append(descriptor.numpy()[0, :], label.numpy(), pose.numpy())
                template_embeddings = np.vstack((template_embeddings, descriptor.numpy()))
                tensorboard_labels.append(label.item())
                tensorboard_image = torch.from_numpy(image.numpy() * self.dataset_dev + self.dataset_mean).int().permute(0, 3, 1, 2).squeeze()
                torch.cat((tensorboard_images, tensorboard_image.view(-1, *tensorboard_image.shape)))

            if self.writer_descriptor is not None:
                self.writer_descriptor.add_embedding(template_embeddings, metadata=tensorboard_labels, label_img=tensorboard_images, tag='validation_epoch_' + str(current_epoch))

        if save_path is not None:
            torch.save(knn_dataset, f'{save_path}/{datetime.now().strftime("%m-%d-%Y")}/template_descriptor_epoch_{current_epoch}_{datetime.now().strftime("%H-%S")}.pt')

        return knn_dataset

    def build_template_space(self, model, template_loader, current_epoch, device, save_path=None):
        knn_dataset = self.build_template_space_prep(model, template_loader, current_epoch, device, save_path=save_path)
        knn_dataset_features = knn_dataset[:, 0:16]
        knn_dataset_label = knn_dataset[:, 16:17]
        knn_dataset_pose = knn_dataset[:, 17:]
        nearest_neighbours = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(knn_dataset_features)
        return nearest_neighbours, knn_dataset_label, knn_dataset_pose

    def build_histogram(self, angular_differences, len_valid_dataset, current_epoch, save_path=None):
        bins = list(range(0, 181, 10))
        hist, bin_edges = np.histogram(angular_differences, bins=bins)
        print('Histogram for each degree', hist)
        bins = [0, 10, 20, 40, 180]
        hist, bin_edges = np.histogram(angular_differences, bins=bins)
        hist = hist / len_valid_dataset
        print(hist)
        hist = np.cumsum(hist)
        print(hist)
        fig, ax = plt.subplots()
        # Plot the histogram heights against integers on the x axis
        ax.bar(range(len(hist)), hist, width=1)
        # Set the ticks to the middle of the bars
        ax.set_xticks([i for i, j in enumerate(hist)])
        # Set the xticklabels to a string that tells us what the bin edges were
        ax.set_xticklabels(['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)])
        if save_path is not None:
            plt.savefig(f'{save_path}/{datetime.now().strftime("%m-%d-%Y")}/Test_Histogram_Plot_epoch_{current_epoch}_{datetime.now().strftime("%H-%S")}.png')
        plt.show()

    def solve(self, model, train_loader, val_loader, template_loader, num_epochs, save_path, save_model=False, save_state_dict=True,
              log_checkpoint=False, checkpoint_dir=None, resume_training=False, resume_checkpoint_file=None, plot_normalized_confusion_mat=True):
        optimizer = self.optimizer(model.parameters(), **self.optim_args)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        classes = ["ape", "benchvise", "cam", "cat", "duck"]
        torch.backends.cudnn.benchmark = True

        model = model.to(device)

        resume_epoch = 0
        if resume_training:
            model, optimizer, resume_epoch = load_checkpoint(resume_checkpoint_file, model, optimizer)
            print(f'Resuming Training from epoch {resume_epoch - 1}/{num_epochs} epochs')

        best_val_accuracy = 1e-8

        if not resume_epoch:
            print('START TRAINING')

        for epoch in range(resume_epoch, num_epochs):
            self.train_step(model=model, train_loader=train_loader, current_epoch=epoch, optimizer=optimizer, device=device)
            nearest_neighbours, knn_dataset_label, knn_dataset_pose = self.build_template_space(model=model, template_loader=template_loader, current_epoch=epoch,
                                                                                                device=device, save_path=save_path)
            val_accuracy, angular_differences, y_true, y_pred = self.validation_step(model=model, val_loader=val_loader, nearest_neighbours=nearest_neighbours,
                                                                                     knn_dataset_label=knn_dataset_label, knn_dataset_pose=knn_dataset_pose, current_epoch=epoch, device=device)
            confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
            np.set_printoptions(precision=2)
            plt.figure()

            if plot_normalized_confusion_mat:
                plot_confusion_matrix(confusion_matrix=confusion_mat, classes=classes,
                                      normalize=True,
                                      title='Normalized confusion matrix')
            else:
                plot_confusion_matrix(confusion_matrix=confusion_mat, classes=classes,
                                      normalize=False,
                                      title='Without normalization confusion matrix')

            plt.savefig(f'{save_path}/{datetime.now().strftime("%m-%d-%Y")}/Confusion_Matrix_epoch_{epoch}_{datetime.now().strftime("%H-%S")}.png')
            self.build_histogram(angular_differences=angular_differences, len_valid_dataset=len(val_loader.dataset), current_epoch=epoch, save_path=save_path)

            if log_checkpoint:
                checkpoint_dict = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                save_checkpoint(checkpoint_dict, checkpoint_dir=checkpoint_dir)

            if save_model:
                if val_accuracy > best_val_accuracy:
                    model.save(save_path, save_state_dict=save_state_dict)

        print('TRAINING FINISHED')
