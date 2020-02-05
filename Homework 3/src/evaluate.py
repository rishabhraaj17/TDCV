import os

import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils.data_utils import del_theta_quaternion
from utils.vis_utils import plot_confusion_matrix


class Evaluator(object):
    def __init__(self, dataset_mean, dataset_dev, writer=None, k_neighbour_count=5):
        self.k_neighbour_count = k_neighbour_count
        self.dataset_dev = dataset_dev
        self.dataset_mean = dataset_mean
        self.writer = writer

    def test(self, model, test_loader, nearest_neighbours, knn_dataset_label, knn_dataset_pose, device):
        model.eval()

        test_accuracy = 0
        angular_differences = []
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                image, label, pose = data[0].permute(dims=[0, 3, 1, 2]).float(), data[1].float(), torch.tensor(data[2:]).float()
                image, label, pose = image.to(device), label.to(device), pose.to(device)
                val_descriptor = model(image)

                prediction_distance, prediction_idx = nearest_neighbours.kneighbors(val_descriptor.numpy(), self.k_neighbour_count)
                prediction_label = int(np.round(knn_dataset_label[prediction_idx[0][0]]))
                label = label.int().item()
                y_pred.append(prediction_label)
                y_true.append(label)
                if prediction_label == label:
                    test_accuracy += 1
                    angular_difference = del_theta_quaternion(knn_dataset_pose[prediction_idx[0][0]],
                                                              pose.numpy())
                    angular_differences.append(angular_difference)

        test_accuracy /= len(test_loader.dataset)

        if self.writer is not None:
            self.writer.add_scalar('Test/Accuracy', test_accuracy)

        print(f'Test Accuracy: {test_accuracy}')
        return test_accuracy, angular_differences, y_true, y_pred

    @staticmethod
    def load_template_descriptor(load_path, device):
        knn_dataset = torch.load(load_path, map_location=device)
        return knn_dataset.reshape(-1, 21)

    def build_template_space(self, load_path, device):
        knn_dataset = self.load_template_descriptor(load_path=load_path, device=device)
        knn_dataset_features = knn_dataset[:, 0:16]
        knn_dataset_label = knn_dataset[:, 16:17].astype(np.int)
        knn_dataset_pose = knn_dataset[:, 17:]
        nearest_neighbours = NearestNeighbors(n_neighbors=self.k_neighbour_count, algorithm='ball_tree').fit(knn_dataset_features)
        return nearest_neighbours, knn_dataset_label, knn_dataset_pose

    def build_histogram(self, angular_differences, len_test_dataset, save_path=None):
        bins = list(range(0, 181, 10))
        hist, bin_edges = np.histogram(angular_differences, bins=bins)
        bins = [0, 10, 20, 40, 180]
        hist, bin_edges = np.histogram(angular_differences, bins=bins)
        hist = hist / len_test_dataset
        hist = np.cumsum(hist)
        fig, ax = plt.subplots()
        ax.bar(range(len(hist)), hist, width=1)
        ax.set_xticks([i for i, j in enumerate(hist)])
        ax.set_xticklabels(['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)])
        if save_path is not None:
            plt.savefig(f'{save_path}test_{datetime.now().strftime("%m-%d-%Y_T_%H")}/Test_Histogram_Plot_{datetime.now().strftime("%H-%S")}.png')
        plt.show()
        return fig

    def test_embeddings_visualizer(self, model, test_loader, device, save_path=None):
        knn_dataset = np.ndarray([len(test_loader.dataset), 16 + 1 + 4], dtype=np.float64)
        test_embeddings = np.ndarray([0, 16], dtype=np.float64)
        tensorboard_labels = []
        tensorboard_images = torch.zeros(size=(0, 3, 64, 64))
        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                image, label, pose = data[0].permute(dims=[0, 3, 1, 2]).float(), data[1].float(), torch.tensor(data[2:]).float()
                image, label, pose = image.to(device), label.to(device), pose.to(device)
                descriptor = model(image)
                knn_dataset[batch, :] = np.append(descriptor.numpy()[0, :], np.append(label.numpy(), pose.numpy()))
                test_embeddings = np.vstack((test_embeddings, descriptor.numpy()))
                tensorboard_labels.append(label.int().item())
                tensorboard_image = torch.from_numpy(image.numpy().transpose((0, 2, 3, 1)) * self.dataset_dev + self.dataset_mean).float().permute(0, 3, 1, 2)
                tensorboard_images = torch.cat((tensorboard_images, tensorboard_image))

        if self.writer is not None:
            self.writer.add_embedding(test_embeddings, metadata=tensorboard_labels, label_img=tensorboard_images.int(), tag='Test_Embeddings/model_' + str(model.created_at))

        if save_path is not None:
            os.makedirs(f'{save_path}{datetime.now().strftime("%m-%d-%Y_T_%H")}', exist_ok=True)
            torch.save(knn_dataset, f'{save_path}{datetime.now().strftime("%m-%d-%Y_T_%H")}/Test_Embeddings_model_{str(model.created_at)}_{datetime.now().strftime("%H-%S")}.pt')

        return knn_dataset

    def evaluate(self, model, test_loader, template_descriptor_path, plot_normalized_confusion_mat, save_path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        classes = ["ape", "benchvise", "cam", "cat", "duck"]
        torch.backends.cudnn.benchmark = True

        model = model.to(device)
        nearest_neighbours, knn_dataset_label, knn_dataset_pose = self.build_template_space(load_path=template_descriptor_path, device=device)
        test_accuracy, angular_differences, y_true, y_pred = self.test(model=model, test_loader=test_loader, nearest_neighbours=nearest_neighbours,
                                                                       knn_dataset_label=knn_dataset_label, knn_dataset_pose=knn_dataset_pose, device=device)
        # normalize must be one of {'true', 'pred', 'all', None}
        confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        np.set_printoptions(precision=2)
        plt.figure()

        if plot_normalized_confusion_mat:
            plot_confusion_matrix(confusion_matrix=confusion_mat, classes=classes,
                                  normalize=True, test=True)
        else:
            plot_confusion_matrix(confusion_matrix=confusion_mat, classes=classes,
                                  normalize=False, test=True)
        os.makedirs(f'{save_path}test_{datetime.now().strftime("%m-%d-%Y_T_%H")}', exist_ok=True)
        plt.savefig(f'{save_path}test_{datetime.now().strftime("%m-%d-%Y_T_%H")}/Confusion_Matrix_epoch_{datetime.now().strftime("%H-%S")}.png')
        self.build_histogram(angular_differences=angular_differences, len_test_dataset=len(test_loader.dataset), save_path=save_path)
        self.test_embeddings_visualizer(model=model, test_loader=test_loader, device=device, save_path=save_path)
